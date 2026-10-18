"""Evaluate a bit-explicit fixed-point reference for the pursuit SNN.

The reference folds BatchNorm and the learned threshold into each layer's
weights and constant, quantizes those static parameters to power-of-two scales,
and executes the inference equations used by the planned RTL.  Tensor values
that represent integers are kept in float32 only so PyTorch can use its fast
CPU convolution; the script checks that every convolution result is still an
exact integer.  With the supported operand bounds, all convolution sums remain
below float32's exact-integer limit.

This program is for post-training quantization evaluation.  It never modifies
the checkpoint and refuses to overwrite an existing report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Literal

import numpy as np
import sinabs
import torch
from torch.nn import functional as F

from code.pursuit.fixed_event_window.ann.metrics import angular_error_deg, pixel_error
from code.pursuit.fixed_event_window.snn.model import SpikingPursuitSNN
from code.pursuit.fixed_event_window.snn.training import (
    load_persistent_model_state_dict,
)


SCHEMA_VERSION = 1
ScaleMode = Literal["per_layer", "per_output_channel"]
ScalePolicy = Literal["no_clipping", "minimum_weight_mse"]
AlphaRounding = Literal["floor", "toward_zero", "nearest_away"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_half_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    return torch.sign(values) * torch.floor(values.abs() + 0.5)


def signed_limits(bits: int) -> tuple[int, int]:
    if bits < 2:
        raise ValueError("signed values require at least two bits")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def choose_fraction_bits(
    weights: torch.Tensor,
    bits: int,
    mode: ScaleMode,
    policy: ScalePolicy,
    maximum_fraction_bits: int,
) -> torch.Tensor:
    """Choose power-of-two weight scales along output dimension zero."""

    _, maximum_integer = signed_limits(bits)
    flat = weights.detach().abs().reshape(weights.shape[0], -1)
    per_output_maximum = flat.max(dim=1).values
    if mode == "per_layer":
        per_output_maximum = per_output_maximum.max().expand_as(per_output_maximum)
    elif mode != "per_output_channel":
        raise ValueError(f"unsupported scale mode: {mode}")

    no_clipping = torch.zeros_like(per_output_maximum, dtype=torch.int64)
    nonzero = per_output_maximum > 0
    no_clipping[nonzero] = torch.floor(
        torch.log2(maximum_integer / per_output_maximum[nonzero])
    ).to(torch.int64)
    no_clipping = no_clipping.clamp(min=0, max=maximum_fraction_bits)
    if policy == "no_clipping":
        return no_clipping
    if policy != "minimum_weight_mse":
        raise ValueError(f"unsupported scale policy: {policy}")

    minimum_integer, maximum_integer = signed_limits(bits)
    candidates = range(maximum_fraction_bits + 1)
    if mode == "per_layer":
        errors = []
        for fraction_bits in candidates:
            scale = float(1 << fraction_bits)
            integer = round_half_away_from_zero(weights.detach().cpu() * scale)
            integer = integer.clamp(minimum_integer, maximum_integer)
            errors.append(float(((integer / scale - weights.detach().cpu()) ** 2).mean()))
        chosen = min(range(len(errors)), key=errors.__getitem__)
        return torch.full_like(no_clipping, chosen)

    flattened = weights.detach().cpu().reshape(weights.shape[0], -1)
    chosen_values = []
    for output_weights in flattened:
        errors = []
        for fraction_bits in candidates:
            scale = float(1 << fraction_bits)
            integer = round_half_away_from_zero(output_weights * scale)
            integer = integer.clamp(minimum_integer, maximum_integer)
            errors.append(float(((integer / scale - output_weights) ** 2).mean()))
        chosen_values.append(min(range(len(errors)), key=errors.__getitem__))
    return torch.tensor(chosen_values, dtype=torch.int64)


def output_channel_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    return values.view(1, -1, *([1] * (ndim - 2)))


def quantize_static(
    values: torch.Tensor,
    fraction_bits: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    minimum, maximum = signed_limits(bits)
    scale = torch.pow(2.0, fraction_bits.to(torch.float32))
    shape = (len(scale), *([1] * (values.ndim - 1)))
    quantized = round_half_away_from_zero(values.detach().cpu() * scale.view(shape))
    return quantized.clamp(minimum, maximum).to(torch.float32)


@dataclass(frozen=True)
class FixedLayer:
    name: str
    weight_integer: torch.Tensor
    constant_integer: torch.Tensor
    weight_fraction_bits: torch.Tensor
    current_fraction_bits: torch.Tensor
    product_left_shift: torch.Tensor
    pool: bool

    @property
    def scale(self) -> torch.Tensor:
        return torch.pow(2.0, self.current_fraction_bits.to(torch.float64))


@dataclass
class RuntimeCounters:
    accumulator_saturated: dict[str, int]
    spike_saturated: dict[str, int]
    accumulator_maximum_absolute: dict[str, int]
    spike_maximum: dict[str, int]

    @classmethod
    def empty(cls) -> "RuntimeCounters":
        return cls({}, {}, {}, {})

    def record_accumulator(
        self,
        name: str,
        before_saturation: torch.Tensor,
        after_saturation: torch.Tensor,
    ) -> None:
        saturated = int(torch.count_nonzero(before_saturation != after_saturation))
        self.accumulator_saturated[name] = self.accumulator_saturated.get(name, 0) + saturated
        maximum = int(before_saturation.abs().max())
        self.accumulator_maximum_absolute[name] = max(
            self.accumulator_maximum_absolute.get(name, 0), maximum
        )

    def record_spikes(
        self,
        name: str,
        before_saturation: torch.Tensor,
        after_saturation: torch.Tensor,
    ) -> None:
        saturated = int(torch.count_nonzero(before_saturation != after_saturation))
        self.spike_saturated[name] = self.spike_saturated.get(name, 0) + saturated
        maximum = int(before_saturation.max())
        self.spike_maximum[name] = max(self.spike_maximum.get(name, 0), maximum)


class FixedPointPursuitReference:
    def __init__(
        self,
        model: SpikingPursuitSNN,
        *,
        weight_bits: int,
        readout_weight_bits: int,
        constant_bits: int,
        accumulator_bits: int,
        spike_bits: int,
        scale_mode: ScaleMode,
        scale_policy: ScalePolicy,
        maximum_fraction_bits: int,
        state_guard_bits: int,
        alpha_rounding: AlphaRounding,
        stateless_layer_names: tuple[str, ...] = ("L1", "L2"),
    ) -> None:
        if model.alpha != 0.75:
            raise ValueError("fixed reference currently requires alpha=0.75")
        self.weight_bits = weight_bits
        self.readout_weight_bits = readout_weight_bits
        self.constant_bits = constant_bits
        self.accumulator_bits = accumulator_bits
        self.spike_bits = spike_bits
        self.scale_mode = scale_mode
        self.scale_policy = scale_policy
        self.maximum_fraction_bits = maximum_fraction_bits
        self.state_guard_bits = state_guard_bits
        self.alpha_rounding = alpha_rounding
        self.stateless_layer_names = frozenset(stateless_layer_names)
        self.accumulator_minimum, self.accumulator_maximum = signed_limits(
            accumulator_bits
        )
        self.spike_maximum_integer = (1 << spike_bits) - 1
        self.layers: list[FixedLayer] = []

        for index, block in enumerate(model.blocks, start=1):
            inverse_sigma = torch.rsqrt(
                block.batch_norm.running_var + block.batch_norm.eps
            )
            normalized_scale = (
                block.batch_norm.weight
                * inverse_sigma
                / block.spike.threshold
            )
            weight = block.conv.weight * normalized_scale[:, None, None, None]
            constant = (
                block.batch_norm.bias
                - block.batch_norm.weight
                * block.batch_norm.running_mean
                * inverse_sigma
            ) / block.spike.threshold
            self.layers.append(
                self._make_layer(
                    f"L{index}",
                    weight,
                    constant,
                    pool=index <= 5,
                    bits=weight_bits,
                )
            )

        for name, block in (("H1", model.hidden1), ("H2", model.hidden2)):
            weight = block.linear.weight / block.spike.threshold[:, None]
            constant = block.linear.bias / block.spike.threshold
            self.layers.append(
                self._make_layer(
                    name,
                    weight,
                    constant,
                    pool=False,
                    bits=weight_bits,
                )
            )

        readout_weight = model.readout.weight.detach().cpu()
        self.readout_fraction_bits = choose_fraction_bits(
            readout_weight,
            readout_weight_bits,
            scale_mode,
            scale_policy,
            maximum_fraction_bits,
        )
        self.readout_weight_integer = quantize_static(
            readout_weight,
            self.readout_fraction_bits,
            readout_weight_bits,
        )

    def _make_layer(
        self,
        name: str,
        weight: torch.Tensor,
        constant: torch.Tensor,
        *,
        pool: bool,
        bits: int,
    ) -> FixedLayer:
        weight_fraction_bits = choose_fraction_bits(
            weight,
            bits,
            self.scale_mode,
            self.scale_policy,
            self.maximum_fraction_bits,
        )
        current_fraction_bits = (weight_fraction_bits + self.state_guard_bits).clamp(
            max=self.maximum_fraction_bits
        )
        product_left_shift = current_fraction_bits - weight_fraction_bits
        weight_integer = quantize_static(weight, weight_fraction_bits, bits)
        constant_integer = quantize_static(
            constant,
            current_fraction_bits,
            self.constant_bits,
        )
        return FixedLayer(
            name=name,
            weight_integer=weight_integer,
            constant_integer=constant_integer,
            weight_fraction_bits=weight_fraction_bits,
            current_fraction_bits=current_fraction_bits,
            product_left_shift=product_left_shift,
            pool=pool,
        )

    def _saturate_accumulator(
        self,
        values: torch.Tensor,
        layer_name: str,
        counters: RuntimeCounters,
    ) -> torch.Tensor:
        rounded = torch.round(values)
        if not torch.equal(values, rounded):
            maximum_error = float((values - rounded).abs().max())
            raise AssertionError(
                f"{layer_name}: convolution lost integer exactness; "
                f"maximum residual={maximum_error}"
            )
        saturated = rounded.clamp(self.accumulator_minimum, self.accumulator_maximum)
        counters.record_accumulator(layer_name, rounded, saturated)
        return saturated

    def _linear_or_convolution(
        self,
        source_spikes: torch.Tensor,
        layer: FixedLayer,
        counters: RuntimeCounters,
    ) -> torch.Tensor:
        source_spikes = source_spikes.to(torch.float32)
        if layer.weight_integer.ndim == 4:
            current = F.conv2d(
                source_spikes,
                layer.weight_integer,
                bias=None,
                stride=1,
                padding=1,
            )
        else:
            current = F.linear(
                source_spikes,
                layer.weight_integer,
                bias=None,
            )
        rounded_product_sum = torch.round(current)
        if not torch.equal(current, rounded_product_sum):
            maximum_error = float((current - rounded_product_sum).abs().max())
            raise AssertionError(
                f"{layer.name}: product sum lost integer exactness; "
                f"maximum residual={maximum_error}"
            )
        current = rounded_product_sum.to(torch.float64)
        product_shift_scale = torch.pow(
            2.0, layer.product_left_shift.to(torch.float64)
        )
        current = current * output_channel_view(product_shift_scale, current.ndim)
        current = current + output_channel_view(
            layer.constant_integer.to(torch.float64), current.ndim
        )
        current = self._saturate_accumulator(current, layer.name, counters)
        if layer.pool:
            current = F.max_pool2d(current, kernel_size=2, stride=2)
        return current

    def _round_alpha(self, membrane_integer: torch.Tensor) -> torch.Tensor:
        numerator = 3.0 * membrane_integer
        if self.alpha_rounding == "floor":
            return torch.floor(numerator / 4.0)
        if self.alpha_rounding == "toward_zero":
            return torch.trunc(numerator / 4.0)
        if self.alpha_rounding == "nearest_away":
            return round_half_away_from_zero(numerator / 4.0)
        raise ValueError(f"unsupported alpha rounding: {self.alpha_rounding}")

    def _lif(
        self,
        current_integer: torch.Tensor,
        previous_membrane_integer: torch.Tensor,
        layer: FixedLayer,
        counters: RuntimeCounters,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = output_channel_view(layer.scale, current_integer.ndim)
        pre_reset = self._round_alpha(previous_membrane_integer) + current_integer
        unsaturated_spikes = torch.clamp_min(torch.floor(pre_reset / scale), 0.0)
        spikes = unsaturated_spikes.clamp(max=self.spike_maximum_integer)
        counters.record_spikes(layer.name, unsaturated_spikes, spikes)
        membrane = torch.maximum(pre_reset - spikes * scale, -scale)
        return spikes, membrane

    def predict(
        self,
        event_frames: torch.Tensor,
        counters: RuntimeCounters,
        trace: dict[str, np.ndarray] | None = None,
    ) -> torch.Tensor:
        if event_frames.ndim != 4 or tuple(event_frames.shape[1:]) != (2, 96, 128):
            raise ValueError(
                f"expected [batch,2,96,128] event frames, got {tuple(event_frames.shape)}"
            )
        if not torch.equal(event_frames, torch.round(event_frames)):
            raise ValueError("event frames must contain integer counts")

        states: dict[str, torch.Tensor] = {}
        h2_spikes = []
        for timestep in range(2):
            values = event_frames
            for layer in self.layers:
                current = self._linear_or_convolution(values, layer, counters)
                previous = states.get(layer.name)
                if previous is None or layer.name in self.stateless_layer_names:
                    previous = torch.zeros_like(current)
                values, states[layer.name] = self._lif(
                    current,
                    previous,
                    layer,
                    counters,
                )
                if trace is not None:
                    prefix = f"t{timestep}.{layer.name}"
                    trace[f"{prefix}.current_integer"] = (
                        current.cpu().numpy().astype(np.int32)
                    )
                    trace[f"{prefix}.spike_integer"] = (
                        values.cpu().numpy().astype(np.uint8)
                    )
                    trace[f"{prefix}.membrane_integer"] = (
                        states[layer.name].cpu().numpy().astype(np.int32)
                    )
                if layer.name == "L8":
                    values = values.reshape(values.shape[0], 480)
            h2_spikes.append(values)

        temporal_sum = h2_spikes[0] + h2_spikes[1]
        # float64 keeps the integer dot product exact for the readout range.
        output_integer = F.linear(
            temporal_sum.to(torch.float64),
            self.readout_weight_integer.to(torch.float64),
        )
        readout_scale = torch.pow(
            2.0, self.readout_fraction_bits.to(torch.float64)
        ).view(1, -1)
        prediction = (output_integer / (2.0 * readout_scale)).to(torch.float32)
        if trace is not None:
            trace["readout.temporal_sum_integer"] = (
                temporal_sum.cpu().numpy().astype(np.uint16)
            )
            trace["readout.output_integer_before_temporal_divide"] = (
                output_integer.cpu().numpy().astype(np.int64)
            )
            trace["readout.prediction_normalized_xy"] = prediction.cpu().numpy()
        return prediction

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for layer in self.layers:
            arrays[f"{layer.name}.weight_integer"] = (
                layer.weight_integer.cpu().numpy().astype(np.int16)
            )
            arrays[f"{layer.name}.constant_integer"] = (
                layer.constant_integer.cpu().numpy().astype(np.int32)
            )
            arrays[f"{layer.name}.weight_fraction_bits"] = (
                layer.weight_fraction_bits.cpu().numpy().astype(np.uint8)
            )
            arrays[f"{layer.name}.current_and_state_fraction_bits"] = (
                layer.current_fraction_bits.cpu().numpy().astype(np.uint8)
            )
        arrays["readout.weight_integer"] = (
            self.readout_weight_integer.cpu().numpy().astype(np.int16)
        )
        arrays["readout.weight_fraction_bits"] = (
            self.readout_fraction_bits.cpu().numpy().astype(np.uint8)
        )
        return arrays

    def parameter_report(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for layer in self.layers:
            result[layer.name] = {
                "weight_fraction_bits": [
                    int(value) for value in layer.weight_fraction_bits
                ],
                "current_and_state_fraction_bits": [
                    int(value) for value in layer.current_fraction_bits
                ],
                "product_left_shift": [
                    int(value) for value in layer.product_left_shift
                ],
                "weight_integer_minimum": int(layer.weight_integer.min()),
                "weight_integer_maximum": int(layer.weight_integer.max()),
                "weight_zero_fraction": float(
                    torch.count_nonzero(layer.weight_integer == 0)
                    / layer.weight_integer.numel()
                ),
                "constant_integer_minimum": int(layer.constant_integer.min()),
                "constant_integer_maximum": int(layer.constant_integer.max()),
            }
        result["readout"] = {
            "fraction_bits": [int(value) for value in self.readout_fraction_bits],
            "weight_integer_minimum": int(self.readout_weight_integer.min()),
            "weight_integer_maximum": int(self.readout_weight_integer.max()),
            "weight_zero_fraction": float(
                torch.count_nonzero(self.readout_weight_integer == 0)
                / self.readout_weight_integer.numel()
            ),
        }
        return result


def load_model(
    checkpoint_path: Path,
    configuration: dict[str, object],
) -> tuple[SpikingPursuitSNN, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SpikingPursuitSNN(
        alpha=float(configuration["model"]["alpha"]),
        threshold_init=1.0,
        min_v_mem=float(configuration["model"]["minimum_membrane_potential"]),
        backend="sinabs",
        stateless_channels=configuration["model"]["stateless_channels"],
    )
    load_persistent_model_state_dict(model, checkpoint["model"])
    model.eval()
    return model, checkpoint


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    configuration_path = args.configuration or args.checkpoint.parent / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    if int(configuration["input"]["temporal_steps"]) != 2:
        raise ValueError("fixed reference currently requires T=2")
    if configuration["input"]["temporal_organization"] != (
        "repeat_identical_complete_400_event_count_frame"
    ):
        raise ValueError("fixed reference requires repeated complete input")
    expected_stateless = {"L1": list(range(8)), "L2": list(range(16))}
    if configuration["model"]["stateless_channels"] != expected_stateless:
        raise ValueError("this reference currently targets the final L1/L2-stateless model")

    model, checkpoint = load_model(args.checkpoint, configuration)
    reference = FixedPointPursuitReference(
        model,
        weight_bits=args.weight_bits,
        readout_weight_bits=args.readout_weight_bits,
        constant_bits=args.constant_bits,
        accumulator_bits=args.accumulator_bits,
        spike_bits=args.spike_bits,
        scale_mode=args.scale_mode,
        scale_policy=args.scale_policy,
        maximum_fraction_bits=args.maximum_fraction_bits,
        state_guard_bits=args.state_guard_bits,
        alpha_rounding=args.alpha_rounding,
    )

    cache_path = args.cache_dir / f"{args.split}.npz"
    with np.load(cache_path) as cached:
        event_frames = cached["event_frames"].copy()
        labels = cached["labels"].copy()
    if event_frames.ndim != 4 or tuple(event_frames.shape[1:]) != (2, 96, 128):
        raise ValueError(f"unexpected event frame shape: {event_frames.shape}")
    if labels.shape != (len(event_frames), 2):
        raise ValueError(f"unexpected label shape: {labels.shape}")
    if not np.all(event_frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400):
        raise ValueError("every hardware input must contain exactly 400 events")

    counters = RuntimeCounters.empty()
    float_pixel_total = 0.0
    float_angle_total = 0.0
    fixed_pixel_total = 0.0
    fixed_angle_total = 0.0
    prediction_absolute_total = torch.zeros(2, dtype=torch.float64)
    prediction_absolute_maximum = torch.zeros(2, dtype=torch.float64)
    predictions = []
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(event_frames), args.batch_size):
            stop = min(start + args.batch_size, len(event_frames))
            frames = torch.from_numpy(event_frames[start:stop]).to(torch.float32)
            target = torch.from_numpy(labels[start:stop]).to(torch.float32)
            bins = frames.unsqueeze(1).expand(-1, 2, -1, -1, -1)
            float_prediction = model(bins)
            fixed_prediction = reference.predict(frames, counters)

            float_pixel_total += float(pixel_error(float_prediction, target).sum())
            float_angle_total += float(angular_error_deg(float_prediction, target).sum())
            fixed_pixel_total += float(pixel_error(fixed_prediction, target).sum())
            fixed_angle_total += float(angular_error_deg(fixed_prediction, target).sum())
            absolute_difference = (fixed_prediction - float_prediction).abs().to(torch.float64)
            prediction_absolute_total += absolute_difference.sum(dim=0)
            prediction_absolute_maximum = torch.maximum(
                prediction_absolute_maximum,
                absolute_difference.max(dim=0).values,
            )
            if args.predictions_output is not None:
                predictions.append(fixed_prediction.cpu().numpy())

    elapsed = time.monotonic() - started
    samples = len(event_frames)
    if args.predictions_output is not None:
        if args.predictions_output.exists():
            raise FileExistsError(f"refusing to overwrite {args.predictions_output}")
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.predictions_output,
            predictions_normalized_xy=np.concatenate(predictions, axis=0),
            labels_row_col_px=labels,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "bit_explicit_post_training_fixed_point_reference",
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "execution_environment": {
            "backend_for_float_reference": "sinabs_cpu",
            "torch_version": torch.__version__,
            "sinabs_version": sinabs.__version__,
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "epoch": int(checkpoint["epoch"]),
        },
        "configuration": {
            "path": str(configuration_path),
            "sha256": sha256_file(configuration_path),
        },
        "data": {
            "cache_file": str(cache_path),
            "cache_file_sha256": sha256_file(cache_path),
            "split": args.split,
            "samples": samples,
            "batch_size": args.batch_size,
            "selection_warning": (
                "test must not be used to choose a fixed-point configuration"
                if args.split == "test"
                else None
            ),
        },
        "arithmetic_contract": {
            "weight_bits": args.weight_bits,
            "readout_weight_bits": args.readout_weight_bits,
            "constant_bits": args.constant_bits,
            "accumulator_bits": args.accumulator_bits,
            "input_count_bits": 9,
            "spike_bits": args.spike_bits,
            "scale_mode": args.scale_mode,
            "scale_policy": args.scale_policy,
            "scale": "positive power of two per layer or output channel",
            "maximum_fraction_bits": args.maximum_fraction_bits,
            "state_guard_bits": args.state_guard_bits,
            "static_parameter_rounding": "nearest, ties away from zero",
            "alpha": "0.75 implemented as rounded (3 * signed_membrane_integer) / 4",
            "alpha_rounding": args.alpha_rounding,
            "spike": "unsigned floor(max(u,0)) with explicit saturation",
            "reset": "subtractive reset followed by clamp to -1.0",
            "accumulator_saturation": "symmetric two-complement limits after complete dot product",
            "readout": "integer H2 temporal sum times quantized weights, followed by exact divide by two",
        },
        "parameters": reference.parameter_report(),
        "runtime_counters": {
            "accumulator_saturated_values": counters.accumulator_saturated,
            "spike_saturated_values": counters.spike_saturated,
            "accumulator_maximum_absolute_before_saturation": (
                counters.accumulator_maximum_absolute
            ),
            "spike_maximum_before_saturation": counters.spike_maximum,
        },
        "metrics": {
            "float_sinabs_cpu": {
                "mean_pixel_error": float_pixel_total / samples,
                "mean_angular_error_deg": float_angle_total / samples,
            },
            "fixed_point": {
                "mean_pixel_error": fixed_pixel_total / samples,
                "mean_angular_error_deg": fixed_angle_total / samples,
            },
            "fixed_minus_float": {
                "mean_pixel_error": (fixed_pixel_total - float_pixel_total) / samples,
                "mean_angular_error_deg": (fixed_angle_total - float_angle_total) / samples,
            },
            "prediction_normalized_xy_absolute_difference": {
                "mean": [float(value / samples) for value in prediction_absolute_total],
                "maximum": [float(value) for value in prediction_absolute_maximum],
            },
        },
        "predictions_output": (
            {
                "path": str(args.predictions_output),
                "sha256": sha256_file(args.predictions_output),
            }
            if args.predictions_output is not None
            else None
        ),
        "runtime_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--readout-weight-bits", type=int, default=16)
    parser.add_argument("--constant-bits", type=int, default=24)
    parser.add_argument("--accumulator-bits", type=int, default=32)
    parser.add_argument("--spike-bits", type=int, default=8)
    parser.add_argument(
        "--scale-mode",
        choices=("per_layer", "per_output_channel"),
        default="per_output_channel",
    )
    parser.add_argument(
        "--scale-policy",
        choices=("no_clipping", "minimum_weight_mse"),
        default="no_clipping",
    )
    parser.add_argument("--maximum-fraction-bits", type=int, default=20)
    parser.add_argument(
        "--state-guard-bits",
        type=int,
        default=8,
        help=(
            "extra fractional bits for current, folded constants, and membrane; "
            "products are left-shifted by this amount"
        ),
    )
    parser.add_argument(
        "--alpha-rounding",
        choices=("floor", "toward_zero", "nearest_away"),
        default="floor",
    )
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.maximum_fraction_bits < 0 or args.state_guard_bits < 0:
        raise ValueError("fraction and guard bit counts must be nonnegative")
    for name in (
        "weight_bits",
        "readout_weight_bits",
        "constant_bits",
        "accumulator_bits",
        "spike_bits",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
