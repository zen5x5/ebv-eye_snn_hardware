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

from .metrics import angular_error_deg, pixel_error
from .model import SpikingPursuitSNN
from .model_io import (
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
