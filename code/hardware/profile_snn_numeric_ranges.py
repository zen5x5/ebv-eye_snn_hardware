"""Profile hardware-visible numeric ranges of a trained pursuit SNN.

This is a read-only analysis of an existing checkpoint and cached split.  It
runs the ordinary floating-point model with the portable Sinabs backend and
records the tensors that a fixed-point implementation must represent.  It does
not quantize the model or change its predictions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np
import sinabs
import torch

from code.pursuit.fixed_event_window.ann.metrics import (
    angular_error_deg,
    pixel_error,
)
from code.pursuit.fixed_event_window.snn.model import (
    LearnableThresholdSpikingLayer,
    SpikingPursuitSNN,
)
from code.pursuit.fixed_event_window.snn.training import (
    load_persistent_model_state_dict,
)


PROFILE_SCHEMA_VERSION = 2
SAMPLE_VALUES_PER_UPDATE = 2_048
QUANTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class TensorProfile:
    count: int = 0
    nonzero: int = 0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0
    square_total: float = 0.0
    samples: list[torch.Tensor] = field(default_factory=list)

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
        if flat.numel() == 0:
            return
        self.count += flat.numel()
        self.nonzero += int(torch.count_nonzero(flat))
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.total += float(flat.sum())
        self.square_total += float(flat.square().sum())

        take = min(SAMPLE_VALUES_PER_UPDATE, flat.numel())
        if take == flat.numel():
            selected = flat
        else:
            indices = torch.linspace(0, flat.numel() - 1, take).round().long()
            selected = flat[indices]
        self.samples.append(selected.to(dtype=torch.float32))

    def finalize(self) -> dict[str, object]:
        if self.count == 0:
            raise ValueError("cannot finalize an empty tensor profile")
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        sampled = torch.cat(self.samples).to(dtype=torch.float64)
        absolute = sampled.abs()

        def sampled_quantiles(values: torch.Tensor) -> dict[str, float]:
            result = torch.quantile(
                values,
                torch.tensor(QUANTILES, dtype=values.dtype),
            )
            return {
                f"q{quantile:g}": float(value)
                for quantile, value in zip(QUANTILES, result)
            }

        return {
            "elements": self.count,
            "nonzero_elements": self.nonzero,
            "nonzero_fraction": self.nonzero / self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "maximum_absolute": max(abs(self.minimum), abs(self.maximum)),
            "mean": mean,
            "standard_deviation": math.sqrt(variance),
            "sampled_elements_for_quantiles": int(sampled.numel()),
            "sampled_value_quantiles": sampled_quantiles(sampled),
            "sampled_absolute_quantiles": sampled_quantiles(absolute),
        }


def describe_tensor(tensor: torch.Tensor) -> dict[str, object]:
    profile = TensorProfile()
    profile.update(tensor)
    return profile.finalize()


def threshold_view(layer: LearnableThresholdSpikingLayer, ndim: int) -> torch.Tensor:
    shape = (1, 1, layer.channels, *([1] * (ndim - 3)))
    return layer.threshold.detach().view(shape)


def folded_parameter_profiles(model: SpikingPursuitSNN) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, block in enumerate(model.blocks, start=1):
        batch_norm = block.batch_norm
        inverse_sigma = torch.rsqrt(batch_norm.running_var + batch_norm.eps)
        normalized_scale = batch_norm.weight * inverse_sigma / block.spike.threshold
        folded_weight = block.conv.weight * normalized_scale[:, None, None, None]
        folded_constant = (
            batch_norm.bias
            - batch_norm.weight * batch_norm.running_mean * inverse_sigma
        ) / block.spike.threshold
        result[f"L{index}"] = {
            "effective_threshold": describe_tensor(block.spike.threshold),
            "normalized_bn_scale": describe_tensor(normalized_scale),
            "folded_weight": describe_tensor(folded_weight),
            "folded_constant": describe_tensor(folded_constant),
        }

    for name, block in (("H1", model.hidden1), ("H2", model.hidden2)):
        result[name] = {
            "effective_threshold": describe_tensor(block.spike.threshold),
            "normalized_weight": describe_tensor(
                block.linear.weight / block.spike.threshold[:, None]
            ),
            "normalized_constant": describe_tensor(
                block.linear.bias / block.spike.threshold
            ),
        }
    result["readout"] = {"weight": describe_tensor(model.readout.weight)}
    return result


def make_spike_hook(
    name: str,
    profiles: dict[str, TensorProfile],
) -> Callable:
    def record(metric: str, value: torch.Tensor) -> None:
        profiles.setdefault(f"{name}.{metric}", TensorProfile()).update(value)

    def hook(
        layer: LearnableThresholdSpikingLayer,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        current = inputs[0]
        normalized = current / threshold_view(layer, current.ndim)
        record("lif_input_current", normalized)
        record("spike_count", output)

        membrane = torch.zeros_like(normalized[:, 0])
        minimum_membrane = float(layer.neuron.min_v_mem)
        retain = (~layer._stateless_channels).to(dtype=normalized.dtype)
        retain = retain.view(1, layer.channels, *([1] * (membrane.ndim - 2)))
        for timestep in range(normalized.shape[1]):
            pre_reset = layer.alpha * membrane + normalized[:, timestep]
            expected_spikes = torch.where(
                pre_reset > 0,
                torch.trunc(pre_reset),
                torch.zeros_like(pre_reset),
            )
            if not torch.equal(expected_spikes, output[:, timestep]):
                raise AssertionError(f"{name} timestep {timestep}: spike reconstruction mismatch")
            post_reset = torch.clamp_min(
                pre_reset - output[:, timestep],
                minimum_membrane,
            )
            record(f"t{timestep}.membrane_pre_reset", pre_reset)
            record(f"t{timestep}.membrane_post_reset", post_reset)
            membrane = post_reset
            if timestep + 1 < normalized.shape[1]:
                membrane = membrane * retain
        if not torch.allclose(membrane, layer.neuron.v_mem, rtol=0.0, atol=1e-6):
            raise AssertionError(f"{name}: final membrane reconstruction mismatch")
        if bool(layer._stateless_channels.all()) and normalized.shape[1] == 2:
            if not torch.equal(output[:, 0], output[:, 1]):
                raise AssertionError(f"{name}: repeated stateless outputs differ")

    return hook


def profile_checkpoint(
    checkpoint_path: Path,
    configuration_path: Path,
    cache_dir: Path,
    split: str,
    batch_size: int,
) -> dict[str, object]:
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    if int(configuration["input"]["temporal_steps"]) != 2:
        raise ValueError("hardware range profiling currently requires T=2")
    if configuration["input"]["temporal_organization"] != (
        "repeat_identical_complete_400_event_count_frame"
    ):
        raise ValueError("hardware range profiling requires repeated complete input")

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

    cache_path = cache_dir / f"{split}.npz"
    with np.load(cache_path) as cached:
        event_frames = cached["event_frames"].copy()
        labels = cached["labels"].copy()
    if event_frames.ndim != 4 or tuple(event_frames.shape[1:]) != (2, 96, 128):
        raise ValueError(
            f"expected cached event frames [N, 2, 96, 128], got {event_frames.shape}"
        )
    if labels.shape != (len(event_frames), 2):
        raise ValueError(f"expected labels [N, 2], got {labels.shape}")
    if len(event_frames) == 0:
        raise ValueError(f"cached split is empty: {cache_path}")
    events_per_sample = event_frames.sum(axis=(1, 2, 3), dtype=np.uint64)
    if not np.all(events_per_sample == 400):
        raise ValueError(
            "hardware profiling requires exactly 400 cached events per sample"
        )

    profiles: dict[str, TensorProfile] = {}
    hooks = []

    def add_profile(name: str, tensor: torch.Tensor) -> None:
        profiles.setdefault(name, TensorProfile()).update(tensor)

    for index, block in enumerate(model.blocks, start=1):
        name = f"L{index}"
        threshold = block.spike.threshold.detach().view(1, -1, 1, 1)
        hooks.append(
            block.conv.register_forward_hook(
                lambda _module, _inputs, output, layer=name: add_profile(
                    f"{layer}.raw_convolution", output
                )
            )
        )
        hooks.append(
            block.batch_norm.register_forward_hook(
                lambda _module, _inputs, output, layer=name, scale=threshold: (
                    add_profile(f"{layer}.prepool_normalized_current", output / scale)
                )
            )
        )
        hooks.append(block.spike.register_forward_hook(make_spike_hook(name, profiles)))

    hooks.append(
        model.hidden1.spike.register_forward_hook(make_spike_hook("H1", profiles))
    )
    hooks.append(
        model.hidden2.spike.register_forward_hook(make_spike_hook("H2", profiles))
    )

    def record_readout(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        add_profile("readout.input_mean_spikes", inputs[0])
        add_profile("readout.output_normalized_xy", output)

    hooks.append(
        model.readout.register_forward_hook(record_readout)
    )

    pixel_total = 0.0
    angular_total = 0.0
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(event_frames), batch_size):
            stop = min(start + batch_size, len(event_frames))
            frame = torch.from_numpy(event_frames[start:stop]).float()
            target = torch.from_numpy(labels[start:stop]).float()
            add_profile("input.event_count_per_pixel", frame)
            add_profile("input.events_per_sample", frame.sum(dim=(1, 2, 3)))
            event_bins = frame.unsqueeze(1).expand(-1, 2, -1, -1, -1)
            prediction = model(event_bins)
            pixel_total += float(pixel_error(prediction, target).sum())
            angular_total += float(angular_error_deg(prediction, target).sum())

    for hook in hooks:
        hook.remove()

    elapsed = time.monotonic() - started
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "scope": "floating_point_range_profile_without_quantization",
        "analysis_code": {
            "profiler_path": str(Path(__file__).resolve()),
            "profiler_sha256": sha256_file(Path(__file__).resolve()),
        },
        "execution_environment": {
            "backend": "sinabs_cpu",
            "torch_version": torch.__version__,
            "sinabs_version": sinabs.__version__,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "configuration": {
            "path": str(configuration_path),
            "sha256": sha256_file(configuration_path),
            "alpha": float(configuration["model"]["alpha"]),
            "stateless_channels": configuration["model"]["stateless_channels"],
        },
        "data": {
            "cache_file": str(cache_path),
            "cache_file_sha256": sha256_file(cache_path),
            "split": split,
            "input_scope": "cached_physical_windows_without_online_recombination",
            "samples": len(event_frames),
            "batch_size": batch_size,
        },
        "floating_point_metrics": {
            "mean_pixel_error": pixel_total / len(event_frames),
            "mean_angular_error_deg": angular_total / len(event_frames),
        },
        "parameter_profiles": folded_parameter_profiles(model),
        "activation_profiles": {
            name: profile.finalize() for name, profile in sorted(profiles.items())
        },
        "runtime_seconds": elapsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    configuration = args.configuration or args.checkpoint.parent / "configuration.json"
    report = profile_checkpoint(
        args.checkpoint,
        configuration,
        args.cache_dir,
        args.split,
        args.batch_size,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
