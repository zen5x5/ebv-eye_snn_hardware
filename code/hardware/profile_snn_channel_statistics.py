"""Profile per-channel activity and folded parameters for hardware mapping.

The existing numeric-range profiler answers how wide each tensor must be.  This
companion answers how uneven the work is across channels, which is needed for
sparse scheduling, PE load balancing, and per-output-channel quantization.
It is read-only and never trains or modifies a checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

import numpy as np
import sinabs
import torch

from code.hardware.profile_snn_numeric_ranges import sha256_file, threshold_view
from code.pursuit.fixed_event_window.ann.metrics import (
    angular_error_deg,
    pixel_error,
)
from code.pursuit.fixed_event_window.snn.model import SpikingPursuitSNN
from code.pursuit.fixed_event_window.snn.training import (
    load_persistent_model_state_dict,
)


SCHEMA_VERSION = 1
CONV_DESTINATION_CHANNELS = {
    "input_to_L1": 8,
    "L1_to_L2": 16,
    "L2_to_L3": 24,
    "L3_to_L4": 32,
    "L4_to_L5": 40,
    "L5_to_L6": 40,
    "L6_to_L7": 40,
    "L7_to_L8": 40,
}


@dataclass
class ChannelAccumulator:
    channels: int
    elements_per_channel: int = 0
    nonzero: torch.Tensor = field(init=False)
    total: torch.Tensor = field(init=False)
    square_total: torch.Tensor = field(init=False)
    minimum: torch.Tensor = field(init=False)
    maximum: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.nonzero = torch.zeros(self.channels, dtype=torch.int64)
        self.total = torch.zeros(self.channels, dtype=torch.float64)
        self.square_total = torch.zeros(self.channels, dtype=torch.float64)
        self.minimum = torch.full((self.channels,), math.inf, dtype=torch.float64)
        self.maximum = torch.full((self.channels,), -math.inf, dtype=torch.float64)

    def update(self, tensor: torch.Tensor) -> None:
        if tensor.ndim < 3 or tensor.shape[2] != self.channels:
            raise ValueError(
                f"expected channel dimension 2 with {self.channels} channels, "
                f"got {tuple(tensor.shape)}"
            )
        values = (
            tensor.detach()
            .to(device="cpu", dtype=torch.float64)
            .movedim(2, 0)
            .reshape(self.channels, -1)
        )
        self.elements_per_channel += values.shape[1]
        self.nonzero += torch.count_nonzero(values, dim=1)
        self.total += values.sum(dim=1)
        self.square_total += values.square().sum(dim=1)
        self.minimum = torch.minimum(self.minimum, values.min(dim=1).values)
        self.maximum = torch.maximum(self.maximum, values.max(dim=1).values)

    def finalize(self, samples: int) -> list[dict[str, float | int]]:
        if self.elements_per_channel == 0:
            raise ValueError("cannot finalize an empty channel accumulator")
        mean = self.total / self.elements_per_channel
        variance = torch.clamp(
            self.square_total / self.elements_per_channel - mean.square(), min=0
        )
        return [
            {
                "channel": index,
                "elements": self.elements_per_channel,
                "sum_per_sample": float(self.total[index] / samples),
                "nonzero_positions_per_sample": float(
                    self.nonzero[index] / samples
                ),
                "nonzero_fraction": float(
                    self.nonzero[index] / self.elements_per_channel
                ),
                "mean": float(mean[index]),
                "standard_deviation": float(torch.sqrt(variance[index])),
                "minimum": float(self.minimum[index]),
                "maximum": float(self.maximum[index]),
            }
            for index in range(self.channels)
        ]


def reduction_stats(values: torch.Tensor, reduce_dims: tuple[int, ...]) -> list[dict]:
    values = values.detach().to(device="cpu", dtype=torch.float64)
    minimum = values.amin(dim=reduce_dims)
    maximum = values.amax(dim=reduce_dims)
    maximum_absolute = values.abs().amax(dim=reduce_dims)
    mean_absolute = values.abs().mean(dim=reduce_dims)
    rms = values.square().mean(dim=reduce_dims).sqrt()
    exact_zero_fraction = values.eq(0).to(torch.float64).mean(dim=reduce_dims)
    return [
        {
            "channel": index,
            "minimum": float(minimum[index]),
            "maximum": float(maximum[index]),
            "maximum_absolute": float(maximum_absolute[index]),
            "mean_absolute": float(mean_absolute[index]),
            "rms": float(rms[index]),
            "exact_zero_fraction": float(exact_zero_fraction[index]),
        }
        for index in range(len(minimum))
    ]


def folded_parameter_statistics(model: SpikingPursuitSNN) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, block in enumerate(model.blocks, start=1):
        batch_norm = block.batch_norm
        inverse_sigma = torch.rsqrt(batch_norm.running_var + batch_norm.eps)
        scale = batch_norm.weight * inverse_sigma / block.spike.threshold
        weight = block.conv.weight * scale[:, None, None, None]
        constant = (
            batch_norm.bias
            - batch_norm.weight * batch_norm.running_mean * inverse_sigma
        ) / block.spike.threshold
        per_output = reduction_stats(weight, (1, 2, 3))
        per_input = reduction_stats(weight.movedim(1, 0), (1, 2, 3))
        for channel, values in enumerate(per_output):
            values["effective_threshold"] = float(block.spike.threshold[channel])
            values["folded_constant"] = float(constant[channel])
        result[f"L{index}"] = {
            "shape": list(weight.shape),
            "per_output_channel": per_output,
            "per_input_channel": per_input,
        }

    for name, block in (("H1", model.hidden1), ("H2", model.hidden2)):
        weight = block.linear.weight / block.spike.threshold[:, None]
        constant = block.linear.bias / block.spike.threshold
        per_output = reduction_stats(weight, (1,))
        per_input = reduction_stats(weight.movedim(1, 0), (1,))
        for channel, values in enumerate(per_output):
            values["effective_threshold"] = float(block.spike.threshold[channel])
            values["normalized_constant"] = float(constant[channel])
        result[name] = {
            "shape": list(weight.shape),
            "per_output_channel": per_output,
            "per_input_channel": per_input,
        }

    readout = model.readout.weight.detach()
    result["readout"] = {
        "shape": list(readout.shape),
        "per_output_channel": reduction_stats(readout, (1,)),
        "per_input_channel": reduction_stats(readout.movedim(1, 0), (1,)),
    }
    return result


def convolution_source_work(
    source: torch.Tensor,
    destination_channels: int,
) -> torch.Tensor:
    """Return sparse SynOps contribution for every source channel."""
    if source.ndim != 5:
        raise ValueError(f"expected [B,T,C,H,W], got {tuple(source.shape)}")
    height, width = source.shape[-2:]
    row_fanout = torch.full((height,), 3.0, dtype=torch.float64)
    column_fanout = torch.full((width,), 3.0, dtype=torch.float64)
    row_fanout[[0, -1]] = 2.0
    column_fanout[[0, -1]] = 2.0
    spatial_fanout = row_fanout[:, None] * column_fanout[None, :]
    values = source.detach().to(device="cpu", dtype=torch.float64)
    return (
        (values * spatial_fanout).sum(dim=(0, 1, 3, 4))
        * destination_channels
    )


def summarize_channel_imbalance(
    channels: list[dict[str, float | int]],
    field: str,
) -> dict[str, float]:
    values = np.asarray([float(channel[field]) for channel in channels])
    mean = float(values.mean())
    return {
        "minimum": float(values.min()),
        "mean": mean,
        "maximum": float(values.max()),
        "maximum_over_mean": float(values.max() / mean) if mean else 0.0,
        "coefficient_of_variation": float(values.std() / mean) if mean else 0.0,
    }


@torch.no_grad()
def profile(
    checkpoint_path: Path,
    configuration_path: Path,
    cache_dir: Path,
    split: str,
    batch_size: int,
) -> dict[str, object]:
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    if configuration["input"]["temporal_organization"] != (
        "repeat_identical_complete_400_event_count_frame"
    ):
        raise ValueError("channel profiling requires repeated complete input")
    timesteps = int(configuration["input"]["temporal_steps"])
    if timesteps != 2:
        raise ValueError("channel profiling currently requires T=2")

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
        raise ValueError(f"unexpected event frame shape: {event_frames.shape}")
    if labels.shape != (len(event_frames), 2) or len(event_frames) == 0:
        raise ValueError(f"unexpected label shape: {labels.shape}")
    event_counts = event_frames.sum(axis=(1, 2, 3), dtype=np.uint64)
    if not np.all(event_counts == 400):
        raise ValueError("every hardware profile sample must contain 400 events")

    widths = {
        **{f"L{index}": block.spike.channels for index, block in enumerate(model.blocks, 1)},
        "H1": model.hidden1.spike.channels,
        "H2": model.hidden2.spike.channels,
    }
    accumulators = {
        name: {
            "current": ChannelAccumulator(width),
            "spikes": ChannelAccumulator(width),
            "timestep_spikes": [ChannelAccumulator(width) for _ in range(timesteps)],
        }
        for name, width in widths.items()
    }
    input_accumulator = ChannelAccumulator(2)
    latest_outputs: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            current = inputs[0]
            normalized = current / threshold_view(module, current.ndim)
            accumulators[name]["current"].update(normalized)
            accumulators[name]["spikes"].update(output)
            for timestep in range(timesteps):
                accumulators[name]["timestep_spikes"][timestep].update(
                    output[:, timestep : timestep + 1]
                )
            latest_outputs[name] = output.detach()

        return hook

    for index, block in enumerate(model.blocks, start=1):
        hooks.append(block.spike.register_forward_hook(make_hook(f"L{index}")))
    hooks.append(model.hidden1.spike.register_forward_hook(make_hook("H1")))
    hooks.append(model.hidden2.spike.register_forward_hook(make_hook("H2")))

    transition_widths = {
        "input_to_L1": 2,
        **{f"L{index}_to_L{index + 1}": widths[f"L{index}"] for index in range(1, 8)},
        "L8_to_H1": 40,
        "H1_to_H2": 60,
    }
    transition_work = {
        name: torch.zeros(width, dtype=torch.float64)
        for name, width in transition_widths.items()
    }
    pixel_total = 0.0
    angular_total = 0.0
    started = time.monotonic()
    for start in range(0, len(event_frames), batch_size):
        stop = min(start + batch_size, len(event_frames))
        frame = torch.from_numpy(event_frames[start:stop]).float()
        target = torch.from_numpy(labels[start:stop]).float()
        event_bins = frame.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
        input_accumulator.update(event_bins)
        latest_outputs.clear()
        prediction = model(event_bins)
        pixel_total += float(pixel_error(prediction, target).sum())
        angular_total += float(angular_error_deg(prediction, target).sum())

        transition_work["input_to_L1"] += convolution_source_work(event_bins, 8)
        for index in range(1, 8):
            name = f"L{index}_to_L{index + 1}"
            transition_work[name] += convolution_source_work(
                latest_outputs[f"L{index}"],
                CONV_DESTINATION_CHANNELS[name],
            )
        transition_work["L8_to_H1"] += (
            latest_outputs["L8"].to(torch.float64).sum(dim=(0, 1, 3, 4)) * 60
        )
        transition_work["H1_to_H2"] += (
            latest_outputs["H1"].to(torch.float64).sum(dim=(0, 1)) * 12
        )

    for hook in hooks:
        hook.remove()

    samples = len(event_frames)
    layer_statistics = {}
    for name, values in accumulators.items():
        current = values["current"].finalize(samples)
        spikes = values["spikes"].finalize(samples)
        timestep_spikes = [
            accumulator.finalize(samples)
            for accumulator in values["timestep_spikes"]
        ]
        if name in ("L1", "L2") and model.stateless_channels().get(name) == list(
            range(widths[name])
        ):
            for channel in range(widths[name]):
                if timestep_spikes[0][channel] != timestep_spikes[1][channel]:
                    raise AssertionError(
                        f"{name} channel {channel}: repeated stateless statistics differ"
                    )
        layer_statistics[name] = {
            "channels": widths[name],
            "normalized_current_per_channel": current,
            "spike_per_channel": spikes,
            "spike_per_channel_by_timestep": timestep_spikes,
            "spike_sum_imbalance": summarize_channel_imbalance(
                spikes, "sum_per_sample"
            ),
            "active_position_imbalance": summarize_channel_imbalance(
                spikes, "nonzero_fraction"
            ),
        }

    transitions = {}
    for name, values in transition_work.items():
        per_channel = [
            {
                "source_channel": index,
                "synops_per_sample": float(value / samples),
            }
            for index, value in enumerate(values)
        ]
        transitions[name] = {
            "per_source_channel": per_channel,
            "total_synops_per_sample": float(values.sum() / samples),
            "source_work_imbalance": summarize_channel_imbalance(
                [
                    {"synops_per_sample": item["synops_per_sample"]}
                    for item in per_channel
                ],
                "synops_per_sample",
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "floating_point_per_channel_hardware_profile_without_quantization",
        "analysis_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
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
            "samples": samples,
            "batch_size": batch_size,
        },
        "floating_point_metrics": {
            "mean_pixel_error": pixel_total / samples,
            "mean_angular_error_deg": angular_total / samples,
        },
        "input_per_channel": input_accumulator.finalize(samples),
        "layer_statistics": layer_statistics,
        "transition_sparse_work": transitions,
        "folded_parameter_statistics": folded_parameter_statistics(model),
        "runtime_seconds": time.monotonic() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    configuration = args.configuration or args.checkpoint.parent / "configuration.json"
    report = profile(
        checkpoint_path=args.checkpoint,
        configuration_path=configuration,
        cache_dir=args.cache_dir,
        split=args.split,
        batch_size=args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
