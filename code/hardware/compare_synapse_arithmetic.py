"""Compare multiplier, unary-add, and binary shift-add synapse arithmetic.

For an integer source spike count ``n`` and a fixed-point weight ``w`` the same
mathematical contribution can be formed as one ``n*w`` multiplication, ``n``
repeated additions of ``w``, or additions of shifted ``w`` terms selected by
the set bits of ``n``.  This script counts those alternatives on real model
activity.  It excludes routing conflicts, accumulator clearing, folded
constants, pooling, LIF, and pipeline stalls; those belong to the cycle model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import sinabs
import torch

from code.hardware.profile_snn_channel_statistics import convolution_source_work
from code.hardware.profile_snn_numeric_ranges import sha256_file
from code.pursuit.fixed_event_window.snn.model import SpikingPursuitSNN
from code.pursuit.fixed_event_window.snn.training import (
    load_persistent_model_state_dict,
)


SCHEMA_VERSION = 1
CONV_TRANSITIONS = (
    ("input_to_L1", "input", 8),
    ("L1_to_L2", "L1", 16),
    ("L2_to_L3", "L2", 24),
    ("L3_to_L4", "L3", 32),
    ("L4_to_L5", "L4", 40),
    ("L5_to_L6", "L5", 40),
    ("L6_to_L7", "L6", 40),
    ("L7_to_L8", "L7", 40),
)
LINEAR_TRANSITIONS = (
    ("L8_to_H1", "L8", 60),
    ("H1_to_H2", "H1", 12),
)
PREFIX_REUSED_TRANSITIONS = {"input_to_L1", "L1_to_L2", "L2_to_L3"}
DENSE_MACS_PER_TIMESTEP = {
    "input_to_L1": 1_769_472,
    "L1_to_L2": 3_538_944,
    "L2_to_L3": 2_654_208,
    "L3_to_L4": 1_327_104,
    "L4_to_L5": 552_960,
    "L5_to_L6": 172_800,
    "L6_to_L7": 172_800,
    "L7_to_L8": 172_800,
    "L8_to_H1": 28_800,
    "H1_to_H2": 720,
}


def integer_counts(values: torch.Tensor) -> torch.Tensor:
    if bool((values < 0).any()):
        raise ValueError("spike/event counts must be nonnegative")
    rounded = values.round()
    if not torch.equal(values, rounded):
        raise ValueError("spike/event counts must be exact integers")
    return rounded.to(torch.int64)


def population_count(values: torch.Tensor) -> torch.Tensor:
    counts = integer_counts(values)
    result = torch.zeros_like(counts)
    remaining = counts.clone()
    while bool((remaining != 0).any()):
        result += remaining & 1
        remaining >>= 1
    return result


def arithmetic_views(source: torch.Tensor) -> dict[str, torch.Tensor]:
    counts = integer_counts(source)
    return {
        "unary_weight_adds": counts.to(torch.float64),
        "nonzero_count_weight_macs": counts.ne(0).to(torch.float64),
        "binary_shifted_weight_terms": population_count(source).to(torch.float64),
    }


def empty_totals(timesteps: int) -> dict[str, object]:
    return {
        "per_timestep": [
            {
                "unary_weight_adds": 0.0,
                "nonzero_count_weight_macs": 0.0,
                "binary_shifted_weight_terms": 0.0,
            }
            for _ in range(timesteps)
        ]
    }


@torch.no_grad()
def compare(
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
        raise ValueError("synapse comparison requires repeated complete input")
    timesteps = int(configuration["input"]["temporal_steps"])
    if timesteps != 2:
        raise ValueError("synapse comparison currently requires T=2")

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
    if event_frames.ndim != 4 or tuple(event_frames.shape[1:]) != (2, 96, 128):
        raise ValueError(f"unexpected event frame shape: {event_frames.shape}")
    if len(event_frames) == 0:
        raise ValueError("cached split is empty")
    if not np.all(event_frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400):
        raise ValueError("every sample must contain exactly 400 events")

    latest: dict[str, torch.Tensor] = {}
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            latest[name] = output.detach()

        return hook

    for index, block in enumerate(model.blocks, start=1):
        hooks.append(block.spike.register_forward_hook(capture(f"L{index}")))
    hooks.append(model.hidden1.spike.register_forward_hook(capture("H1")))
    hooks.append(model.hidden2.spike.register_forward_hook(capture("H2")))

    transitions = {
        name: empty_totals(timesteps)
        for name, _, _ in (*CONV_TRANSITIONS, *LINEAR_TRANSITIONS)
    }
    readout_totals = {
        "unary_weight_adds": 0.0,
        "nonzero_count_weight_macs": 0.0,
        "binary_shifted_weight_terms": 0.0,
    }
    started = time.monotonic()
    for start in range(0, len(event_frames), batch_size):
        stop = min(start + batch_size, len(event_frames))
        frame = torch.from_numpy(event_frames[start:stop]).float()
        event_bins = frame.unsqueeze(1).expand(-1, timesteps, -1, -1, -1)
        latest.clear()
        model(event_bins)

        sources = {"input": event_bins, **latest}
        for name, source_name, destination_channels in CONV_TRANSITIONS:
            source = sources[source_name]
            for timestep in range(timesteps):
                step = source[:, timestep : timestep + 1]
                for arithmetic, values in arithmetic_views(step).items():
                    transitions[name]["per_timestep"][timestep][arithmetic] += float(
                        convolution_source_work(values, destination_channels).sum()
                    )

        for name, source_name, destination_features in LINEAR_TRANSITIONS:
            source = sources[source_name]
            for timestep in range(timesteps):
                step = source[:, timestep : timestep + 1]
                for arithmetic, values in arithmetic_views(step).items():
                    transitions[name]["per_timestep"][timestep][arithmetic] += (
                        float(values.sum()) * destination_features
                    )

        temporal_h2_count = latest["H2"].sum(dim=1)
        for arithmetic, values in arithmetic_views(temporal_h2_count).items():
            readout_totals[arithmetic] += float(values.sum()) * 2

    for hook in hooks:
        hook.remove()

    samples = len(event_frames)
    arithmetic_names = tuple(readout_totals)
    for name, values in transitions.items():
        values["dense_macs_per_timestep"] = DENSE_MACS_PER_TIMESTEP[name]
        values["dense_macs_raw_t2"] = 2 * DENSE_MACS_PER_TIMESTEP[name]
        for step in values["per_timestep"]:
            for arithmetic in arithmetic_names:
                step[arithmetic] /= samples
        values["raw_t2"] = {
            arithmetic: sum(
                step[arithmetic] for step in values["per_timestep"]
            )
            for arithmetic in arithmetic_names
        }
        values["sl_hyb"] = {
            arithmetic: (
                values["per_timestep"][0][arithmetic]
                if name in PREFIX_REUSED_TRANSITIONS
                else values["raw_t2"][arithmetic]
            )
            for arithmetic in arithmetic_names
        }
        values["dense_macs_sl_hyb"] = (
            DENSE_MACS_PER_TIMESTEP[name]
            if name in PREFIX_REUSED_TRANSITIONS
            else 2 * DENSE_MACS_PER_TIMESTEP[name]
        )

    readout = {
        arithmetic: value / samples
        for arithmetic, value in readout_totals.items()
    }
    readout["dense_macs"] = 24
    totals = {}
    for schedule in ("raw_t2", "sl_hyb"):
        totals[schedule] = {
            arithmetic: sum(
                values[schedule][arithmetic] for values in transitions.values()
            )
            + readout[arithmetic]
            for arithmetic in arithmetic_names
        }
        dense_key = "dense_macs_raw_t2" if schedule == "raw_t2" else "dense_macs_sl_hyb"
        totals[schedule]["dense_macs"] = sum(
            int(values[dense_key]) for values in transitions.values()
        ) + 24

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "real_activity_synapse_arithmetic_operation_counts",
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
            "stateless_channels": configuration["model"]["stateless_channels"],
        },
        "data": {
            "cache_file": str(cache_path),
            "cache_file_sha256": sha256_file(cache_path),
            "split": split,
            "samples": samples,
            "batch_size": batch_size,
        },
        "definitions": {
            "unary_weight_adds": "one weight accumulation for every unit in an integer spike count",
            "nonzero_count_weight_macs": "one count-times-weight MAC for every nonzero source position and destination connection",
            "binary_shifted_weight_terms": "one shifted-weight accumulation for every set bit in the integer count",
            "dense_macs": "all dense tensor connections regardless of source activity",
        },
        "excluded_costs": [
            "address routing and accumulator conflicts",
            "accumulator initialization and clearing",
            "folded constants, pooling, LIF, state traffic, and pipeline stalls",
        ],
        "transitions": transitions,
        "h2_temporal_sum_to_readout": readout,
        "totals_per_sample": totals,
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
    report = compare(
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
