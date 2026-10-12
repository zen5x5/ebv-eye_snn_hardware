"""Generate traceable fixed-point parameters and RTL golden vectors.

The output contains all quantized network parameters, complete per-layer traces
for selected cached samples, hexadecimal ``.mem`` files for one replay sample,
and a manifest binding every artifact to the checkpoint, configuration, cache,
and generator hashes.  Existing output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_pursuit_fixed_point import (
    FixedPointPursuitReference,
    RuntimeCounters,
    load_model,
    sha256_file,
)


SCHEMA_VERSION = 1


def array_description(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "minimum": float(array.min()) if array.size else None,
        "maximum": float(array.max()) if array.size else None,
        "layout": "NumPy C order; model tensors retain PyTorch OIHW/OI conventions",
    }


def write_hex_mem(path: Path, values: np.ndarray, bits: int) -> None:
    if bits <= 0:
        raise ValueError("memory word width must be positive")
    flattened = np.asarray(values).reshape(-1)
    if not np.issubdtype(flattened.dtype, np.integer):
        raise TypeError(f"{path.name}: .mem output requires integer values")
    mask = (1 << bits) - 1
    digits = (bits + 3) // 4
    with path.open("w", encoding="ascii") as handle:
        for raw_value in flattened:
            value = int(raw_value)
            handle.write(f"{value & mask:0{digits}x}\n")


def memory_width(name: str) -> int | None:
    if name == "input.event_count_integer":
        return 9
    if name.endswith(".weight_integer"):
        return 16
    if name.endswith(".constant_integer"):
        return 24
    if name.endswith(".weight_fraction_bits"):
        return 8
    if name.endswith(".current_and_state_fraction_bits"):
        return 8
    if name.endswith(".current_integer"):
        return 24
    if name.endswith(".spike_integer"):
        return 8
    if name.endswith(".membrane_integer"):
        return 16
    if name == "readout.temporal_sum_integer":
        return 9
    if name == "readout.output_integer_before_temporal_divide":
        return 64
    return None


def build_reference(
    model,
    args: argparse.Namespace,
) -> FixedPointPursuitReference:
    stateless_layer_names = (
        ("L1", "L2") if args.state_mode == "l1_l2_stateless" else ()
    )
    return FixedPointPursuitReference(
        model,
        weight_bits=args.weight_bits,
        readout_weight_bits=args.readout_weight_bits,
        constant_bits=args.constant_bits,
        accumulator_bits=args.current_bits,
        spike_bits=args.spike_bits,
        scale_mode="per_output_channel",
        scale_policy="no_clipping",
        maximum_fraction_bits=args.maximum_fraction_bits,
        state_guard_bits=0,
        alpha_rounding="floor",
        stateless_layer_names=stateless_layer_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--sample-indices", type=int, nargs="*")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--mem-sample-offset", type=int, default=0)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--readout-weight-bits", type=int, default=16)
    parser.add_argument("--constant-bits", type=int, default=24)
    parser.add_argument("--current-bits", type=int, default=24)
    parser.add_argument("--state-bits", type=int, default=16)
    parser.add_argument("--spike-bits", type=int, default=8)
    parser.add_argument("--maximum-fraction-bits", type=int, default=15)
    parser.add_argument(
        "--state-mode",
        choices=("l1_l2_stateless", "full_state"),
        default="l1_l2_stateless",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    configuration_path = args.configuration or args.checkpoint.parent / "configuration.json"
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    expected_stateless = (
        {"L1": list(range(8)), "L2": list(range(16))}
        if args.state_mode == "l1_l2_stateless"
        else {}
    )
    if configuration["model"]["stateless_channels"] != expected_stateless:
        raise ValueError(
            f"configuration stateless channels do not match {args.state_mode}"
        )
    model, checkpoint = load_model(args.checkpoint, configuration)
    reference = build_reference(model, args)

    cache_path = args.cache_dir / f"{args.split}.npz"
    with np.load(cache_path) as cached:
        all_frames = cached["event_frames"].copy()
        all_labels = cached["labels"].copy()
    if args.sample_indices:
        sample_indices = np.asarray(args.sample_indices, dtype=np.int64)
    else:
        if args.sample_count <= 0:
            raise ValueError("sample count must be positive")
        sample_indices = np.linspace(
            0,
            len(all_frames) - 1,
            min(args.sample_count, len(all_frames)),
            dtype=np.int64,
        )
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("sample indices must be unique")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(all_frames)):
        raise IndexError("sample index outside cached split")
    if not 0 <= args.mem_sample_offset < len(sample_indices):
        raise IndexError("mem sample offset outside selected sample list")

    frames = all_frames[sample_indices]
    labels = all_labels[sample_indices]
    if not np.all(frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400):
        raise ValueError("each selected input must contain exactly 400 events")
    input_tensor = torch.from_numpy(frames).to(torch.float32)
    trace: dict[str, np.ndarray] = {}
    counters = RuntimeCounters.empty()
    with torch.no_grad():
        fixed_prediction = reference.predict(input_tensor, counters, trace=trace)
        float_prediction = model(
            input_tensor.unsqueeze(1).expand(-1, 2, -1, -1, -1)
        )

    parameter_arrays = reference.parameter_arrays()
    vector_arrays: dict[str, np.ndarray] = {
        "sample_indices": sample_indices,
        "input.event_count_integer": frames.astype(np.uint16),
        "target.row_col_px": labels.astype(np.float32),
        "float_sinabs_cpu.prediction_normalized_xy": (
            float_prediction.cpu().numpy().astype(np.float32)
        ),
        "fixed.prediction_normalized_xy": (
            fixed_prediction.cpu().numpy().astype(np.float32)
        ),
        **trace,
    }

    args.output_dir.mkdir(parents=True)
    parameters_path = args.output_dir / "parameters.npz"
    vectors_path = args.output_dir / "vectors.npz"
    np.savez_compressed(parameters_path, **parameter_arrays)
    np.savez_compressed(vectors_path, **vector_arrays)

    mem_dir = args.output_dir / "mem"
    mem_dir.mkdir()
    memory_files: dict[str, dict[str, object]] = {}
    for name, array in parameter_arrays.items():
        width = memory_width(name)
        if width is None:
            continue
        path = mem_dir / f"parameter_{name.replace('.', '_')}.mem"
        write_hex_mem(path, array, width)
        memory_files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "word_bits": width,
            "elements": int(array.size),
            "scope": "complete_parameter_tensor",
        }

    mem_offset = args.mem_sample_offset
    for name, full_array in vector_arrays.items():
        width = memory_width(name)
        if width is None:
            continue
        if name == "input.event_count_integer" or name.startswith("t") or name.startswith("readout"):
            array = full_array[mem_offset : mem_offset + 1]
        else:
            continue
        path = mem_dir / f"vector_{name.replace('.', '_')}.mem"
        write_hex_mem(path, array, width)
        memory_files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "word_bits": width,
            "elements": int(array.size),
            "scope": "one_replay_sample",
            "sample_index": int(sample_indices[mem_offset]),
            "flattening": "NumPy C order",
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "pursuit_fixed_point_parameters_and_full_layer_golden_vectors",
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "fixed_reference": {
            "path": str(Path(__file__).with_name("evaluate_pursuit_fixed_point.py").resolve()),
            "sha256": sha256_file(
                Path(__file__).with_name("evaluate_pursuit_fixed_point.py").resolve()
            ),
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
            "sample_indices": [int(value) for value in sample_indices],
            "mem_replay_sample_index": int(sample_indices[mem_offset]),
        },
        "arithmetic_contract": {
            "state_mode": args.state_mode,
            "weight_bits": args.weight_bits,
            "readout_weight_bits": args.readout_weight_bits,
            "constant_bits": args.constant_bits,
            "current_bits_after_complete_dot_product": args.current_bits,
            "internal_running_accumulator_requirement": (
                "at least 33 signed bits by worst-case layer bound; use DSP48/40+ bits"
            ),
            "state_bits": args.state_bits,
            "input_count_bits": 9,
            "spike_bits": args.spike_bits,
            "scale_mode": "per_output_channel_power_of_two",
            "scale_policy": "maximum precision without weight clipping",
            "maximum_fraction_bits": args.maximum_fraction_bits,
            "static_rounding": "nearest, ties away from zero",
            "alpha": "floor((3 * signed_membrane_integer) / 4)",
            "spike": "min(floor(max(u,0)), 255)",
            "reset": "max(u - spike * scale, -scale)",
            "tensor_order": "NCHW; convolution weights OIHW; linear weights OI",
        },
        "runtime_counters": {
            "accumulator_saturated_values": counters.accumulator_saturated,
            "spike_saturated_values": counters.spike_saturated,
            "accumulator_maximum_absolute_before_saturation": (
                counters.accumulator_maximum_absolute
            ),
            "spike_maximum_before_saturation": counters.spike_maximum,
        },
        "artifacts": {
            "parameters_npz": {
                "path": str(parameters_path),
                "sha256": sha256_file(parameters_path),
                "arrays": {
                    name: array_description(array)
                    for name, array in sorted(parameter_arrays.items())
                },
            },
            "vectors_npz": {
                "path": str(vectors_path),
                "sha256": sha256_file(vectors_path),
                "arrays": {
                    name: array_description(array)
                    for name, array in sorted(vector_arrays.items())
                },
            },
            "memory_files": memory_files,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
