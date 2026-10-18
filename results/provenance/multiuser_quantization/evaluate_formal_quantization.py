"""Evaluate frozen 16-bit quantization on the formal L40S-trained models.

This reuses the existing BN/threshold folding and fixed-point arithmetic. CPU
float64 convolution holds exact integer sums (the worst-case bound is below
2**53), avoiding float32's limited integer range for newly trained checkpoints.
No training, calibration on test labels, or checkpoint modification occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from code import paths
from code.hardware.evaluate_pursuit_fixed_point import (
    FixedPointPursuitReference,
    RuntimeCounters,
    load_model,
    output_channel_view,
    sha256_file,
)
from code.pursuit.fixed_event_window.ann.metrics import (
    MM_PER_PX,
    SCREEN_CENTER_X_PX,
    SCREEN_CENTER_Y_PX,
    VIEW_DISTANCE_MM,
)


CONTRACT = dict(
    weight_bits=16, readout_weight_bits=16, constant_bits=24,
    accumulator_bits=24, spike_bits=8, scale_mode="per_output_channel",
    scale_policy="no_clipping", maximum_fraction_bits=15,
    state_guard_bits=0, alpha_rounding="floor",
)
GRID_ROWS = np.arange(150, 931, 78, dtype=np.float64)
GRID_COLS = np.arange(150, 1771, 162, dtype=np.float64)


class ExactFixedPointReference(FixedPointPursuitReference):
    """Keep the existing arithmetic, with exact double-precision dot products."""

    def _linear_or_convolution(self, source_spikes, layer, counters):
        source = source_spikes.to(torch.float64)
        weight = layer.weight_integer.to(torch.float64)
        bound = float(source.abs().max()) * float(
            weight.abs().reshape(weight.shape[0], -1).sum(dim=1).max()
        )
        if bound >= 2**53:
            raise ArithmeticError(f"{layer.name}: exceeds exact integer range")
        if weight.ndim == 4:
            current = F.conv2d(source, weight, padding=1)
        else:
            current = F.linear(source, weight)
        scale = torch.pow(2.0, layer.product_left_shift.to(torch.float64))
        current = current * output_channel_view(scale, current.ndim)
        current = current + output_channel_view(
            layer.constant_integer.to(torch.float64), current.ndim
        )
        current = self._saturate_accumulator(current, layer.name, counters)
        return F.max_pool2d(current, 2) if layer.pool else current

    def _lif(self, current_integer, previous_membrane_integer, layer, counters):
        spikes, membrane = super()._lif(
            current_integer, previous_membrane_integer, layer, counters
        )
        if membrane.min() < -32768 or membrane.max() > 32767:
            raise ArithmeticError(f"{layer.name}: membrane exceeds signed 16 bits")
        return spikes, membrane


def normalized_to_pixels(prediction):
    xy = np.asarray(prediction, dtype=np.float64)
    return np.column_stack((
        xy[:, 1] * SCREEN_CENTER_Y_PX + SCREEN_CENTER_Y_PX,
        xy[:, 0] * SCREEN_CENTER_X_PX + SCREEN_CENTER_X_PX,
    ))


def project_grid(prediction_px):
    return np.column_stack((
        GRID_ROWS[np.abs(prediction_px[:, :1] - GRID_ROWS).argmin(axis=1)],
        GRID_COLS[np.abs(prediction_px[:, 1:] - GRID_COLS).argmin(axis=1)],
    ))


def errors(prediction_px, labels):
    def rays(points):
        return np.column_stack((
            (points[:, 1] - SCREEN_CENTER_X_PX) * MM_PER_PX,
            (points[:, 0] - SCREEN_CENTER_Y_PX) * MM_PER_PX,
            np.full(len(points), VIEW_DISTANCE_MM),
        ))
    a, b = rays(prediction_px), rays(labels)
    angles = np.rad2deg(np.arctan2(
        np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1)
    ))
    pixels = np.linalg.norm(prediction_px - labels, axis=1)
    return angles, pixels


def means(pair):
    return dict(mean_angular_error_deg=float(pair[0].mean()),
                mean_pixel_error=float(pair[1].mean()))


def result_directory(root, task, user):
    ordinary = "ordinary_" if task == "saccade" else ""
    experiment = (
        f"all_physical_source_recombination_{ordinary}t2_alpha0p75_"
        "l1_l2_stateless_continue_epoch100_to200"
    )
    return root / task / "fixed_event_window" / "snn" / f"user{user}" / "eye0" / experiment / "seed2021"


def evaluate_subject(args, task, user):
    started = time.monotonic()
    root = result_directory(args.result_root, task, user)
    cache = args.cache_root / task / "fixed_event_window_400_stride200_96x128" / f"user{user}" / "eye0"
    names = [root / n for n in ("configuration.json", "summary.json", "best_checkpoint.pt")]
    names += [cache / "test.npz", cache / "manifest.json"]
    if task == "saccade":
        names.append(root / "predictions_test.npz")
    before = {str(p): sha256_file(p) for p in names}
    configuration = json.loads(names[0].read_text())
    summary = json.loads(names[1].read_text())
    assert configuration["training"]["seed"] == 2021
    assert configuration["training"]["epochs"] == 200
    assert configuration["input"]["temporal_steps"] == 2
    assert configuration["input"]["temporal_organization"] == "repeat_identical_complete_400_event_count_frame"
    assert configuration["model"]["stateless_channels"] == {"L1": list(range(8)), "L2": list(range(16))}
    model, checkpoint = load_model(names[2], configuration)
    reference = ExactFixedPointReference(model, **CONTRACT)
    with np.load(cache / "test.npz", allow_pickle=False) as data:
        frames, labels = data["event_frames"].copy(), data["labels"].astype(np.float64)
    assert np.all(frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400)
    assert frames.shape[1:] == (2, 96, 128) and labels.shape == (len(frames), 2)
    assert np.isfinite(labels).all()
    float_metrics = summary["test"] if task == "pursuit" else summary["metrics"]["test"]
    assert float_metrics["samples"] == len(frames)

    # One real-input probe independently checks the inherited arithmetic path.
    # Layer currents/spikes/membranes, not just final outputs, are compared.
    old = FixedPointPursuitReference(model, **CONTRACT)
    probe = torch.from_numpy(frames[:1]).float()
    trace_old, trace_exact = {}, {}
    with torch.inference_mode():
        old.predict(probe, RuntimeCounters.empty(), trace_old)
        reference.predict(probe, RuntimeCounters.empty(), trace_exact)
    for key in trace_old:
        if not np.array_equal(trace_old[key], trace_exact[key]):
            raise AssertionError(f"{task} user{user}: trace mismatch at {key}")

    counters = RuntimeCounters.empty()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frames), args.batch_size):
            inputs = torch.from_numpy(frames[start:start + args.batch_size]).float()
            outputs.append(reference.predict(inputs, counters).numpy())
    prediction = np.concatenate(outputs)
    assert np.isfinite(prediction).all()
    prediction_px = normalized_to_pixels(prediction)
    continuous = errors(prediction_px, labels)
    fixed_metrics = {"continuous": means(continuous)}
    official = {"continuous": {k: float(float_metrics[k]) for k in
                              ("mean_angular_error_deg", "mean_pixel_error")}}
    arrays = dict(prediction_normalized_xy=prediction, prediction_row_col_px=prediction_px,
                  target_row_col_px=labels, angular_error_deg=continuous[0],
                  pixel_error=continuous[1], representation_index=np.arange(len(frames)))
    if task == "saccade":
        assert np.isin(labels[:, 0], GRID_ROWS).all() and np.isin(labels[:, 1], GRID_COLS).all()
        projected = project_grid(prediction_px)
        rounded = errors(projected, labels)
        fixed_metrics["grid_rounded"] = means(rounded)
        arrays.update(prediction_grid_row_col_px=projected, grid_angular_error_deg=rounded[0],
                      grid_pixel_error=rounded[1])
        with np.load(root / "predictions_test.npz", allow_pickle=False) as stored:
            assert np.array_equal(labels, stored["target_row_col_px"])
            assert np.array_equal(stored["sample_index"], np.arange(len(frames)))
            official["grid_rounded"] = means(errors(project_grid(stored["prediction_row_col_px"].astype(np.float64)), labels))

    destination = args.output / task / f"user{user}"
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "predictions_test.npz", **arrays)
    np.savez_compressed(destination / "parameters.npz", **reference.parameter_arrays())
    after = {str(p): sha256_file(p) for p in names}
    assert before == after
    result = dict(task=task, user=user, seed=2021, representations=len(frames),
                  checkpoint_epoch=int(checkpoint["epoch"]), input_sha256=before,
                  input_files_unchanged=True, official_float=official,
                  fixed_point=fixed_metrics, runtime_counters=asdict(counters),
                  parameters=reference.parameter_report(),
                  verification=dict(legacy_trace_arrays_equal=len(trace_old),
                                    float64_exact_integer_dot_products=True,
                                    every_membrane_within_signed16=True),
                  runtime_seconds=time.monotonic() - started)
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    selected = "grid_rounded" if task == "saccade" else "continuous"
    print(json.dumps(dict(task=task, user=user, n=len(frames),
                         float_deg=official[selected]["mean_angular_error_deg"],
                         quantized_deg=fixed_metrics[selected]["mean_angular_error_deg"],
                         seconds=result["runtime_seconds"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=paths.RESULT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=paths.CACHE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("pursuit", "saccade"), default=["pursuit", "saccade"])
    parser.add_argument("--users", nargs="+", type=int, default=list(range(4, 28)))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.users)) != len(args.users) or not set(args.users) <= set(range(4, 28)):
        raise ValueError("unique subjects 4--27 required")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("duplicate tasks")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    code_files = [Path(__file__), Path(__file__).with_name("evaluate_pursuit_fixed_point.py"),
                  paths.PROJECT_ROOT / "code/pursuit/fixed_event_window/snn/model.py",
                  paths.PROJECT_ROOT / "code/pursuit/fixed_event_window/snn/training.py"]
    config = dict(scope="Post-training quantization of formal L40S-trained seed2021 final models",
                  execution=dict(host=platform.node(), device="cpu", torch=torch.__version__,
                                 threads=args.threads, batch_size=args.batch_size),
                  arithmetic=CONTRACT, state_bits=16, tasks=args.tasks, users=args.users,
                  split="test", aggregation="Mean within each subject, then equally across subjects",
                  source_sha256={str(p.resolve()): sha256_file(p) for p in code_files})
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    source_dir = args.output / "source_snapshot"
    source_dir.mkdir()
    for p in code_files:
        (source_dir / p.name).write_bytes(p.read_bytes())
    rows = [evaluate_subject(args, task, user) for task in args.tasks for user in args.users]
    aggregates = {}
    for task in args.tasks:
        group = [r for r in rows if r["task"] == task]
        modes = ["continuous", "grid_rounded"] if task == "saccade" else ["continuous"]
        result = dict(subjects=len(group), representations=sum(r["representations"] for r in group))
        for mode in modes:
            result[mode] = {}
            for precision in ("official_float", "fixed_point"):
                best = min(group, key=lambda r: r[precision][mode]["mean_angular_error_deg"])
                result[mode][precision] = {
                    **{key: float(np.mean([r[precision][mode][key] for r in group]))
                       for key in ("mean_angular_error_deg", "mean_pixel_error")},
                    "best_subject": best["user"],
                    "best_subject_error_deg": best[precision][mode]["mean_angular_error_deg"],
                }
        aggregates[task] = result
    (args.output / "results.json").write_text(json.dumps(dict(configuration=config, aggregates=aggregates, subjects=rows), indent=2) + "\n")
    print("COMPLETE", json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
