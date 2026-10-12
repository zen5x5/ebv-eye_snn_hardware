"""Aggregate the three-seed fixed-point reports without rerunning inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


SCHEMA_VERSION = 1
SEEDS = (2021, 2022, 2023)
SPLITS = ("train", "validation", "test")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_and_sample_standard_deviation(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    source_files: dict[str, dict[str, str]] = {}
    reports: dict[str, list[dict[str, object]]] = {split: [] for split in SPLITS}
    reference_contract = None
    reference_code = None
    for split in SPLITS:
        for seed in SEEDS:
            path = args.input_dir / f"seed{seed}_{split}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["data"]["split"] != split:
                raise ValueError(f"split mismatch in {path}")
            if reference_contract is None:
                reference_contract = report["arithmetic_contract"]
                reference_code = report["analysis_code"]
            if report["arithmetic_contract"] != reference_contract:
                raise ValueError(f"arithmetic contract mismatch in {path}")
            if report["analysis_code"] != reference_code:
                raise ValueError(f"analysis code mismatch in {path}")
            reports[split].append(report)
            source_files[f"seed{seed}_{split}"] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    aggregates = {}
    for split, split_reports in reports.items():
        metric_paths = {
            "float_sinabs_cpu_mean_angular_error_deg": (
                "float_sinabs_cpu",
                "mean_angular_error_deg",
            ),
            "fixed_point_mean_angular_error_deg": (
                "fixed_point",
                "mean_angular_error_deg",
            ),
            "fixed_minus_float_mean_angular_error_deg": (
                "fixed_minus_float",
                "mean_angular_error_deg",
            ),
            "float_sinabs_cpu_mean_pixel_error": (
                "float_sinabs_cpu",
                "mean_pixel_error",
            ),
            "fixed_point_mean_pixel_error": (
                "fixed_point",
                "mean_pixel_error",
            ),
            "fixed_minus_float_mean_pixel_error": (
                "fixed_minus_float",
                "mean_pixel_error",
            ),
        }
        metrics = {
            name: mean_and_sample_standard_deviation(
                [float(report["metrics"][group][metric]) for report in split_reports]
            )
            for name, (group, metric) in metric_paths.items()
        }
        accumulator_saturations = sum(
            sum(report["runtime_counters"]["accumulator_saturated_values"].values())
            for report in split_reports
        )
        spike_saturations = sum(
            sum(report["runtime_counters"]["spike_saturated_values"].values())
            for report in split_reports
        )
        maximum_accumulator = max(
            max(
                report["runtime_counters"][
                    "accumulator_maximum_absolute_before_saturation"
                ].values()
            )
            for report in split_reports
        )
        maximum_spike = max(
            max(report["runtime_counters"]["spike_maximum_before_saturation"].values())
            for report in split_reports
        )
        aggregates[split] = {
            "samples_per_seed": [int(report["data"]["samples"]) for report in split_reports],
            "metrics": metrics,
            "runtime_range_audit": {
                "accumulator_saturated_values": accumulator_saturations,
                "spike_saturated_values": spike_saturations,
                "maximum_accumulator_absolute_before_saturation": maximum_accumulator,
                "maximum_spike_before_saturation": maximum_spike,
            },
        }

    script = Path(__file__).resolve()
    summary = {
        "schema_version": SCHEMA_VERSION,
        "scope": "three_seed_fixed_point_summary_for_frozen_arithmetic_contract",
        "summarizer": {"path": str(script), "sha256": sha256_file(script)},
        "analysis_code": reference_code,
        "arithmetic_contract": reference_contract,
        "seeds": list(SEEDS),
        "source_reports": source_files,
        "aggregates": aggregates,
        "interpretation_limit": (
            "The float comparison uses the Sinabs CPU backend in the same evaluator; "
            "it does not replace the official Exodus endpoint."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
