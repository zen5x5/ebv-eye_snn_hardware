"""Replay the paired-time H1/H2 linear LIF engine against formal goldens."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re

from .run_rtl_checks import (
    RTL_DIRECTORY,
    checked,
    executable,
    relative,
    run_command,
    sha256_file,
    source_record,
    tool_version,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Layer:
    name: str
    previous: str
    input_features: int
    output_features: int

    @property
    def output_groups(self) -> int:
        return (self.output_features + 15) // 16


LAYERS = (
    Layer("H1", "L8", 480, 60),
    Layer("H2", "H1", 60, 12),
)


def memory_path(manifest: dict[str, object], key: str) -> Path:
    record = manifest["artifacts"]["memory_files"][key]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"golden memory hash mismatch: {key}")
    return path


def artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def mapping_summary(statistics: dict[str, object]) -> dict[str, int]:
    cells = statistics["design"]["num_cells_by_type"]
    return {
        "dsp48e2": int(cells.get("DSP48E2", 0)),
        "ramb36e2": int(cells.get("RAMB36E2", 0)),
        "ramb18e2": int(cells.get("RAMB18E2", 0)),
        "flip_flops": int(cells.get("FDRE", 0) + cells.get("FDSE", 0)),
        "luts": int(
            sum(value for key, value in cells.items() if key.startswith("LUT"))
        ),
        "carry4": int(cells.get("CARRY4", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--skip-mapping", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    golden_manifest_path = args.golden_dir / "manifest.json"
    golden_manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    image_layout_path = args.image_dir / "layout.json"
    image_layout = json.loads(image_layout_path.read_text(encoding="utf-8"))
    for record in image_layout["artifacts"].values():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"hardware image hash mismatch: {path}")

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")
    sources = [
        RTL_DIRECTORY / "lif_multispike.sv",
        RTL_DIRECTORY / "paired_linear_lif_layer_engine.sv",
    ]
    testbench = RTL_DIRECTORY / "tb" / "tb_paired_linear_lif_layer_engine.sv"
    binary = args.output_dir / "tb_paired_linear_lif_layer_engine.vvp"
    compile_result = run_command(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_paired_linear_lif_layer_engine",
            "-o",
            str(binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "iverilog.log",
    )
    checked(compile_result, "paired linear compile")

    weight_path = Path(image_layout["artifacts"]["linear_beats"]["path"])
    constant_path = Path(image_layout["artifacts"]["constants"]["path"])
    fraction_path = Path(image_layout["artifacts"]["fractions"]["path"])
    replays = []
    totals = {
        "values_checked": 0,
        "cycles_start_to_done_sum": 0,
        "weight_beats_read": 0,
        "input_values_read": 0,
    }
    replay_pattern = re.compile(
        r"PASS: transactions=(\d+) values=(\d+) weight_reads=(\d+) "
        r"input_reads=(\d+) inference_cycles=(\d+)"
    )
    for layer in LAYERS:
        layout = image_layout["linear"]["layers"][layer.name]
        parameters = image_layout["channel_parameters"]["layers"][layer.name]
        inputs = {
            time: memory_path(
                golden_manifest, f"t{time}.{layer.previous}.spike_integer"
            )
            for time in (0, 1)
        }
        expected = {
            (time, quantity): memory_path(
                golden_manifest, f"t{time}.{layer.name}.{quantity}_integer"
            )
            for time in (0, 1)
            for quantity in ("current", "spike", "membrane")
        }
        log_path = args.output_dir / f"{layer.name.lower()}_paired.log"
        result = run_command(
            [
                vvp,
                str(binary),
                f"+WEIGHT_FILE={weight_path}",
                f"+CONSTANT_FILE={constant_path}",
                f"+FRACTION_FILE={fraction_path}",
                f"+INPUT_T0_FILE={inputs[0]}",
                f"+INPUT_T1_FILE={inputs[1]}",
                f"+EXPECTED_CURRENT_T0_FILE={expected[(0, 'current')]}",
                f"+EXPECTED_CURRENT_T1_FILE={expected[(1, 'current')]}",
                f"+EXPECTED_SPIKE_T0_FILE={expected[(0, 'spike')]}",
                f"+EXPECTED_SPIKE_T1_FILE={expected[(1, 'spike')]}",
                f"+EXPECTED_MEMBRANE_T0_FILE={expected[(0, 'membrane')]}",
                f"+EXPECTED_MEMBRANE_T1_FILE={expected[(1, 'membrane')]}",
                f"+INPUT_FEATURES={layer.input_features}",
                f"+OUTPUT_FEATURES={layer.output_features}",
                f"+OUTPUT_GROUPS={layer.output_groups}",
                f"+WEIGHT_BASE={layout['beat_offset']}",
                f"+PARAMETER_BASE={parameters['record_offset']}",
                "+TIMEOUT_CYCLES=10000",
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=log_path,
        )
        checked(result, f"{layer.name} paired replay")
        match = replay_pattern.search(log_path.read_text(encoding="utf-8"))
        if match is None:
            raise RuntimeError(f"missing replay summary for {layer.name}")
        transactions, values, weight_reads, input_reads, cycles = (
            int(value) for value in match.groups()
        )
        totals["values_checked"] += values
        totals["cycles_start_to_done_sum"] += cycles
        totals["weight_beats_read"] += weight_reads
        totals["input_values_read"] += input_reads
        replays.append(
            {
                "layer": layer.name,
                "transactions": transactions,
                "values_checked": values,
                "weight_beats_read": weight_reads,
                "input_values_read": input_reads,
                "cycles_start_to_done": cycles,
                "run": result,
            }
        )
        print(
            f"PASS {layer.name}: {values} values, {weight_reads} weight beats, "
            f"{cycles} cycles",
            flush=True,
        )

    lint = run_command(
        [
            verilator,
            "--lint-only",
            "--Wall",
            "--top-module",
            "paired_linear_lif_layer_engine",
            *(relative(path) for path in sources),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "verilator.log",
    )
    checked(lint, "paired linear lint")

    mapping = None
    if not args.skip_mapping:
        stat_path = args.output_dir / "paired_linear_lif_layer_engine_stat.json"
        mapping = run_command(
            [
                yosys,
                "-q",
                "-p",
                "; ".join(
                    (
                        "read_verilog -sv "
                        + " ".join(relative(path) for path in sources),
                        "hierarchy -top paired_linear_lif_layer_engine",
                        "synth_xilinx -family xcup -noiopad "
                        "-top paired_linear_lif_layer_engine",
                        f"tee -o {stat_path} stat -json",
                    )
                ),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / "yosys.log",
        )
        checked(mapping, "paired linear mapping")
        statistics = json.loads(stat_path.read_text(encoding="utf-8"))
        mapping["statistics_file"] = str(stat_path)
        mapping["statistics_sha256"] = sha256_file(stat_path)
        mapping["design_statistics"] = statistics["design"]
        mapping["summary"] = mapping_summary(statistics)
        mapping["scope"] = (
            "Yosys synth_xilinx xcup primitive mapping without I/O pads; "
            "no placement, routing, timing, or power"
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "paired_time_h1_h2_golden_replay_before_vivado",
        "contract": {
            "schedule": (
                "output-group-major; each 256-bit weight beat is read once "
                "and used on consecutive t0/t1 multiplier cycles"
            ),
            "arithmetic": (
                "16 shared S16xU8 lanes, separate t0/t1 S40 accumulators, "
                "S24 current, S16 membrane, U8 spike"
            ),
            "state": "t0 LIF membrane directly feeds t1 LIF",
        },
        "golden_manifest": artifact_record(golden_manifest_path),
        "hardware_image": artifact_record(image_layout_path),
        "sources": source_record([*sources, testbench]),
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "vvp": tool_version([vvp, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "replays": replays,
        "totals": totals,
        "compile": compile_result,
        "lint": lint,
        "mapping": mapping,
    }
    output = args.output_dir / "report.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
