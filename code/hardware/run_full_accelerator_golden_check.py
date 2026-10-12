"""Verify the complete pursuit accelerator before Vivado.

The driver builds consolidated trace memories, compiles lean and full-state
tops, replays SL-TM, SL-HYB, and FS-TM, runs strict lint for both storage
builds, and binds optional Yosys xcup primitive mappings into one report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

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
LAYERS = tuple(f"L{index}" for index in range(1, 9)) + ("H1", "H2")
QUANTITIES = ("current", "spike", "membrane")


def load_and_validate_golden(directory: Path, expected_mode: str) -> dict[str, object]:
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual_mode = manifest["arithmetic_contract"].get(
        "state_mode", "l1_l2_stateless"
    )
    if actual_mode != expected_mode:
        raise ValueError(f"{directory}: state mode {actual_mode} != {expected_mode}")
    for record in manifest["artifacts"]["memory_files"].values():
        artifact = Path(record["path"])
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"golden artifact hash mismatch: {artifact}")
    return manifest


def load_and_validate_image(directory: Path) -> dict[str, object]:
    path = directory / "layout.json"
    layout = json.loads(path.read_text(encoding="utf-8"))
    for record in layout["artifacts"].values():
        artifact = Path(record["path"])
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"hardware image hash mismatch: {artifact}")
    return layout


def memory_path(manifest: dict[str, object], key: str) -> Path:
    return Path(manifest["artifacts"]["memory_files"][key]["path"])


def artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_trace_memories(
    manifest: dict[str, object], output_directory: Path, prefix: str
) -> dict[str, dict[str, object]]:
    records = {}
    for quantity in QUANTITIES:
        output = output_directory / f"{prefix}_all_{quantity}.mem"
        lines = 0
        with output.open("w", encoding="ascii") as destination:
            for time_index in (0, 1):
                for layer in LAYERS:
                    source = memory_path(
                        manifest, f"t{time_index}.{layer}.{quantity}_integer"
                    )
                    text = source.read_text(encoding="ascii")
                    destination.write(text)
                    lines += len(text.splitlines())
        if lines != 90000:
            raise AssertionError(f"{output}: expected 90000 values, got {lines}")
        records[quantity] = {
            "path": str(output),
            "sha256": sha256_file(output),
            "lines": lines,
        }
    return records


def simulation_arguments(
    *,
    binary: Path,
    mode: int,
    manifest: dict[str, object],
    image: dict[str, object],
    trace_records: dict[str, dict[str, object]],
) -> list[str]:
    artifacts = image["artifacts"]
    return [
        str(binary),
        f"+CONV_WEIGHT_FILE={artifacts['convolution_beats']['path']}",
        f"+LINEAR_WEIGHT_FILE={artifacts['linear_beats']['path']}",
        f"+CONSTANT_FILE={artifacts['constants']['path']}",
        f"+FRACTION_FILE={artifacts['fractions']['path']}",
        f"+READOUT_FRACTION_FILE={artifacts['readout_fractions']['path']}",
        f"+INPUT_FILE={memory_path(manifest, 'input.event_count_integer')}",
        f"+EXPECTED_CURRENT_FILE={trace_records['current']['path']}",
        f"+EXPECTED_SPIKE_FILE={trace_records['spike']['path']}",
        f"+EXPECTED_MEMBRANE_FILE={trace_records['membrane']['path']}",
        "+EXPECTED_RESULT_FILE="
        + str(memory_path(manifest, "readout.output_integer_before_temporal_divide")),
        f"+DATAFLOW_MODE={mode}",
    ]


def parse_replay(log_path: Path, mode: int) -> dict[str, int]:
    text = log_path.read_text(encoding="utf-8")
    match = re.search(
        r"PASS: mode=(\d+) transactions=(\d+) values=(\d+) "
        r"total_cycles=(\d+) conv_cycles=(\d+) dense_cycles=(\d+) "
        r"replay_cycles=(\d+)",
        text,
    )
    if match is None or int(match.group(1)) != mode:
        raise RuntimeError(f"missing mode {mode} replay summary in {log_path}")
    names = (
        "mode",
        "transactions",
        "values_checked",
        "total_cycles",
        "convolution_cycles",
        "dense_cycles",
        "replay_cycles",
    )
    return {name: int(value) for name, value in zip(names, match.groups())}


def import_mapping(
    stat_source: Path | None,
    log_source: Path | None,
    output_directory: Path,
    name: str,
) -> dict[str, object] | None:
    if stat_source is None and log_source is None:
        return None
    if stat_source is None or log_source is None:
        raise ValueError(f"{name}: mapping stat and log must be supplied together")
    stat_destination = output_directory / f"{name}_yosys_stat.json"
    log_destination = output_directory / f"{name}_yosys.log"
    shutil.copy2(stat_source, stat_destination)
    shutil.copy2(log_source, log_destination)
    statistics = json.loads(stat_destination.read_text(encoding="utf-8"))
    design = statistics["design"]
    cells = design["num_cells_by_type"]
    return {
        "statistics_file": str(stat_destination),
        "statistics_sha256": sha256_file(stat_destination),
        "log_file": str(log_destination),
        "log_sha256": sha256_file(log_destination),
        "design_statistics": design,
        "summary": {
            "dsp48e2": int(cells.get("DSP48E2", 0)),
            "ramb36e2": int(cells.get("RAMB36E2", 0)),
            "ramb18e2": int(cells.get("RAMB18E2", 0)),
            "flip_flops": int(cells.get("FDRE", 0) + cells.get("FDSE", 0)),
            "luts": int(
                sum(value for key, value in cells.items() if key.startswith("LUT"))
            ),
            "carry4": int(cells.get("CARRY4", 0)),
        },
        "scope": (
            "Yosys synth_xilinx xcup primitive mapping without I/O pads; "
            "no placement, routing, timing, or power"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stateless-golden-dir", type=Path, required=True)
    parser.add_argument("--stateless-image-dir", type=Path, required=True)
    parser.add_argument("--full-golden-dir", type=Path, required=True)
    parser.add_argument("--full-image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lean-mapping-stat", type=Path)
    parser.add_argument("--lean-mapping-log", type=Path)
    parser.add_argument("--full-mapping-stat", type=Path)
    parser.add_argument("--full-mapping-log", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    stateless_manifest = load_and_validate_golden(
        args.stateless_golden_dir, "l1_l2_stateless"
    )
    full_manifest = load_and_validate_golden(args.full_golden_dir, "full_state")
    stateless_image = load_and_validate_image(args.stateless_image_dir)
    full_image = load_and_validate_image(args.full_image_dir)
    stateless_trace = build_trace_memories(
        stateless_manifest, args.output_dir, "stateless"
    )
    full_trace = build_trace_memories(full_manifest, args.output_dir, "full")

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")
    sources = [
        RTL_DIRECTORY / name
        for name in (
            "programmable_padded_line_buffer_3x3.sv",
            "programmable_maxpool2x2_stream.sv",
            "pipelined_count_mac_array.sv",
            "lif_multispike.sv",
            "conv3x3_layer_engine.sv",
            "linear_lif_layer_engine.sv",
            "readout_engine.sv",
            "cached_current_lif_replay.sv",
            "pursuit_activation_store.sv",
            "pursuit_conv_state_store.sv",
            "pursuit_l3_current_cache.sv",
            "pursuit_snn_accelerator_top.sv",
        )
    ]
    testbench = RTL_DIRECTORY / "tb" / "tb_pursuit_snn_accelerator_top.sv"
    common_compile = [
        iverilog,
        "-g2012",
        "-Wall",
        "-s",
        "tb_pursuit_snn_accelerator_top",
    ]
    lean_binary = args.output_dir / "tb_pursuit_top_lean.vvp"
    lean_compile = run_command(
        [
            *common_compile,
            "-o",
            str(lean_binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "lean_iverilog.log",
    )
    checked(lean_compile, "lean top compile")
    full_binary = args.output_dir / "tb_pursuit_top_full.vvp"
    full_compile = run_command(
        [
            *common_compile,
            "-P",
            "tb_pursuit_snn_accelerator_top.FULL_STATE_CAPABLE=1",
            "-o",
            str(full_binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "full_iverilog.log",
    )
    checked(full_compile, "full top compile")

    replay_definitions = (
        ("sl_tm", 0, lean_binary, stateless_manifest, stateless_image, stateless_trace),
        ("sl_hyb", 1, lean_binary, stateless_manifest, stateless_image, stateless_trace),
        ("fs_tm", 2, full_binary, full_manifest, full_image, full_trace),
    )
    replays = {}
    for name, mode, binary, manifest, image, trace in replay_definitions:
        log_path = args.output_dir / f"{name}_simulation.log"
        result = run_command(
            [
                vvp,
                *simulation_arguments(
                    binary=binary,
                    mode=mode,
                    manifest=manifest,
                    image=image,
                    trace_records=trace,
                ),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=log_path,
        )
        checked(result, f"{name} full replay")
        summary = parse_replay(log_path, mode)
        summary["latency_at_200mhz_ms"] = summary["total_cycles"] / 200_000.0
        replays[name] = {"run": result, **summary}
        print(
            f"PASS {name}: {summary['values_checked']} values, "
            f"{summary['total_cycles']} cycles",
            flush=True,
        )

    lint = {}
    for name, extra in (("lean", []), ("full", ["-GFULL_STATE_CAPABLE=1"])):
        result = run_command(
            [
                verilator,
                "--lint-only",
                "--Wall",
                "--top-module",
                "pursuit_snn_accelerator_top",
                *extra,
                *(relative(path) for path in sources),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / f"{name}_verilator.log",
        )
        checked(result, f"{name} top lint")
        lint[name] = result

    mappings = {
        "lean": import_mapping(
            args.lean_mapping_stat,
            args.lean_mapping_log,
            args.output_dir,
            "lean",
        ),
        "full": import_mapping(
            args.full_mapping_stat,
            args.full_mapping_log,
            args.output_dir,
            "full",
        ),
    }
    cycle_saving = (
        replays["sl_tm"]["total_cycles"] - replays["sl_hyb"]["total_cycles"]
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "complete_pursuit_pre_vivado_rtl_replay_and_mapping",
        "driver": source_record([Path(__file__).resolve()])[0],
        "rtl_sources": source_record(sources),
        "testbench": source_record([testbench])[0],
        "inputs": {
            "stateless_golden_manifest": artifact_record(
                args.stateless_golden_dir / "manifest.json"
            ),
            "stateless_hardware_image": artifact_record(
                args.stateless_image_dir / "layout.json"
            ),
            "full_golden_manifest": artifact_record(
                args.full_golden_dir / "manifest.json"
            ),
            "full_hardware_image": artifact_record(
                args.full_image_dir / "layout.json"
            ),
            "consolidated_traces": {
                "stateless": stateless_trace,
                "full": full_trace,
            },
        },
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "compile": {"lean": lean_compile, "full": full_compile},
        "replays": replays,
        "comparison": {
            "sl_hyb_cycles_saved_vs_sl_tm": cycle_saving,
            "sl_hyb_latency_reduction_fraction": cycle_saving
            / replays["sl_tm"]["total_cycles"],
            "fs_tm_and_sl_tm_cycle_equal": replays["fs_tm"]["total_cycles"]
            == replays["sl_tm"]["total_cycles"],
        },
        "lint": lint,
        "primitive_mapping": mappings,
        "interpretation_limit": (
            "All three routes are bit-exact for the bound sample and model. "
            "Yosys counts are pre-place-and-route. Only Vivado on "
            "xczu3eg-sfvc784-2-e can establish WNS, routed utilization, BRAM "
            "fragmentation, and power."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
