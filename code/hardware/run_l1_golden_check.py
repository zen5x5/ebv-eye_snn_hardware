"""Check the complete streaming L1 RTL against a formal golden replay sample.

The check binds the RTL sources and every consumed ``.mem`` file by SHA-256,
runs a full bit-exact Icarus replay, applies strict Verilator lint, and performs
an UltraScale+ primitive mapping with Yosys.  Mapping is pre-place-and-route
evidence only; Vivado remains authoritative for timing and routed utilization.
"""

from __future__ import annotations

import argparse
import json
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
REQUIRED_MEMORY_KEYS = (
    "input.event_count_integer",
    "L1.weight_integer",
    "L1.constant_integer",
    "L1.weight_fraction_bits",
    "L1.current_and_state_fraction_bits",
    "t0.L1.current_integer",
    "t0.L1.spike_integer",
    "t0.L1.membrane_integer",
)


def validate_golden_directory(directory: Path) -> dict[str, object]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    memories = manifest["artifacts"]["memory_files"]
    records: dict[str, object] = {}
    for key in REQUIRED_MEMORY_KEYS:
        record = memories[key]
        path = Path(record["path"])
        if path.parent.parent.resolve() != directory.resolve():
            raise ValueError(f"{key}: memory path is outside golden directory")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"{key}: SHA-256 mismatch")
        nonempty_lines = sum(
            bool(line.strip()) for line in path.read_text(encoding="ascii").splitlines()
        )
        if nonempty_lines != int(record["elements"]):
            raise ValueError(
                f"{key}: line count {nonempty_lines} != {record['elements']}"
            )
        records[key] = {
            "path": str(path),
            "sha256": actual_hash,
            "elements": int(record["elements"]),
            "word_bits": int(record["word_bits"]),
        }
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "replay_sample_index": int(manifest["data"]["mem_replay_sample_index"]),
        "memory_files": records,
    }


def yosys_default_mapping(
    *,
    yosys: str,
    sources: list[Path],
    output_directory: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    """Map the default L1 configuration without renaming its top module."""
    stat_path = output_directory / "l1_streaming_layer_stat.json"
    log_path = output_directory / "l1_streaming_layer_yosys.log"
    script = "; ".join(
        (
            "read_verilog -sv " + " ".join(relative(path) for path in sources),
            "hierarchy -top l1_streaming_layer",
            "synth_xilinx -family xcup -noiopad -top l1_streaming_layer",
            f"tee -o {stat_path} stat -json",
        )
    )
    result = run_command(
        [yosys, "-q", "-p", script],
        timeout_seconds=timeout_seconds,
        log_path=log_path,
    )
    result.update(
        {
            "top": "l1_streaming_layer",
            "parameters": "module defaults fixed by the L1 wrapper contract",
            "sources": source_record(sources),
            "mapping_scope": (
                "Yosys synth_xilinx xcup primitive mapping without I/O pads; "
                "no placement, routing, timing, or power"
            ),
        }
    )
    if result["status"] == "passed":
        statistics = json.loads(stat_path.read_text(encoding="utf-8"))
        result["statistics_file"] = str(stat_path)
        result["statistics_sha256"] = sha256_file(stat_path)
        result["design_statistics"] = statistics["design"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    golden = validate_golden_directory(args.golden_dir)
    args.output_dir.mkdir(parents=True)

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")

    sources = [
        RTL_DIRECTORY / "padded_line_buffer_3x3.sv",
        RTL_DIRECTORY / "pipelined_count_mac_array.sv",
        RTL_DIRECTORY / "maxpool2x2_stream.sv",
        RTL_DIRECTORY / "lif_multispike.sv",
        RTL_DIRECTORY / "l1_streaming_layer.sv",
    ]
    testbench = RTL_DIRECTORY / "tb" / "tb_l1_streaming_layer.sv"
    executable_path = args.output_dir / "tb_l1_streaming_layer.vvp"
    compile_result = run_command(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_l1_streaming_layer",
            "-o",
            str(executable_path),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "l1_iverilog.log",
    )
    checked(compile_result, "L1 Icarus compile")
    simulation_result = run_command(
        [vvp, str(executable_path), f"+GOLDEN_DIR={args.golden_dir.resolve()}"],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "l1_simulation.log",
    )
    checked(simulation_result, "L1 golden simulation")
    simulation_text = Path(simulation_result["log"]).read_text(encoding="utf-8")
    match = re.search(
        r"PASS: outputs=(\d+) inference_cycles=(\d+)", simulation_text
    )
    if match is None:
        raise RuntimeError("L1 simulation passed without the expected audit summary")

    lint_result = run_command(
        [
            verilator,
            "--lint-only",
            "--Wall",
            "--top-module",
            "l1_streaming_layer",
            *(relative(path) for path in sources),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "l1_verilator.log",
    )
    checked(lint_result, "L1 Verilator lint")

    mapping = yosys_default_mapping(
        yosys=yosys,
        sources=sources,
        output_directory=args.output_dir,
        timeout_seconds=args.timeout_seconds,
    )
    checked(mapping, "L1 Yosys mapping")

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "complete_pursuit_l1_streaming_rtl_golden_replay_and_mapping",
        "driver": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "rtl_sources": source_record(sources),
        "testbench": source_record([testbench])[0],
        "golden_vectors": golden,
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "simulation": {
            "compile": compile_result,
            "run": simulation_result,
            "outputs_checked": int(match.group(1)),
            "values_checked": int(match.group(1)) * 8 * 3,
            "inference_cycles_start_to_done": int(match.group(2)),
            "comparison": "S24 pooled current, U8 spike, and S16 membrane",
        },
        "lint": lint_result,
        "primitive_mapping": mapping,
        "interpretation_limit": (
            "The replay establishes bit-exact L1 behavior and Yosys establishes "
            "an UltraScale+ primitive structure. Only Vivado implementation for "
            "XCZU3EG can establish Fmax, WNS, routed utilization, BRAM "
            "fragmentation, and power."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report_path)


if __name__ == "__main__":
    main()
