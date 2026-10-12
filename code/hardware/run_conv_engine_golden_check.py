"""Replay the shared L1--L8 convolution engine against formal vectors.

The sixteen checks cover every convolution layer at both time steps.  L1/L2
use the final stateless contract at t1; L3--L8 read the corresponding t0
membrane.  Every emitted S24 current, U8 spike, and S16 membrane is compared.
"""

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
    previous: str | None
    width_log2: int
    height: int
    input_channels: int
    output_channels: int
    input_spatial: int
    output_spatial: int
    pool: int

    @property
    def input_tiles(self) -> int:
        return (self.input_channels + 3) // 4

    @property
    def output_tiles(self) -> int:
        return (self.output_channels + 7) // 8


LAYERS = (
    Layer("L1", None, 7, 96, 2, 8, 12288, 3072, 1),
    Layer("L2", "L1", 6, 48, 8, 16, 3072, 768, 1),
    Layer("L3", "L2", 5, 24, 16, 24, 768, 192, 1),
    Layer("L4", "L3", 4, 12, 24, 32, 192, 48, 1),
    Layer("L5", "L4", 3, 6, 32, 40, 48, 12, 1),
    Layer("L6", "L5", 2, 3, 40, 40, 12, 12, 0),
    Layer("L7", "L6", 2, 3, 40, 40, 12, 12, 0),
    Layer("L8", "L7", 2, 3, 40, 40, 12, 12, 0),
)


def memory_path(manifest: dict[str, object], key: str) -> Path:
    record = manifest["artifacts"]["memory_files"][key]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"golden memory hash mismatch: {key}")
    return path


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
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"hardware image hash mismatch: {record['path']}")

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")
    sources = [
        RTL_DIRECTORY / "programmable_padded_line_buffer_3x3.sv",
        RTL_DIRECTORY / "programmable_maxpool2x2_stream.sv",
        RTL_DIRECTORY / "pipelined_count_mac_array.sv",
        RTL_DIRECTORY / "lif_multispike.sv",
        RTL_DIRECTORY / "conv3x3_layer_engine.sv",
    ]
    testbench = RTL_DIRECTORY / "tb" / "tb_conv3x3_layer_engine.sv"
    simulation_binary = args.output_dir / "tb_conv3x3_layer_engine.vvp"
    compile_result = run_command(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_conv3x3_layer_engine",
            "-o",
            str(simulation_binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "iverilog.log",
    )
    checked(compile_result, "convolution-engine compile")

    weight_path = Path(image_layout["artifacts"]["convolution_beats"]["path"])
    constant_path = Path(image_layout["artifacts"]["constants"]["path"])
    fraction_path = Path(image_layout["artifacts"]["fractions"]["path"])
    replay_results = []
    total_values = 0
    for layer in LAYERS:
        layer_layout = image_layout["convolution"]["layers"][layer.name]
        parameter_layout = image_layout["channel_parameters"]["layers"][layer.name]
        for time_index in (0, 1):
            if layer.previous is None:
                input_path = memory_path(
                    golden_manifest, "input.event_count_integer"
                )
            else:
                input_path = memory_path(
                    golden_manifest,
                    f"t{time_index}.{layer.previous}.spike_integer",
                )
            previous_state_path = memory_path(
                golden_manifest, f"t0.{layer.name}.membrane_integer"
            )
            expected_paths = {
                quantity: memory_path(
                    golden_manifest,
                    f"t{time_index}.{layer.name}.{quantity}_integer",
                )
                for quantity in ("current", "spike", "membrane")
            }
            use_previous = int(time_index == 1 and layer.name not in ("L1", "L2"))
            log_path = args.output_dir / f"{layer.name.lower()}_t{time_index}.log"
            result = run_command(
                [
                    vvp,
                    str(simulation_binary),
                    f"+WEIGHT_FILE={weight_path}",
                    f"+CONSTANT_FILE={constant_path}",
                    f"+FRACTION_FILE={fraction_path}",
                    f"+INPUT_FILE={input_path}",
                    f"+PREVIOUS_STATE_FILE={previous_state_path}",
                    f"+EXPECTED_CURRENT_FILE={expected_paths['current']}",
                    f"+EXPECTED_SPIKE_FILE={expected_paths['spike']}",
                    f"+EXPECTED_MEMBRANE_FILE={expected_paths['membrane']}",
                    f"+WIDTH_LOG2={layer.width_log2}",
                    f"+HEIGHT={layer.height}",
                    f"+INPUT_CHANNELS={layer.input_channels}",
                    f"+OUTPUT_CHANNELS={layer.output_channels}",
                    f"+INPUT_SPATIAL={layer.input_spatial}",
                    f"+OUTPUT_SPATIAL={layer.output_spatial}",
                    f"+INPUT_TILES={layer.input_tiles}",
                    f"+OUTPUT_TILES={layer.output_tiles}",
                    f"+POOL_ENABLE={layer.pool}",
                    f"+USE_PREVIOUS_STATE={use_previous}",
                    f"+WRITE_STATE={int(layer.name not in ('L1', 'L2'))}",
                    f"+WEIGHT_BASE={layer_layout['beat_offset']}",
                    f"+PARAMETER_BASE={parameter_layout['record_offset']}",
                    "+TIMEOUT_CYCLES=100000",
                ],
                timeout_seconds=args.timeout_seconds,
                log_path=log_path,
            )
            checked(result, f"{layer.name} t{time_index} replay")
            text = log_path.read_text(encoding="utf-8")
            match = re.search(
                r"PASS: transactions=(\d+) values=(\d+) inference_cycles=(\d+)",
                text,
            )
            if match is None:
                raise RuntimeError(f"missing replay summary: {layer.name} t{time_index}")
            values = int(match.group(2))
            total_values += values
            replay_results.append(
                {
                    "layer": layer.name,
                    "time_index": time_index,
                    "use_previous_state": bool(use_previous),
                    "transactions": int(match.group(1)),
                    "values_checked": values,
                    "cycles_start_to_done": int(match.group(3)),
                    "run": result,
                    "inputs": {
                        "activation": str(input_path),
                        "previous_state": str(previous_state_path)
                        if use_previous
                        else None,
                    },
                    "expected": {
                        key: str(value) for key, value in expected_paths.items()
                    },
                }
            )
            print(
                f"PASS {layer.name} t{time_index}: "
                f"{values} values, {match.group(3)} cycles",
                flush=True,
            )

    lint_result = run_command(
        [
            verilator,
            "--lint-only",
            "--Wall",
            "--top-module",
            "conv3x3_layer_engine",
            *(relative(path) for path in sources),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "verilator.log",
    )
    checked(lint_result, "convolution-engine lint")

    mapping: dict[str, object] | None = None
    if not args.skip_mapping:
        stat_path = args.output_dir / "conv3x3_layer_engine_stat.json"
        mapping = run_command(
            [
                yosys,
                "-q",
                "-p",
                "; ".join(
                    (
                        "read_verilog -sv "
                        + " ".join(relative(path) for path in sources),
                        "hierarchy -top conv3x3_layer_engine",
                        "synth_xilinx -family xcup -noiopad "
                        "-top conv3x3_layer_engine",
                        f"tee -o {stat_path} stat -json",
                    )
                ),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / "yosys.log",
        )
        checked(mapping, "convolution-engine Yosys mapping")
        statistics = json.loads(stat_path.read_text(encoding="utf-8"))
        mapping["statistics_file"] = str(stat_path)
        mapping["statistics_sha256"] = sha256_file(stat_path)
        mapping["design_statistics"] = statistics["design"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "shared_pursuit_l1_l8_convolution_engine_bit_exact_replay",
        "driver": source_record([Path(__file__).resolve()])[0],
        "rtl_sources": source_record(sources),
        "testbench": source_record([testbench])[0],
        "golden_manifest": {
            "path": str(golden_manifest_path),
            "sha256": sha256_file(golden_manifest_path),
        },
        "hardware_image_layout": {
            "path": str(image_layout_path),
            "sha256": sha256_file(image_layout_path),
        },
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "compile": compile_result,
        "replays": replay_results,
        "summary": {
            "replays_passed": len(replay_results),
            "values_checked": total_values,
            "mismatches": 0,
            "range_errors": 0,
        },
        "lint": lint_result,
        "primitive_mapping": mapping,
        "interpretation_limit": (
            "Simulation establishes bit-exact layer behavior. Yosys mapping is "
            "pre-place-and-route only; XCZU3EG timing, routed utilization, "
            "BRAM fragmentation, and power require Vivado."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path)


if __name__ == "__main__":
    main()
