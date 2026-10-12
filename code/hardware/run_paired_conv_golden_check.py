"""Replay the layer-major paired-time L3--L8 convolution engine."""

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
    width_log2: int
    height: int
    input_channels: int
    output_channels: int
    input_spatial: int
    output_spatial: int
    pool: int
    reuse_current: int

    @property
    def input_tiles(self) -> int:
        return (self.input_channels + 3) // 4

    @property
    def output_tiles(self) -> int:
        return (self.output_channels + 7) // 8


LAYERS = (
    Layer("L3", "L2", 5, 24, 16, 24, 768, 192, 1, 1),
    Layer("L4", "L3", 4, 12, 24, 32, 192, 48, 1, 0),
    Layer("L5", "L4", 3, 6, 32, 40, 48, 12, 1, 0),
    Layer("L6", "L5", 2, 3, 40, 40, 12, 12, 0, 0),
    Layer("L7", "L6", 2, 3, 40, 40, 12, 12, 0, 0),
    Layer("L8", "L7", 2, 3, 40, 40, 12, 12, 0, 0),
)


def memory_path(manifest: dict[str, object], key: str) -> Path:
    record = manifest["artifacts"]["memory_files"][key]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"golden memory hash mismatch: {key}")
    return path


def artifact_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


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
        RTL_DIRECTORY / "paired_conv3x3_layer_engine.sv",
    ]
    testbench = RTL_DIRECTORY / "tb" / "tb_paired_conv3x3_layer_engine.sv"
    binary = args.output_dir / "tb_paired_conv3x3_layer_engine.vvp"
    compile_result = run_command(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_paired_conv3x3_layer_engine",
            "-o",
            str(binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "iverilog.log",
    )
    checked(compile_result, "paired convolution compile")

    weight_path = Path(image_layout["artifacts"]["convolution_beats"]["path"])
    constant_path = Path(image_layout["artifacts"]["constants"]["path"])
    fraction_path = Path(image_layout["artifacts"]["fractions"]["path"])
    replays = []
    total_values = 0
    total_cycles = 0
    for layer in LAYERS:
        layout = image_layout["convolution"]["layers"][layer.name]
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
                f"+WIDTH_LOG2={layer.width_log2}",
                f"+HEIGHT={layer.height}",
                f"+INPUT_CHANNELS={layer.input_channels}",
                f"+OUTPUT_CHANNELS={layer.output_channels}",
                f"+INPUT_SPATIAL={layer.input_spatial}",
                f"+OUTPUT_SPATIAL={layer.output_spatial}",
                f"+INPUT_TILES={layer.input_tiles}",
                f"+OUTPUT_TILES={layer.output_tiles}",
                f"+POOL_ENABLE={layer.pool}",
                f"+REUSE_CURRENT={layer.reuse_current}",
                f"+WEIGHT_BASE={layout['beat_offset']}",
                f"+PARAMETER_BASE={parameters['record_offset']}",
                "+TIMEOUT_CYCLES=100000",
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=log_path,
        )
        checked(result, f"{layer.name} paired replay")
        match = re.search(
            r"PASS: transactions=(\d+) values=(\d+) inference_cycles=(\d+)",
            log_path.read_text(encoding="utf-8"),
        )
        if match is None:
            raise RuntimeError(f"missing replay summary for {layer.name}")
        values = int(match.group(2))
        cycles = int(match.group(3))
        total_values += values
        total_cycles += cycles
        replays.append(
            {
                "layer": layer.name,
                "transactions": int(match.group(1)),
                "values_checked": values,
                "cycles_start_to_done": cycles,
                "run": result,
            }
        )
        print(f"PASS {layer.name}: {values} values, {cycles} cycles", flush=True)

    lint = run_command(
        [
            verilator,
            "--lint-only",
            "--Wall",
            "--top-module",
            "paired_conv3x3_layer_engine",
            *(relative(path) for path in sources),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "verilator.log",
    )
    checked(lint, "paired convolution lint")

    mapping = None
    if not args.skip_mapping:
        stat_path = args.output_dir / "paired_conv3x3_layer_engine_stat.json"
        mapping = run_command(
            [
                yosys,
                "-q",
                "-p",
                "; ".join(
                    (
                        "read_verilog -sv "
                        + " ".join(relative(path) for path in sources),
                        "hierarchy -top paired_conv3x3_layer_engine",
                        "synth_xilinx -family xcup -noiopad "
                        "-top paired_conv3x3_layer_engine",
                        f"tee -o {stat_path} stat -json",
                    )
                ),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / "yosys.log",
        )
        checked(mapping, "paired convolution mapping")
        statistics = json.loads(stat_path.read_text(encoding="utf-8"))
        mapping["statistics_file"] = str(stat_path)
        mapping["statistics_sha256"] = sha256_file(stat_path)
        mapping["design_statistics"] = statistics["design"]

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "paired_time_l3_l8_golden_replay_before_vivado",
        "contract": {
            "schedule": "layer-major; one weight tile load followed by t0 and t1 scans",
            "arithmetic": "S16 weights, S40 partial sums, pool in S40, add constant once, S24 current, S16 state, U8 spike",
            "state": "local t0 membrane RAM consumed by t1",
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
        "totals": {
            "values_checked": total_values,
            "cycles_start_to_done_sum": total_cycles,
        },
        "compile": compile_result,
        "lint": lint,
        "mapping": mapping,
    }
    output = args.output_dir / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
