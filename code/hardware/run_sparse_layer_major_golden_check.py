"""Verify the complete Sparse-LM-A pursuit accelerator against seed2022."""

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
LAYERS = tuple(f"L{index}" for index in range(1, 9)) + ("H1", "H2")
QUANTITIES = ("current", "spike", "membrane")
EXPECTED_TRACE_VALUES = 159_408
EXPECTED_TRACE_TRANSACTIONS = 5_866
EXPECTED_SIMULATION_COUNTS = {
    "total_cycles": 67_929,
    "frontend_cycles": 15_259,
    "convolution_cycles": 48_579,
    "dense_cycles": 4_074,
    "conv_weight_read_beats": 4_068,
    "linear_weight_read_beats": 1_980,
    "readout_weight_read_beats": 12,
    "feature_read_requests": 25_872,
    "feature_write_transactions": 2_016,
    "accepted_events": 400,
}


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "lines": len(path.read_text(encoding="ascii").splitlines()),
    }


def memory_path(manifest: dict[str, object], key: str) -> Path:
    record = manifest["artifacts"]["memory_files"][key]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"golden memory hash mismatch: {key}")
    lines = len(path.read_text(encoding="ascii").splitlines())
    if lines != int(record["elements"]):
        raise ValueError(f"golden memory length mismatch: {key}")
    return path


def read_hex_words(path: Path) -> list[int]:
    return [
        int(line.strip(), 16)
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]


def build_sorted_event_file(
    manifest: dict[str, object], output_directory: Path
) -> tuple[Path, dict[str, object]]:
    input_path = memory_path(manifest, "input.event_count_integer")
    counts = read_hex_words(input_path)
    if len(counts) != 2 * 96 * 128:
        raise ValueError(f"unexpected input tensor length: {len(counts)}")
    events: list[tuple[int, int, int]] = []
    spatial = 96 * 128
    for polarity in range(2):
        for row in range(96):
            for column in range(128):
                count = counts[polarity * spatial + row * 128 + column]
                events.extend((row, column, polarity) for _ in range(count))
    events.sort(key=lambda event: (event[0], event[1], event[2]))
    if len(events) != 400:
        raise ValueError(f"formal input expands to {len(events)} events, not 400")
    output = output_directory / "input_events_sorted_row_column_polarity.mem"
    with output.open("w", encoding="ascii") as handle:
        for row, column, polarity in events:
            packed = (row << 8) | (column << 1) | polarity
            handle.write(f"{packed:04x}\n")
    unique_addresses = len(set(events))
    maximum_multiplicity = max(events.count(event) for event in set(events))
    return output, {
        "source_count_tensor": artifact_record(input_path),
        "generated_event_stream": artifact_record(output),
        "order": "row, column, polarity; repeated entries preserve count",
        "events": len(events),
        "unique_polarity_positions": unique_addresses,
        "maximum_multiplicity": maximum_multiplicity,
    }


def build_trace_memories(
    manifest: dict[str, object], output_directory: Path
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for quantity in QUANTITIES:
        output = output_directory / f"all_{quantity}.mem"
        lines = 0
        with output.open("w", encoding="ascii") as destination:
            for time_index in (0, 1):
                for layer in LAYERS:
                    source = memory_path(
                        manifest,
                        f"t{time_index}.{layer}.{quantity}_integer",
                    )
                    text = source.read_text(encoding="ascii")
                    destination.write(text)
                    lines += len(text.splitlines())
        if lines != 90_000:
            raise AssertionError(f"{output}: expected 90000 lines, got {lines}")
        records[quantity] = artifact_record(output)
    return records


LUTRAM_HALF_LUT_CAPACITY_BOUNDS = {
    # Bounds are in half-LUT units.  A single-port 16x1 or 32x1 can share a
    # LUT6_2 when controls and addresses permit; the other listed macros have
    # fixed LUT capacity at the primitive level.
    "RAM16X1S": (1, 2),
    "RAM16X1S_1": (1, 2),
    "RAM32X1S": (1, 2),
    "RAM32X1S_1": (1, 2),
    "RAM64X1S": (2, 2),
    "RAM64X1S_1": (2, 2),
    "RAM128X1S": (4, 4),
    "RAM128X1S_1": (4, 4),
    "RAM256X1S": (8, 8),
    "RAM512X1S": (16, 16),
    "RAM16X2S": (2, 2),
    "RAM32X2S": (2, 2),
    "RAM64X2S": (4, 4),
    "RAM16X4S": (4, 4),
    "RAM32X4S": (4, 4),
    "RAM16X8S": (8, 8),
    "RAM32X8S": (8, 8),
    "RAM16X1D": (2, 2),
    "RAM16X1D_1": (2, 2),
    "RAM32X1D": (2, 2),
    "RAM32X1D_1": (2, 2),
    "RAM64X1D": (4, 4),
    "RAM64X1D_1": (4, 4),
    "RAM128X1D": (8, 8),
    "RAM256X1D": (16, 16),
    "RAM32M": (8, 8),
    "RAM32M16": (16, 16),
    "RAM64M": (8, 8),
    "RAM64M8": (16, 16),
    "RAM32X16DR8": (16, 16),
    "RAM64X8SW": (16, 16),
}


def mapping_summary(statistics: dict[str, object]) -> dict[str, object]:
    cells = statistics["design"]["num_cells_by_type"]
    logic_luts = {
        f"LUT{inputs}": int(cells.get(f"LUT{inputs}", 0))
        for inputs in range(1, 7)
    }
    distributed_ram = {
        primitive: int(count)
        for primitive, count in sorted(cells.items())
        if primitive.startswith("RAM") and not primitive.startswith("RAMB")
    }
    unknown_lutram = sorted(
        set(distributed_ram) - set(LUTRAM_HALF_LUT_CAPACITY_BOUNDS)
    )
    lutram_lower_half_units = sum(
        count * LUTRAM_HALF_LUT_CAPACITY_BOUNDS.get(primitive, (0, 0))[0]
        for primitive, count in distributed_ram.items()
    )
    lutram_upper_half_units = sum(
        count * LUTRAM_HALF_LUT_CAPACITY_BOUNDS.get(primitive, (0, 0))[1]
        for primitive, count in distributed_ram.items()
    )
    distributed_ram_lut_lower_bound = (lutram_lower_half_units + 1) // 2
    distributed_ram_lut_upper_bound = (lutram_upper_half_units + 1) // 2
    logical_lut_count = sum(logic_luts.values())
    physical_lut_lower_bound = (
        distributed_ram_lut_lower_bound
        + max(logic_luts["LUT6"], (logical_lut_count + 1) // 2)
    )
    physical_lut_upper_bound = (
        distributed_ram_lut_upper_bound + logical_lut_count
    )
    ramb36e2 = int(cells.get("RAMB36E2", 0))
    ramb18e2 = int(cells.get("RAMB18E2", 0))
    return {
        "dsp48e2": int(cells.get("DSP48E2", 0)),
        "ramb36e2": ramb36e2,
        "ramb18e2": ramb18e2,
        "bram36_equivalent": ramb36e2 + ramb18e2 / 2,
        "flip_flops": int(
            sum(cells.get(primitive, 0) for primitive in ("FDCE", "FDPE", "FDRE", "FDSE"))
        ),
        "luts": logical_lut_count,
        "logic_lut_primitives": logic_luts,
        "distributed_ram_primitives": distributed_ram,
        "distributed_ram_lut_lower_bound": (
            distributed_ram_lut_lower_bound
        ),
        "distributed_ram_lut_upper_bound": (
            distributed_ram_lut_upper_bound
        ),
        "physical_lut_lower_bound": physical_lut_lower_bound,
        "physical_lut_upper_bound": physical_lut_upper_bound,
        "unpacked_logic_plus_lutram_capacity": physical_lut_upper_bound,
        "unknown_lutram_capacity_primitives": unknown_lutram,
        "aup_zu3_dsp48e2_capacity": 360,
        "aup_zu3_bram36_equivalent_capacity": 216,
        "aup_zu3_physical_lut_capacity": 70_560,
        "aup_zu3_flip_flop_capacity": 141_120,
        "carry4": int(cells.get("CARRY4", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--skip-mapping", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    args.output_dir.mkdir(parents=True)

    golden_manifest_path = args.golden_dir / "manifest.json"
    golden_manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    state_mode = golden_manifest["arithmetic_contract"].get(
        "state_mode", "l1_l2_stateless"
    )
    if state_mode != "l1_l2_stateless":
        raise ValueError(f"Sparse-LM-A requires stateless golden, got {state_mode}")
    image_layout_path = args.image_dir / "layout.json"
    image_layout = json.loads(image_layout_path.read_text(encoding="utf-8"))
    for name, record in image_layout["artifacts"].items():
        path = Path(record["path"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"hardware image hash mismatch: {name}")

    event_path, event_record = build_sorted_event_file(
        golden_manifest, args.output_dir
    )
    trace_records = build_trace_memories(golden_manifest, args.output_dir)
    l1_spike_path = memory_path(golden_manifest, "t0.L1.spike_integer")
    expected_l1_packets = sum(value != 0 for value in read_hex_words(l1_spike_path))

    l1_paths = {
        "weight": memory_path(golden_manifest, "L1.weight_integer"),
        "constant": memory_path(golden_manifest, "L1.constant_integer"),
        "fraction": memory_path(
            golden_manifest, "L1.current_and_state_fraction_bits"
        ),
    }
    l2_paths = {
        "weight": memory_path(golden_manifest, "L2.weight_integer"),
        "constant": memory_path(golden_manifest, "L2.constant_integer"),
        "fraction": memory_path(
            golden_manifest, "L2.current_and_state_fraction_bits"
        ),
    }
    result_path = memory_path(
        golden_manifest, "readout.output_integer_before_temporal_divide"
    )

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")
    sources = [
        RTL_DIRECTORY / name
        for name in (
            "sparse_conv3x3_pool_lif_core.sv",
            "pursuit_sparse_l1_event_layer.sv",
            "pursuit_sparse_l2_packet_layer.sv",
            "pursuit_sparse_frontend.sv",
            "programmable_padded_line_buffer_3x3.sv",
            "programmable_maxpool2x2_stream.sv",
            "pipelined_count_mac_array.sv",
            "lif_multispike.sv",
            "paired_conv3x3_layer_engine.sv",
            "pursuit_paired_feature_store.sv",
            "paired_linear_lif_layer_engine.sv",
            "readout_engine.sv",
            "pursuit_sparse_layer_major_accelerator_top.sv",
        )
    ]
    testbench = (
        RTL_DIRECTORY
        / "tb"
        / "tb_pursuit_sparse_layer_major_accelerator_top.sv"
    )
    binary = args.output_dir / "tb_pursuit_sparse_layer_major.vvp"
    compile_result = run_command(
        [
            iverilog,
            "-g2012",
            "-Wall",
            "-s",
            "tb_pursuit_sparse_layer_major_accelerator_top",
            "-o",
            str(binary),
            *(relative(path) for path in [*sources, testbench]),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "iverilog.log",
    )
    checked(compile_result, "Sparse-LM-A Icarus compile")

    artifacts = image_layout["artifacts"]
    simulation_log = args.output_dir / "simulation.log"
    simulation = run_command(
        [
            vvp,
            str(binary),
            f"+EVENT_FILE={event_path}",
            f"+L1_WEIGHT_FILE={l1_paths['weight']}",
            f"+L1_CONSTANT_FILE={l1_paths['constant']}",
            f"+L1_FRACTION_FILE={l1_paths['fraction']}",
            f"+L2_WEIGHT_FILE={l2_paths['weight']}",
            f"+L2_CONSTANT_FILE={l2_paths['constant']}",
            f"+L2_FRACTION_FILE={l2_paths['fraction']}",
            f"+CONV_WEIGHT_FILE={artifacts['convolution_beats']['path']}",
            f"+LINEAR_WEIGHT_FILE={artifacts['linear_beats']['path']}",
            f"+CONSTANT_FILE={artifacts['constants']['path']}",
            f"+FRACTION_FILE={artifacts['fractions']['path']}",
            f"+READOUT_FRACTION_FILE={artifacts['readout_fractions']['path']}",
            f"+EXPECTED_CURRENT_FILE={trace_records['current']['path']}",
            f"+EXPECTED_SPIKE_FILE={trace_records['spike']['path']}",
            f"+EXPECTED_MEMBRANE_FILE={trace_records['membrane']['path']}",
            f"+EXPECTED_RESULT_FILE={result_path}",
            f"+EXPECTED_L1_PACKETS={expected_l1_packets}",
            "+TIMEOUT_CYCLES=200000",
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=simulation_log,
    )
    checked(simulation, "Sparse-LM-A full golden replay")
    pattern = re.compile(
        r"PASS: transactions=(\d+) values=(\d+) total_cycles=(\d+) "
        r"frontend_cycles=(\d+) convolution_cycles=(\d+) dense_cycles=(\d+) "
        r"conv_weight_reads=(\d+) linear_weight_reads=(\d+) "
        r"readout_weight_reads=(\d+) feature_reads=(\d+) "
        r"feature_writes=(\d+) events=(\d+) l1_packets=(\d+) "
        r"l2_packets=(\d+) frontend_trace_collisions=(\d+)"
    )
    match = pattern.search(simulation_log.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError("Sparse-LM-A replay omitted its audit summary")
    names = (
        "trace_transactions",
        "values_checked",
        "total_cycles",
        "frontend_cycles",
        "convolution_cycles",
        "dense_cycles",
        "conv_weight_read_beats",
        "linear_weight_read_beats",
        "readout_weight_read_beats",
        "feature_read_requests",
        "feature_write_transactions",
        "accepted_events",
        "l1_emitted_packets",
        "l2_accepted_packets",
        "frontend_trace_collisions",
    )
    summary = {name: int(value) for name, value in zip(names, match.groups())}
    if summary["trace_transactions"] != EXPECTED_TRACE_TRANSACTIONS:
        raise AssertionError("unexpected trace transaction total")
    if summary["values_checked"] != EXPECTED_TRACE_VALUES:
        raise AssertionError("unexpected checked value total")
    for name, expected in EXPECTED_SIMULATION_COUNTS.items():
        if summary[name] != expected:
            raise AssertionError(
                f"unexpected {name}: {summary[name]} != {expected}"
            )
    if summary["l1_emitted_packets"] != expected_l1_packets:
        raise AssertionError("unexpected L1 sparse packet total")
    if summary["l2_accepted_packets"] != expected_l1_packets:
        raise AssertionError("unexpected L2 sparse packet total")
    if summary["frontend_trace_collisions"] != 0:
        raise AssertionError("frontend trace collisions must be zero")

    lint = run_command(
        [
            verilator,
            "--lint-only",
            "--Wall",
            "--top-module",
            "pursuit_sparse_layer_major_accelerator_top",
            *(relative(path) for path in sources),
        ],
        timeout_seconds=args.timeout_seconds,
        log_path=args.output_dir / "verilator.log",
    )
    checked(lint, "Sparse-LM-A Verilator lint")

    mapping = None
    if not args.skip_mapping:
        stat_path = args.output_dir / "sparse_layer_major_stat.json"
        mapping = run_command(
            [
                yosys,
                "-q",
                "-p",
                "; ".join(
                    (
                        "read_verilog -sv "
                        + " ".join(relative(path) for path in sources),
                        "hierarchy -top pursuit_sparse_layer_major_accelerator_top",
                        "synth_xilinx -family xcup -noiopad "
                        "-top pursuit_sparse_layer_major_accelerator_top",
                        f"tee -o {stat_path} stat -json",
                    )
                ),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / "yosys.log",
        )
        checked(mapping, "Sparse-LM-A Yosys mapping")
        statistics = json.loads(stat_path.read_text(encoding="utf-8"))
        mapping["statistics_file"] = str(stat_path)
        mapping["statistics_sha256"] = sha256_file(stat_path)
        mapping["design_statistics"] = statistics["design"]
        mapping["summary"] = mapping_summary(statistics)
        mapping["scope"] = (
            "Yosys synth_xilinx xcup primitive mapping without I/O pads; "
            "no placement, routing, timing, or power"
        )

    consumed = {
        **{f"L1_{name}": artifact_record(path) for name, path in l1_paths.items()},
        **{f"L2_{name}": artifact_record(path) for name, path in l2_paths.items()},
        "expected_readout": artifact_record(result_path),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "complete_pursuit_sparse_layer_major_a_before_vivado",
        "runner": artifact_record(Path(__file__).resolve()),
        "contract": {
            "frontend": "event-driven L1/L2; one stateless realization",
            "backend": (
                "L3 current reuse, paired t0/t1 L4-L8, paired H1/H2"
            ),
            "arithmetic": (
                "S16 weights, S40 accumulation, S24 current, "
                "S16 membrane, U8 spike"
            ),
        },
        "golden_manifest": artifact_record(golden_manifest_path),
        "hardware_image": artifact_record(image_layout_path),
        "event_stream": event_record,
        "consumed_flat_artifacts": consumed,
        "generated_trace_memories": trace_records,
        "rtl_sources": source_record(sources),
        "testbench": source_record([testbench])[0],
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "vvp": tool_version([vvp, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "simulation": {
            "compile": compile_result,
            "run": simulation,
            **summary,
            "expected_l1_nonzero_packets": expected_l1_packets,
        },
        "lint": lint,
        "mapping": mapping,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"PASS Sparse-LM-A: {summary['values_checked']} values, "
        f"{summary['total_cycles']} cycles"
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
