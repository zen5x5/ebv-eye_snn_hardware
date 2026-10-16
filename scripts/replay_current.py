"""Replay the current user17 sample using Vivado XSim and portable evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.hardware.run_sparse_layer_major_golden_check import (
    build_sorted_event_file,
    build_trace_memories,
    memory_path,
    read_hex_words,
)

SOURCES = (
    "sparse_conv3x3_pool_lif_core.sv", "pursuit_sparse_l1_event_layer.sv",
    "pursuit_sparse_l2_packet_layer.sv", "pursuit_sparse_frontend.sv",
    "programmable_padded_line_buffer_3x3.sv", "programmable_maxpool2x2_stream.sv",
    "pipelined_count_mac_array.sv", "lif_multispike.sv",
    "paired_conv3x3_layer_engine.sv", "pursuit_paired_feature_store.sv",
    "paired_linear_lif_layer_engine.sv", "readout_engine.sv",
    "pursuit_sparse_layer_major_accelerator_top.sv",
)
SUMMARY_NAMES = (
    "trace_transactions", "values_checked", "total_cycles", "frontend_cycles",
    "convolution_cycles", "dense_cycles", "conv_weight_read_beats",
    "linear_weight_read_beats", "readout_weight_read_beats",
    "feature_read_requests", "feature_write_transactions", "accepted_events",
    "l1_emitted_packets", "l2_accepted_packets", "frontend_trace_collisions",
)
EXPECTED = (5866, 159408, 104005, 51305, 48609, 4074, 4068, 1980, 12,
            25872, 2016, 400, 3315, 3315, 0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    os.chdir(ROOT)
    current = ROOT / "evidence/current"
    golden_path = current / "golden_vectors/manifest.json"
    layout_path = current / "hardware_image/layout.json"
    golden = json.loads(golden_path.read_text())
    layout = json.loads(layout_path.read_text())
    for record in layout["artifacts"].values():
        path = ROOT / record["path"]
        assert sha256(path) == record["sha256"], path
    event_path, event_record = build_sorted_event_file(golden, output)
    traces = build_trace_memories(golden, output)
    params = {f"L{layer}_{name.upper()}_FILE": memory_path(golden, f"L{layer}.{key}").resolve()
              for layer in (1, 2)
              for name, key in (("weight", "weight_integer"),
                                ("constant", "constant_integer"),
                                ("fraction", "current_and_state_fraction_bits"))}
    for argument, key in (
        ("CONV_WEIGHT_FILE", "convolution_beats"),
        ("LINEAR_WEIGHT_FILE", "linear_beats"),
        ("CONSTANT_FILE", "constants"), ("FRACTION_FILE", "fractions"),
        ("READOUT_FRACTION_FILE", "readout_fractions"),
    ):
        params[argument] = ROOT / layout["artifacts"][key]["path"]
    params["EVENT_FILE"] = event_path
    for quantity in ("current", "spike", "membrane"):
        params[f"EXPECTED_{quantity.upper()}_FILE"] = traces[quantity]["path"]
    params["EXPECTED_RESULT_FILE"] = memory_path(
        golden, "readout.output_integer_before_temporal_divide").resolve()
    params["EXPECTED_L1_PACKETS"] = sum(v != 0 for v in read_hex_words(
        memory_path(golden, "t0.L1.spike_integer")))
    params["TIMEOUT_CYCLES"] = 200000
    sources = [ROOT / "code/hardware/rtl" / name for name in SOURCES]
    testbench = ROOT / "code/hardware/rtl/tb/tb_pursuit_sparse_layer_major_accelerator_top.sv"
    monitor = current / "rtl_replay/full_walkthrough_monitor.sv"
    commands = [
        ("xvlog", ["--sv", *map(str, sources), str(testbench), str(monitor)], "compile.log"),
        ("xelab", ["tb_pursuit_sparse_layer_major_accelerator_top", "full_walkthrough_monitor",
                    "--snapshot", "current_replay", "--debug", "typical"], "elaborate.log"),
        ("xsim", ["current_replay", "--runall",
                  *[item for key, value in params.items()
                    for item in ("--testplusarg", f"{key}={value}")]], "simulation.log"),
    ]
    for name, arguments, log_name in commands:
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f"{name}: source Vivado/settings64.sh before running")
        print(f"Running {name}", flush=True)
        result = subprocess.run([executable, *arguments], cwd=output,
                                text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (output / log_name).write_text(result.stdout)
        result.check_returncode()
    log = (output / "simulation.log").read_text()
    match = re.search(r"PASS: transactions=(\d+) values=(\d+) total_cycles=(\d+) "
                      r"frontend_cycles=(\d+) convolution_cycles=(\d+) dense_cycles=(\d+) "
                      r"conv_weight_reads=(\d+) linear_weight_reads=(\d+) readout_weight_reads=(\d+) "
                      r"feature_reads=(\d+) feature_writes=(\d+) events=(\d+) l1_packets=(\d+) "
                      r"l2_packets=(\d+) frontend_trace_collisions=(\d+)", log)
    assert match is not None, "XSim omitted the passing golden comparison summary"
    actual = tuple(map(int, match.groups()))
    assert actual == EXPECTED, (actual, EXPECTED)
    summary = dict(zip(SUMMARY_NAMES, actual))
    report = {
        "model": "Pursuit user17 seed2022", "simulator": "Vivado XSim",
        "scope": "RTL functional simulation", "status": "passed",
        "mismatches": 0, "simulation_summary": summary, "event_stream": event_record,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p)
                          for p in [*sources, testbench, monitor]},
        "golden_manifest_sha256": sha256(golden_path),
        "image_layout_sha256": sha256(layout_path),
        "simulation_log_sha256": sha256(output / "simulation.log"),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Golden comparison: passed, 0 mismatches")


if __name__ == "__main__":
    main()
