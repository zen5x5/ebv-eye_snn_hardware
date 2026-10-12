"""Run reproducible open-source checks for the hardware arithmetic RTL.

The report covers behavior simulation, lint, and Xilinx UltraScale+ primitive
mapping.  Yosys mapping is useful for architecture rejection and DSP counts;
it is not a substitute for Vivado placement, routing, timing, or power.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


SCHEMA_VERSION = 1
RTL_DIRECTORY = Path(__file__).resolve().parent / "rtl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable(name: str) -> str:
    result = shutil.which(name)
    if result is None:
        sibling = Path(sys.executable).resolve().parent / name
        if sibling.is_file():
            result = str(sibling)
    if result is None:
        raise FileNotFoundError(f"required executable is not on PATH: {name}")
    return result


def run_command(
    command: list[str],
    *,
    timeout_seconds: float,
    log_path: Path,
) -> dict[str, object]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        log_path.write_text(output, encoding="utf-8")
        return {
            "status": "timeout",
            "runtime_seconds": time.monotonic() - started,
            "command": command,
            "log": str(log_path),
            "log_sha256": sha256_file(log_path),
        }
    log_path.write_text(completed.stdout, encoding="utf-8")
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "runtime_seconds": time.monotonic() - started,
        "command": command,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def checked(result: dict[str, object], label: str) -> None:
    if result["status"] != "passed":
        raise RuntimeError(f"{label} failed; inspect {result['log']}")


def relative(path: Path) -> str:
    return str(path.relative_to(Path(__file__).resolve().parents[2]))


def source_record(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": relative(path), "sha256": sha256_file(path)}
        for path in paths
    ]


def yosys_mapping(
    *,
    yosys: str,
    name: str,
    top: str,
    sources: list[Path],
    parameters: dict[str, int],
    output_directory: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    stat_path = output_directory / f"{name}_stat.json"
    log_path = output_directory / f"{name}_yosys.log"
    parameter_command = " ".join(
        f"-set {key} {value}" for key, value in parameters.items()
    )
    script = "; ".join(
        (
            "read_verilog -sv " + " ".join(relative(path) for path in sources),
            f"chparam {parameter_command} {top}",
            f"hierarchy -top {top}",
            f"synth_xilinx -family xcup -noiopad -top {top}",
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
            "top": top,
            "parameters": parameters,
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


def tool_version(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    return completed.stdout.strip().splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    args.output_dir.mkdir(parents=True)

    iverilog = executable("iverilog")
    vvp = executable("vvp")
    verilator = executable("verilator")
    yosys = executable("yosys")
    tree = RTL_DIRECTORY / "balanced_signed_adder_tree.sv"
    arithmetic_sources = [
        tree,
        RTL_DIRECTORY / "synapse_count_mac_array.sv",
        RTL_DIRECTORY / "synapse_shift_add_array.sv",
        RTL_DIRECTORY / "synapse_unary_add_array.sv",
        RTL_DIRECTORY / "tb" / "tb_synapse_arrays.sv",
    ]
    lif_sources = [
        RTL_DIRECTORY / "lif_multispike.sv",
        RTL_DIRECTORY / "tb" / "tb_lif_multispike.sv",
    ]
    pipeline_sources = [
        RTL_DIRECTORY / "pipelined_count_mac_array.sv",
        RTL_DIRECTORY / "tb" / "tb_pipelined_count_mac_array.sv",
    ]

    with tempfile.TemporaryDirectory(prefix="ebv_eye_rtl_") as temporary:
        temporary_path = Path(temporary)
        simulations: dict[str, object] = {}
        for name, top, sources in (
            ("synapse_arrays", "tb_synapse_arrays", arithmetic_sources),
            ("lif_multispike", "tb_lif_multispike", lif_sources),
            (
                "pipelined_count_mac_array",
                "tb_pipelined_count_mac_array",
                pipeline_sources,
            ),
        ):
            executable_path = temporary_path / f"{name}.vvp"
            compile_result = run_command(
                [
                    iverilog,
                    "-g2012",
                    "-Wall",
                    "-s",
                    top,
                    "-o",
                    str(executable_path),
                    *(relative(path) for path in sources),
                ],
                timeout_seconds=args.timeout_seconds,
                log_path=args.output_dir / f"{name}_iverilog.log",
            )
            checked(compile_result, f"{name} Icarus compile")
            run_result = run_command(
                [vvp, str(executable_path)],
                timeout_seconds=args.timeout_seconds,
                log_path=args.output_dir / f"{name}_simulation.log",
            )
            checked(run_result, f"{name} simulation")
            simulations[name] = {
                "sources": source_record(sources),
                "compile": compile_result,
                "run": run_result,
            }

    lint: dict[str, object] = {}
    for top, sources in (
        (
            "synapse_count_mac_array",
            [tree, RTL_DIRECTORY / "synapse_count_mac_array.sv"],
        ),
        (
            "synapse_shift_add_array",
            [tree, RTL_DIRECTORY / "synapse_shift_add_array.sv"],
        ),
        (
            "synapse_unary_add_array",
            [tree, RTL_DIRECTORY / "synapse_unary_add_array.sv"],
        ),
        ("lif_multispike", [RTL_DIRECTORY / "lif_multispike.sv"]),
        (
            "pipelined_count_mac_array",
            [RTL_DIRECTORY / "pipelined_count_mac_array.sv"],
        ),
    ):
        result = run_command(
            [
                verilator,
                "--lint-only",
                "--Wall",
                "--top-module",
                top,
                *(relative(path) for path in sources),
            ],
            timeout_seconds=args.timeout_seconds,
            log_path=args.output_dir / f"{top}_verilator.log",
        )
        checked(result, f"{top} Verilator lint")
        result["sources"] = source_record(sources)
        lint[top] = result

    common_parameters = {
        "OUTPUT_PES": 8,
        "WEIGHT_BITS": 16,
        "ACCUMULATOR_BITS": 40,
    }
    mappings = {
        "count_mac_8x18": yosys_mapping(
            yosys=yosys,
            name="count_mac_8x18",
            top="synapse_count_mac_array",
            sources=[tree, RTL_DIRECTORY / "synapse_count_mac_array.sv"],
            parameters={**common_parameters, "LANES": 18, "COUNT_BITS": 9},
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "count_mac_8x36": yosys_mapping(
            yosys=yosys,
            name="count_mac_8x36",
            top="synapse_count_mac_array",
            sources=[tree, RTL_DIRECTORY / "synapse_count_mac_array.sv"],
            parameters={**common_parameters, "LANES": 36, "COUNT_BITS": 9},
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "binary_term_8x18": yosys_mapping(
            yosys=yosys,
            name="binary_term_8x18",
            top="synapse_shift_add_array",
            sources=[tree, RTL_DIRECTORY / "synapse_shift_add_array.sv"],
            parameters={**common_parameters, "LANES": 18, "SHIFT_BITS": 4},
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "unary_add_8x18": yosys_mapping(
            yosys=yosys,
            name="unary_add_8x18",
            top="synapse_unary_add_array",
            sources=[tree, RTL_DIRECTORY / "synapse_unary_add_array.sv"],
            parameters={**common_parameters, "LANES": 18},
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "lif_scalar": yosys_mapping(
            yosys=yosys,
            name="lif_scalar",
            top="lif_multispike",
            sources=[RTL_DIRECTORY / "lif_multispike.sv"],
            parameters={
                "CURRENT_BITS": 24,
                "STATE_BITS": 16,
                "SPIKE_BITS": 8,
                "FRACTION_BITS_WIDTH": 5,
            },
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "pipelined_count_mac_8x18": yosys_mapping(
            yosys=yosys,
            name="pipelined_count_mac_8x18",
            top="pipelined_count_mac_array",
            sources=[RTL_DIRECTORY / "pipelined_count_mac_array.sv"],
            parameters={
                **common_parameters,
                "LANES": 18,
                "COUNT_BITS": 9,
                "TAG_BITS": 16,
            },
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
        "pipelined_count_mac_8x36": yosys_mapping(
            yosys=yosys,
            name="pipelined_count_mac_8x36",
            top="pipelined_count_mac_array",
            sources=[RTL_DIRECTORY / "pipelined_count_mac_array.sv"],
            parameters={
                **common_parameters,
                "LANES": 36,
                "COUNT_BITS": 9,
                "TAG_BITS": 16,
            },
            output_directory=args.output_dir,
            timeout_seconds=args.timeout_seconds,
        ),
    }
    for name, result in mappings.items():
        checked(result, f"{name} Yosys mapping")

    report = {
        "schema_version": SCHEMA_VERSION,
        "scope": "open_source_rtl_simulation_lint_and_ultrascale_plus_primitive_mapping",
        "driver": {
            "path": relative(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "tools": {
            "iverilog": tool_version([iverilog, "-V"]),
            "verilator": tool_version([verilator, "--version"]),
            "yosys": tool_version([yosys, "--version"]),
        },
        "simulation": simulations,
        "lint": lint,
        "primitive_mapping": mappings,
        "interpretation_limit": (
            "These results establish arithmetic behavior and primitive counts. "
            "Only Vivado implementation for XCZU3EG can establish Fmax, WNS, "
            "routed utilization, BRAM fragmentation, and power."
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
