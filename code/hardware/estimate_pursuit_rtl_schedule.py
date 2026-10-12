"""Estimate a realizable schedule for the selected pursuit convolution core.

Unlike the raw MAC lower bound, this model charges padded line-buffer scans,
pipeline fill/drain, finite-width weight loading, and the SL-HYB L3-current
replay.  It still precedes complete multi-layer controller simulation and
Vivado timing, so both cycle estimates and BRAM counts remain design estimates.
The default fill/drain allowance is calibrated to the complete bit-exact L1
RTL replay.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

from .explore_pursuit_dataflow import (
    CONV_LAYERS,
    FOLDED_CONSTANT_COUNT,
    FULL_STATE_COUNT,
    L3_CURRENT_CACHE_COUNT,
    LINEAR_LAYERS,
    READOUT,
    STATELESS_STATE_COUNT,
    TARGET_PART,
    WEIGHT_COUNT,
)


SCHEMA_VERSION = 1
AUP_ZU3_BRAM36_BLOCKS = 216
BRAM36_BITS = 36 * 1024


@dataclass(frozen=True)
class Core:
    output_pes: int
    input_channels_per_pe: int
    kernel_taps: int

    @property
    def lanes_per_pe(self) -> int:
        return self.input_channels_per_pe * self.kernel_taps

    @property
    def multiplier_lanes(self) -> int:
        return self.output_pes * self.lanes_per_pe


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convolution_schedule(
    core: Core,
    *,
    weights_per_load_cycle: int,
    pipeline_latency: int,
) -> list[dict[str, object]]:
    result = []
    for layer in CONV_LAYERS:
        tile_load_cycles = []
        tile_compute_cycles = []
        for output_start in range(0, layer.out_channels, core.output_pes):
            valid_outputs = min(core.output_pes, layer.out_channels - output_start)
            for input_start in range(
                0, layer.in_channels, core.input_channels_per_pe
            ):
                valid_inputs = min(
                    core.input_channels_per_pe,
                    layer.in_channels - input_start,
                )
                weights = valid_outputs * valid_inputs * layer.kernel_elements
                tile_load_cycles.append(
                    math.ceil(weights / weights_per_load_cycle)
                )
                padded_scan = (
                    layer.out_height + 2
                ) * (layer.out_width + 2) * math.ceil(
                    layer.kernel_elements / core.kernel_taps
                )
                tile_compute_cycles.append(padded_scan + pipeline_latency)

        serialized = sum(tile_load_cycles) + sum(tile_compute_cycles)
        double_buffered = tile_load_cycles[0] + tile_compute_cycles[0]
        for index in range(1, len(tile_compute_cycles)):
            uncovered_load = max(
                0,
                tile_load_cycles[index] - tile_compute_cycles[index - 1],
            )
            double_buffered += uncovered_load + tile_compute_cycles[index]
        raw = (
            layer.out_height
            * layer.out_width
            * math.ceil(layer.out_channels / core.output_pes)
            * math.ceil(layer.in_channels / core.input_channels_per_pe)
            * math.ceil(layer.kernel_elements / core.kernel_taps)
        )
        result.append(
            {
                "name": layer.name,
                "output_shape_before_pool": [
                    layer.out_channels,
                    layer.out_height,
                    layer.out_width,
                ],
                "output_channel_tiles": math.ceil(
                    layer.out_channels / core.output_pes
                ),
                "input_channel_tiles": math.ceil(
                    layer.in_channels / core.input_channels_per_pe
                ),
                "spatial_sweeps": len(tile_compute_cycles),
                "raw_arithmetic_lower_bound_cycles": raw,
                "padded_scan_and_pipeline_cycles": sum(tile_compute_cycles),
                "weight_load_cycles_if_serialized": sum(tile_load_cycles),
                "serialized_weight_load_cycles": serialized,
                "double_buffered_weight_load_cycles": double_buffered,
            }
        )
    return result


def linear_cycles(core: Core, weights_per_cycle: int) -> dict[str, int]:
    result = {}
    for name, inputs, outputs in (*LINEAR_LAYERS, READOUT):
        arithmetic = math.ceil(outputs / core.output_pes) * math.ceil(
            inputs / core.lanes_per_pe
        )
        weight_stream = math.ceil(inputs * outputs / weights_per_cycle)
        result[name] = max(arithmetic, weight_stream)
    return result


def storage_account(core: Core, args: argparse.Namespace) -> dict[str, int]:
    input_and_l1 = (
        CONV_LAYERS[0].input_elements * args.input_count_bits
        + CONV_LAYERS[0].output_elements * args.spike_bits
    )
    later_ping_pong = max(
        previous.output_elements * args.spike_bits
        + current.output_elements * args.spike_bits
        for previous, current in zip(CONV_LAYERS, CONV_LAYERS[1:])
    )
    activation_ping_pong = max(input_and_l1, later_ping_pong)

    partial_sum_candidates = []
    for layer in CONV_LAYERS:
        if math.ceil(layer.in_channels / core.input_channels_per_pe) > 1:
            partial_sum_candidates.append(
                min(core.output_pes, layer.out_channels)
                * layer.out_height
                * layer.out_width
                * args.internal_accumulator_bits
            )
    partial_sum = max(partial_sum_candidates, default=0)
    line_buffer = max(
        3
        * layer.out_width
        * min(core.input_channels_per_pe, layer.in_channels)
        * (args.input_count_bits if layer.name == "L1" else args.spike_bits)
        for layer in CONV_LAYERS
    )
    local_weight_double_buffer = (
        2
        * core.output_pes
        * core.input_channels_per_pe
        * 9
        * args.weight_bits
    )
    pooling_row = max(
        core.output_pes * layer.out_width * args.current_bits
        for layer in CONV_LAYERS[:5]
    )
    common = {
        "network_weights": WEIGHT_COUNT * args.weight_bits,
        "folded_constants": FOLDED_CONSTANT_COUNT * args.current_bits,
        "activation_ping_pong": activation_ping_pong,
        "temporary_partial_sums": partial_sum,
        "input_line_buffer": line_buffer,
        "double_buffered_local_weight_tile": local_weight_double_buffer,
        "streaming_maxpool_row": pooling_row,
    }
    common_total = sum(common.values())
    result = {
        **common,
        "full_state": FULL_STATE_COUNT * args.state_bits,
        "l1_l2_stateless_state": STATELESS_STATE_COUNT * args.state_bits,
        "sl_hyb_l3_current_cache": L3_CURRENT_CACHE_COUNT * args.current_bits,
    }
    result["fs_tm_total"] = common_total + result["full_state"]
    result["sl_tm_total"] = common_total + result["l1_l2_stateless_state"]
    result["sl_hyb_total"] = (
        result["sl_tm_total"] + result["sl_hyb_l3_current_cache"]
    )
    return result


def build_core_report(core: Core, args: argparse.Namespace) -> dict[str, object]:
    convolution = convolution_schedule(
        core,
        weights_per_load_cycle=args.weights_per_load_cycle,
        pipeline_latency=args.pipeline_latency,
    )
    linear = linear_cycles(core, args.weights_per_load_cycle)
    prefix_serialized = sum(
        item["serialized_weight_load_cycles"] for item in convolution[:3]
    )
    prefix_double_buffered = sum(
        item["double_buffered_weight_load_cycles"] for item in convolution[:3]
    )
    one_step_serialized = sum(
        item["serialized_weight_load_cycles"] for item in convolution
    ) + linear["H1"] + linear["H2"]
    one_step_double_buffered = sum(
        item["double_buffered_weight_load_cycles"] for item in convolution
    ) + linear["H1"] + linear["H2"]
    replay = math.ceil(L3_CURRENT_CACHE_COUNT / core.output_pes)
    schedules = {
        "serialized_weight_loading": {
            "fs_tm_or_sl_tm": 2 * one_step_serialized + linear["readout"],
            "sl_hyb": (
                2 * one_step_serialized
                + linear["readout"]
                - prefix_serialized
                + replay
            ),
        },
        "double_buffered_weight_loading": {
            "fs_tm_or_sl_tm": 2 * one_step_double_buffered + linear["readout"],
            "sl_hyb": (
                2 * one_step_double_buffered
                + linear["readout"]
                - prefix_double_buffered
                + replay
            ),
        },
    }
    latency = {
        mode: {
            f"{clock:g}_mhz": {
                schedule: cycles / (clock * 1000)
                for schedule, cycles in values.items()
            }
            for clock in args.clock_mhz
        }
        for mode, values in schedules.items()
    }
    return {
        "core": {
            "output_pes": core.output_pes,
            "input_channels_per_pe": core.input_channels_per_pe,
            "kernel_taps_per_input_channel": core.kernel_taps,
            "lanes_per_output_pe": core.lanes_per_pe,
            "multiplier_lanes_and_assumed_dsp48": core.multiplier_lanes,
        },
        "loop_order": [
            "time step",
            "layer",
            "output-channel tile",
            "input-channel tile",
            "padded spatial raster",
        ],
        "convolution_layers": convolution,
        "linear_and_readout_cycles": linear,
        "l3_current_replay_cycles_in_sl_hyb": replay,
        "schedule_cycles": schedules,
        "latency_ms": latency,
        "storage_bits_without_bram_fragmentation": storage_account(core, args),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.output_pes,
        *args.input_channels_per_pe,
        args.kernel_taps,
        args.weights_per_load_cycle,
        args.pipeline_latency,
        *args.clock_mhz,
    ) <= 0:
        raise ValueError("parallelism, bandwidth, latency, and clocks must be positive")
    script = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "pre_controller_realizable_schedule_and_storage_estimate",
        "analysis_code": {"path": str(script), "sha256": sha256_file(script)},
        "target": {
            "board": "RealDigital AUP-ZU3",
            "part": TARGET_PART,
            "bram36_blocks": AUP_ZU3_BRAM36_BLOCKS,
            "bram36_capacity_bits": AUP_ZU3_BRAM36_BLOCKS * BRAM36_BITS,
            "clock_mhz": args.clock_mhz,
        },
        "fixed_point_widths": {
            "input_count_bits": args.input_count_bits,
            "spike_bits": args.spike_bits,
            "weight_bits": args.weight_bits,
            "current_bits": args.current_bits,
            "state_bits": args.state_bits,
            "internal_accumulator_bits": args.internal_accumulator_bits,
        },
        "schedule_assumptions": {
            "weights_per_load_cycle": args.weights_per_load_cycle,
            "weight_load_bus_bits": args.weights_per_load_cycle * args.weight_bits,
            "pipeline_fill_and_drain_cycles_per_spatial_sweep": args.pipeline_latency,
            "pipeline_calibration": (
                "complete L1 RTL produces 3072 pooled output vectors in "
                "12754 start-to-done cycles: (96+2)*(128+2)+14"
            ),
            "spatial_scan": "(H+2)*(W+2) cycles for same-padded 3x3 line buffer",
            "partial_sums": "one P_OC-wide vector read/write per cycle when needed",
            "pool_and_lif": "stream behind the final input-channel sweep at one P_OC-wide vector per cycle",
            "linear_layers": "reuse the same core at the larger of arithmetic cycles and finite weight-stream cycles",
            "serialized_case": "weight-tile loading never overlaps arithmetic",
            "double_buffered_case": "next weight tile loads from a separate weight RAM port while the current tile computes",
        },
        "cores": [
            build_core_report(
                Core(args.output_pes, input_channels, args.kernel_taps), args
            )
            for input_channels in args.input_channels_per_pe
        ],
        "limitations": [
            "L1 controller and BRAM streaming are simulated; L2-L8 input-tile partial-sum arbitration is not yet implemented",
            "BRAM width/depth fragmentation and placement are excluded from the bit totals",
            "clock frequency, routed latency, and power require Vivado implementation",
            "event aggregation time is separate from accelerator compute latency",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-pes", type=int, default=8)
    parser.add_argument("--input-channels-per-pe", type=int, nargs="+", default=(2, 4))
    parser.add_argument("--kernel-taps", type=int, default=9)
    parser.add_argument("--weights-per-load-cycle", type=int, default=16)
    parser.add_argument(
        "--pipeline-latency",
        type=int,
        default=14,
        help=(
            "fill/drain cycles per spatial sweep; default is calibrated to "
            "the bit-exact complete L1 RTL replay"
        ),
    )
    parser.add_argument("--clock-mhz", type=float, nargs="+", default=(100.0, 200.0))
    parser.add_argument("--input-count-bits", type=int, default=9)
    parser.add_argument("--spike-bits", type=int, default=8)
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--current-bits", type=int, default=24)
    parser.add_argument("--state-bits", type=int, default=16)
    parser.add_argument("--internal-accumulator-bits", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
