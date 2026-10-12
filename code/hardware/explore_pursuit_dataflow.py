"""Enumerate dense shared-engine parallelism for the pursuit SNN.

This is an analytical scheduling lower bound, not an RTL timing result.  One
output PE owns one partial sum and contains ``P_IC * P_K`` multiplier lanes.
There are ``P_OC`` output PEs, so the shared engine contains
``P_OC * P_IC * P_K`` multiplier lanes.  Pooling/LIF can be streamed behind the
convolution but controller, pipeline fill, RAM conflicts, and stalls are not
included here; those are measured later in the cycle model and RTL.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 1
TARGET_PART = "xczu3eg-sfvc784-2-e"
AUP_ZU3_ON_CHIP_RAM_BITS = 9_400_000
WEIGHT_COUNT = 95_928
FOLDED_CONSTANT_COUNT = 312
FULL_STATE_COUNT = 45_000
STATELESS_STATE_COUNT = 8_136
L3_CURRENT_CACHE_COUNT = 4_608


@dataclass(frozen=True)
class ConvLayer:
    name: str
    in_channels: int
    out_channels: int
    out_height: int
    out_width: int
    pooled_height: int
    pooled_width: int
    kernel_elements: int = 9

    @property
    def weights(self) -> int:
        return self.in_channels * self.out_channels * self.kernel_elements

    @property
    def dense_macs(self) -> int:
        return self.weights * self.out_height * self.out_width

    @property
    def input_elements(self) -> int:
        return self.in_channels * self.out_height * self.out_width

    @property
    def output_elements(self) -> int:
        return self.out_channels * self.pooled_height * self.pooled_width


CONV_LAYERS = (
    ConvLayer("L1", 2, 8, 96, 128, 48, 64),
    ConvLayer("L2", 8, 16, 48, 64, 24, 32),
    ConvLayer("L3", 16, 24, 24, 32, 12, 16),
    ConvLayer("L4", 24, 32, 12, 16, 6, 8),
    ConvLayer("L5", 32, 40, 6, 8, 3, 4),
    ConvLayer("L6", 40, 40, 3, 4, 3, 4),
    ConvLayer("L7", 40, 40, 3, 4, 3, 4),
    ConvLayer("L8", 40, 40, 3, 4, 3, 4),
)
LINEAR_LAYERS = (
    ("H1", 480, 60),
    ("H2", 60, 12),
)
READOUT = ("readout", 12, 2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conv_cycles(layer: ConvLayer, p_oc: int, p_ic: int, p_k: int) -> int:
    return (
        layer.out_height
        * layer.out_width
        * math.ceil(layer.out_channels / p_oc)
        * math.ceil(layer.in_channels / p_ic)
        * math.ceil(layer.kernel_elements / p_k)
    )


def linear_cycles(in_features: int, out_features: int, p_oc: int, reduction_lanes: int) -> int:
    return math.ceil(out_features / p_oc) * math.ceil(in_features / reduction_lanes)


def enumerate_candidates(
    clock_mhz: float,
    max_multiplier_lanes: int,
    p_oc_values: tuple[int, ...],
    p_ic_values: tuple[int, ...],
    p_k_values: tuple[int, ...],
    weight_bits: int,
    activation_bits: int,
    accumulator_bits: int,
) -> list[dict[str, object]]:
    dense_macs_per_timestep = sum(layer.dense_macs for layer in CONV_LAYERS)
    dense_macs_per_timestep += sum(
        in_features * out_features
        for _, in_features, out_features in LINEAR_LAYERS
    )
    readout_macs = READOUT[1] * READOUT[2]
    full_dense_macs = 2 * dense_macs_per_timestep + readout_macs
    prefix_dense_macs = sum(layer.dense_macs for layer in CONV_LAYERS[:3])
    hybrid_dense_macs = full_dense_macs - prefix_dense_macs

    candidates = []
    for p_oc in p_oc_values:
        for p_ic in p_ic_values:
            for p_k in p_k_values:
                multiplier_lanes = p_oc * p_ic * p_k
                if multiplier_lanes > max_multiplier_lanes:
                    continue
                reduction_lanes = p_ic * p_k
                per_conv = {
                    layer.name: conv_cycles(layer, p_oc, p_ic, p_k)
                    for layer in CONV_LAYERS
                }
                per_linear = {
                    name: linear_cycles(
                        in_features, out_features, p_oc, reduction_lanes
                    )
                    for name, in_features, out_features in LINEAR_LAYERS
                }
                readout_cycles = linear_cycles(
                    READOUT[1], READOUT[2], p_oc, reduction_lanes
                )
                timestep_cycles = sum(per_conv.values()) + sum(per_linear.values())
                full_cycles = 2 * timestep_cycles + readout_cycles
                hybrid_cycles = full_cycles - sum(
                    per_conv[layer.name] for layer in CONV_LAYERS[:3]
                )
                max_weight_tile_bits = max(
                    p_oc
                    * layer.in_channels
                    * layer.kernel_elements
                    * weight_bits
                    for layer in CONV_LAYERS
                )
                max_line_buffer_bits = max(
                    3 * layer.out_width * layer.in_channels * activation_bits
                    for layer in CONV_LAYERS
                )
                candidates.append(
                    {
                        "p_oc_output_pes": p_oc,
                        "p_ic_input_channels_per_pe": p_ic,
                        "p_k_kernel_taps_per_input_channel": p_k,
                        "reduction_lanes_per_output_pe": reduction_lanes,
                        "multiplier_lanes": multiplier_lanes,
                        "assumed_dsp48_if_one_multiplier_per_dsp": multiplier_lanes,
                        "per_cycle_operand_bits": {
                            "activation_broadcast": reduction_lanes * activation_bits,
                            "weights": multiplier_lanes * weight_bits,
                            "resident_partial_sums": p_oc * accumulator_bits,
                        },
                        "local_storage_upper_bounds_bits": {
                            "largest_output_channel_weight_tile": max_weight_tile_bits,
                            "largest_three_row_input_buffer": max_line_buffer_bits,
                        },
                        "cycles": {
                            "per_conv_per_timestep": per_conv,
                            "per_linear_per_timestep": per_linear,
                            "readout_once": readout_cycles,
                            "fs_tm_or_sl_tm": full_cycles,
                            "sl_hyb_l3_current_reuse": hybrid_cycles,
                        },
                        "compute_lower_bound_ms_at_clock": {
                            "fs_tm_or_sl_tm": full_cycles / (clock_mhz * 1_000),
                            "sl_hyb_l3_current_reuse": hybrid_cycles
                            / (clock_mhz * 1_000),
                        },
                        "arithmetic_lane_utilization": {
                            "fs_tm_or_sl_tm": full_dense_macs
                            / (full_cycles * multiplier_lanes),
                            "sl_hyb_l3_current_reuse": hybrid_dense_macs
                            / (hybrid_cycles * multiplier_lanes),
                        },
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            item["multiplier_lanes"],
            item["cycles"]["fs_tm_or_sl_tm"],
        ),
    )


def build_report(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.clock_mhz,
        args.max_multiplier_lanes,
        args.weight_bits,
        args.activation_bits,
        args.current_bits,
        args.state_bits,
        args.accumulator_bits,
    ) <= 0:
        raise ValueError("clock, lane limit, and all widths must be positive")
    script = Path(__file__).resolve()
    candidates = enumerate_candidates(
        clock_mhz=args.clock_mhz,
        max_multiplier_lanes=args.max_multiplier_lanes,
        p_oc_values=tuple(args.p_oc),
        p_ic_values=tuple(args.p_ic),
        p_k_values=tuple(args.p_k),
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        accumulator_bits=args.accumulator_bits,
    )
    conv_macs = {layer.name: layer.dense_macs for layer in CONV_LAYERS}
    hidden_macs = {
        name: in_features * out_features
        for name, in_features, out_features in LINEAR_LAYERS
    }
    one_timestep_without_readout = sum(conv_macs.values()) + sum(hidden_macs.values())
    full_dense_macs = 2 * one_timestep_without_readout + READOUT[1] * READOUT[2]
    prefix_dense_macs = sum(layer.dense_macs for layer in CONV_LAYERS[:3])

    activation_ping_pong_elements = max(
        CONV_LAYERS[0].input_elements + CONV_LAYERS[0].output_elements,
        *(
            previous.output_elements + current.output_elements
            for previous, current in zip(CONV_LAYERS, CONV_LAYERS[1:])
        ),
    )
    storage = {
        "network_weights": WEIGHT_COUNT * args.weight_bits,
        "folded_constants": FOLDED_CONSTANT_COUNT * args.current_bits,
        "time_major_activation_ping_pong_upper_bound": (
            activation_ping_pong_elements * args.activation_bits
        ),
        "full_state": FULL_STATE_COUNT * args.state_bits,
        "l1_l2_stateless_state": STATELESS_STATE_COUNT * args.state_bits,
        "sl_hyb_l3_current_cache": L3_CURRENT_CACHE_COUNT * args.current_bits,
    }
    storage["fs_tm_total_accounted"] = (
        storage["network_weights"]
        + storage["folded_constants"]
        + storage["time_major_activation_ping_pong_upper_bound"]
        + storage["full_state"]
    )
    storage["sl_tm_total_accounted"] = (
        storage["network_weights"]
        + storage["folded_constants"]
        + storage["time_major_activation_ping_pong_upper_bound"]
        + storage["l1_l2_stateless_state"]
    )
    storage["sl_hyb_total_accounted"] = (
        storage["sl_tm_total_accounted"] + storage["sl_hyb_l3_current_cache"]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "analytical_shared_dense_engine_design_space_before_rtl",
        "analysis_code": {"path": str(script), "sha256": sha256_file(script)},
        "target": {
            "board": "RealDigital AUP-ZU3",
            "part": TARGET_PART,
            "clock_mhz": args.clock_mhz,
            "published_total_on_chip_ram_bits": AUP_ZU3_ON_CHIP_RAM_BITS,
            "max_multiplier_lanes_swept": args.max_multiplier_lanes,
        },
        "arithmetic_definition": {
            "output_pe": "one resident output partial sum with P_IC*P_K reduction lanes",
            "multiplier_lanes": "P_OC*P_IC*P_K; baseline assumes one DSP48 per lane",
            "limitations": [
                "controller, pipeline fill/drain, RAM conflicts, pooling, LIF, and stalls are excluded",
                "latencies are scheduling lower bounds, not RTL or FPGA measurements",
                "DSP packing and LUT multipliers are not assumed",
            ],
        },
        "model": {
            "convolution_layers": [asdict(layer) for layer in CONV_LAYERS],
            "linear_layers": [
                {"name": name, "in_features": inputs, "out_features": outputs}
                for name, inputs, outputs in LINEAR_LAYERS
            ],
            "readout": {
                "name": READOUT[0],
                "in_features": READOUT[1],
                "out_features": READOUT[2],
            },
            "dense_macs": {
                "per_conv_per_timestep": conv_macs,
                "per_hidden_per_timestep": hidden_macs,
                "fs_tm_or_sl_tm": full_dense_macs,
                "l1_through_l3_prefix_one_timestep": prefix_dense_macs,
                "sl_hyb_l3_current_reuse": full_dense_macs - prefix_dense_macs,
            },
        },
        "width_assumptions_for_capacity_only": {
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "current_bits": args.current_bits,
            "state_bits": args.state_bits,
            "accumulator_bits": args.accumulator_bits,
        },
        "on_chip_storage_account_bits_without_bram_fragmentation": storage,
        "dataflow_axes_still_to_measure": {
            "within_layer_loop_orders": [
                "output-channel tile outer: keep a weight tile close, rescan input rows per output tile",
                "spatial window outer: keep an input window close, stream banked weights for output tiles",
                "row-stationary hybrid: line-buffer inputs, bank weights, and keep partial sums in output PEs",
            ],
            "across_time_schedules": [
                "time-major whole network: preserves the direct 45000 versus 8136 state comparison",
                "layer-major two timesteps: reuses layer weights but needs two-timestep activation buffers",
                "SL-HYB: compute through L3 current once and cache 4608 currents",
            ],
            "sparsity_schedules": [
                "dense fixed schedule",
                "zero-skip with the same dense tensor layout",
                "event-scatter for layers where touched-position sparsity pays for routing",
            ],
        },
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-mhz", type=float, default=100.0)
    parser.add_argument("--max-multiplier-lanes", type=int, default=128)
    parser.add_argument("--p-oc", type=int, nargs="+", default=(1, 2, 4, 8, 16))
    parser.add_argument("--p-ic", type=int, nargs="+", default=(1, 2, 4, 8))
    parser.add_argument("--p-k", type=int, nargs="+", default=(1, 3, 9))
    parser.add_argument("--weight-bits", type=int, default=16)
    parser.add_argument("--activation-bits", type=int, default=16)
    parser.add_argument("--current-bits", type=int, default=16)
    parser.add_argument("--state-bits", type=int, default=16)
    parser.add_argument("--accumulator-bits", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
