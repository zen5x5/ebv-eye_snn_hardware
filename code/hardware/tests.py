from argparse import Namespace

import torch

from .explore_pursuit_dataflow import build_report
from .estimate_pursuit_rtl_schedule import build_report as build_schedule_report
from .compare_synapse_arithmetic import arithmetic_views, population_count
from .evaluate_pursuit_fixed_point import (
    choose_fraction_bits,
    quantize_static,
    round_half_away_from_zero,
)
from .profile_snn_channel_statistics import (
    ChannelAccumulator,
    convolution_source_work,
)


def _design_space_args() -> Namespace:
    return Namespace(
        clock_mhz=100.0,
        max_multiplier_lanes=128,
        p_oc=[1, 2, 4, 8, 16],
        p_ic=[1, 2, 4, 8],
        p_k=[1, 3, 9],
        weight_bits=16,
        activation_bits=16,
        current_bits=16,
        state_bits=16,
        accumulator_bits=32,
    )


def _schedule_args() -> Namespace:
    return Namespace(
        output_pes=8,
        input_channels_per_pe=[2, 4],
        kernel_taps=9,
        weights_per_load_cycle=16,
        pipeline_latency=14,
        clock_mhz=[100.0, 200.0],
        input_count_bits=9,
        spike_bits=8,
        weight_bits=16,
        current_bits=24,
        state_bits=16,
        internal_accumulator_bits=40,
    )


def test_reference_design_space_point() -> None:
    report = build_report(_design_space_args())
    assert report["model"]["dense_macs"]["fs_tm_or_sl_tm"] == 20_781_240
    assert (
        report["model"]["dense_macs"]["l1_through_l3_prefix_one_timestep"]
        == 7_962_624
    )
    candidate = next(
        item
        for item in report["candidates"]
        if (
            item["p_oc_output_pes"],
            item["p_ic_input_channels_per_pe"],
            item["p_k_kernel_taps_per_input_channel"],
        )
        == (8, 4, 1)
    )
    assert candidate["multiplier_lanes"] == 32
    assert candidate["cycles"]["fs_tm_or_sl_tm"] == 760_143
    assert candidate["cycles"]["sl_hyb_l3_current_reuse"] == 456_015
    assert candidate["per_cycle_operand_bits"]["weights"] == 512


def test_state_capacity_difference_is_72_kib() -> None:
    report = build_report(_design_space_args())
    storage = report["on_chip_storage_account_bits_without_bram_fragmentation"]
    assert storage["full_state"] - storage["l1_l2_stateless_state"] == (
        72 * 1024 * 8
    )


def test_selected_fast_core_schedule_and_storage() -> None:
    report = build_schedule_report(_schedule_args())
    slow, fast = report["cores"]
    assert slow["core"]["multiplier_lanes_and_assumed_dsp48"] == 144
    assert fast["core"]["multiplier_lanes_and_assumed_dsp48"] == 288
    assert fast["schedule_cycles"]["serialized_weight_loading"] == {
        "fs_tm_or_sl_tm": 119_050,
        "sl_hyb": 82_543,
    }
    assert fast["convolution_layers"][0][
        "padded_scan_and_pipeline_cycles"
    ] == 12_754
    assert (
        fast["storage_bits_without_bram_fragmentation"]["fs_tm_total"]
        - fast["storage_bits_without_bram_fragmentation"]["sl_tm_total"]
        == 72 * 1024 * 8
    )


def test_channel_accumulator_uses_explicit_channel_dimension() -> None:
    values = torch.tensor(
        [
            [
                [[[0.0, 1.0]], [[3.0, 0.0]]],
                [[[2.0, 0.0]], [[0.0, 4.0]]],
            ]
        ]
    )
    accumulator = ChannelAccumulator(2)
    accumulator.update(values)
    result = accumulator.finalize(samples=1)
    assert result[0]["sum_per_sample"] == 3.0
    assert result[1]["sum_per_sample"] == 7.0
    assert result[0]["nonzero_fraction"] == 0.5
    assert result[1]["maximum"] == 4.0


def test_sparse_convolution_work_accounts_for_image_edges() -> None:
    source = torch.zeros(1, 1, 2, 3, 3)
    source[0, 0, 0, 1, 1] = 1.0
    source[0, 0, 1, 0, 0] = 1.0
    work = convolution_source_work(source, destination_channels=8)
    assert torch.equal(work, torch.tensor([72.0, 32.0], dtype=torch.float64))


def test_integer_spike_arithmetic_views() -> None:
    values = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 173.0, 400.0]]])
    assert torch.equal(
        population_count(values),
        torch.tensor([[[0, 1, 1, 2, 5, 3]]]),
    )
    views = arithmetic_views(values)
    assert float(views["unary_weight_adds"].sum()) == 579.0
    assert float(views["nonzero_count_weight_macs"].sum()) == 5.0
    assert float(views["binary_shifted_weight_terms"].sum()) == 12.0


def test_fixed_point_rounding_and_power_of_two_scales() -> None:
    values = torch.tensor([-1.5, -0.5, -0.49, 0.49, 0.5, 1.5])
    assert torch.equal(
        round_half_away_from_zero(values),
        torch.tensor([-2.0, -1.0, -0.0, 0.0, 1.0, 2.0]),
    )
    weights = torch.tensor([[1.0, -2.0], [0.125, -0.25]])
    per_channel = choose_fraction_bits(
        weights,
        bits=4,
        mode="per_output_channel",
        policy="no_clipping",
        maximum_fraction_bits=8,
    )
    per_layer = choose_fraction_bits(
        weights,
        bits=4,
        mode="per_layer",
        policy="no_clipping",
        maximum_fraction_bits=8,
    )
    assert torch.equal(per_channel, torch.tensor([1, 4]))
    assert torch.equal(per_layer, torch.tensor([1, 1]))
    quantized = quantize_static(weights, per_channel, bits=4)
    assert torch.equal(quantized, torch.tensor([[2.0, -4.0], [2.0, -4.0]]))


def main() -> int:
    test_reference_design_space_point()
    test_state_capacity_difference_is_72_kib()
    test_selected_fast_core_schedule_and_storage()
    test_channel_accumulator_uses_explicit_channel_dimension()
    test_sparse_convolution_work_accounts_for_image_edges()
    test_integer_spike_arithmetic_views()
    test_fixed_point_rounding_and_power_of_two_scales()
    print("hardware analysis checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
