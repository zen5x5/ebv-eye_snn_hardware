"""Focused checks for the sparse-front/dense-back fixed-point reference."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .evaluate_pursuit_sparse_hybrid import (
    SampleWork,
    _one_or_two_step_dense_schedule,
    _schedule_memory_traffic,
    _scheduled_lane_operation_report,
    _three_schedule_cycles,
    dense_backend_schedule,
    pooled_touched_mask,
    replay_sparse_frontend_control,
    sparse_same_conv3x3,
    valid_fanout_work_per_sample,
)


def test_event_add_only_sparse_convolution_matches_dense_integer_conv() -> None:
    source = torch.zeros(2, 2, 5, 6, dtype=torch.int64)
    source[0, 0, 0, 0] = 3
    source[0, 1, 2, 4] = 2
    source[1, 0, 4, 5] = 7
    source[1, 1, 3, 1] = 1
    weights = torch.arange(-72, 0, dtype=torch.int64).reshape(4, 2, 3, 3)
    sparse = sparse_same_conv3x3(
        source,
        weights,
        expand_unit_events=True,
    )
    dense = F.conv2d(source, weights, padding=1)
    assert torch.equal(sparse, dense)


def test_nonzero_count_mac_sparse_convolution_matches_dense_integer_conv() -> None:
    source = torch.zeros(1, 3, 4, 5, dtype=torch.int64)
    source[0, 0, 0, 4] = 13
    source[0, 1, 2, 2] = 173
    source[0, 2, 3, 0] = 5
    weights = torch.arange(-135, 0, dtype=torch.int64).reshape(5, 3, 3, 3)
    sparse = sparse_same_conv3x3(
        source,
        weights,
        expand_unit_events=False,
    )
    dense = F.conv2d(source, weights, padding=1)
    assert torch.equal(sparse, dense)


def test_pool_before_constant_and_clamp_is_exact() -> None:
    raw = torch.tensor(
        [[[[9_000_000, -20], [8_000_000, 5]], [[-9_000_000, 7], [3, 2]]]],
        dtype=torch.int64,
    )
    constant = torch.tensor([500_000, -1_000_000], dtype=torch.int64).reshape(
        1, 2, 1, 1
    )
    minimum, maximum = -(1 << 23), (1 << 23) - 1
    old_order = F.max_pool2d((raw + constant).clamp(minimum, maximum), 2)
    new_order = (F.max_pool2d(raw, 2) + constant).clamp(minimum, maximum)
    assert torch.equal(old_order, new_order)


def test_touched_mask_includes_convolution_and_pool_support() -> None:
    source = torch.zeros(1, 2, 6, 8, dtype=torch.int64)
    source[0, 1, 2, 3] = 4
    touched = pooled_touched_mask(source)
    expected = F.max_pool2d(
        F.max_pool2d(
            source.ne(0).any(dim=1, keepdim=True).float(),
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        kernel_size=2,
        stride=2,
    ).bool()[:, 0]
    assert torch.equal(touched, expected)


def test_valid_fanout_counts_edges_and_count_semantics() -> None:
    source = torch.zeros(1, 1, 3, 4, dtype=torch.int64)
    source[0, 0, 0, 0] = 3  # four valid taps
    source[0, 0, 1, 2] = 2  # nine valid taps
    unit = valid_fanout_work_per_sample(
        source,
        destination_channels=8,
        unit_events=True,
    )
    count_mac = valid_fanout_work_per_sample(
        source,
        destination_channels=8,
        unit_events=False,
    )
    assert int(unit[0]) == (3 * 4 + 2 * 9) * 8
    assert int(count_mac[0]) == (4 + 9) * 8


def test_layer_adjacent_schedule_reuses_weights() -> None:
    l4 = _one_or_two_step_dense_schedule(3, steps=2)
    assert l4["double_buffered_cycles"] == 12_786
    assert l4["separate_invocation_cycles"] == 12_804
    assert l4["weight_values_saved_by_layer_adjacent_reuse"] == 6_912
    backend = dense_backend_schedule()
    assert backend["convolution_cycles"] == 44_372
    assert backend["linear_and_readout_cycles"] == 3_692
    assert backend["total_cycles"] == 48_064


def test_three_schedule_memory_and_cycle_accounting() -> None:
    work = SampleWork(
        l1_unit_events=[400],
        l1_valid_weight_additions=[28_680],
        l1_output_packets=[2_575],
        l2_valid_count_weight_macs=[365_072],
        sparse_frontend_cycle_exact=[15_259],
        sparse_frontend_l1_wrapper_cycles=[15_192],
        sparse_frontend_l1_core_cycles=[15_190],
        sparse_frontend_l2_core_cycles=[15_259],
    )
    traffic = _schedule_memory_traffic(work)
    assert traffic["Dense_TM"]["weight"][
        "runtime_weight_BRAM_read_beats_256b"
    ] == 11_982
    assert traffic["Sparse_TM"]["weight"][
        "runtime_weight_BRAM_read_beats_256b"
    ] == 11_892
    assert traffic["Sparse_LM_A"]["weight"][
        "runtime_weight_BRAM_read_beats_256b"
    ] == 6_060
    assert traffic["Dense_TM"]["activation"][
        "activation_BRAM_read_beats_mixed_36b_8b"
    ]["mean"] == 54_420
    assert traffic["Sparse_LM_A"]["activation"][
        "activation_BRAM_write_beats_mixed_64b_128b_256b"
    ]["mean"] == 2_794
    assert traffic["Sparse_TM"]["state"][
        "external_state_elements_written"
    ] == 16_272
    assert traffic["Sparse_LM_A"]["state"][
        "external_state_elements_written"
    ] == 0
    assert traffic["Sparse_LM_A"]["state"][
        "local_state_elements_written"
    ] == 8_136
    assert traffic["Sparse_TM"]["front_local_partial_sum_traffic"][
        "L1_8xS40_vector_reads"
    ]["mean"] == 18_945
    assert traffic["Sparse_TM"]["front_local_partial_sum_traffic"][
        "L2_16xS40_vector_reads"
    ]["mean"] == 26_657
    assert traffic["Sparse_LM_A"]["external_L3_current_cache"][
        "S24_elements_written"
    ] == 0
    assert traffic["Sparse_LM_A"]["tile_local_L3_current_buffer"][
        "capacity_bytes"
    ] == 4_608

    cycles = _three_schedule_cycles(work, checkpoint_seed=2022)
    assert cycles["front_end"]["serialized_idealized_stage_sum"][
        "sparse_front_total_cycles"
    ]["mean"] == 18_380
    assert cycles["front_end"]["cycle_exact_RTL_control_model"][
        "frontend_cycles"
    ]["mean"] == 15_259
    assert cycles["front_end"][
        "seed2022_sample_zero_calibration_exact_match"
    ]
    assert cycles["Dense_TM"]["comparison_distribution"]["mean"] == 83_286
    assert cycles["Sparse_TM"]["rtl_calibrated_distribution"]["mean"] == 72_431
    assert cycles["Sparse_LM_A"]["rtl_measured_fixed_backend_component_sum_cycles"] == 52_671
    assert cycles["Sparse_LM_A"]["integrated_boundary_adjustment_vs_component_sum_cycles"] == -1
    assert cycles["Sparse_LM_A"]["rtl_calibrated_distribution"]["mean"] == 67_929

    scheduled = _scheduled_lane_operation_report(work)
    assert scheduled["Dense_TM"]["convolution_multiplier_lane_operations"] == {
        "L1": 3_538_944,
        "L2": 3_538_944,
        "L3": 2_654_208,
        "L4": 2_654_208,
        "L5": 1_105_920,
        "L6": 345_600,
        "L7": 345_600,
        "L8": 345_600,
    }
    assert scheduled["Dense_TM"][
        "H1_H2_readout_multiplier_lane_operations"
    ] == 63_384
    assert scheduled["Dense_TM"][
        "total_scheduled_multiplier_lane_operations"
    ]["mean"] == 14_592_408
    assert scheduled["Sparse_TM"][
        "L1_scheduled_add_lane_operations"
    ]["mean"] == 28_800
    assert scheduled["Sparse_TM"][
        "L2_scheduled_multiplier_lane_operations"
    ]["mean"] == 370_800
    assert scheduled["Sparse_TM"][
        "total_scheduled_add_and_multiplier_lane_operations"
    ]["mean"] == 7_914_120


def test_sparse_frontend_control_replay_handles_empty_L1_packet_stream() -> None:
    replay = replay_sparse_frontend_control([0] * 400, [0] * (48 * 64))
    assert replay.frontend_cycles == 8_344
    assert replay.l1_wrapper_cycles == 6_735
    assert replay.l1_core_cycles == 6_733
    assert replay.l2_core_cycles == 8_344
    assert replay.emitted_l1_packets == 0
    assert replay.accepted_l2_packets == 0


def main() -> int:
    test_event_add_only_sparse_convolution_matches_dense_integer_conv()
    test_nonzero_count_mac_sparse_convolution_matches_dense_integer_conv()
    test_pool_before_constant_and_clamp_is_exact()
    test_touched_mask_includes_convolution_and_pool_support()
    test_valid_fanout_counts_edges_and_count_semantics()
    test_layer_adjacent_schedule_reuses_weights()
    test_three_schedule_memory_and_cycle_accounting()
    test_sparse_frontend_control_replay_handles_empty_L1_packet_stream()
    print("sparse hybrid reference checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
