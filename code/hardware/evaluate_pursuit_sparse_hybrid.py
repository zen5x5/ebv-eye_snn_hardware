"""Validate and account the selected sparse-front/dense-back pursuit schedule.

The selected functional schedule is deliberately separate from the existing
dense-raster RTL baseline:

* L1 consumes the 400 spatially pooled unit events and accumulates S16 weights
  with add-only event scatter;
* L2 consumes only nonzero L1 MultiSpike packets and performs one U8-count by
  S16-weight MAC per valid synapse;
* L2 nevertheless finalizes every output position, so its folded constant can
  create the required per-channel default spike plane;
* L3 convolution is evaluated once and its current is passed through two
  consecutive stateful LIF updates;
* L4--L8, H1, and H2 execute t0 and t1 next to each other while retaining the
  layer weights; the shared convolution core has 8 output PEs, 4 input
  channels per PE, and 9 taps (288 multiplier lanes).

All arithmetic follows :mod:`evaluate_pursuit_fixed_point`.  The sparse
reference uses integer scatter-adds rather than dense convolution for L1/L2
and checks every logical trace value against the frozen dense fixed-point
reference.  The cycle section combines a cycle-exact sparse-front control
replay with fixed back-end counts calibrated by complete RTL simulation; only
the calibration sample is a complete RTL run.  No cycle count establishes
Vivado timing.  The command refuses to overwrite its JSON.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from code.hardware.evaluate_pursuit_fixed_point import (
    FixedLayer,
    FixedPointPursuitReference,
    RuntimeCounters,
    load_model,
    output_channel_view,
    sha256_file,
)
from code.hardware.explore_pursuit_dataflow import (
    CONV_LAYERS,
    LINEAR_LAYERS,
    READOUT,
)
from code.pursuit.fixed_event_window.ann.metrics import (
    angular_error_deg,
    pixel_error,
)


SCHEMA_VERSION = 3
FROZEN_FIXED_POINT_ARGUMENTS = {
    "weight_bits": 16,
    "readout_weight_bits": 16,
    "constant_bits": 24,
    "accumulator_bits": 24,
    "spike_bits": 8,
    "scale_mode": "per_output_channel",
    "scale_policy": "no_clipping",
    "maximum_fraction_bits": 15,
    "state_guard_bits": 0,
    "alpha_rounding": "floor",
}
FIXED_POINT_REPORT_CONTRACT = {
    **FROZEN_FIXED_POINT_ARGUMENTS,
    "current_bits": 24,
    "internal_partial_sum_bits": 40,
    "accumulator_bits_field_meaning": (
        "legacy fixed-reference name for the S24 current clamp applied only "
        "after the complete dot product; RTL running partial sums are S40"
    ),
}

# Conventional dense-network MAC count: every output evaluates the full 3x3
# kernel, including same-padding zero taps.  These constants exclude unused
# channel/output lanes in the fixed-width RTL engines; that separate hardware
# issue count is reported under ``scheduled_lane_operations``.
DEEP_DENSE_CONV_MACS = {
    "L3_once": 2_654_208,
    "L4_t0_t1": 2_654_208,
    "L5_t0_t1": 1_105_920,
    "L6_t0_t1": 345_600,
    "L7_t0_t1": 345_600,
    "L8_t0_t1": 345_600,
}
DEEP_DENSE_LINEAR_MACS = {
    "H1_t0_t1": 57_600,
    "H2_t0_t1": 1_440,
    "readout_once": 24,
}
OLD_DENSE_SL_HYB_MACS = 12_818_616
OLD_DENSE_SL_HYB_RTL_CYCLES = 83_286
S40_MINIMUM = -(1 << 39)
S40_MAXIMUM = (1 << 39) - 1

# Isolated dense-engine cycles were replayed from the formal seed2022 golden
# vectors.  Their scheduled sum differs from the integrated old top by 13
# cycles; the schedule comparison uses the integrated top wherever available.
DENSE_CONV_LAYER_RTL_CYCLES = {
    "L1": 12_779,
    "L2": 13_339,
    "L3": 11_007,
    "L4": 6_835,
    "L5": 4_503,
    "L6": 3_113,
    "L7": 3_113,
    "L8": 3_113,
}
PAIRED_CONV_LAYER_RTL_CYCLES = {
    "L4": 13_171,
    "L5": 8_183,
    "L6": 5_213,
    "L7": 5_213,
    "L8": 5_213,
}
PAIRED_L3_CURRENT_REUSE_RTL_CYCLES = 11_592
PAIRED_LINEAR_LAYER_RTL_CYCLES = {"H1": 3_928, "H2": 142}
READOUT_RTL_CYCLES = 16
OLD_DENSE_TOP_RTL_COMPONENTS = {
    "convolution": 78_466,
    "dense_and_readout": 4_188,
    "L3_current_replay": 617,
    "top_control_residual": 15,
}

# Integrated sparse-front RTL replay of seed2022 validation sample zero.  The
# software control replay below consumes the complete event and L1 packet
# streams and reproduces all four counters exactly for this calibration case.
SPARSE_FRONT_RTL_CALIBRATION = {
    "sample_unit_events": 400,
    "sample_L1_output_packets": 2_575,
    "integrated_frontend_measured_cycles": 15_259,
    "L1_wrapper_measured_cycles": 15_192,
    "L1_core_measured_cycles": 15_190,
    "L2_core_measured_cycles": 15_259,
    "model_status": "cycle-exact match for all four counters",
}
SPARSE_LM_A_RTL_CALIBRATION = {
    "seed": 2022,
    "validation_sample_index": 0,
    "whole_network_measured_cycles": 67_929,
    "frontend_measured_cycles": 15_259,
    "fixed_backend_calibrated_cycles": 52_670,
    "convolution_measured_cycles": 48_579,
    "dense_measured_cycles": 4_074,
    "top_control_cycles_outside_phase_counters": 17,
    "convolution_weight_BRAM_read_beats_256b": 4_068,
    "linear_weight_BRAM_read_beats_256b": 1_980,
    "readout_weight_memory_read_transactions_256b_port_low32_used": 12,
    "convolution_feature_read_requests": 25_872,
    "convolution_feature_write_transactions": 2_016,
    "trace_values_checked": 159_408,
    "trace_mismatches": 0,
}
SPARSE_TM_RTL_CALIBRATION = {
    "seed": 2022,
    "validation_sample_index": 0,
    "whole_network_measured_cycles": 72_431,
    "frontend_measured_cycles": 15_259,
    "fixed_backend_calibrated_cycles": 57_172,
    "convolution_measured_cycles": 52_350,
    "dense_measured_cycles": 4_188,
    "L3_current_replay_measured_cycles": 617,
    "top_control_cycles_outside_phase_counters": 17,
    "convolution_weight_BRAM_read_beats_256b": 7_920,
    "linear_and_readout_weight_memory_read_transactions": 3_972,
    "external_state_vector_read_beats": 1_013,
    "external_state_vector_write_beats": 2_026,
    "trace_values_checked": 159_408,
    "trace_mismatches": 0,
}


def _require_integer_counts(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"{name}: expected [B,C,H,W], got {tuple(values.shape)}")
    if bool((values < 0).any()):
        raise ValueError(f"{name}: source counts must be nonnegative")
    rounded = torch.round(values)
    if not torch.equal(values, rounded):
        raise ValueError(f"{name}: source counts must be exact integers")
    return rounded.to(device="cpu", dtype=torch.int64)


def _integer_weights(layer: FixedLayer) -> torch.Tensor:
    rounded = torch.round(layer.weight_integer)
    if not torch.equal(layer.weight_integer, rounded):
        raise AssertionError(f"{layer.name}: quantized weights are not integers")
    return rounded.to(device="cpu", dtype=torch.int64)


def _row_major_packets(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nonzero [batch,channel,row,column] indices in spatial-major order."""

    nonzero = source.nonzero(as_tuple=False)
    if len(nonzero) == 0:
        return nonzero, torch.empty(0, dtype=torch.int64)
    values = source[tuple(nonzero.t())]
    _, channels, height, width = source.shape
    key = (
        ((nonzero[:, 0] * height + nonzero[:, 2]) * width + nonzero[:, 3])
        * channels
        + nonzero[:, 1]
    )
    order = torch.argsort(key)
    return nonzero[order], values[order]


def sparse_same_conv3x3(
    source_counts: torch.Tensor,
    weight_integer: torch.Tensor,
    *,
    expand_unit_events: bool,
) -> torch.Tensor:
    """Integer 3x3 cross-correlation by source-driven scatter.

    ``expand_unit_events=True`` literally repeats a source record ``n`` times
    and adds an unscaled weight for every unit event.  It is the bit-exact L1
    add-only interpretation.  ``False`` retains one nonzero record and forms
    ``n*w`` once, which is the selected L2 count-MAC interpretation.  No
    partial sum is saturated; callers apply S24 saturation only after the
    complete dot product, matching the frozen S40 internal contract.
    """

    source = _require_integer_counts(source_counts, "sparse_same_conv3x3")
    weights = torch.as_tensor(weight_integer, dtype=torch.int64, device="cpu")
    if weights.ndim != 4 or tuple(weights.shape[2:]) != (3, 3):
        raise ValueError(f"expected [O,I,3,3] weights, got {tuple(weights.shape)}")
    batch, input_channels, height, width = source.shape
    output_channels, weight_inputs, _, _ = weights.shape
    if input_channels != weight_inputs:
        raise ValueError(
            f"source has {input_channels} channels but weights require {weight_inputs}"
        )

    packets, packet_counts = _row_major_packets(source)
    if expand_unit_events and len(packets):
        packets = torch.repeat_interleave(packets, packet_counts, dim=0)
        packet_counts = torch.ones(len(packets), dtype=torch.int64)

    flat = torch.zeros(
        batch * output_channels * height * width,
        dtype=torch.int64,
    )
    if len(packets) == 0:
        return flat.reshape(batch, output_channels, height, width)

    output_channel = torch.arange(output_channels, dtype=torch.int64)
    for kernel_row in range(3):
        output_row = packets[:, 2] + 1 - kernel_row
        for kernel_column in range(3):
            output_column = packets[:, 3] + 1 - kernel_column
            valid = (
                (output_row >= 0)
                & (output_row < height)
                & (output_column >= 0)
                & (output_column < width)
            )
            selected = packets[valid]
            if len(selected) == 0:
                continue
            selected_row = output_row[valid]
            selected_column = output_column[valid]
            selected_counts = packet_counts[valid]
            flat_indices = (
                (
                    (
                        selected[:, 0, None] * output_channels
                        + output_channel[None, :]
                    )
                    * height
                    + selected_row[:, None]
                )
                * width
                + selected_column[:, None]
            ).reshape(-1)
            selected_weights = weights[
                :, selected[:, 1], kernel_row, kernel_column
            ].transpose(0, 1)
            if expand_unit_events:
                # Every expanded record is a unit event.  The hardware action
                # is a signed weight add; no count multiplier is present.
                contributions = selected_weights.reshape(-1)
            else:
                contributions = (
                    selected_counts[:, None] * selected_weights
                ).reshape(-1)
            flat.index_add_(0, flat_indices, contributions)
    return flat.reshape(batch, output_channels, height, width)


def pooled_touched_mask(source_counts: torch.Tensor) -> torch.Tensor:
    """Return pool-output spatial sites touched by any 3x3 source fanout."""

    source = _require_integer_counts(source_counts, "pooled_touched_mask")
    support = source.ne(0).any(dim=1, keepdim=True).to(torch.float32)
    convolution_support = F.max_pool2d(support, kernel_size=3, stride=1, padding=1)
    return F.max_pool2d(convolution_support, kernel_size=2, stride=2).ne(0)[:, 0]


def valid_fanout_work_per_sample(
    source_counts: torch.Tensor,
    destination_channels: int,
    *,
    unit_events: bool,
) -> torch.Tensor:
    """Count valid weight additions/MACs, excluding disabled padding taps."""

    source = _require_integer_counts(source_counts, "valid_fanout_work_per_sample")
    height, width = source.shape[-2:]
    row_fanout = torch.full((height,), 3, dtype=torch.int64)
    column_fanout = torch.full((width,), 3, dtype=torch.int64)
    row_fanout[[0, -1]] = 2
    column_fanout[[0, -1]] = 2
    spatial_fanout = row_fanout[:, None] * column_fanout[None, :]
    multiplicity = source if unit_events else source.ne(0).to(torch.int64)
    return (
        (multiplicity * spatial_fanout).sum(dim=(1, 2, 3))
        * destination_channels
    )


def _post_pool_current(
    raw_dot: torch.Tensor,
    layer: FixedLayer,
    reference: FixedPointPursuitReference,
    counters: RuntimeCounters,
) -> torch.Tensor:
    """Apply per-output shift, optional pool, one constant add, and S24 clamp."""

    current = raw_dot.to(torch.int64)
    shifts = layer.product_left_shift.to(torch.int64)
    current = current * output_channel_view(1 << shifts, current.ndim)
    if bool(((current < S40_MINIMUM) | (current > S40_MAXIMUM)).any()):
        raise OverflowError(f"{layer.name}: complete dot product exceeds S40")
    if layer.pool:
        current = F.max_pool2d(current, kernel_size=2, stride=2)
    current = current + output_channel_view(
        layer.constant_integer.to(torch.int64), current.ndim
    )
    return reference._saturate_accumulator(current, layer.name, counters).to(
        torch.int64
    )


def _integer_lif(
    current: torch.Tensor,
    previous_membrane: torch.Tensor,
    layer: FixedLayer,
    reference: FixedPointPursuitReference,
    counters: RuntimeCounters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The frozen floor(3v/4), MultiSpike, subtractive-reset LIF in int64."""

    current = current.to(torch.int64)
    previous = previous_membrane.to(torch.int64)
    decayed = torch.div(3 * previous, 4, rounding_mode="floor")
    pre_reset = decayed + current
    scale = output_channel_view(
        (1 << layer.current_fraction_bits.to(torch.int64)), current.ndim
    )
    unsaturated = torch.clamp_min(
        torch.div(pre_reset, scale, rounding_mode="floor"), 0
    )
    spikes = unsaturated.clamp(max=reference.spike_maximum_integer)
    counters.record_spikes(layer.name, unsaturated, spikes)
    membrane = torch.maximum(pre_reset - spikes * scale, -scale)
    return spikes.to(torch.int64), membrane.to(torch.int64)


def _record_trace(
    trace: dict[str, np.ndarray] | None,
    timestep: int,
    layer_name: str,
    current: torch.Tensor,
    spikes: torch.Tensor,
    membrane: torch.Tensor,
) -> None:
    if trace is None:
        return
    prefix = f"t{timestep}.{layer_name}"
    trace[f"{prefix}.current_integer"] = current.cpu().numpy().astype(np.int32)
    trace[f"{prefix}.spike_integer"] = spikes.cpu().numpy().astype(np.uint8)
    trace[f"{prefix}.membrane_integer"] = membrane.cpu().numpy().astype(np.int32)


_SPARSE_IDLE = 0
_SPARSE_CLEAR = 1
_SPARSE_WAIT_PACKET = 2
_SPARSE_UPDATE = 3
_SPARSE_FINALIZE = 4
_SPARSE_FINALIZE_DRAIN = 5


@dataclass(frozen=True)
class SparseFrontendCycleResult:
    frontend_cycles: int
    l1_wrapper_cycles: int
    l1_core_cycles: int
    l2_core_cycles: int
    emitted_l1_packets: int
    accepted_l2_packets: int


@dataclass
class _SparseCoreControl:
    """Control-only mirror of ``sparse_conv3x3_pool_lif_core``."""

    height: int
    width: int
    taps_per_cycle: int
    state: int = _SPARSE_IDLE
    busy: bool = False
    done: bool = False
    clear_index: int = 0
    input_row_cursor: int = 0
    end_seen: bool = False
    tap_base: int = 0
    finalize_output_row: int = 0
    finalize_pair_column: int = 0
    finalize_is_last_row: bool = False
    output_valid: bool = False
    output_address: int = 0
    cycle_count: int = 0
    accepted_packets: int = 0

    def input_ready(self, input_row: int, input_end: bool) -> bool:
        final_update = (
            self.state == _SPARSE_UPDATE
            and self.tap_base + self.taps_per_cycle >= 9
        )
        return (
            self.busy
            and (self.state == _SPARSE_WAIT_PACKET or final_update)
            and not self.end_seen
            and not input_end
            and input_row == self.input_row_cursor
        )

    def step(
        self,
        *,
        start: bool,
        input_valid: bool,
        input_row: int,
        input_end: bool,
        output_ready: bool,
    ) -> None:
        # Build next-state values from the pre-edge state.  This preserves the
        # nonblocking-assignment behavior of the RTL, including simultaneous
        # output consume/replace and pipelined packet acceptance.
        next_values = dict(self.__dict__)
        next_values["done"] = False
        if self.busy:
            next_values["cycle_count"] = self.cycle_count + 1
        if self.busy and input_end:
            next_values["end_seen"] = True
        if self.output_valid and output_ready:
            next_values["output_valid"] = False

        def complete_finalized_row() -> None:
            if self.finalize_is_last_row:
                next_values["busy"] = False
                next_values["done"] = True
                next_values["state"] = _SPARSE_IDLE
            else:
                next_values["input_row_cursor"] = self.input_row_cursor + 1
                next_values["state"] = _SPARSE_WAIT_PACKET

        if self.state == _SPARSE_IDLE:
            if start:
                next_values.update(
                    busy=True,
                    end_seen=input_end,
                    clear_index=0,
                    input_row_cursor=0,
                    output_valid=False,
                    cycle_count=0,
                    accepted_packets=0,
                    state=_SPARSE_CLEAR,
                )
        elif self.state == _SPARSE_CLEAR:
            column_bank_depth = (self.width + 2) // 3
            if self.clear_index == column_bank_depth - 1:
                next_values["state"] = _SPARSE_WAIT_PACKET
            else:
                next_values["clear_index"] = self.clear_index + 1
        elif self.state == _SPARSE_WAIT_PACKET:
            if (
                input_valid
                and input_row == self.input_row_cursor
                and not self.end_seen
            ):
                next_values.update(
                    tap_base=0,
                    accepted_packets=self.accepted_packets + 1,
                    state=_SPARSE_UPDATE,
                )
            elif (
                (input_valid and input_row > self.input_row_cursor)
                or self.end_seen
                or input_end
            ):
                if self.input_row_cursor == 0:
                    next_values["input_row_cursor"] = 1
                elif self.input_row_cursor < self.height:
                    next_values.update(
                        finalize_output_row=self.input_row_cursor - 1,
                        finalize_pair_column=0,
                        finalize_is_last_row=False,
                        state=_SPARSE_FINALIZE,
                    )
                else:
                    next_values.update(
                        finalize_output_row=self.height - 1,
                        finalize_pair_column=0,
                        finalize_is_last_row=True,
                        state=_SPARSE_FINALIZE,
                    )
        elif self.state == _SPARSE_UPDATE:
            if self.tap_base + self.taps_per_cycle >= 9:
                if (
                    input_valid
                    and input_row == self.input_row_cursor
                    and not self.end_seen
                    and not input_end
                ):
                    next_values["tap_base"] = 0
                    next_values["accepted_packets"] = self.accepted_packets + 1
                else:
                    next_values["state"] = _SPARSE_WAIT_PACKET
            else:
                next_values["tap_base"] = (
                    self.tap_base + self.taps_per_cycle
                )
        elif self.state == _SPARSE_FINALIZE:
            half_width = self.width // 2
            if self.finalize_output_row % 2 == 0:
                if self.finalize_pair_column == half_width - 1:
                    complete_finalized_row()
                else:
                    next_values["finalize_pair_column"] = (
                        self.finalize_pair_column + 1
                    )
            elif not self.output_valid or output_ready:
                next_values["output_valid"] = True
                next_values["output_address"] = (
                    (self.finalize_output_row // 2) * half_width
                    + self.finalize_pair_column
                )
                if self.finalize_pair_column == half_width - 1:
                    next_values["state"] = _SPARSE_FINALIZE_DRAIN
                else:
                    next_values["finalize_pair_column"] = (
                        self.finalize_pair_column + 1
                    )
        elif self.state == _SPARSE_FINALIZE_DRAIN:
            if not self.output_valid or output_ready:
                complete_finalized_row()
        else:
            raise AssertionError(f"unknown sparse-core state {self.state}")
        self.__dict__.update(next_values)


@dataclass
class _L1SerializerControl:
    """Control-only mirror of the nonzero-channel L1 packet serializer."""

    busy: bool = False
    done: bool = False
    serialize_active: bool = False
    serialize_mask: int = 0
    serialize_address: int = 0
    core_done_seen: bool = False
    cycle_count: int = 0
    emitted_packets: int = 0

    def sparse_output(self) -> tuple[bool, int, int]:
        selected_channel = next(
            (
                channel
                for channel in range(8)
                if self.serialize_mask & (1 << channel)
            ),
            0,
        )
        return (
            self.serialize_active and self.serialize_mask != 0,
            selected_channel,
            self.serialize_address,
        )

    def step(
        self,
        *,
        start: bool,
        core_output_valid: bool,
        core_output_address: int,
        core_output_ready: bool,
        core_done: bool,
        core_busy: bool,
        output_nonzero_masks: list[int],
        sparse_valid: bool,
        sparse_ready: bool,
        sparse_channel: int,
    ) -> None:
        next_values = dict(self.__dict__)
        next_values["done"] = False
        if self.busy:
            next_values["cycle_count"] = self.cycle_count + 1
        if start and not self.busy:
            next_values.update(
                busy=True,
                serialize_active=False,
                serialize_mask=0,
                core_done_seen=False,
                cycle_count=0,
                emitted_packets=0,
            )
        if core_output_valid and core_output_ready:
            mask = output_nonzero_masks[core_output_address]
            next_values.update(
                serialize_address=core_output_address,
                serialize_mask=mask,
                serialize_active=bool(mask),
            )
        if sparse_valid and sparse_ready:
            remaining = self.serialize_mask & ~(1 << sparse_channel)
            next_values["serialize_mask"] = remaining
            next_values["emitted_packets"] = self.emitted_packets + 1
            if remaining == 0:
                next_values["serialize_active"] = False
        if core_done:
            next_values["core_done_seen"] = True
        if self.core_done_seen and not self.serialize_active and not core_busy:
            next_values.update(
                busy=False,
                done=True,
                core_done_seen=False,
            )
        self.__dict__.update(next_values)


def replay_sparse_frontend_control(
    event_rows: Iterable[int],
    l1_output_nonzero_masks: Iterable[int],
) -> SparseFrontendCycleResult:
    """Replay the integrated L1/L2 ready-valid FSMs cycle by cycle.

    Arithmetic is intentionally absent: the caller supplies the already
    bit-exact L1 nonzero-channel mask for every 48x64 address.  The replay then
    models row progression, clear/finalize/drain, L1 serialization, L2's
    three update beats, end flush, and all backpressure exactly as the RTL.
    """

    rows = [int(value) for value in event_rows]
    masks = [int(value) for value in l1_output_nonzero_masks]
    if rows != sorted(rows):
        raise ValueError("sparse frontend requires nondecreasing event rows")
    if any(row < 0 or row >= 96 for row in rows):
        raise ValueError("L1 event row is outside [0,95]")
    if len(masks) != 48 * 64:
        raise ValueError(f"expected 3072 L1 output masks, got {len(masks)}")
    if any(mask < 0 or mask >= (1 << 8) for mask in masks):
        raise ValueError("L1 output mask does not fit eight channels")

    l1_core = _SparseCoreControl(height=96, width=128, taps_per_cycle=9)
    l2_core = _SparseCoreControl(height=48, width=64, taps_per_cycle=3)
    serializer = _L1SerializerControl()
    input_index = 0
    event_end_pulse = False
    frontend_cycles = 0
    maximum_cycles = 1_000_000

    for edge in range(maximum_cycles):
        start = edge == 0
        input_valid = edge >= 1 and input_index < len(rows)
        input_row = rows[input_index] if input_valid else 0
        input_end = event_end_pulse
        event_end_pulse = False

        sparse_valid, sparse_channel, sparse_address = (
            serializer.sparse_output()
        )
        l1_output_ready = not serializer.serialize_active
        l2_input_ready = l2_core.input_ready(
            sparse_address // 64,
            serializer.done,
        )
        event_ready = l1_core.input_ready(input_row, input_end)

        frontend_busy = (
            serializer.busy
            or l2_core.busy
            or (serializer.done and not l2_core.done)
        )
        if start and not frontend_busy:
            frontend_cycles = 0
        elif frontend_busy:
            frontend_cycles += 1

        l1_pre_busy = l1_core.busy
        l1_pre_done = l1_core.done
        l1_pre_output_valid = l1_core.output_valid
        l1_pre_output_address = l1_core.output_address
        l1_sparse_end = serializer.done

        l1_core.step(
            start=start,
            input_valid=input_valid,
            input_row=input_row,
            input_end=input_end,
            output_ready=l1_output_ready,
        )
        l2_core.step(
            start=start,
            input_valid=sparse_valid,
            input_row=sparse_address // 64,
            input_end=l1_sparse_end,
            output_ready=True,
        )
        serializer.step(
            start=start,
            core_output_valid=l1_pre_output_valid,
            core_output_address=l1_pre_output_address,
            core_output_ready=l1_output_ready,
            core_done=l1_pre_done,
            core_busy=l1_pre_busy,
            output_nonzero_masks=masks,
            sparse_valid=sparse_valid,
            sparse_ready=l2_input_ready,
            sparse_channel=sparse_channel,
        )

        if input_valid and event_ready:
            input_index += 1
            if input_index == len(rows):
                event_end_pulse = True

        if l2_core.done:
            expected_packets = sum(mask.bit_count() for mask in masks)
            if input_index != len(rows):
                raise AssertionError("frontend completed before consuming events")
            if serializer.emitted_packets != expected_packets:
                raise AssertionError("L1 serializer packet count mismatch")
            if l2_core.accepted_packets != expected_packets:
                raise AssertionError("L2 accepted packet count mismatch")
            return SparseFrontendCycleResult(
                frontend_cycles=frontend_cycles,
                l1_wrapper_cycles=serializer.cycle_count,
                l1_core_cycles=l1_core.cycle_count,
                l2_core_cycles=l2_core.cycle_count,
                emitted_l1_packets=serializer.emitted_packets,
                accepted_l2_packets=l2_core.accepted_packets,
            )
    raise RuntimeError(f"sparse frontend control replay exceeded {maximum_cycles} cycles")


@dataclass
class SampleWork:
    l1_unit_events: list[int] = field(default_factory=list)
    l1_nonzero_count_addresses: list[int] = field(default_factory=list)
    l1_valid_weight_additions: list[int] = field(default_factory=list)
    l1_touched_pool_sites: list[int] = field(default_factory=list)
    l1_output_packets: list[int] = field(default_factory=list)
    l1_output_spike_units: list[int] = field(default_factory=list)
    l2_valid_count_weight_macs: list[int] = field(default_factory=list)
    l2_touched_pool_sites: list[int] = field(default_factory=list)
    l2_output_packets: list[int] = field(default_factory=list)
    l2_output_spike_units: list[int] = field(default_factory=list)
    l2_default_nonzero_records_at_untouched_sites: list[int] = field(
        default_factory=list
    )
    l2_default_spike_units_at_untouched_sites: list[int] = field(
        default_factory=list
    )
    sparse_frontend_cycle_exact: list[int] = field(default_factory=list)
    sparse_frontend_l1_wrapper_cycles: list[int] = field(default_factory=list)
    sparse_frontend_l1_core_cycles: list[int] = field(default_factory=list)
    sparse_frontend_l2_core_cycles: list[int] = field(default_factory=list)

    def extend(
        self,
        frames: torch.Tensor,
        l1_touched: torch.Tensor,
        l1_spikes: torch.Tensor,
        l2_touched: torch.Tensor,
        l2_spikes: torch.Tensor,
        l1_work: torch.Tensor,
        l2_work: torch.Tensor,
        l2_default_spikes: torch.Tensor,
    ) -> None:
        batch = frames.shape[0]
        l1_packets = l1_spikes.ne(0).sum(dim=(1, 2, 3))
        l2_packets = l2_spikes.ne(0).sum(dim=(1, 2, 3))
        l2_untouched = (~l2_touched).sum(dim=(1, 2))
        default_nonzero_channels = int(torch.count_nonzero(l2_default_spikes))
        default_spike_units = int(l2_default_spikes.sum())
        fields = {
            "l1_unit_events": frames.sum(dim=(1, 2, 3)),
            "l1_nonzero_count_addresses": frames.ne(0).sum(dim=(1, 2, 3)),
            "l1_valid_weight_additions": l1_work,
            "l1_touched_pool_sites": l1_touched.sum(dim=(1, 2)),
            "l1_output_packets": l1_packets,
            "l1_output_spike_units": l1_spikes.sum(dim=(1, 2, 3)),
            "l2_valid_count_weight_macs": l2_work,
            "l2_touched_pool_sites": l2_touched.sum(dim=(1, 2)),
            "l2_output_packets": l2_packets,
            "l2_output_spike_units": l2_spikes.sum(dim=(1, 2, 3)),
            "l2_default_nonzero_records_at_untouched_sites": (
                l2_untouched * default_nonzero_channels
            ),
            "l2_default_spike_units_at_untouched_sites": (
                l2_untouched * default_spike_units
            ),
        }
        for name, values in fields.items():
            values = values.to(torch.int64).tolist()
            if len(values) != batch:
                raise AssertionError(f"{name}: expected {batch} values")
            getattr(self, name).extend(int(value) for value in values)

        channel_bits = 1 << torch.arange(8, dtype=torch.int64)
        for sample_index in range(batch):
            events_per_row = frames[sample_index].sum(dim=(0, 2)).to(
                torch.int64
            )
            event_rows = torch.repeat_interleave(
                torch.arange(96, dtype=torch.int64),
                events_per_row,
            ).tolist()
            nonzero_by_address = (
                l1_spikes[sample_index]
                .ne(0)
                .permute(1, 2, 0)
                .reshape(-1, 8)
                .to(torch.int64)
            )
            nonzero_masks = (nonzero_by_address * channel_bits).sum(
                dim=1
            ).tolist()
            replay = replay_sparse_frontend_control(event_rows, nonzero_masks)
            if replay.emitted_l1_packets != int(l1_packets[sample_index]):
                raise AssertionError("cycle replay and arithmetic packet counts differ")
            self.sparse_frontend_cycle_exact.append(replay.frontend_cycles)
            self.sparse_frontend_l1_wrapper_cycles.append(
                replay.l1_wrapper_cycles
            )
            self.sparse_frontend_l1_core_cycles.append(replay.l1_core_cycles)
            self.sparse_frontend_l2_core_cycles.append(replay.l2_core_cycles)


class SparseHybridReference:
    """Bit-exact selected dataflow built from a frozen fixed-point reference."""

    def __init__(self, reference: FixedPointPursuitReference) -> None:
        self.reference = reference
        self.layers = {layer.name: layer for layer in reference.layers}
        if tuple(self.layers) != (
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L6",
            "L7",
            "L8",
            "H1",
            "H2",
        ):
            raise ValueError(f"unexpected layer order: {tuple(self.layers)}")

    def _dense_current(
        self,
        source: torch.Tensor,
        layer: FixedLayer,
        counters: RuntimeCounters,
    ) -> torch.Tensor:
        weights = _integer_weights(layer)
        if bool((source < 0).any()):
            raise ValueError(f"{layer.name}: source spikes must be nonnegative")
        rounded = torch.round(source)
        if not torch.equal(source, rounded):
            raise ValueError(f"{layer.name}: source spikes must be exact integers")
        source = rounded.to(device="cpu", dtype=torch.int64)
        if weights.ndim == 4:
            if source.ndim != 4:
                raise ValueError(
                    f"{layer.name}: convolution source must be [B,C,H,W]"
                )
            raw = F.conv2d(source, weights, padding=1)
        else:
            if source.ndim != 2:
                raise ValueError(f"{layer.name}: linear source must be [B,F]")
            raw = F.linear(source, weights)
        return _post_pool_current(raw, layer, self.reference, counters)

    def _default_lif(
        self,
        layer: FixedLayer,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = layer.constant_integer.to(torch.int64).clamp(
            self.reference.accumulator_minimum,
            self.reference.accumulator_maximum,
        )
        shape = (1, len(current), 1, 1)
        current = current.reshape(shape)
        counters = RuntimeCounters.empty()
        spikes, membrane = _integer_lif(
            current,
            torch.zeros_like(current),
            layer,
            self.reference,
            counters,
        )
        return current[0, :, 0, 0], spikes[0, :, 0, 0], membrane[0, :, 0, 0]

    @staticmethod
    def _check_untouched_defaults(
        layer_name: str,
        touched: torch.Tensor,
        current: torch.Tensor,
        spikes: torch.Tensor,
        membrane: torch.Tensor,
        default_current: torch.Tensor,
        default_spikes: torch.Tensor,
        default_membrane: torch.Tensor,
    ) -> None:
        untouched = (~touched)[:, None, :, :]
        expected = (
            (default_current, current, "current"),
            (default_spikes, spikes, "spikes"),
            (default_membrane, membrane, "membrane"),
        )
        for default, actual, name in expected:
            mismatch = untouched & actual.ne(default[None, :, None, None])
            if bool(mismatch.any()):
                raise AssertionError(
                    f"{layer_name}: {int(mismatch.sum())} untouched {name} values "
                    "do not equal the per-channel default plane"
                )

    def predict(
        self,
        event_frames: torch.Tensor,
        counters: RuntimeCounters,
        work: SampleWork,
        trace: dict[str, np.ndarray] | None = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        frames = _require_integer_counts(event_frames, "event_frames")
        if tuple(frames.shape[1:]) != (2, 96, 128):
            raise ValueError(f"unexpected input shape: {tuple(frames.shape)}")
        if not torch.all(frames.sum(dim=(1, 2, 3)) == 400):
            raise ValueError("every input must contain exactly 400 events")

        l1 = self.layers["L1"]
        l1_raw = sparse_same_conv3x3(
            frames,
            _integer_weights(l1),
            expand_unit_events=True,
        )
        l1_current = _post_pool_current(l1_raw, l1, self.reference, counters)
        l1_spikes, l1_membrane = _integer_lif(
            l1_current,
            torch.zeros_like(l1_current),
            l1,
            self.reference,
            counters,
        )
        l1_touched = pooled_touched_mask(frames)
        l1_defaults = self._default_lif(l1)
        self._check_untouched_defaults(
            "L1",
            l1_touched,
            l1_current,
            l1_spikes,
            l1_membrane,
            *l1_defaults,
        )
        if bool((l1_defaults[1] != 0).any()):
            raise AssertionError("L1 default plane is nonzero; event-only omission is unsafe")
        for timestep in range(2):
            _record_trace(
                trace, timestep, "L1", l1_current, l1_spikes, l1_membrane
            )

        l2 = self.layers["L2"]
        if int(l1_spikes.max()) > 255:
            raise OverflowError("L1 packet count exceeds the selected U8 interface")
        l2_raw = sparse_same_conv3x3(
            l1_spikes,
            _integer_weights(l2),
            expand_unit_events=False,
        )
        l2_current = _post_pool_current(l2_raw, l2, self.reference, counters)
        l2_spikes, l2_membrane = _integer_lif(
            l2_current,
            torch.zeros_like(l2_current),
            l2,
            self.reference,
            counters,
        )
        l2_touched = pooled_touched_mask(l1_spikes)
        l2_defaults = self._default_lif(l2)
        self._check_untouched_defaults(
            "L2",
            l2_touched,
            l2_current,
            l2_spikes,
            l2_membrane,
            *l2_defaults,
        )
        for timestep in range(2):
            _record_trace(
                trace, timestep, "L2", l2_current, l2_spikes, l2_membrane
            )

        work.extend(
            frames,
            l1_touched,
            l1_spikes,
            l2_touched,
            l2_spikes,
            valid_fanout_work_per_sample(
                frames, destination_channels=8, unit_events=True
            ),
            valid_fanout_work_per_sample(
                l1_spikes, destination_channels=16, unit_events=False
            ),
            l2_defaults[1],
        )

        # L3 receives the same L2 map in both bins.  Compute its current once,
        # then apply two stateful LIF updates directly without a current BRAM.
        l3 = self.layers["L3"]
        l3_current = self._dense_current(l2_spikes, l3, counters)
        l3_spikes_0, l3_membrane_0 = _integer_lif(
            l3_current,
            torch.zeros_like(l3_current),
            l3,
            self.reference,
            counters,
        )
        l3_spikes_1, l3_membrane_1 = _integer_lif(
            l3_current,
            l3_membrane_0,
            l3,
            self.reference,
            counters,
        )
        _record_trace(trace, 0, "L3", l3_current, l3_spikes_0, l3_membrane_0)
        _record_trace(trace, 1, "L3", l3_current, l3_spikes_1, l3_membrane_1)

        values_by_time = [l3_spikes_0, l3_spikes_1]
        for layer_name in ("L4", "L5", "L6", "L7", "L8", "H1", "H2"):
            layer = self.layers[layer_name]
            previous = None
            next_values: list[torch.Tensor] = []
            for timestep, values in enumerate(values_by_time):
                if layer_name == "H1" and values.ndim == 4:
                    values = values.reshape(values.shape[0], 480)
                current = self._dense_current(values, layer, counters)
                if previous is None:
                    previous = torch.zeros_like(current)
                spikes, previous = _integer_lif(
                    current,
                    previous,
                    layer,
                    self.reference,
                    counters,
                )
                _record_trace(
                    trace,
                    timestep,
                    layer_name,
                    current,
                    spikes,
                    previous,
                )
                next_values.append(spikes)
            values_by_time = next_values

        temporal_sum = values_by_time[0] + values_by_time[1]
        output_integer = F.linear(
            temporal_sum.to(torch.int64),
            self.reference.readout_weight_integer.to(torch.int64),
        )
        readout_scale = (
            1 << self.reference.readout_fraction_bits.to(torch.int64)
        ).view(1, -1)
        prediction = (output_integer.to(torch.float64) / (2 * readout_scale)).to(
            torch.float32
        )
        if trace is not None:
            trace["readout.temporal_sum_integer"] = (
                temporal_sum.cpu().numpy().astype(np.uint16)
            )
            trace["readout.output_integer_before_temporal_divide"] = (
                output_integer.cpu().numpy().astype(np.int64)
            )
            trace["readout.prediction_normalized_xy"] = prediction.cpu().numpy()

        defaults = {
            "L1": {
                "current_integer": [int(value) for value in l1_defaults[0]],
                "spike_integer": [int(value) for value in l1_defaults[1]],
                "membrane_integer": [int(value) for value in l1_defaults[2]],
            },
            "L2": {
                "current_integer": [int(value) for value in l2_defaults[0]],
                "spike_integer": [int(value) for value in l2_defaults[1]],
                "membrane_integer": [int(value) for value in l2_defaults[2]],
            },
        }
        return prediction, defaults


@dataclass
class TraceComparison:
    values_checked: int = 0
    mismatches: int = 0
    per_tensor_mismatches: dict[str, int] = field(default_factory=dict)

    def update(
        self,
        expected: dict[str, np.ndarray],
        actual: dict[str, np.ndarray],
    ) -> None:
        if expected.keys() != actual.keys():
            missing = sorted(expected.keys() - actual.keys())
            extra = sorted(actual.keys() - expected.keys())
            raise AssertionError(f"trace key mismatch: missing={missing}, extra={extra}")
        for name in expected:
            if expected[name].shape != actual[name].shape:
                raise AssertionError(
                    f"{name}: shape {actual[name].shape} != {expected[name].shape}"
                )
            count = int(expected[name].size)
            mismatch = int(np.count_nonzero(expected[name] != actual[name]))
            self.values_checked += count
            self.mismatches += mismatch
            self.per_tensor_mismatches[name] = (
                self.per_tensor_mismatches.get(name, 0) + mismatch
            )


def _distribution(values: Iterable[int | float]) -> dict[str, float | int]:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(materialized)
    return {
        "count": len(materialized),
        "minimum": ordered[0],
        "mean": statistics.fmean(materialized),
        "median": statistics.median(ordered),
        "p95_linear": float(np.percentile(np.asarray(ordered), 95)),
        "maximum": ordered[-1],
        "sample_standard_deviation": (
            statistics.stdev(materialized) if len(materialized) > 1 else 0.0
        ),
    }


def _one_or_two_step_dense_schedule(
    layer_index: int,
    *,
    steps: int,
    output_pes: int = 8,
    input_channels_per_pe: int = 4,
    weights_per_load_cycle: int = 16,
    pipeline_latency: int = 14,
) -> dict[str, int]:
    """Double-buffered tile schedule with one weight load for one/two steps."""

    layer = CONV_LAYERS[layer_index]
    load_cycles: list[int] = []
    compute_cycles: list[int] = []
    for output_start in range(0, layer.out_channels, output_pes):
        valid_outputs = min(output_pes, layer.out_channels - output_start)
        for input_start in range(0, layer.in_channels, input_channels_per_pe):
            valid_inputs = min(
                input_channels_per_pe, layer.in_channels - input_start
            )
            load_cycles.append(
                math.ceil(
                    valid_outputs
                    * valid_inputs
                    * layer.kernel_elements
                    / weights_per_load_cycle
                )
            )
            compute_cycles.append(
                (layer.out_height + 2) * (layer.out_width + 2)
                + pipeline_latency
            )
    cycles = load_cycles[0] + steps * compute_cycles[0]
    for index in range(1, len(compute_cycles)):
        cycles += max(
            0,
            load_cycles[index] - steps * compute_cycles[index - 1],
        )
        cycles += steps * compute_cycles[index]
    return {
        "steps": steps,
        "weight_values_read_once": layer.weights,
        "weight_values_read_if_separate_invocations": steps * layer.weights,
        "weight_values_saved_by_layer_adjacent_reuse": (steps - 1) * layer.weights,
        "double_buffered_cycles": cycles,
        "separate_invocation_cycles": steps
        * _one_or_two_step_dense_schedule(
            layer_index,
            steps=1,
            output_pes=output_pes,
            input_channels_per_pe=input_channels_per_pe,
            weights_per_load_cycle=weights_per_load_cycle,
            pipeline_latency=pipeline_latency,
        )["double_buffered_cycles"]
        if steps == 2
        else cycles,
    }


def dense_backend_schedule() -> dict[str, object]:
    per_layer = {
        "L3": _one_or_two_step_dense_schedule(2, steps=1),
        **{
            f"L{index + 1}": _one_or_two_step_dense_schedule(index, steps=2)
            for index in range(3, 8)
        },
    }
    convolution_cycles = sum(
        item["double_buffered_cycles"] for item in per_layer.values()
    )
    saved_weight_values = sum(
        item["weight_values_saved_by_layer_adjacent_reuse"]
        for item in per_layer.values()
    )
    saved_cycles = sum(
        item["separate_invocation_cycles"] - item["double_buffered_cycles"]
        for item in per_layer.values()
    )
    linear_per_step = {
        name: max(
            math.ceil(outputs / 8) * math.ceil(inputs / 36),
            math.ceil(inputs * outputs / 16),
        )
        for name, inputs, outputs in LINEAR_LAYERS
    }
    linear_and_readout_cycles = (
        2 * sum(linear_per_step.values())
        + max(
            math.ceil(READOUT[2] / 8) * math.ceil(READOUT[1] / 36),
            math.ceil(READOUT[1] * READOUT[2] / 16),
        )
    )
    return {
        "core": {
            "output_pes": 8,
            "input_channels_per_pe": 4,
            "kernel_taps": 9,
            "multiplier_lanes": 288,
        },
        "convolution_layers": per_layer,
        "convolution_cycles": convolution_cycles,
        "weight_values_saved_by_layer_adjacent_reuse": saved_weight_values,
        "cycles_saved_by_weight_reuse_under_double_buffering": saved_cycles,
        "linear_cycles_per_timestep": linear_per_step,
        "linear_and_readout_cycles": linear_and_readout_cycles,
        "total_cycles": convolution_cycles + linear_and_readout_cycles,
        "l3_current_cache_bytes": 0,
        "l3_current_cache_bytes_saved_vs_old_SL_HYB": 4_608 * 3,
        "l3_dual_lif_assumption": (
            "two cascaded LIF stages sustain the convolution output rate; only "
            "pipeline latency, not a 4608-value replay, remains"
        ),
        "limitations": (
            "analytical double-buffered schedule; controller stalls and routed "
            "clock frequency require RTL simulation and Vivado"
        ),
    }


def _front_cycle_distribution(
    work: SampleWork,
    *,
    l1_taps_per_cycle: int,
    l2_taps_per_cycle: int,
    l2_packets_per_cycle: int,
) -> dict[str, object]:
    l1_tap_beats = math.ceil(9 / l1_taps_per_cycle)
    l2_tap_beats = math.ceil(9 / l2_taps_per_cycle)
    l1_stage = []
    l2_stage = []
    total = []
    for events, packets in zip(work.l1_unit_events, work.l1_output_packets):
        # Finalization visits each pre-pool row and adjacent-column pair.  An
        # even row stores horizontal maxima and the following odd row combines
        # them vertically, hence H*(W/2) rather than pooled H*(W/2) beats.
        l1 = (
            events * l1_tap_beats
            + 96 * 64
            + packets
        )
        l2 = (
            math.ceil(packets / l2_packets_per_cycle) * l2_tap_beats
            + 48 * 32
        )
        l1_stage.append(l1)
        l2_stage.append(l2)
        total.append(l1 + l2)
    return {
        "assumptions": {
            "l1_taps_per_cycle": l1_taps_per_cycle,
            "l2_taps_per_cycle": l2_taps_per_cycle,
            "l2_packets_per_cycle": l2_packets_per_cycle,
            "l1_output_packet_serialization": "one nonzero channel packet per cycle",
            "pool_finalize": "one adjacent-column pair for every pre-pool row per cycle",
            "stages": "serialized; no L1/L2 FIFO overlap credited",
            "padding": "all nine taps occupy issue slots; invalid edge taps suppress writes",
            "excluded": "handshake fill/drain and bank-conflict stalls beyond the stated fixed issue schedule",
        },
        "l1_stage_cycles": _distribution(l1_stage),
        "l2_stage_cycles": _distribution(l2_stage),
        "sparse_front_total_cycles": _distribution(total),
    }


def _work_report(work: SampleWork) -> dict[str, object]:
    return {
        name: _distribution(values)
        for name, values in work.__dict__.items()
    }


def _operation_report(work: SampleWork) -> dict[str, object]:
    deep_conv = sum(DEEP_DENSE_CONV_MACS.values())
    deep_linear = sum(DEEP_DENSE_LINEAR_MACS.values())
    mixed_total = [
        l1 + l2 + deep_conv + deep_linear
        for l1, l2 in zip(
            work.l1_valid_weight_additions,
            work.l2_valid_count_weight_macs,
        )
    ]
    return {
        "L1_add_only_valid_weight_additions": _distribution(
            work.l1_valid_weight_additions
        ),
        "L2_nonzero_count_weight_macs": _distribution(
            work.l2_valid_count_weight_macs
        ),
        "L3_L8_dense_conv_macs": {
            **DEEP_DENSE_CONV_MACS,
            "total": deep_conv,
        },
        "H1_H2_readout_dense_macs": {
            **DEEP_DENSE_LINEAR_MACS,
            "total": deep_linear,
        },
        "mixed_synaptic_arithmetic_total": _distribution(mixed_total),
        "old_dense_SL_HYB_macs": OLD_DENSE_SL_HYB_MACS,
        "mean_reduction_vs_old_dense_SL_HYB_fraction": (
            1
            - float(_distribution(mixed_total)["mean"]) / OLD_DENSE_SL_HYB_MACS
        ),
        "comparison_limit": (
            "L1/L2 count only valid in-bounds sparse synapses, while the "
            "regular dense backend uses the conventional full-kernel count "
            "including same-padding zero taps. The total adds signed additions "
            "to MACs and is not a lane-issue, latency, switching, or energy count"
        ),
    }


def _scheduled_dense_conv_lane_operations() -> dict[str, int]:
    """Multiplier-lane issues of the fixed 8x4x9 shared convolution core."""

    invocations = {
        "L1": 1,
        "L2": 1,
        "L3": 1,
        **{f"L{index}": 2 for index in range(4, 9)},
    }
    result = {}
    for layer in CONV_LAYERS:
        output_tiles = math.ceil(layer.out_channels / 8)
        input_tiles = math.ceil(layer.in_channels / 4)
        result[layer.name] = (
            output_tiles
            * input_tiles
            * layer.out_height
            * layer.out_width
            * 288
            * invocations[layer.name]
        )
    return result


def _scheduled_lane_operation_values(
    work: SampleWork,
) -> dict[str, list[int]]:
    """Per-sample issued arithmetic lanes for the instantiated datapaths.

    Counts include spatial padding and unused channel/output lanes whenever the
    fixed-width engine issues them.  They do not estimate data-dependent
    switching, energy, memory traffic, pipeline fill/drain, Pool, or LIF work.
    """

    samples = len(work.l1_unit_events)
    if len(work.l1_output_packets) != samples:
        raise AssertionError("scheduled-lane packet count length mismatch")
    dense_conv = _scheduled_dense_conv_lane_operations()
    deep_conv = sum(dense_conv[f"L{index}"] for index in range(3, 9))
    # H1: 480 inputs x four 16-output groups x two steps.
    # H2: 60 inputs x one 16-output group x two steps.
    # Readout: twelve inputs x two physical multiplier lanes, once.
    dense_head = 480 * math.ceil(60 / 16) * 16 * 2
    dense_head += 60 * math.ceil(12 / 16) * 16 * 2
    dense_head += 12 * 2
    dense_total = sum(dense_conv.values()) + dense_head
    sparse_total = [
        events * 9 * 8 + packets * 9 * 16 + deep_conv + dense_head
        for events, packets in zip(work.l1_unit_events, work.l1_output_packets)
    ]
    return {
        "Dense_TM": [dense_total] * samples,
        "Sparse_TM": sparse_total,
        "Sparse_LM_A": list(sparse_total),
    }


def _scheduled_lane_operation_report(work: SampleWork) -> dict[str, object]:
    dense_conv = _scheduled_dense_conv_lane_operations()
    deep_conv = sum(dense_conv[f"L{index}"] for index in range(3, 9))
    dense_head = 480 * math.ceil(60 / 16) * 16 * 2
    dense_head += 60 * math.ceil(12 / 16) * 16 * 2
    dense_head += 12 * 2
    values = _scheduled_lane_operation_values(work)
    dense_distribution = _distribution(values["Dense_TM"])
    sparse_distribution = _distribution(values["Sparse_TM"])
    sparse_l1 = [events * 9 * 8 for events in work.l1_unit_events]
    sparse_l2 = [packets * 9 * 16 for packets in work.l1_output_packets]
    reduction = 1 - float(sparse_distribution["mean"]) / float(
        dense_distribution["mean"]
    )
    return {
        "definition": (
            "scheduled/issued arithmetic lanes of the instantiated datapaths; "
            "includes same-padding taps and fixed-width channel/output padding, "
            "but is not a valid-synapse, switching-activity, power, or energy count"
        ),
        "Dense_TM": {
            "convolution_multiplier_lane_operations": dense_conv,
            "L3_L8_multiplier_lane_operations": deep_conv,
            "H1_H2_readout_multiplier_lane_operations": dense_head,
            "total_scheduled_multiplier_lane_operations": dense_distribution,
        },
        "Sparse_TM": {
            "L1_scheduled_add_lane_operations": _distribution(sparse_l1),
            "L2_scheduled_multiplier_lane_operations": _distribution(sparse_l2),
            "L3_L8_multiplier_lane_operations": deep_conv,
            "H1_H2_readout_multiplier_lane_operations": dense_head,
            "total_scheduled_add_and_multiplier_lane_operations": (
                sparse_distribution
            ),
        },
        "Sparse_LM_A": {
            "L1_scheduled_add_lane_operations": _distribution(sparse_l1),
            "L2_scheduled_multiplier_lane_operations": _distribution(sparse_l2),
            "L3_L8_multiplier_lane_operations": deep_conv,
            "H1_H2_readout_multiplier_lane_operations": dense_head,
            "total_scheduled_add_and_multiplier_lane_operations": (
                _distribution(values["Sparse_LM_A"])
            ),
        },
        "mean_reduction_vs_Dense_TM_fraction": reduction,
    }


def _fixed_distribution(value: int | float, samples: int) -> dict[str, float | int]:
    return _distribution([value] * samples)


def _plus_fixed(values: Iterable[int | float], fixed: int) -> list[float]:
    return [float(value) + fixed for value in values]


def _cycle_distribution_with_200mhz(
    values: Iterable[int | float],
) -> dict[str, object]:
    materialized = [float(value) for value in values]
    result = _distribution(materialized)
    result["mean_latency_ms_if_200MHz"] = float(result["mean"]) / 200_000
    result["conditional_latency_ms_if_200MHz"] = {
        name: float(result[name]) / 200_000
        for name in ("minimum", "mean", "median", "p95_linear", "maximum")
    }
    result["mean_throughput_samples_per_second_if_200MHz"] = (
        200_000_000 / float(result["mean"])
    )
    result["mean_reduction_vs_Dense_TM_fraction"] = (
        1 - float(result["mean"]) / OLD_DENSE_SL_HYB_RTL_CYCLES
    )
    return result


def _rtl_calibrated_route_cycle_values(
    work: SampleWork,
) -> dict[str, list[float]]:
    front = work.sparse_frontend_cycle_exact
    return {
        "Dense_TM": [
            float(OLD_DENSE_SL_HYB_RTL_CYCLES)
            for _ in work.l1_unit_events
        ],
        "Sparse_TM": _plus_fixed(
            front,
            SPARSE_TM_RTL_CALIBRATION["fixed_backend_calibrated_cycles"],
        ),
        "Sparse_LM_A": _plus_fixed(
            front,
            SPARSE_LM_A_RTL_CALIBRATION["fixed_backend_calibrated_cycles"],
        ),
    }


def _layer_memory_shapes() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for layer in CONV_LAYERS:
        output_tiles = math.ceil(layer.out_channels / 8)
        input_tiles = math.ceil(layer.in_channels / 4)
        result[layer.name] = {
            "weight_BRAM_read_beats_256b_per_invocation": (
                output_tiles * input_tiles * 18
            ),
            "activation_BRAM_read_beats_4_channels_per_invocation": (
                output_tiles
                * input_tiles
                * layer.out_height
                * layer.out_width
            ),
            "activation_BRAM_write_beats_8_channels_per_output": (
                output_tiles * layer.pooled_height * layer.pooled_width
            ),
        }
    return result


def _schedule_memory_traffic(work: SampleWork) -> dict[str, object]:
    """Logical runtime traffic for the three requested schedule organizations."""

    samples = len(work.l1_unit_events)
    layer = _layer_memory_shapes()
    conv_weight = {
        name: values["weight_BRAM_read_beats_256b_per_invocation"]
        for name, values in layer.items()
    }
    conv_read = {
        name: values["activation_BRAM_read_beats_4_channels_per_invocation"]
        for name, values in layer.items()
    }
    conv_write = {
        name: values["activation_BRAM_write_beats_8_channels_per_output"]
        for name, values in layer.items()
    }
    linear_weight_beats = {"H1": 1_920, "H2": 60, "readout": 12}
    linear_activation_reads = dict(linear_weight_beats)
    linear_activation_writes_per_timestep = {"H1": 4, "H2": 1}

    time_major_conv_weight_beats = (
        conv_weight["L1"]
        + conv_weight["L2"]
        + conv_weight["L3"]
        + 2 * sum(conv_weight[f"L{index}"] for index in range(4, 9))
    )
    sparse_time_major_conv_weight_beats = (
        conv_weight["L3"]
        + 2 * sum(conv_weight[f"L{index}"] for index in range(4, 9))
    )
    sparse_layer_major_conv_weight_beats = conv_weight["L3"] + sum(
        conv_weight[f"L{index}"] for index in range(4, 9)
    )
    time_major_linear_weight_beats = (
        2 * (linear_weight_beats["H1"] + linear_weight_beats["H2"])
        + linear_weight_beats["readout"]
    )
    layer_major_linear_weight_beats = sum(linear_weight_beats.values())

    dense_conv_activation_reads = (
        conv_read["L1"]
        + conv_read["L2"]
        + conv_read["L3"]
        + 2 * sum(conv_read[f"L{index}"] for index in range(4, 9))
    )
    sparse_conv_activation_reads = conv_read["L3"] + 2 * sum(
        conv_read[f"L{index}"] for index in range(4, 9)
    )
    time_activation_linear_reads = (
        2 * (linear_activation_reads["H1"] + linear_activation_reads["H2"])
        + linear_activation_reads["readout"]
    )

    dense_conv_activation_writes = (
        conv_write["L1"]
        + conv_write["L2"]
        + 2 * sum(conv_write[f"L{index}"] for index in range(3, 9))
    )
    deep_conv_activation_writes = 2 * sum(
        conv_write[f"L{index}"] for index in range(3, 9)
    )
    dense_activation_writes = (
        dense_conv_activation_writes
        + 2 * sum(linear_activation_writes_per_timestep.values())
    )
    sparse_activation_writes = (
        768  # one 16xU8 L2 dense vector for every 24x32 address
        + deep_conv_activation_writes
        + 2 * sum(linear_activation_writes_per_timestep.values())
    )

    # L3--L8 vector beats use eight S16 states; H1/H2 use sixteen lanes.
    conv_state_vector_beats = sum(
        math.ceil(item.out_channels / 8) * item.pooled_height * item.pooled_width
        for item in CONV_LAYERS[2:]
    )
    dense_state_vector_beats = math.ceil(60 / 16) + math.ceil(12 / 16)
    external_state_vector_beats = conv_state_vector_beats + dense_state_vector_beats

    l1_local_weight_reads = [events * 72 for events in work.l1_unit_events]
    l2_local_weight_reads = [packets * 144 for packets in work.l1_output_packets]
    packet_reads = [
        events + packets
        for events, packets in zip(work.l1_unit_events, work.l1_output_packets)
    ]
    # The sparse front keeps partial sums in nine distributed-RAM banks.  An
    # interior packet touches all nine banks; padding taps suppress the actual
    # read/write and are removed using the exact valid-fanout counters.
    # Finalization reads and clears two banks per adjacent-column pair, and its
    # horizontal pool scratch is written on even rows and read on odd rows.
    # Keep these local vectors separate from cross-layer activation BRAM beats
    # because their widths differ.
    if any(value % 8 for value in work.l1_valid_weight_additions):
        raise AssertionError("L1 valid additions are not whole 8-channel vectors")
    if any(value % 16 for value in work.l2_valid_count_weight_macs):
        raise AssertionError("L2 valid MACs are not whole 16-channel vectors")
    l1_local_accumulator_reads = [
        valid_additions // 8 + 2 * (96 * 64) + 48 * 64
        for valid_additions in work.l1_valid_weight_additions
    ]
    l1_local_accumulator_writes = list(l1_local_accumulator_reads)
    l2_local_accumulator_reads = [
        valid_macs // 16 + 2 * (48 * 32) + 24 * 32
        for valid_macs in work.l2_valid_count_weight_macs
    ]
    l2_local_accumulator_writes = list(l2_local_accumulator_reads)

    common_time_major_state = {
        "external_state_elements_read": 8_136,
        "external_state_elements_written": 16_272,
        "external_state_payload_bits_read": 8_136 * 16,
        "external_state_payload_bits_written": 16_272 * 16,
        "external_vector_read_beats_mixed_128b_256b": external_state_vector_beats,
        "external_vector_write_beats_mixed_128b_256b": 2
        * external_state_vector_beats,
        "local_state_elements_read": 0,
        "local_state_elements_written": 0,
    }
    layer_major_state = {
        "external_state_elements_read": 0,
        "external_state_elements_written": 0,
        "external_state_payload_bits_read": 0,
        "external_state_payload_bits_written": 0,
        "external_vector_read_beats_mixed_128b_256b": 0,
        "external_vector_write_beats_mixed_128b_256b": 0,
        "local_state_elements_read": 8_136,
        "local_state_elements_written": 8_136,
        "note": (
            "t0 membrane is written once to engine-local storage and read once "
            "by t1; the final t1 membrane is discarded"
        ),
    }

    def activation_record(
        *,
        bram_reads: int,
        bram_writes: int,
        ingress_records: dict[str, int],
        sparse_packets: bool,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "input_ingress_records": ingress_records,
            "activation_BRAM_read_beats_mixed_36b_8b": _fixed_distribution(
                bram_reads, samples
            ),
            "activation_BRAM_write_beats_mixed_64b_128b_256b": (
                _fixed_distribution(bram_writes, samples)
            ),
        }
        if sparse_packets:
            result["sparse_packet_stream_reads"] = _distribution(packet_reads)
            result["L1_nonzero_packet_stream_writes"] = _distribution(
                work.l1_output_packets
            )
        else:
            result["sparse_packet_stream_reads"] = _fixed_distribution(0, samples)
            result["L1_nonzero_packet_stream_writes"] = _fixed_distribution(
                0, samples
            )
        return result

    dense_tm = {
        "weight": {
            "runtime_weight_BRAM_read_beats_256b": time_major_conv_weight_beats
            + time_major_linear_weight_beats,
            "front_local_weight_word_reads": _fixed_distribution(0, samples),
        },
        "activation": activation_record(
            bram_reads=dense_conv_activation_reads + time_activation_linear_reads,
            bram_writes=dense_activation_writes,
            ingress_records={"U9_count_frame_values": 2 * 96 * 128},
            sparse_packets=False,
        ),
        "state": common_time_major_state,
        "external_L3_current_cache": {
            "S24_elements_written": 4_608,
            "S24_elements_read": 4_608,
            "payload_bytes_written": 13_824,
            "payload_bytes_read": 13_824,
        },
    }
    dense_tm["activation"]["logical_transfer_breakdown"] = {
        "convolution_feature_reads": dense_conv_activation_reads,
        "linear_and_readout_activation_reads": time_activation_linear_reads,
        "convolution_feature_writes": dense_conv_activation_writes,
        "linear_feature_writes": 2
        * sum(linear_activation_writes_per_timestep.values()),
    }
    sparse_tm = {
        "weight": {
            "runtime_weight_BRAM_read_beats_256b": (
                sparse_time_major_conv_weight_beats
                + time_major_linear_weight_beats
            ),
            "front_configuration_writes_S16_once_per_model_load": 144 + 1_152,
            "L1_local_S16_weight_word_reads": _distribution(
                l1_local_weight_reads
            ),
            "L2_local_S16_weight_word_reads": _distribution(
                l2_local_weight_reads
            ),
        },
        "activation": activation_record(
            bram_reads=sparse_conv_activation_reads + time_activation_linear_reads,
            bram_writes=sparse_activation_writes,
            ingress_records={"unit_event_packets": 400},
            sparse_packets=True,
        ),
        "front_local_partial_sum_traffic": {
            "L1_8xS40_vector_reads": _distribution(
                l1_local_accumulator_reads
            ),
            "L1_8xS40_vector_writes": _distribution(
                l1_local_accumulator_writes
            ),
            "L2_16xS40_vector_reads": _distribution(
                l2_local_accumulator_reads
            ),
            "L2_16xS40_vector_writes": _distribution(
                l2_local_accumulator_writes
            ),
            "includes": "ring update, finalize read-and-clear, and horizontal-pool scratch",
        },
        "state": common_time_major_state,
        "external_L3_current_cache": dense_tm["external_L3_current_cache"],
        "seed2022_sample0_integrated_RTL_backend_counters": {
            key: SPARSE_TM_RTL_CALIBRATION[key]
            for key in (
                "convolution_weight_BRAM_read_beats_256b",
                "linear_and_readout_weight_memory_read_transactions",
                "external_state_vector_read_beats",
                "external_state_vector_write_beats",
            )
        },
    }
    sparse_tm["activation"]["logical_transfer_breakdown"] = {
        "L2_dense_vector_writes": 768,
        "deep_convolution_feature_reads": sparse_conv_activation_reads,
        "linear_and_readout_activation_reads": time_activation_linear_reads,
        "deep_convolution_feature_writes": deep_conv_activation_writes,
        "linear_feature_writes": 2
        * sum(linear_activation_writes_per_timestep.values()),
    }
    sparse_lm = {
        "weight": {
            "runtime_weight_BRAM_read_beats_256b": (
                sparse_layer_major_conv_weight_beats
                + layer_major_linear_weight_beats
            ),
            "front_configuration_writes_S16_once_per_model_load": 144 + 1_152,
            "L1_local_S16_weight_word_reads": _distribution(
                l1_local_weight_reads
            ),
            "L2_local_S16_weight_word_reads": _distribution(
                l2_local_weight_reads
            ),
        },
        "activation": activation_record(
            bram_reads=sparse_conv_activation_reads + time_activation_linear_reads,
            bram_writes=sparse_activation_writes,
            ingress_records={"unit_event_packets": 400},
            sparse_packets=True,
        ),
        "front_local_partial_sum_traffic": sparse_tm[
            "front_local_partial_sum_traffic"
        ],
        "state": layer_major_state,
        "external_L3_current_cache": {
            "S24_elements_written": 0,
            "S24_elements_read": 0,
            "payload_bytes_written": 0,
            "payload_bytes_read": 0,
        },
        "tile_local_L3_current_buffer": {
            "capacity_S24_elements": 192 * 8,
            "capacity_bytes": 192 * 8 * 3,
            "capacity_KiB": 4.5,
            "S24_elements_written_per_sample": 4_608,
            "S24_elements_read_per_sample": 4_608,
            "payload_bytes_written_per_sample": 13_824,
            "payload_bytes_read_per_sample": 13_824,
            "scope": (
                "one 192-position by 8-output-channel tile retained inside "
                "the paired L3 engine; no full-layer external current cache"
            ),
        },
        "seed2022_sample0_integrated_RTL_backend_counters": {
            key: SPARSE_LM_A_RTL_CALIBRATION[key]
            for key in (
                "convolution_weight_BRAM_read_beats_256b",
                "linear_weight_BRAM_read_beats_256b",
                "readout_weight_memory_read_transactions_256b_port_low32_used",
                "convolution_feature_read_requests",
                "convolution_feature_write_transactions",
            )
        },
    }
    sparse_lm["activation"]["logical_transfer_breakdown"] = dict(
        sparse_tm["activation"]["logical_transfer_breakdown"]
    )
    return {
        "units": {
            "weight_BRAM_beat": "256-bit memory transaction; convolution/H1/H2 use 16 S16 words, while readout uses only the low two S16 words",
            "conv_activation_read_beat": "four zero-extended activation channels (36-bit interface)",
            "conv_activation_write_beat": "eight U8 channels in old store; sparse L2 writes one 16xU8 word",
            "state": "element counts are comparable; vector beats mix 8xS16 conv and 16xS16 dense interfaces",
            "packet": "one addressed event/MultiSpike record; packet bits are kept separate from BRAM beats",
            "front_partial_sum_vectors": "L1 is 8xS40 and L2 is 16xS40; update accesses count only valid in-bounds taps, while finalize and pool scratch are exact fixed traffic; these distributed-RAM accesses are not added to BRAM beats",
        },
        "count_status": {
            "runtime_weight_BRAM_reads": "exact logical reads for the stated packed engines and schedule",
            "activation_BRAM_reads_and_writes": "exact logical cross-layer transfers for the stated engines and schedule",
            "state": "exact element accesses under the stated external-versus-engine-local state contract",
            "front_local_partial_sums": "exact in-bounds distributed-RAM vector accesses; invalid padding taps are excluded",
            "physical_memory_energy": "not inferred from logical traffic counts",
        },
        "per_layer_dense_shapes": layer,
        "Dense_TM": dense_tm,
        "Sparse_TM": sparse_tm,
        "Sparse_LM_A": sparse_lm,
    }


def _three_schedule_cycles(
    work: SampleWork,
    *,
    checkpoint_seed: int | None = None,
) -> dict[str, object]:
    samples = len(work.l1_unit_events)
    serialized_front = _front_cycle_distribution(
        work,
        l1_taps_per_cycle=9,
        l2_taps_per_cycle=3,
        l2_packets_per_cycle=1,
    )
    front_cycle_exact_values = list(work.sparse_frontend_cycle_exact)
    if len(front_cycle_exact_values) != samples:
        raise AssertionError("missing per-sample sparse frontend cycle replay")
    cycle_exact_front = {
        "frontend_cycles": _distribution(front_cycle_exact_values),
        "L1_wrapper_cycles": _distribution(
            work.sparse_frontend_l1_wrapper_cycles
        ),
        "L1_core_cycles": _distribution(work.sparse_frontend_l1_core_cycles),
        "L2_core_cycles": _distribution(work.sparse_frontend_l2_core_cycles),
        "sample_zero": {
            "unit_events": work.l1_unit_events[0],
            "L1_output_packets": work.l1_output_packets[0],
            "frontend_cycles": work.sparse_frontend_cycle_exact[0],
            "L1_wrapper_cycles": work.sparse_frontend_l1_wrapper_cycles[0],
            "L1_core_cycles": work.sparse_frontend_l1_core_cycles[0],
            "L2_core_cycles": work.sparse_frontend_l2_core_cycles[0],
        },
        "modeled": (
            "complete per-sample event rows and L1 nonzero-channel masks; "
            "cycle-by-cycle clear, row advance, update, finalize, drain, "
            "serialization, end flush, and L1-to-L2 backpressure"
        ),
    }
    calibration_exact = None
    if checkpoint_seed == SPARSE_LM_A_RTL_CALIBRATION["seed"]:
        calibration_exact = all(
            cycle_exact_front["sample_zero"][key]
            == SPARSE_FRONT_RTL_CALIBRATION[calibration_key]
            for key, calibration_key in (
                ("unit_events", "sample_unit_events"),
                ("L1_output_packets", "sample_L1_output_packets"),
                ("frontend_cycles", "integrated_frontend_measured_cycles"),
                ("L1_wrapper_cycles", "L1_wrapper_measured_cycles"),
                ("L1_core_cycles", "L1_core_measured_cycles"),
                ("L2_core_cycles", "L2_core_measured_cycles"),
            )
        )

    sparse_tm_analytic_backend = 56_452
    sparse_tm_rtl_calibrated_backend = SPARSE_TM_RTL_CALIBRATION[
        "fixed_backend_calibrated_cycles"
    ]
    sparse_lm_analytic_backend = int(dense_backend_schedule()["total_cycles"])
    sparse_lm_measured_component_sum = (
        PAIRED_L3_CURRENT_REUSE_RTL_CYCLES
        + sum(PAIRED_CONV_LAYER_RTL_CYCLES.values())
        + sum(PAIRED_LINEAR_LAYER_RTL_CYCLES.values())
        + READOUT_RTL_CYCLES
    )
    sparse_lm_rtl_calibrated_backend = SPARSE_LM_A_RTL_CALIBRATION[
        "fixed_backend_calibrated_cycles"
    ]

    rtl_route_values = _rtl_calibrated_route_cycle_values(work)
    dense_values = rtl_route_values["Dense_TM"]
    sparse_tm_theoretical = _plus_fixed(
        front_cycle_exact_values, sparse_tm_analytic_backend
    )
    sparse_tm_calibrated = rtl_route_values["Sparse_TM"]
    sparse_lm_theoretical = _plus_fixed(
        front_cycle_exact_values, sparse_lm_analytic_backend
    )
    sparse_lm_calibrated = rtl_route_values["Sparse_LM_A"]
    sparse_tm_sample0_model = int(sparse_tm_calibrated[0])
    sparse_lm_sample0_model = int(sparse_lm_calibrated[0])
    sparse_tm_sample0_match = None
    sparse_lm_sample0_match = None
    if checkpoint_seed == SPARSE_TM_RTL_CALIBRATION["seed"]:
        sparse_tm_sample0_match = sparse_tm_sample0_model == (
            SPARSE_TM_RTL_CALIBRATION["whole_network_measured_cycles"]
        )
        sparse_lm_sample0_match = sparse_lm_sample0_model == (
            SPARSE_LM_A_RTL_CALIBRATION["whole_network_measured_cycles"]
        )

    return {
        "front_end": {
            "serialized_idealized_stage_sum": serialized_front,
            "serialized_idealized_stage_sum_status": (
                "analytical comparison only: excludes controller fill/drain "
                "but also excludes L1/L2 overlap, so it is neither an exact "
                "cycle count nor a guaranteed upper/lower bound"
            ),
            "cycle_exact_RTL_control_model": cycle_exact_front,
            "integrated_RTL_calibration": SPARSE_FRONT_RTL_CALIBRATION,
            "seed2022_sample_zero_calibration_exact_match": calibration_exact,
            "calibration_limit": (
                "the software FSM replay matches one integrated RTL sample "
                "exactly; the three full validation distributions use the "
                "same complete per-sample packet-dependent control replay"
            ),
        },
        "Dense_TM": {
            "organization": "old dense-raster SL-HYB, time-major deep execution",
            "analytical_cycles_pre_controller": 82_543,
            "rtl_measured_whole_network": _fixed_distribution(
                OLD_DENSE_SL_HYB_RTL_CYCLES, samples
            ),
            "rtl_measured_components": OLD_DENSE_TOP_RTL_COMPONENTS,
            "comparison_distribution": _cycle_distribution_with_200mhz(
                dense_values
            ),
            "cycle_status": "whole-network RTL measured; data-independent dense schedule",
        },
        "Sparse_TM": {
            "organization": (
                "sparse L1/L2, old time-major deep order with L3-current cache"
            ),
            "theoretical_distribution": _cycle_distribution_with_200mhz(
                sparse_tm_theoretical
            ),
            "rtl_calibrated_distribution": _cycle_distribution_with_200mhz(
                sparse_tm_calibrated
            ),
            "rtl_calibrated_fixed_backend_cycles": (
                sparse_tm_rtl_calibrated_backend
            ),
            "seed2022_sample0_integrated_RTL_calibration": (
                SPARSE_TM_RTL_CALIBRATION
            ),
            "sample_zero_cycle_model": sparse_tm_sample0_model,
            "seed2022_sample_zero_integrated_RTL_exact_match": (
                sparse_tm_sample0_match
            ),
            "rtl_measured_whole_validation_distribution": None,
            "cycle_status": (
                "per-sample frontend is cycle-exact control replay; fixed "
                "backend is calibrated from integrated RTL, making seed2022 "
                "sample0 exact; full validation distributions are modeled"
            ),
        },
        "Sparse_LM_A": {
            "organization": (
                "sparse L1/L2, L3 current once, layer-adjacent t0/t1 for "
                "L4-L8 and H1/H2"
            ),
            "theoretical_distribution": _cycle_distribution_with_200mhz(
                sparse_lm_theoretical
            ),
            "rtl_calibrated_distribution": _cycle_distribution_with_200mhz(
                sparse_lm_calibrated
            ),
            "rtl_calibrated_fixed_backend_cycles": (
                sparse_lm_rtl_calibrated_backend
            ),
            "rtl_measured_fixed_backend_component_sum_cycles": (
                sparse_lm_measured_component_sum
            ),
            "rtl_measured_L3_current_reuse_cycles": (
                PAIRED_L3_CURRENT_REUSE_RTL_CYCLES
            ),
            "rtl_measured_paired_layer_cycles": PAIRED_CONV_LAYER_RTL_CYCLES,
            "rtl_measured_paired_linear_cycles": PAIRED_LINEAR_LAYER_RTL_CYCLES,
            "rtl_measured_readout_cycles": READOUT_RTL_CYCLES,
            "integrated_boundary_adjustment_vs_component_sum_cycles": (
                sparse_lm_rtl_calibrated_backend
                - sparse_lm_measured_component_sum
            ),
            "seed2022_sample0_integrated_RTL_calibration": (
                SPARSE_LM_A_RTL_CALIBRATION
            ),
            "sample_zero_cycle_model": sparse_lm_sample0_model,
            "seed2022_sample_zero_integrated_RTL_exact_match": (
                sparse_lm_sample0_match
            ),
            "rtl_measured_whole_validation_distribution": None,
            "cycle_status": (
                "per-sample frontend is cycle-exact control replay; fixed "
                "backend is calibrated from integrated RTL, making seed2022 "
                "sample0 exact; full validation distributions are modeled"
            ),
        },
        "seed2022_sample0_exact_RTL_comparison": {
            "Dense_TM_cycles": OLD_DENSE_SL_HYB_RTL_CYCLES,
            "Sparse_TM_cycles": SPARSE_TM_RTL_CALIBRATION[
                "whole_network_measured_cycles"
            ],
            "Sparse_LM_A_cycles": SPARSE_LM_A_RTL_CALIBRATION[
                "whole_network_measured_cycles"
            ],
            "Sparse_LM_A_cycle_reduction_fraction": (
                1
                - SPARSE_LM_A_RTL_CALIBRATION["whole_network_measured_cycles"]
                / OLD_DENSE_SL_HYB_RTL_CYCLES
            ),
            "Sparse_TM_cycle_reduction_fraction": (
                1
                - SPARSE_TM_RTL_CALIBRATION["whole_network_measured_cycles"]
                / OLD_DENSE_SL_HYB_RTL_CYCLES
            ),
            "conditional_latency_ms_at_200MHz": {
                "Dense_TM": OLD_DENSE_SL_HYB_RTL_CYCLES / 200_000,
                "Sparse_TM": SPARSE_TM_RTL_CALIBRATION[
                    "whole_network_measured_cycles"
                ]
                / 200_000,
                "Sparse_LM_A": SPARSE_LM_A_RTL_CALIBRATION[
                    "whole_network_measured_cycles"
                ]
                / 200_000,
            },
            "conditional_throughput_samples_per_second_at_200MHz": {
                "Dense_TM": 200_000_000 / OLD_DENSE_SL_HYB_RTL_CYCLES,
                "Sparse_TM": 200_000_000
                / SPARSE_TM_RTL_CALIBRATION["whole_network_measured_cycles"],
                "Sparse_LM_A": 200_000_000
                / SPARSE_LM_A_RTL_CALIBRATION["whole_network_measured_cycles"],
            },
        },
        "conditional_200MHz_limit": (
            "latency and throughput divide cycle counts by 200 MHz and assume "
            "sequential samples without inter-sample overlap; Vivado timing "
            "closure and FPGA measurement are still pending"
        ),
        "status": (
            "All three routes are whole-network RTL measured on seed2022 "
            "sample0. Sparse full-validation distributions combine cycle-exact "
            "frontend replay with their measured fixed backend; only sample0, "
            "not every validation sample, was replayed through whole RTL."
        ),
    }


def _three_schedule_comparison(
    work: SampleWork,
    *,
    checkpoint_seed: int | None = None,
) -> dict[str, object]:
    operation_breakdown = _operation_report(work)
    sparse_operations = operation_breakdown["mixed_synaptic_arithmetic_total"]
    return {
        "definitions": {
            "Dense_TM": "old dense-raster SL-HYB reference",
            "Sparse_TM": "selected sparse front with old time-major deep order",
            "Sparse_LM_A": "selected sparse front with layer-adjacent t0/t1 deep order",
        },
        "synaptic_arithmetic_per_sample": {
            "Dense_TM": {
                "dense_MACs": _fixed_distribution(
                    OLD_DENSE_SL_HYB_MACS, len(work.l1_unit_events)
                )
            },
            "Sparse_TM": {
                "mixed_L1_adds_and_L2_deep_MACs": sparse_operations
            },
            "Sparse_LM_A": {
                "mixed_L1_adds_and_L2_deep_MACs": sparse_operations
            },
            "limit": "signed additions and MACs are counted, not equated in energy",
        },
        "selected_sparse_operation_breakdown": operation_breakdown,
        "operation_count_status": (
            "L1/L2 are exact valid in-bounds synapses from each sample; Dense-TM "
            "and L3-readout use the conventional network MAC definition. Fixed-"
            "width hardware lane issues are reported separately"
        ),
        "scheduled_lane_operations": _scheduled_lane_operation_report(work),
        "nominal_compute_resources_before_vivado": {
            "Dense_TM": {
                "shared_convolution_DSP": 288,
                "dense_DSP": 16,
                "readout_DSP": 2,
                "total_DSP": 306,
            },
            "Sparse_TM": {
                "L1_add_lanes_in_LUT_fabric": 72,
                "separate_L2_DSP": 48,
                "shared_deep_convolution_DSP": 288,
                "dense_DSP": 16,
                "readout_DSP": 2,
                "total_DSP": 354,
            },
            "Sparse_LM_A": {
                "L1_add_lanes_in_LUT_fabric": 72,
                "separate_L2_DSP": 48,
                "shared_deep_convolution_DSP": 288,
                "paired_dense_DSP": 16,
                "readout_DSP": 2,
                "total_DSP": 354,
            },
            "AUP_ZU3_DSP48E2_capacity": 360,
            "limit": "nominal lane allocation; Vivado placement, timing, and inferred primitive counts remain unmeasured",
        },
        "runtime_memory_traffic_per_sample": _schedule_memory_traffic(work),
        "cycles": _three_schedule_cycles(work, checkpoint_seed=checkpoint_seed),
    }


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    configuration_path: Path,
    event_frames: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    expected_stateless = {"L1": list(range(8)), "L2": list(range(16))}
    if configuration["model"]["stateless_channels"] != expected_stateless:
        raise ValueError("selected hybrid route requires fully stateless L1/L2")
    if int(configuration["input"]["temporal_steps"]) != 2:
        raise ValueError("selected hybrid route requires T=2")
    if configuration["input"]["temporal_organization"] != (
        "repeat_identical_complete_400_event_count_frame"
    ):
        raise ValueError("selected hybrid route requires repeated complete input")

    model, checkpoint = load_model(checkpoint_path, configuration)
    dense_reference = FixedPointPursuitReference(
        model,
        **FROZEN_FIXED_POINT_ARGUMENTS,
    )
    hybrid_reference = SparseHybridReference(dense_reference)
    dense_counters = RuntimeCounters.empty()
    hybrid_counters = RuntimeCounters.empty()
    work = SampleWork()
    comparison = TraceComparison()
    pixel_total = 0.0
    angle_total = 0.0
    defaults: dict[str, object] | None = None
    started = time.monotonic()

    for start in range(0, len(event_frames), batch_size):
        stop = min(start + batch_size, len(event_frames))
        frames = torch.from_numpy(event_frames[start:stop]).to(torch.int64)
        target = torch.from_numpy(labels[start:stop]).to(torch.float32)
        expected_trace: dict[str, np.ndarray] = {}
        actual_trace: dict[str, np.ndarray] = {}
        expected_prediction = dense_reference.predict(
            frames.to(torch.float32), dense_counters, trace=expected_trace
        )
        actual_prediction, batch_defaults = hybrid_reference.predict(
            frames,
            hybrid_counters,
            work,
            trace=actual_trace,
        )
        comparison.update(expected_trace, actual_trace)
        if not torch.equal(expected_prediction, actual_prediction):
            mismatch = int(torch.count_nonzero(expected_prediction != actual_prediction))
            raise AssertionError(f"fixed predictions differ at {mismatch} coordinates")
        if defaults is None:
            defaults = batch_defaults
        elif defaults != batch_defaults:
            raise AssertionError("per-channel default plane changed between batches")
        pixel_total += float(pixel_error(actual_prediction, target).sum())
        angle_total += float(angular_error_deg(actual_prediction, target).sum())

    if comparison.mismatches:
        raise AssertionError(f"hybrid trace has {comparison.mismatches} mismatches")
    samples = len(event_frames)
    if len(work.l1_unit_events) != samples:
        raise AssertionError("work-account sample count mismatch")
    if any(value != 400 for value in work.l1_unit_events):
        raise AssertionError("L1 event stream is not exactly 400 unit events")

    return {
        "seed": int(configuration["training"]["seed"]),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
        },
        "configuration": {
            "path": str(configuration_path),
            "sha256": sha256_file(configuration_path),
        },
        "samples": samples,
        "bit_exact_equivalence": {
            "dense_fixed_reference_values_checked": comparison.values_checked,
            "mismatches": comparison.mismatches,
            "per_tensor_mismatches": comparison.per_tensor_mismatches,
            "prediction_exact_match": True,
            "pool_before_constant_identity_checked": True,
            "L1_L2_sparse_accumulation_checked": True,
            "L3_current_reuse_checked": True,
        },
        "fixed_point_metrics": {
            "mean_pixel_error": pixel_total / samples,
            "mean_angular_error_deg": angle_total / samples,
        },
        "default_planes": defaults,
        "activity_and_sparse_work": _work_report(work),
        "operation_counts_per_sample": _operation_report(work),
        "three_schedule_comparison": _three_schedule_comparison(
            work,
            checkpoint_seed=int(configuration["training"]["seed"]),
        ),
        "analytical_layer_adjacent_backend_details": dense_backend_schedule(),
        "_internal_rtl_calibrated_route_cycle_values": (
            _rtl_calibrated_route_cycle_values(work)
        ),
        "_internal_scheduled_lane_operation_values": (
            _scheduled_lane_operation_values(work)
        ),
        "runtime_counters_for_executed_hybrid_schedule": {
            "accumulator_saturated_values": hybrid_counters.accumulator_saturated,
            "spike_saturated_values": hybrid_counters.spike_saturated,
            "accumulator_maximum_absolute_before_saturation": (
                hybrid_counters.accumulator_maximum_absolute
            ),
            "spike_maximum_before_saturation": hybrid_counters.spike_maximum,
        },
        "runtime_seconds": time.monotonic() - started,
    }


def _aggregate_seed_field(
    reports: list[dict[str, object]],
    path: tuple[str, ...],
) -> dict[str, object]:
    values = []
    for report in reports:
        current: object = report
        for key in path:
            if not isinstance(current, dict):
                raise KeyError(path)
            current = current[key]
        values.append(float(current))
    return {
        "per_seed": [
            {"seed": int(report["seed"]), "value": value}
            for report, value in zip(reports, values)
        ],
        "mean": statistics.fmean(values),
        "sample_standard_deviation": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    cache_path = args.cache_dir / f"{args.split}.npz"
    with np.load(cache_path) as cached:
        event_frames = cached["event_frames"].copy()
        labels = cached["labels"].copy()
    if event_frames.ndim != 4 or tuple(event_frames.shape[1:]) != (2, 96, 128):
        raise ValueError(f"unexpected event frame shape: {event_frames.shape}")
    if labels.shape != (len(event_frames), 2):
        raise ValueError(f"unexpected label shape: {labels.shape}")
    if args.sample_limit is not None:
        event_frames = event_frames[: args.sample_limit]
        labels = labels[: args.sample_limit]
    if len(event_frames) == 0:
        raise ValueError("selected split is empty")
    if not np.all(event_frames.sum(axis=(1, 2, 3), dtype=np.uint64) == 400):
        raise ValueError("every selected sample must contain exactly 400 events")

    reports = []
    for checkpoint in args.checkpoint:
        configuration = checkpoint.parent / "configuration.json"
        reports.append(
            evaluate_checkpoint(
                checkpoint,
                configuration,
                event_frames,
                labels,
                args.batch_size,
            )
        )
    reports.sort(key=lambda item: int(item["seed"]))

    pooled_route_cycles: dict[str, list[float]] = {
        "Dense_TM": [],
        "Sparse_TM": [],
        "Sparse_LM_A": [],
    }
    pooled_scheduled_lane_operations: dict[str, list[float]] = {
        "Dense_TM": [],
        "Sparse_TM": [],
        "Sparse_LM_A": [],
    }
    for report in reports:
        internal_values = report.pop(
            "_internal_rtl_calibrated_route_cycle_values"
        )
        if not isinstance(internal_values, dict):
            raise TypeError("missing internal route cycle values")
        for route in pooled_route_cycles:
            values = internal_values[route]
            if not isinstance(values, list):
                raise TypeError(f"{route}: internal cycles are not a list")
            pooled_route_cycles[route].extend(float(value) for value in values)
        scheduled_values = report.pop(
            "_internal_scheduled_lane_operation_values"
        )
        if not isinstance(scheduled_values, dict):
            raise TypeError("missing internal scheduled-lane values")
        for route in pooled_scheduled_lane_operations:
            values = scheduled_values[route]
            if not isinstance(values, list):
                raise TypeError(
                    f"{route}: internal scheduled-lane values are not a list"
                )
            pooled_scheduled_lane_operations[route].extend(
                float(value) for value in values
            )
    expected_pooled_samples = len(event_frames) * len(reports)
    if any(
        len(values) != expected_pooled_samples
        for values in pooled_route_cycles.values()
    ):
        raise AssertionError("pooled route cycle sample count mismatch")
    if any(
        len(values) != expected_pooled_samples
        for values in pooled_scheduled_lane_operations.values()
    ):
        raise AssertionError("pooled scheduled-lane sample count mismatch")

    aggregate_paths = {
        "fixed_mean_pixel_error": (
            "fixed_point_metrics",
            "mean_pixel_error",
        ),
        "fixed_mean_angular_error_deg": (
            "fixed_point_metrics",
            "mean_angular_error_deg",
        ),
        "L1_valid_weight_additions": (
            "activity_and_sparse_work",
            "l1_valid_weight_additions",
            "mean",
        ),
        "L1_output_packets": (
            "activity_and_sparse_work",
            "l1_output_packets",
            "mean",
        ),
        "L2_valid_count_weight_macs": (
            "activity_and_sparse_work",
            "l2_valid_count_weight_macs",
            "mean",
        ),
        "mixed_synaptic_arithmetic_total": (
            "operation_counts_per_sample",
            "mixed_synaptic_arithmetic_total",
            "mean",
        ),
        "Dense_TM_rtl_measured_cycles": (
            "three_schedule_comparison",
            "cycles",
            "Dense_TM",
            "comparison_distribution",
            "mean",
        ),
        "Sparse_TM_theoretical_cycles": (
            "three_schedule_comparison",
            "cycles",
            "Sparse_TM",
            "theoretical_distribution",
            "mean",
        ),
        "Sparse_TM_rtl_calibrated_cycles": (
            "three_schedule_comparison",
            "cycles",
            "Sparse_TM",
            "rtl_calibrated_distribution",
            "mean",
        ),
        "Sparse_LM_A_theoretical_cycles": (
            "three_schedule_comparison",
            "cycles",
            "Sparse_LM_A",
            "theoretical_distribution",
            "mean",
        ),
        "Sparse_LM_A_rtl_calibrated_cycles": (
            "three_schedule_comparison",
            "cycles",
            "Sparse_LM_A",
            "rtl_calibrated_distribution",
            "mean",
        ),
        "Dense_TM_scheduled_multiplier_lane_operations": (
            "three_schedule_comparison",
            "scheduled_lane_operations",
            "Dense_TM",
            "total_scheduled_multiplier_lane_operations",
            "mean",
        ),
        "Sparse_TM_scheduled_mixed_lane_operations": (
            "three_schedule_comparison",
            "scheduled_lane_operations",
            "Sparse_TM",
            "total_scheduled_add_and_multiplier_lane_operations",
            "mean",
        ),
        "Sparse_LM_A_scheduled_mixed_lane_operations": (
            "three_schedule_comparison",
            "scheduled_lane_operations",
            "Sparse_LM_A",
            "total_scheduled_add_and_multiplier_lane_operations",
            "mean",
        ),
    }
    script = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "bit_exact_sparse_L1_L2_and_layer_adjacent_dense_backend_validation",
        "analysis_code": {"path": str(script), "sha256": sha256_file(script)},
        "data": {
            "cache_file": str(cache_path),
            "cache_file_sha256": sha256_file(cache_path),
            "split": args.split,
            "samples": len(event_frames),
            "sample_limit": args.sample_limit,
            "batch_size": args.batch_size,
            "input_scope": "cached physical windows without online recombination",
        },
        "fixed_point_contract": FIXED_POINT_REPORT_CONTRACT,
        "selected_schedule": [
            "L1: 400 sorted unit events, add-only S16 weight scatter into S40",
            "L2: nonzero L1 U8 packets, one count-times-S16-weight MAC per synapse",
            "L2: finalize complete output plane so folded-constant defaults are preserved",
            "L3: one dense convolution/current followed by cascaded t0/t1 LIF",
            "L4-L8: shared 288-lane core, layer-adjacent t0/t1 with one weight-tile load",
            "H1/H2: layer-adjacent t0/t1; readout consumes their temporal sum",
        ],
        "per_checkpoint": reports,
        "aggregate_across_checkpoint_seeds": {
            name: _aggregate_seed_field(reports, path)
            for name, path in aggregate_paths.items()
        },
        "pooled_rtl_calibrated_cycles_across_all_seed_samples": {
            "scope": (
                "true pooled distribution over every (checkpoint seed, "
                "validation sample) pair; percentiles are computed from all "
                f"{expected_pooled_samples} cycle values"
            ),
            "routes": {
                route: _cycle_distribution_with_200mhz(values)
                for route, values in pooled_route_cycles.items()
            },
        },
        "pooled_scheduled_lane_operations_across_all_seed_samples": {
            "scope": (
                "true pooled distribution over every (checkpoint seed, "
                "validation sample) pair; fixed-width issued arithmetic lanes, "
                "not valid synapses or switching activity"
            ),
            "routes": {
                route: _distribution(values)
                for route, values in pooled_scheduled_lane_operations.items()
            },
            "Sparse_TM_mean_reduction_vs_Dense_TM_fraction": (
                1
                - statistics.fmean(pooled_scheduled_lane_operations["Sparse_TM"])
                / statistics.fmean(pooled_scheduled_lane_operations["Dense_TM"])
            ),
            "Sparse_LM_A_mean_reduction_vs_Dense_TM_fraction": (
                1
                - statistics.fmean(
                    pooled_scheduled_lane_operations["Sparse_LM_A"]
                )
                / statistics.fmean(pooled_scheduled_lane_operations["Dense_TM"])
            ),
        },
        "evidence_boundary": [
            "bit-exactness is established against the frozen CPU fixed-point reference",
            (
                "Dense-TM, Sparse-TM, and Sparse-LM-A are whole-network RTL "
                "measured for seed2022 validation sample0; both sparse sample0 "
                "cycle models match those integrated RTL measurements exactly"
            ),
            (
                "the pooled sparse full-validation cycle distributions use an "
                "exact per-sample frontend ready/valid FSM replay plus the fixed "
                "backend cycle count calibrated by integrated RTL; they are not "
                "whole-network RTL measurements for all 1278 seed/sample pairs"
            ),
            "200 MHz latency is conditional on Vivado timing closure",
            "mixed adds and MACs are not converted to energy",
            (
                "scheduled lane operations count issued fixed-width arithmetic "
                "lanes, not data-dependent switching activity, power, or energy"
            ),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--sample-limit",
        type=int,
        help="optional smoke-only prefix; omit for a formal complete-split report",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.sample_limit is not None and args.sample_limit <= 0:
        raise ValueError("sample limit must be positive")
    report = build_report(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
