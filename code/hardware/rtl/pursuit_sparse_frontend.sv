`timescale 1ns/1ps
`default_nettype none

// Integrated pursuit sparse front-end shared by Sparse-TM and Sparse-LM-A.
// L1 directly consumes 400 sorted unit events, emits nonzero MultiSpike
// packets, and L2 consumes those packets with backpressure.  L2 materializes
// the complete 16x24x32 tensor in a repeatedly readable dense RAM.
module pursuit_sparse_frontend #(
    parameter integer WEIGHT_BITS = 16,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer FRACTION_BITS_WIDTH = 5
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output wire                                      busy,
    output wire                                      done,

    input  wire                                      event_valid,
    output wire                                      event_ready,
    input  wire                                      event_polarity,
    input  wire [6:0]                               event_row,
    input  wire [6:0]                               event_column,
    input  wire                                      events_end,

    input  wire                                      l1_weight_write_enable,
    input  wire [2:0]                               l1_weight_write_output_channel,
    input  wire                                      l1_weight_write_input_channel,
    input  wire [3:0]                               l1_weight_write_kernel_index,
    input  wire signed [WEIGHT_BITS-1:0]            l1_weight_write_data,
    input  wire                                      l1_channel_write_enable,
    input  wire [2:0]                               l1_channel_write_channel,
    input  wire signed [CURRENT_BITS-1:0]           l1_constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]           l1_fraction_bits_write_data,

    input  wire                                      l2_weight_write_enable,
    input  wire [3:0]                               l2_weight_write_output_channel,
    input  wire [2:0]                               l2_weight_write_input_channel,
    input  wire [3:0]                               l2_weight_write_kernel_index,
    input  wire signed [WEIGHT_BITS-1:0]            l2_weight_write_data,
    input  wire                                      l2_channel_write_enable,
    input  wire [3:0]                               l2_channel_write_channel,
    input  wire signed [CURRENT_BITS-1:0]           l2_constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]           l2_fraction_bits_write_data,

    input  wire                                      dense_read_enable,
    input  wire [9:0]                               dense_read_address,
    input  wire [3:0]                               dense_read_channel_base,
    output wire                                      dense_read_valid,
    output wire [31:0]                              dense_read_data_flat,

    output wire                                      l1_trace_valid,
    output wire [11:0]                              l1_trace_address,
    output wire [8*CURRENT_BITS-1:0]                l1_trace_current_flat,
    output wire [8*SPIKE_BITS-1:0]                  l1_trace_spike_flat,
    output wire [8*STATE_BITS-1:0]                  l1_trace_membrane_flat,
    output wire                                      l1_trace_range_error,
    output wire                                      l2_trace_valid,
    output wire [9:0]                               l2_trace_address,
    output wire [16*CURRENT_BITS-1:0]               l2_trace_current_flat,
    output wire [16*SPIKE_BITS-1:0]                 l2_trace_spike_flat,
    output wire [16*STATE_BITS-1:0]                 l2_trace_membrane_flat,
    output wire                                      l2_trace_range_error,

    // Monitor of the exact stream transferred between the layers.
    output wire                                      l1_packet_valid,
    output wire                                      l1_packet_ready,
    output wire [11:0]                              l1_packet_address,
    output wire [2:0]                               l1_packet_channel,
    output wire [SPIKE_BITS-1:0]                    l1_packet_count,

    output wire                                      l1_protocol_error,
    output wire                                      l2_protocol_error,
    output wire                                      protocol_error,
    output reg  [31:0]                              cycle_count,
    output wire [31:0]                              l1_cycle_count,
    output wire [31:0]                              l1_core_cycle_count,
    output wire [31:0]                              l2_cycle_count,
    output wire [31:0]                              accepted_event_count,
    output wire [31:0]                              l1_emitted_packet_count,
    output wire [31:0]                              l2_accepted_packet_count,
    output wire [31:0]                              l2_dense_write_count
);
    wire l1_busy;
    wire l1_done;
    wire l1_sparse_end;
    wire l2_busy;
    wire l2_done;

    pursuit_sparse_l1_event_layer #(
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .CURRENT_BITS(CURRENT_BITS),
        .STATE_BITS(STATE_BITS),
        .SPIKE_BITS(SPIKE_BITS),
        .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
    ) l1 (
        .clk(clk), .rst(rst), .start(start),
        .busy(l1_busy), .done(l1_done),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_polarity(event_polarity), .event_row(event_row),
        .event_column(event_column), .events_end(events_end),
        .weight_write_enable(l1_weight_write_enable),
        .weight_write_output_channel(l1_weight_write_output_channel),
        .weight_write_input_channel(l1_weight_write_input_channel),
        .weight_write_kernel_index(l1_weight_write_kernel_index),
        .weight_write_data(l1_weight_write_data),
        .channel_write_enable(l1_channel_write_enable),
        .channel_write_channel(l1_channel_write_channel),
        .constant_write_data(l1_constant_write_data),
        .fraction_bits_write_data(l1_fraction_bits_write_data),
        .sparse_valid(l1_packet_valid), .sparse_ready(l1_packet_ready),
        .sparse_address(l1_packet_address),
        .sparse_channel(l1_packet_channel),
        .sparse_count(l1_packet_count), .sparse_end(l1_sparse_end),
        .trace_valid(l1_trace_valid), .trace_address(l1_trace_address),
        .trace_current_flat(l1_trace_current_flat),
        .trace_spike_flat(l1_trace_spike_flat),
        .trace_membrane_flat(l1_trace_membrane_flat),
        .trace_range_error(l1_trace_range_error),
        .protocol_error(l1_protocol_error),
        .cycle_count(l1_cycle_count),
        .core_cycle_count(l1_core_cycle_count),
        .accepted_event_count(accepted_event_count),
        .emitted_packet_count(l1_emitted_packet_count)
    );

    pursuit_sparse_l2_packet_layer #(
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .CURRENT_BITS(CURRENT_BITS),
        .STATE_BITS(STATE_BITS),
        .SPIKE_BITS(SPIKE_BITS),
        .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH),
        .READ_LANES(4)
    ) l2 (
        .clk(clk), .rst(rst), .start(start),
        .busy(l2_busy), .done(l2_done),
        .sparse_valid(l1_packet_valid),
        .sparse_ready(l1_packet_ready),
        .sparse_address(l1_packet_address),
        .sparse_channel(l1_packet_channel),
        .sparse_count(l1_packet_count),
        .sparse_end(l1_sparse_end),
        .weight_write_enable(l2_weight_write_enable),
        .weight_write_output_channel(l2_weight_write_output_channel),
        .weight_write_input_channel(l2_weight_write_input_channel),
        .weight_write_kernel_index(l2_weight_write_kernel_index),
        .weight_write_data(l2_weight_write_data),
        .channel_write_enable(l2_channel_write_enable),
        .channel_write_channel(l2_channel_write_channel),
        .constant_write_data(l2_constant_write_data),
        .fraction_bits_write_data(l2_fraction_bits_write_data),
        .dense_read_enable(dense_read_enable),
        .dense_read_address(dense_read_address),
        .dense_read_channel_base(dense_read_channel_base),
        .dense_read_valid(dense_read_valid),
        .dense_read_data_flat(dense_read_data_flat),
        .trace_valid(l2_trace_valid), .trace_address(l2_trace_address),
        .trace_current_flat(l2_trace_current_flat),
        .trace_spike_flat(l2_trace_spike_flat),
        .trace_membrane_flat(l2_trace_membrane_flat),
        .trace_range_error(l2_trace_range_error),
        .protocol_error(l2_protocol_error),
        .cycle_count(l2_cycle_count),
        .accepted_packet_count(l2_accepted_packet_count),
        .dense_write_count(l2_dense_write_count)
    );

    assign busy = l1_busy || l2_busy || (l1_done && !l2_done);
    assign done = l2_done;
    assign protocol_error = l1_protocol_error || l2_protocol_error;

    always @(posedge clk) begin
        if (rst) begin
            cycle_count <= 32'd0;
        end else if (start && !busy) begin
            cycle_count <= 32'd0;
        end else if (busy) begin
            cycle_count <= cycle_count + 1'b1;
        end
    end
endmodule

`default_nettype wire
