`timescale 1ns/1ps
`default_nettype none

// Pursuit L1 sparse front-end.
//
// The 400 input packets are unit events after the model's 2x2 spatial
// SumPool.  They must be sorted by nondecreasing pooled row; sorting does not
// change the frozen model because its L1 input is the accumulated count image.
// All nine kernel taps and all eight output channels are evaluated in one
// update cycle (72 signed additions, no multiplier).  Pool/LIF results are
// serialized into nonzero (address, channel, U8 count) packets for L2.
module pursuit_sparse_l1_event_layer #(
    parameter integer INPUT_WIDTH = 128,
    parameter integer INPUT_HEIGHT = 96,
    parameter integer INPUT_CHANNELS = 2,
    parameter integer OUTPUT_CHANNELS = 8,
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
    output reg                                       busy,
    output reg                                       done,

    input  wire                                      event_valid,
    output wire                                      event_ready,
    input  wire                                      event_polarity,
    input  wire [$clog2(INPUT_HEIGHT)-1:0]           event_row,
    input  wire [$clog2(INPUT_WIDTH)-1:0]            event_column,
    input  wire                                      events_end,

    input  wire                                      weight_write_enable,
    input  wire [$clog2(OUTPUT_CHANNELS)-1:0]
                                                    weight_write_output_channel,
    input  wire                                      weight_write_input_channel,
    input  wire [3:0]                               weight_write_kernel_index,
    input  wire signed [WEIGHT_BITS-1:0]            weight_write_data,
    input  wire                                      channel_write_enable,
    input  wire [$clog2(OUTPUT_CHANNELS)-1:0]
                                                    channel_write_channel,
    input  wire signed [CURRENT_BITS-1:0]           constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]           fraction_bits_write_data,

    output wire                                      sparse_valid,
    input  wire                                      sparse_ready,
    output wire [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    sparse_address,
    output wire [$clog2(OUTPUT_CHANNELS)-1:0]        sparse_channel,
    output wire [SPIKE_BITS-1:0]                    sparse_count,
    output wire                                      sparse_end,

    // One-cycle pulse for every dense pool position, before zero filtering.
    output reg                                       trace_valid,
    output reg  [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    trace_address,
    output reg  [OUTPUT_CHANNELS*CURRENT_BITS-1:0]  trace_current_flat,
    output reg  [OUTPUT_CHANNELS*SPIKE_BITS-1:0]    trace_spike_flat,
    output reg  [OUTPUT_CHANNELS*STATE_BITS-1:0]    trace_membrane_flat,
    output reg                                       trace_range_error,

    output wire                                      protocol_error,
    output reg  [31:0]                              cycle_count,
    output wire [31:0]                              core_cycle_count,
    output wire [31:0]                              accepted_event_count,
    output reg  [31:0]                              emitted_packet_count
);
    localparam integer OUTPUT_PIXELS =
        (INPUT_WIDTH/2)*(INPUT_HEIGHT/2);
    localparam integer OUTPUT_ADDRESS_BITS = $clog2(OUTPUT_PIXELS);
    localparam integer OUTPUT_CHANNEL_BITS = $clog2(OUTPUT_CHANNELS);

    wire core_busy;
    wire core_done;
    wire core_output_valid;
    wire core_output_ready;
    wire [OUTPUT_ADDRESS_BITS-1:0] core_output_address;
    wire [OUTPUT_CHANNELS*CURRENT_BITS-1:0]
        core_output_current_flat;
    wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0]
        core_output_spike_flat;
    wire [OUTPUT_CHANNELS*STATE_BITS-1:0]
        core_output_membrane_flat;
    wire core_output_range_error;

    sparse_conv3x3_pool_lif_core #(
        .INPUT_CHANNELS(INPUT_CHANNELS),
        .OUTPUT_CHANNELS(OUTPUT_CHANNELS),
        .INPUT_WIDTH(INPUT_WIDTH),
        .INPUT_HEIGHT(INPUT_HEIGHT),
        .COUNT_BITS(1),
        .UNIT_INPUT(1),
        .KERNEL_TAPS_PER_CYCLE(9),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .CURRENT_BITS(CURRENT_BITS),
        .STATE_BITS(STATE_BITS),
        .SPIKE_BITS(SPIKE_BITS),
        .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
    ) core (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(core_busy),
        .done(core_done),
        .input_valid(event_valid),
        .input_ready(event_ready),
        .input_channel(event_polarity),
        .input_row(event_row),
        .input_column(event_column),
        .input_count(1'b1),
        .input_end(events_end),
        .weight_write_enable(weight_write_enable),
        .weight_write_output_channel(weight_write_output_channel),
        .weight_write_input_channel(weight_write_input_channel),
        .weight_write_kernel_index(weight_write_kernel_index),
        .weight_write_data(weight_write_data),
        .channel_write_enable(channel_write_enable),
        .channel_write_channel(channel_write_channel),
        .constant_write_data(constant_write_data),
        .fraction_bits_write_data(fraction_bits_write_data),
        .output_valid(core_output_valid),
        .output_ready(core_output_ready),
        .output_address(core_output_address),
        .output_current_flat(core_output_current_flat),
        .output_spike_flat(core_output_spike_flat),
        .output_membrane_flat(core_output_membrane_flat),
        .output_range_error(core_output_range_error),
        .protocol_error(protocol_error),
        .cycle_count(core_cycle_count),
        .accepted_packet_count(accepted_event_count)
    );

    reg serialize_active;
    reg [OUTPUT_CHANNELS-1:0] serialize_mask;
    reg [OUTPUT_ADDRESS_BITS-1:0] serialize_address;
    reg [OUTPUT_CHANNELS*SPIKE_BITS-1:0] serialize_spikes_flat;
    reg core_done_seen;

    reg [OUTPUT_CHANNEL_BITS-1:0] selected_channel;
    reg selected_channel_found;
    integer select_index;
    reg [OUTPUT_CHANNELS-1:0] incoming_nonzero_mask;

    always @* begin
        selected_channel = {OUTPUT_CHANNEL_BITS{1'b0}};
        selected_channel_found = 1'b0;
        incoming_nonzero_mask = {OUTPUT_CHANNELS{1'b0}};
        for (select_index = 0; select_index < OUTPUT_CHANNELS;
                select_index = select_index + 1) begin
            if (serialize_mask[select_index]
                    && !selected_channel_found) begin
                selected_channel = OUTPUT_CHANNEL_BITS'(select_index);
                selected_channel_found = 1'b1;
            end
            if (core_output_spike_flat[
                    select_index*SPIKE_BITS +: SPIKE_BITS
                ] != 0) begin
                incoming_nonzero_mask[select_index] = 1'b1;
            end
        end
    end

    assign core_output_ready = !serialize_active;
    assign sparse_valid = serialize_active && selected_channel_found;
    assign sparse_address = serialize_address;
    assign sparse_channel = selected_channel;
    assign sparse_count = serialize_spikes_flat[
        selected_channel*SPIKE_BITS +: SPIKE_BITS
    ];
    assign sparse_end = done;

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            trace_valid <= 1'b0;
            trace_address <= {OUTPUT_ADDRESS_BITS{1'b0}};
            trace_current_flat <= {OUTPUT_CHANNELS*CURRENT_BITS{1'b0}};
            trace_spike_flat <= {OUTPUT_CHANNELS*SPIKE_BITS{1'b0}};
            trace_membrane_flat <= {OUTPUT_CHANNELS*STATE_BITS{1'b0}};
            trace_range_error <= 1'b0;
            serialize_active <= 1'b0;
            serialize_mask <= {OUTPUT_CHANNELS{1'b0}};
            serialize_address <= {OUTPUT_ADDRESS_BITS{1'b0}};
            serialize_spikes_flat <= {OUTPUT_CHANNELS*SPIKE_BITS{1'b0}};
            core_done_seen <= 1'b0;
            cycle_count <= 32'd0;
            emitted_packet_count <= 32'd0;
        end else begin
            done <= 1'b0;
            trace_valid <= 1'b0;
            if (busy) begin
                cycle_count <= cycle_count + 1'b1;
            end
            if (start && !busy) begin
                busy <= 1'b1;
                serialize_active <= 1'b0;
                serialize_mask <= {OUTPUT_CHANNELS{1'b0}};
                core_done_seen <= 1'b0;
                cycle_count <= 32'd0;
                emitted_packet_count <= 32'd0;
            end
            if (core_output_valid && core_output_ready) begin
                trace_valid <= 1'b1;
                trace_address <= core_output_address;
                trace_current_flat <= core_output_current_flat;
                trace_spike_flat <= core_output_spike_flat;
                trace_membrane_flat <= core_output_membrane_flat;
                trace_range_error <= core_output_range_error;
                serialize_address <= core_output_address;
                serialize_spikes_flat <= core_output_spike_flat;
                serialize_mask <= incoming_nonzero_mask;
                serialize_active <= |incoming_nonzero_mask;
            end
            if (sparse_valid && sparse_ready) begin
                serialize_mask[selected_channel] <= 1'b0;
                emitted_packet_count <= emitted_packet_count + 1'b1;
                if ((serialize_mask
                        & ~({{(OUTPUT_CHANNELS-1){1'b0}}, 1'b1}
                            << selected_channel)) == 0) begin
                    serialize_active <= 1'b0;
                end
            end
            if (core_done) begin
                core_done_seen <= 1'b1;
            end
            if (core_done_seen && !serialize_active && !core_busy) begin
                busy <= 1'b0;
                done <= 1'b1;
                core_done_seen <= 1'b0;
            end
        end
    end

    initial begin
        if (INPUT_CHANNELS != 2 || OUTPUT_CHANNELS != 8
                || INPUT_WIDTH != 128 || INPUT_HEIGHT != 96) begin
            $error("pursuit sparse L1 wrapper is specialized for 2x96x128 -> 8x48x64");
        end
    end
endmodule

`default_nettype wire
