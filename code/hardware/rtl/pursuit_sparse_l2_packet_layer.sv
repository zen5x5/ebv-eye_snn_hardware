`timescale 1ns/1ps
`default_nettype none

// Pursuit L2 sparse front-end.
//
// Each input packet is one nonzero L1 MultiSpike value at a row-major address.
// Three kernel taps and all sixteen output channels are evaluated per update
// cycle: 3 x 16 S16-by-U8 multipliers, so one packet takes three update cycles
// after pipeline fill.  The layer still emits every 24x32 output position and
// writes all sixteen U8 spikes into dense_activation_memory.  Consequently an
// untouched position receives max(0,0,0,0)+folded_constant and the exact
// bias-only default spike; L3 can read a conventional dense tensor.
module pursuit_sparse_l2_packet_layer #(
    parameter integer INPUT_WIDTH = 64,
    parameter integer INPUT_HEIGHT = 48,
    parameter integer INPUT_CHANNELS = 8,
    parameter integer OUTPUT_CHANNELS = 16,
    parameter integer COUNT_BITS = 8,
    parameter integer WEIGHT_BITS = 16,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer FRACTION_BITS_WIDTH = 5,
    parameter integer READ_LANES = 4
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output wire                                      busy,
    output wire                                      done,

    input  wire                                      sparse_valid,
    output wire                                      sparse_ready,
    input  wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT)-1:0]
                                                    sparse_address,
    input  wire [$clog2(INPUT_CHANNELS)-1:0]        sparse_channel,
    input  wire [COUNT_BITS-1:0]                    sparse_count,
    input  wire                                      sparse_end,

    input  wire                                      weight_write_enable,
    input  wire [$clog2(OUTPUT_CHANNELS)-1:0]
                                                    weight_write_output_channel,
    input  wire [$clog2(INPUT_CHANNELS)-1:0]
                                                    weight_write_input_channel,
    input  wire [3:0]                               weight_write_kernel_index,
    input  wire signed [WEIGHT_BITS-1:0]            weight_write_data,
    input  wire                                      channel_write_enable,
    input  wire [$clog2(OUTPUT_CHANNELS)-1:0]
                                                    channel_write_channel,
    input  wire signed [CURRENT_BITS-1:0]           constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]           fraction_bits_write_data,

    // Synchronous, repeatable dense read interface for either complete
    // back-end schedule.  read_valid is asserted one cycle after read_enable.
    input  wire                                      dense_read_enable,
    input  wire [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    dense_read_address,
    input  wire [$clog2(OUTPUT_CHANNELS)-1:0]       dense_read_channel_base,
    output reg                                       dense_read_valid,
    output wire [READ_LANES*SPIKE_BITS-1:0]         dense_read_data_flat,

    output wire                                      trace_valid,
    output wire [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    trace_address,
    output wire [OUTPUT_CHANNELS*CURRENT_BITS-1:0]  trace_current_flat,
    output wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0]    trace_spike_flat,
    output wire [OUTPUT_CHANNELS*STATE_BITS-1:0]    trace_membrane_flat,
    output wire                                      trace_range_error,

    output wire                                      protocol_error,
    output wire [31:0]                              cycle_count,
    output wire [31:0]                              accepted_packet_count,
    output reg  [31:0]                              dense_write_count
);
    localparam integer INPUT_COLUMN_BITS = $clog2(INPUT_WIDTH);
    localparam integer INPUT_ROW_BITS = $clog2(INPUT_HEIGHT);
    localparam integer OUTPUT_PIXELS =
        (INPUT_WIDTH/2)*(INPUT_HEIGHT/2);
    localparam integer OUTPUT_ADDRESS_BITS = $clog2(OUTPUT_PIXELS);
    wire [INPUT_ROW_BITS-1:0] packet_row =
        sparse_address[INPUT_COLUMN_BITS +: INPUT_ROW_BITS];
    wire [INPUT_COLUMN_BITS-1:0] packet_column =
        sparse_address[INPUT_COLUMN_BITS-1:0];

    wire core_output_valid;
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
        .COUNT_BITS(COUNT_BITS),
        .UNIT_INPUT(0),
        .KERNEL_TAPS_PER_CYCLE(3),
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
        .busy(busy),
        .done(done),
        .input_valid(sparse_valid),
        .input_ready(sparse_ready),
        .input_channel(sparse_channel),
        .input_row(packet_row),
        .input_column(packet_column),
        .input_count(sparse_count),
        .input_end(sparse_end),
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
        .output_ready(1'b1),
        .output_address(core_output_address),
        .output_current_flat(core_output_current_flat),
        .output_spike_flat(core_output_spike_flat),
        .output_membrane_flat(core_output_membrane_flat),
        .output_range_error(core_output_range_error),
        .protocol_error(protocol_error),
        .cycle_count(cycle_count),
        .accepted_packet_count(accepted_packet_count)
    );

    // Eight two-channel banks are read together, then a registered group
    // selector chooses four channels.  Every physical bank has a conventional
    // full-word write plus synchronous full-word read; this is directly
    // inferable as simple dual-port BRAM by both Yosys and Vivado.
    wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0] dense_bank_read_flat;
    reg [1:0] dense_read_group_delayed;
    reg dense_read_alignment_delayed;
    genvar dense_bank_index;
    generate
        for (dense_bank_index = 0;
                dense_bank_index < OUTPUT_CHANNELS/2;
                dense_bank_index = dense_bank_index + 1) begin : dense_banks
            (* ram_style = "block" *) reg [2*SPIKE_BITS-1:0] memory
                [0:OUTPUT_PIXELS-1];
            reg [2*SPIKE_BITS-1:0] read_data;
            assign dense_bank_read_flat[
                dense_bank_index*2*SPIKE_BITS +: 2*SPIKE_BITS
            ] = read_data;
            always @(posedge clk) begin
                if (core_output_valid) begin
                    memory[core_output_address]
                        <= core_output_spike_flat[
                            dense_bank_index*2*SPIKE_BITS +: 2*SPIKE_BITS
                        ];
                end
                if (dense_read_enable) begin
                    read_data <= memory[dense_read_address];
                end
            end
        end
    endgenerate

    function automatic [READ_LANES*SPIKE_BITS-1:0] select_dense_group;
        input [OUTPUT_CHANNELS*SPIKE_BITS-1:0] values;
        input [1:0] group_index;
        begin
            case (group_index)
                2'd0: select_dense_group = values[0 +: 4*SPIKE_BITS];
                2'd1: select_dense_group = values[
                    4*SPIKE_BITS +: 4*SPIKE_BITS
                ];
                2'd2: select_dense_group = values[
                    8*SPIKE_BITS +: 4*SPIKE_BITS
                ];
                default: select_dense_group = values[
                    12*SPIKE_BITS +: 4*SPIKE_BITS
                ];
            endcase
        end
    endfunction
    assign dense_read_data_flat = dense_read_alignment_delayed
        ? select_dense_group(dense_bank_read_flat, dense_read_group_delayed)
        : {READ_LANES*SPIKE_BITS{1'b0}};

    assign trace_valid = core_output_valid;
    assign trace_address = core_output_address;
    assign trace_current_flat = core_output_current_flat;
    assign trace_spike_flat = core_output_spike_flat;
    assign trace_membrane_flat = core_output_membrane_flat;
    assign trace_range_error = core_output_range_error;

    always @(posedge clk) begin
        if (rst) begin
            dense_read_valid <= 1'b0;
            dense_read_group_delayed <= 2'd0;
            dense_read_alignment_delayed <= 1'b0;
            dense_write_count <= 32'd0;
        end else begin
            dense_read_valid <= dense_read_enable;
            if (dense_read_enable) begin
                dense_read_group_delayed <= dense_read_channel_base[3:2];
                dense_read_alignment_delayed
                    <= dense_read_channel_base[1:0] == 2'b00;
            end
            if (start && !busy) begin
                dense_write_count <= 32'd0;
            end
            if (core_output_valid) begin
                dense_write_count <= dense_write_count + 1'b1;
            end
        end
    end

    initial begin
        if (INPUT_CHANNELS != 8 || OUTPUT_CHANNELS != 16
                || INPUT_WIDTH != 64 || INPUT_HEIGHT != 48) begin
            $error("pursuit sparse L2 wrapper is specialized for 8x48x64 -> 16x24x32");
        end
        if (READ_LANES != 4) begin
            $error("pursuit L3 interface requires four read lanes");
        end
    end
endmodule

`default_nettype wire
