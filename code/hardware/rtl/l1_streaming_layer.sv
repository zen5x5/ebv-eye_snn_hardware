`timescale 1ns/1ps
`default_nettype none

// Complete streaming implementation of pursuit L1:
//   2x96x128 U9 input -> same-pad 3x3 convolution -> folded constant
//   -> signed S24 clamp -> 2x2 max pool -> stateless MultiSpike LIF.
//
// The two input channels for one spatial address are returned together by a
// one-cycle-latency input memory.  All eight output channels are computed in
// parallel with 8x18 count/weight lanes.  L1 has equal weight/current binary
// scale in the frozen fixed-point contract, so no product left shift is needed.
module l1_streaming_layer #(
    parameter integer INPUT_CHANNELS = 2,
    parameter integer OUTPUT_PES = 8,
    parameter integer INPUT_WIDTH = 128,
    parameter integer INPUT_HEIGHT = 96,
    parameter integer COUNT_BITS = 9,
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

    output wire                                      input_read_enable,
    output wire [$clog2(INPUT_WIDTH*INPUT_HEIGHT)-1:0]
                                                    input_read_address,
    input  wire [INPUT_CHANNELS*COUNT_BITS-1:0]      input_read_data_flat,

    input  wire                                      weight_write_enable,
    input  wire [$clog2(OUTPUT_PES*INPUT_CHANNELS*9)-1:0]
                                                    weight_write_address,
    input  wire signed [WEIGHT_BITS-1:0]             weight_write_data,
    input  wire                                      channel_write_enable,
    input  wire [$clog2(OUTPUT_PES)-1:0]             channel_write_address,
    input  wire signed [CURRENT_BITS-1:0]            constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]            fraction_bits_write_data,

    output wire                                      output_valid,
    output wire [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    output_address,
    output wire [OUTPUT_PES*CURRENT_BITS-1:0]        output_current_flat,
    output wire [OUTPUT_PES*SPIKE_BITS-1:0]          output_spike_flat,
    output wire [OUTPUT_PES*STATE_BITS-1:0]          output_membrane_flat,
    output wire                                      output_range_error
);
    localparam integer LANES = INPUT_CHANNELS * 9;
    localparam integer INPUT_TAG_BITS = $clog2(INPUT_WIDTH*INPUT_HEIGHT);
    localparam integer OUTPUT_TAG_BITS =
        $clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2));
    localparam integer INPUT_COLUMN_BITS = $clog2(INPUT_WIDTH);
    localparam integer INPUT_ROW_ADDRESS_BITS =
        INPUT_TAG_BITS - INPUT_COLUMN_BITS;
    localparam integer PADDED_ROW_BITS = $clog2(INPUT_HEIGHT + 2);
    localparam integer PADDED_COLUMN_BITS = $clog2(INPUT_WIDTH + 2);
    localparam integer WEIGHT_COUNT = OUTPUT_PES * LANES;
    localparam [PADDED_ROW_BITS-1:0] LAST_INTERIOR_ROW =
        PADDED_ROW_BITS'(INPUT_HEIGHT);
    localparam [PADDED_COLUMN_BITS-1:0] LAST_INTERIOR_COLUMN =
        PADDED_COLUMN_BITS'(INPUT_WIDTH);
    localparam [PADDED_ROW_BITS-1:0] LAST_PADDED_ROW =
        PADDED_ROW_BITS'(INPUT_HEIGHT + 1);
    localparam [PADDED_COLUMN_BITS-1:0] LAST_PADDED_COLUMN =
        PADDED_COLUMN_BITS'(INPUT_WIDTH + 1);
    localparam [OUTPUT_TAG_BITS-1:0] LAST_OUTPUT_ADDRESS =
        OUTPUT_TAG_BITS'((INPUT_WIDTH/2)*(INPUT_HEIGHT/2)-1);

    localparam signed [ACCUMULATOR_BITS-1:0] CURRENT_MAXIMUM = {
        {(ACCUMULATOR_BITS-CURRENT_BITS){1'b0}},
        1'b0,
        {(CURRENT_BITS-1){1'b1}}
    };
    localparam signed [ACCUMULATOR_BITS-1:0] CURRENT_MINIMUM = {
        {(ACCUMULATOR_BITS-CURRENT_BITS){1'b1}},
        1'b1,
        {(CURRENT_BITS-1){1'b0}}
    };

    reg signed [WEIGHT_BITS-1:0] weight_registers [0:WEIGHT_COUNT-1];
    reg signed [CURRENT_BITS-1:0] constant_registers [0:OUTPUT_PES-1];
    reg [FRACTION_BITS_WIDTH-1:0] fraction_bits_registers [0:OUTPUT_PES-1];

    always @(posedge clk) begin
        if (weight_write_enable) begin
            weight_registers[weight_write_address] <= weight_write_data;
        end
        if (channel_write_enable) begin
            constant_registers[channel_write_address] <= constant_write_data;
            fraction_bits_registers[channel_write_address]
                <= fraction_bits_write_data;
        end
    end

    reg scan_active;
    reg [PADDED_ROW_BITS-1:0] scan_row;
    reg [PADDED_COLUMN_BITS-1:0] scan_column;
    reg coordinate_valid_delayed;
    reg [PADDED_ROW_BITS-1:0] row_delayed;
    reg [PADDED_COLUMN_BITS-1:0] column_delayed;
    reg interior_delayed;

    wire scan_interior = (scan_row > 0)
        && (scan_row <= LAST_INTERIOR_ROW)
        && (scan_column > 0)
        && (scan_column <= LAST_INTERIOR_COLUMN);
    wire [INPUT_ROW_ADDRESS_BITS-1:0] unpadded_input_row =
        INPUT_ROW_ADDRESS_BITS'(scan_row)
        - INPUT_ROW_ADDRESS_BITS'(1);
    wire [INPUT_COLUMN_BITS-1:0] unpadded_input_column =
        INPUT_COLUMN_BITS'(scan_column)
        - INPUT_COLUMN_BITS'(1);
    assign input_read_enable = scan_active && scan_interior;
    assign input_read_address = input_read_enable
        ? {unpadded_input_row, unpadded_input_column}
        : {INPUT_TAG_BITS{1'b0}};

    wire [INPUT_CHANNELS*COUNT_BITS-1:0] padded_pixels_flat =
        interior_delayed
        ? input_read_data_flat
        : {INPUT_CHANNELS*COUNT_BITS{1'b0}};

    wire line_valid;
    wire [INPUT_TAG_BITS-1:0] line_tag;
    wire [LANES*COUNT_BITS-1:0] line_window_flat;

    padded_line_buffer_3x3 #(
        .CHANNELS(INPUT_CHANNELS),
        .WIDTH(INPUT_WIDTH),
        .HEIGHT(INPUT_HEIGHT),
        .PIXEL_BITS(COUNT_BITS),
        .TAG_BITS(INPUT_TAG_BITS)
    ) line_buffer (
        .clk(clk),
        .rst(rst),
        .in_valid(coordinate_valid_delayed),
        .padded_row(row_delayed),
        .padded_column(column_delayed),
        .pixels_flat(padded_pixels_flat),
        .out_valid(line_valid),
        .tag_out(line_tag),
        .window_flat(line_window_flat)
    );

    wire [OUTPUT_PES*LANES*WEIGHT_BITS-1:0] weights_flat;
    genvar weight_index;
    generate
        for (weight_index = 0; weight_index < WEIGHT_COUNT;
                weight_index = weight_index + 1) begin : gen_weight_pack
            assign weights_flat[weight_index*WEIGHT_BITS +: WEIGHT_BITS]
                = weight_registers[weight_index];
        end
    endgenerate

    wire mac_valid;
    wire [INPUT_TAG_BITS-1:0] mac_tag;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] mac_accumulators_flat;

    pipelined_count_mac_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .COUNT_BITS(COUNT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .TAG_BITS(INPUT_TAG_BITS)
    ) mac_array (
        .clk(clk),
        .rst(rst),
        .in_valid(line_valid),
        .tag_in(line_tag),
        .counts_flat(line_window_flat),
        .weights_flat(weights_flat),
        .accumulators_in_flat(
            {OUTPUT_PES*ACCUMULATOR_BITS{1'b0}}
        ),
        .out_valid(mac_valid),
        .tag_out(mac_tag),
        .accumulators_out_flat(mac_accumulators_flat)
    );

    wire [OUTPUT_PES*CURRENT_BITS-1:0] biased_currents_flat;
    genvar output_index;
    generate
        for (output_index = 0; output_index < OUTPUT_PES;
                output_index = output_index + 1) begin : gen_bias_and_clamp
            wire signed [ACCUMULATOR_BITS-1:0] product_sum;
            wire signed [ACCUMULATOR_BITS-1:0] constant_extended;
            wire signed [ACCUMULATOR_BITS-1:0] biased_sum;
            wire signed [CURRENT_BITS-1:0] clipped_sum;

            assign product_sum = $signed(mac_accumulators_flat[
                output_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ]);
            assign constant_extended = {
                {(ACCUMULATOR_BITS-CURRENT_BITS){
                    constant_registers[output_index][CURRENT_BITS-1]
                }},
                constant_registers[output_index]
            };
            assign biased_sum = product_sum + constant_extended;
            assign clipped_sum = (biased_sum > CURRENT_MAXIMUM)
                ? CURRENT_MAXIMUM[CURRENT_BITS-1:0]
                : ((biased_sum < CURRENT_MINIMUM)
                    ? CURRENT_MINIMUM[CURRENT_BITS-1:0]
                    : biased_sum[CURRENT_BITS-1:0]);
            assign biased_currents_flat[
                output_index*CURRENT_BITS +: CURRENT_BITS
            ] = clipped_sum;
        end
    endgenerate

    wire pool_valid;
    wire [OUTPUT_TAG_BITS-1:0] pool_tag;
    wire [OUTPUT_PES*CURRENT_BITS-1:0] pool_currents_flat;

    maxpool2x2_stream #(
        .OUTPUT_PES(OUTPUT_PES),
        .WIDTH(INPUT_WIDTH),
        .HEIGHT(INPUT_HEIGHT),
        .CURRENT_BITS(CURRENT_BITS),
        .INPUT_TAG_BITS(INPUT_TAG_BITS),
        .OUTPUT_TAG_BITS(OUTPUT_TAG_BITS)
    ) pool (
        .clk(clk),
        .rst(rst),
        .in_valid(mac_valid),
        .tag_in(mac_tag),
        .currents_flat(biased_currents_flat),
        .out_valid(pool_valid),
        .tag_out(pool_tag),
        .pooled_currents_flat(pool_currents_flat)
    );

    wire [OUTPUT_PES-1:0] lif_valid;
    wire [OUTPUT_PES-1:0] lif_range_error;
    wire [OUTPUT_PES*SPIKE_BITS-1:0] lif_spikes_flat;
    wire [OUTPUT_PES*STATE_BITS-1:0] lif_membranes_flat;

    generate
        for (output_index = 0; output_index < OUTPUT_PES;
                output_index = output_index + 1) begin : gen_lif
            lif_multispike #(
                .CURRENT_BITS(CURRENT_BITS),
                .STATE_BITS(STATE_BITS),
                .SPIKE_BITS(SPIKE_BITS),
                .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
            ) neuron (
                .clk(clk),
                .rst(rst),
                .in_valid(pool_valid),
                .current_integer($signed(pool_currents_flat[
                    output_index*CURRENT_BITS +: CURRENT_BITS
                ])),
                .previous_membrane_integer({STATE_BITS{1'b0}}),
                .fraction_bits(fraction_bits_registers[output_index]),
                .out_valid(lif_valid[output_index]),
                .spike_integer(lif_spikes_flat[
                    output_index*SPIKE_BITS +: SPIKE_BITS
                ]),
                .membrane_integer(lif_membranes_flat[
                    output_index*STATE_BITS +: STATE_BITS
                ]),
                .range_error(lif_range_error[output_index])
            );
        end
    endgenerate

    reg [OUTPUT_TAG_BITS-1:0] output_address_register;
    reg [OUTPUT_PES*CURRENT_BITS-1:0] output_current_register;
    always @(posedge clk) begin
        if (rst) begin
            output_address_register <= {OUTPUT_TAG_BITS{1'b0}};
            output_current_register <= {OUTPUT_PES*CURRENT_BITS{1'b0}};
        end else if (pool_valid) begin
            output_address_register <= pool_tag;
            output_current_register <= pool_currents_flat;
        end
    end

    assign output_valid = &lif_valid;
    assign output_address = output_address_register;
    assign output_current_flat = output_current_register;
    assign output_spike_flat = lif_spikes_flat;
    assign output_membrane_flat = lif_membranes_flat;
    assign output_range_error = output_valid && (|lif_range_error);

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            scan_active <= 1'b0;
            scan_row <= {PADDED_ROW_BITS{1'b0}};
            scan_column <= {PADDED_COLUMN_BITS{1'b0}};
            coordinate_valid_delayed <= 1'b0;
            row_delayed <= {PADDED_ROW_BITS{1'b0}};
            column_delayed <= {PADDED_COLUMN_BITS{1'b0}};
            interior_delayed <= 1'b0;
        end else begin
            done <= 1'b0;
            coordinate_valid_delayed <= scan_active;
            if (scan_active) begin
                row_delayed <= scan_row;
                column_delayed <= scan_column;
                interior_delayed <= scan_interior;
                if (scan_column == LAST_PADDED_COLUMN) begin
                    scan_column <= {PADDED_COLUMN_BITS{1'b0}};
                    if (scan_row == LAST_PADDED_ROW) begin
                        scan_active <= 1'b0;
                    end else begin
                        scan_row <= scan_row + 1'b1;
                    end
                end else begin
                    scan_column <= scan_column + 1'b1;
                end
            end else begin
                interior_delayed <= 1'b0;
            end

            if (start && !busy) begin
                busy <= 1'b1;
                scan_active <= 1'b1;
                scan_row <= {PADDED_ROW_BITS{1'b0}};
                scan_column <= {PADDED_COLUMN_BITS{1'b0}};
                coordinate_valid_delayed <= 1'b0;
                interior_delayed <= 1'b0;
            end

            if (output_valid
                    && output_address == LAST_OUTPUT_ADDRESS) begin
                busy <= 1'b0;
                done <= 1'b1;
            end
        end
    end

    initial begin
        if (INPUT_CHANNELS != 2 || OUTPUT_PES != 8) begin
            $error("this checked wrapper is specialized for pursuit L1 2->8");
        end
        if (ACCUMULATOR_BITS < CURRENT_BITS) begin
            $error("accumulator must be at least as wide as current");
        end
        if ((1 << INPUT_COLUMN_BITS) != INPUT_WIDTH) begin
            $error("input address packing requires power-of-two INPUT_WIDTH");
        end
    end
endmodule

`default_nettype wire
