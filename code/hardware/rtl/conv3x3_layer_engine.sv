`timescale 1ns/1ps
`default_nettype none

// Layer-programmable 3x3 convolution engine used by pursuit L1--L8.
//
// The physical datapath is 8 output channels x 4 input channels x 9 taps.
// Layers with more channels are folded over output and input tiles.  A block
// RAM holds S40 partial sums between input-channel tiles.  The engine streams
// an explicitly padded raster through a reusable line buffer, optionally
// performs 2x2 max pooling, and then applies the bit-exact MultiSpike LIF.
//
// All external read ports have one-cycle latency.  Convolution weights use the
// layout emitted by pack_pursuit_hardware_image.py: one tile is eighteen
// 256-bit beats in output-lane, input-lane, kernel-row-major order.
module conv3x3_layer_engine #(
    parameter integer OUTPUT_PES = 8,
    parameter integer INPUT_LANES = 4,
    parameter integer KERNEL_TAPS = 9,
    parameter integer MAX_WIDTH = 128,
    parameter integer MAX_HEIGHT = 96,
    parameter integer MAX_PARTIAL_SPATIAL = 3072,
    parameter integer COUNT_BITS = 9,
    parameter integer WEIGHT_BITS = 16,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6,
    parameter integer WEIGHT_ADDRESS_BITS = 13,
    parameter integer PARAMETER_ADDRESS_BITS = 9,
    parameter integer FRACTION_BITS_WIDTH = 5,
    parameter integer WIDTH_LOG2_BITS = 4
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output reg                                       busy,
    output reg                                       done,

    input  wire [WIDTH_LOG2_BITS-1:0]                config_width_log2,
    input  wire [$clog2(MAX_HEIGHT+1)-1:0]           config_height,
    input  wire [3:0]                                config_input_tiles,
    input  wire [3:0]                                config_output_tiles,
    input  wire [TAG_BITS-1:0]                       config_output_spatial,
    input  wire                                      config_pool_enable,
    input  wire                                      config_use_previous_state,
    input  wire                                      config_write_state,
    input  wire [WEIGHT_ADDRESS_BITS-1:0]            config_weight_beat_base,
    input  wire [PARAMETER_ADDRESS_BITS-1:0]         config_parameter_base,

    output wire                                      weight_read_enable,
    output wire [WEIGHT_ADDRESS_BITS-1:0]            weight_read_address,
    input  wire [255:0]                              weight_read_data,

    output wire                                      parameter_read_enable,
    output wire [PARAMETER_ADDRESS_BITS-1:0]         parameter_read_address,
    input  wire signed [CURRENT_BITS-1:0]            parameter_constant_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]            parameter_fraction_data,

    output wire                                      input_read_enable,
    output wire [TAG_BITS-1:0]                       input_read_address,
    output wire [CHANNEL_BITS-1:0]                   input_read_channel_base,
    input  wire [INPUT_LANES*COUNT_BITS-1:0]         input_read_data_flat,

    output wire                                      state_read_enable,
    output wire [TAG_BITS-1:0]                       state_read_address,
    output wire [CHANNEL_BITS-1:0]                   state_read_channel_base,
    input  wire [OUTPUT_PES*STATE_BITS-1:0]          state_read_data_flat,

    output wire                                      state_write_enable,
    output wire [TAG_BITS-1:0]                       state_write_address,
    output wire [CHANNEL_BITS-1:0]                   state_write_channel_base,
    output wire [OUTPUT_PES*STATE_BITS-1:0]          state_write_data_flat,

    output wire                                      output_valid,
    output wire [TAG_BITS-1:0]                       output_address,
    output wire [CHANNEL_BITS-1:0]                   output_channel_base,
    output wire [OUTPUT_PES*CURRENT_BITS-1:0]        output_current_flat,
    output wire [OUTPUT_PES*SPIKE_BITS-1:0]          output_spike_flat,
    output wire [OUTPUT_PES*STATE_BITS-1:0]          output_membrane_flat,
    output wire                                      output_range_error
);
    localparam integer LANES = INPUT_LANES * KERNEL_TAPS;
    localparam integer WEIGHTS_PER_TILE = OUTPUT_PES * LANES;
    localparam integer WEIGHTS_PER_BEAT = 256 / WEIGHT_BITS;
    localparam integer WEIGHT_BEATS_PER_TILE =
        WEIGHTS_PER_TILE / WEIGHTS_PER_BEAT;
    localparam integer PARTIAL_ADDRESS_BITS = $clog2(MAX_PARTIAL_SPATIAL);
    localparam integer PADDED_ROW_BITS = $clog2(MAX_HEIGHT + 2);
    localparam integer PADDED_COLUMN_BITS = $clog2(MAX_WIDTH + 2);

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_LOAD = 3'd1;
    localparam [2:0] STATE_SCAN = 3'd2;

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

    function automatic signed [CURRENT_BITS-1:0] clamp_current;
        input signed [ACCUMULATOR_BITS-1:0] value;
        begin
            if (value > CURRENT_MAXIMUM) begin
                clamp_current = CURRENT_MAXIMUM[CURRENT_BITS-1:0];
            end else if (value < CURRENT_MINIMUM) begin
                clamp_current = CURRENT_MINIMUM[CURRENT_BITS-1:0];
            end else begin
                clamp_current = value[CURRENT_BITS-1:0];
            end
        end
    endfunction

    reg [2:0] control_state;
    reg [WIDTH_LOG2_BITS-1:0] width_log2_register;
    reg [$clog2(MAX_HEIGHT+1)-1:0] height_register;
    reg [3:0] input_tiles_register;
    reg [3:0] output_tiles_register;
    reg [TAG_BITS-1:0] output_spatial_register;
    reg pool_enable_register;
    reg use_previous_state_register;
    reg write_state_register;

    reg [3:0] input_tile_index;
    reg [3:0] output_tile_index;
    reg [WEIGHT_ADDRESS_BITS-1:0] weight_tile_base;
    reg [PARAMETER_ADDRESS_BITS-1:0] parameter_tile_base;
    reg load_parameters;

    reg [4:0] weight_issue_count;
    reg [4:0] weight_loaded_count;
    reg [4:0] weight_response_index;
    reg weight_response_valid;
    reg [3:0] parameter_issue_count;
    reg [3:0] parameter_loaded_count;
    reg [2:0] parameter_response_index;
    reg parameter_response_valid;

    reg signed [WEIGHT_BITS-1:0] weight_registers
        [0:WEIGHTS_PER_TILE-1];
    reg signed [CURRENT_BITS-1:0] constant_registers [0:OUTPUT_PES-1];
    reg [FRACTION_BITS_WIDTH-1:0] fraction_registers [0:OUTPUT_PES-1];

    assign weight_read_enable = (control_state == STATE_LOAD)
        && (weight_issue_count < 5'(WEIGHT_BEATS_PER_TILE));
    assign weight_read_address = weight_tile_base
        + WEIGHT_ADDRESS_BITS'(weight_issue_count);
    assign parameter_read_enable = (control_state == STATE_LOAD)
        && load_parameters && (parameter_issue_count < 4'(OUTPUT_PES));
    assign parameter_read_address = parameter_tile_base
        + PARAMETER_ADDRESS_BITS'(parameter_issue_count);

    integer load_word_index;
    always @(posedge clk) begin
        if (rst) begin
            weight_issue_count <= 0;
            weight_loaded_count <= 0;
            weight_response_index <= 0;
            weight_response_valid <= 1'b0;
            parameter_issue_count <= 0;
            parameter_loaded_count <= 0;
            parameter_response_index <= 0;
            parameter_response_valid <= 1'b0;
        end else begin
            weight_response_valid <= weight_read_enable;
            if (weight_read_enable) begin
                weight_response_index <= weight_issue_count;
                weight_issue_count <= weight_issue_count + 1'b1;
            end
            if (weight_response_valid) begin
                for (load_word_index = 0;
                        load_word_index < WEIGHTS_PER_BEAT;
                        load_word_index = load_word_index + 1) begin
                    weight_registers[
                        weight_response_index*WEIGHTS_PER_BEAT
                        + load_word_index
                    ] <= weight_read_data[
                        load_word_index*WEIGHT_BITS +: WEIGHT_BITS
                    ];
                end
                weight_loaded_count <= weight_loaded_count + 1'b1;
            end

            parameter_response_valid <= parameter_read_enable;
            if (parameter_read_enable) begin
                parameter_response_index <= parameter_issue_count[2:0];
                parameter_issue_count <= parameter_issue_count + 1'b1;
            end
            if (parameter_response_valid) begin
                constant_registers[parameter_response_index]
                    <= parameter_constant_data;
                fraction_registers[parameter_response_index]
                    <= parameter_fraction_data;
                parameter_loaded_count <= parameter_loaded_count + 1'b1;
            end

            if (control_state != STATE_LOAD) begin
                weight_issue_count <= 0;
                weight_loaded_count <= 0;
                weight_response_valid <= 1'b0;
                parameter_issue_count <= 0;
                parameter_loaded_count <= 0;
                parameter_response_valid <= 1'b0;
            end
        end
    end

    wire [OUTPUT_PES*LANES*WEIGHT_BITS-1:0] weights_flat;
    genvar packed_weight_index;
    generate
        for (packed_weight_index = 0;
                packed_weight_index < WEIGHTS_PER_TILE;
                packed_weight_index = packed_weight_index + 1) begin : gen_weights
            assign weights_flat[
                packed_weight_index*WEIGHT_BITS +: WEIGHT_BITS
            ] = weight_registers[packed_weight_index];
        end
    endgenerate

    reg scan_issue_active;
    reg [PADDED_ROW_BITS-1:0] scan_row;
    reg [PADDED_COLUMN_BITS-1:0] scan_column;
    reg coordinate_valid_delayed;
    reg [PADDED_ROW_BITS-1:0] row_delayed;
    reg [PADDED_COLUMN_BITS-1:0] column_delayed;
    reg interior_delayed;

    wire [PADDED_COLUMN_BITS-1:0] input_width =
        PADDED_COLUMN_BITS'(1) << width_log2_register;
    wire [PADDED_ROW_BITS-1:0] last_interior_row =
        PADDED_ROW_BITS'(height_register);
    wire [PADDED_COLUMN_BITS-1:0] last_interior_column = input_width;
    wire [PADDED_ROW_BITS-1:0] last_padded_row =
        PADDED_ROW_BITS'(height_register) + 1'b1;
    wire [PADDED_COLUMN_BITS-1:0] last_padded_column = input_width + 1'b1;
    wire scan_coordinate_valid = (control_state == STATE_SCAN)
        && scan_issue_active;
    wire scan_interior = (scan_row > 0)
        && (scan_row <= last_interior_row)
        && (scan_column > 0)
        && (scan_column <= last_interior_column);
    wire [TAG_BITS-1:0] unpadded_row = TAG_BITS'(scan_row) - 1'b1;
    wire [TAG_BITS-1:0] unpadded_column =
        TAG_BITS'(scan_column) - 1'b1;

    assign input_read_enable = scan_coordinate_valid && scan_interior;
    assign input_read_address = input_read_enable
        ? ((unpadded_row << width_log2_register) | unpadded_column)
        : {TAG_BITS{1'b0}};
    assign input_read_channel_base = CHANNEL_BITS'(input_tile_index * INPUT_LANES);

    always @(posedge clk) begin
        if (rst) begin
            coordinate_valid_delayed <= 1'b0;
            row_delayed <= 0;
            column_delayed <= 0;
            interior_delayed <= 1'b0;
        end else begin
            coordinate_valid_delayed <= scan_coordinate_valid;
            if (scan_coordinate_valid) begin
                row_delayed <= scan_row;
                column_delayed <= scan_column;
                interior_delayed <= scan_interior;
            end
        end
    end

    wire [INPUT_LANES*COUNT_BITS-1:0] padded_pixels_flat =
        interior_delayed ? input_read_data_flat
                         : {INPUT_LANES*COUNT_BITS{1'b0}};
    wire line_valid;
    wire [TAG_BITS-1:0] line_tag;
    wire [LANES*COUNT_BITS-1:0] line_window_flat;

    programmable_padded_line_buffer_3x3 #(
        .CHANNELS(INPUT_LANES),
        .MAX_WIDTH(MAX_WIDTH),
        .MAX_HEIGHT(MAX_HEIGHT),
        .PIXEL_BITS(COUNT_BITS),
        .TAG_BITS(TAG_BITS),
        .WIDTH_LOG2_BITS(WIDTH_LOG2_BITS)
    ) line_buffer (
        .clk(clk),
        .rst(rst),
        .in_valid(coordinate_valid_delayed),
        .width_log2(width_log2_register),
        .padded_row(row_delayed),
        .padded_column(column_delayed),
        .pixels_flat(padded_pixels_flat),
        .out_valid(line_valid),
        .tag_out(line_tag),
        .window_flat(line_window_flat)
    );

    (* ram_style = "block" *) reg [OUTPUT_PES*ACCUMULATOR_BITS-1:0]
        partial_sum_memory [0:MAX_PARTIAL_SPATIAL-1];
    reg [OUTPUT_PES*ACCUMULATOR_BITS-1:0] partial_sum_read_data;
    reg mac_issue_valid;
    reg [TAG_BITS-1:0] mac_issue_tag;
    reg [LANES*COUNT_BITS-1:0] mac_issue_window;

    always @(posedge clk) begin
        if (line_valid) begin
            partial_sum_read_data <= partial_sum_memory[
                line_tag[PARTIAL_ADDRESS_BITS-1:0]
            ];
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            mac_issue_valid <= 1'b0;
            mac_issue_tag <= 0;
            mac_issue_window <= 0;
        end else begin
            mac_issue_valid <= line_valid;
            if (line_valid) begin
                mac_issue_tag <= line_tag;
                mac_issue_window <= line_window_flat;
            end
        end
    end

    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] mac_accumulator_input =
        (input_tile_index == 0)
        ? {OUTPUT_PES*ACCUMULATOR_BITS{1'b0}}
        : partial_sum_read_data;
    wire mac_output_valid;
    wire [TAG_BITS-1:0] mac_output_tag;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] mac_output_flat;

    pipelined_count_mac_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .COUNT_BITS(COUNT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .TAG_BITS(TAG_BITS)
    ) mac_array (
        .clk(clk),
        .rst(rst),
        .in_valid(mac_issue_valid),
        .tag_in(mac_issue_tag),
        .counts_flat(mac_issue_window),
        .weights_flat(weights_flat),
        .accumulators_in_flat(mac_accumulator_input),
        .out_valid(mac_output_valid),
        .tag_out(mac_output_tag),
        .accumulators_out_flat(mac_output_flat)
    );

    wire final_input_tile = input_tile_index == (input_tiles_register - 1'b1);
    integer partial_output_index;
    always @(posedge clk) begin
        if (mac_output_valid && !final_input_tile) begin
            for (partial_output_index = 0;
                    partial_output_index < OUTPUT_PES;
                    partial_output_index = partial_output_index + 1) begin
                partial_sum_memory[
                    mac_output_tag[PARTIAL_ADDRESS_BITS-1:0]
                ][partial_output_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS]
                    <= mac_output_flat[
                        partial_output_index*ACCUMULATOR_BITS
                        +: ACCUMULATOR_BITS
                    ];
            end
        end
    end

    wire [OUTPUT_PES*CURRENT_BITS-1:0] clamped_currents_flat;
    genvar current_index;
    generate
        for (current_index = 0; current_index < OUTPUT_PES;
                current_index = current_index + 1) begin : gen_current_clamp
            wire signed [ACCUMULATOR_BITS-1:0] accumulator_value;
            wire signed [ACCUMULATOR_BITS-1:0] constant_extended;
            wire signed [ACCUMULATOR_BITS-1:0] complete_current;
            assign accumulator_value = $signed(mac_output_flat[
                current_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ]);
            assign constant_extended = {
                {(ACCUMULATOR_BITS-CURRENT_BITS){
                    constant_registers[current_index][CURRENT_BITS-1]
                }},
                constant_registers[current_index]
            };
            assign complete_current = accumulator_value + constant_extended;
            assign clamped_currents_flat[
                current_index*CURRENT_BITS +: CURRENT_BITS
            ] = clamp_current(complete_current);
        end
    endgenerate

    wire final_current_valid = mac_output_valid && final_input_tile;
    wire pool_output_valid;
    wire [TAG_BITS-1:0] pool_output_tag;
    wire [OUTPUT_PES*CURRENT_BITS-1:0] pool_output_currents;

    programmable_maxpool2x2_stream #(
        .OUTPUT_PES(OUTPUT_PES),
        .MAX_WIDTH(MAX_WIDTH),
        .CURRENT_BITS(CURRENT_BITS),
        .TAG_BITS(TAG_BITS),
        .WIDTH_LOG2_BITS(WIDTH_LOG2_BITS)
    ) max_pool (
        .clk(clk),
        .rst(rst),
        .in_valid(final_current_valid && pool_enable_register),
        .width_log2(width_log2_register),
        .tag_in(mac_output_tag),
        .currents_flat(clamped_currents_flat),
        .out_valid(pool_output_valid),
        .tag_out(pool_output_tag),
        .pooled_currents_flat(pool_output_currents)
    );

    wire neuron_current_valid = pool_enable_register
        ? pool_output_valid : final_current_valid;
    wire [TAG_BITS-1:0] neuron_current_tag = pool_enable_register
        ? pool_output_tag : mac_output_tag;
    wire [OUTPUT_PES*CURRENT_BITS-1:0] neuron_currents_flat =
        pool_enable_register ? pool_output_currents : clamped_currents_flat;

    reg neuron_delay_valid;
    reg [TAG_BITS-1:0] neuron_tag_delayed;
    reg [OUTPUT_PES*CURRENT_BITS-1:0] neuron_currents_delayed;
    reg [CHANNEL_BITS-1:0] neuron_channel_base_delayed;
    always @(posedge clk) begin
        if (rst) begin
            neuron_delay_valid <= 1'b0;
            neuron_tag_delayed <= 0;
            neuron_currents_delayed <= 0;
            neuron_channel_base_delayed <= 0;
        end else begin
            neuron_delay_valid <= neuron_current_valid;
            if (neuron_current_valid) begin
                neuron_tag_delayed <= neuron_current_tag;
                neuron_currents_delayed <= neuron_currents_flat;
                neuron_channel_base_delayed <=
                    CHANNEL_BITS'(output_tile_index * OUTPUT_PES);
            end
        end
    end

    assign state_read_enable = neuron_current_valid
        && use_previous_state_register;
    assign state_read_address = neuron_current_tag;
    assign state_read_channel_base =
        CHANNEL_BITS'(output_tile_index * OUTPUT_PES);

    wire [OUTPUT_PES*SPIKE_BITS-1:0] lif_spikes_flat;
    wire [OUTPUT_PES*STATE_BITS-1:0] lif_membranes_flat;
    wire [OUTPUT_PES-1:0] lif_valid_lanes;
    wire [OUTPUT_PES-1:0] lif_range_errors;
    genvar neuron_index;
    generate
        for (neuron_index = 0; neuron_index < OUTPUT_PES;
                neuron_index = neuron_index + 1) begin : gen_lif
            lif_multispike #(
                .CURRENT_BITS(CURRENT_BITS),
                .STATE_BITS(STATE_BITS),
                .SPIKE_BITS(SPIKE_BITS),
                .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
            ) neuron (
                .clk(clk),
                .rst(rst),
                .in_valid(neuron_delay_valid),
                .current_integer($signed(neuron_currents_delayed[
                    neuron_index*CURRENT_BITS +: CURRENT_BITS
                ])),
                .previous_membrane_integer(use_previous_state_register
                    ? $signed(state_read_data_flat[
                        neuron_index*STATE_BITS +: STATE_BITS
                    ])
                    : {STATE_BITS{1'b0}}),
                .fraction_bits(fraction_registers[neuron_index]),
                .out_valid(lif_valid_lanes[neuron_index]),
                .spike_integer(lif_spikes_flat[
                    neuron_index*SPIKE_BITS +: SPIKE_BITS
                ]),
                .membrane_integer(lif_membranes_flat[
                    neuron_index*STATE_BITS +: STATE_BITS
                ]),
                .range_error(lif_range_errors[neuron_index])
            );
        end
    endgenerate

    reg [TAG_BITS-1:0] lif_tag_delayed;
    reg [CHANNEL_BITS-1:0] lif_channel_base_delayed;
    reg [OUTPUT_PES*CURRENT_BITS-1:0] lif_currents_delayed;
    always @(posedge clk) begin
        if (rst) begin
            lif_tag_delayed <= 0;
            lif_channel_base_delayed <= 0;
            lif_currents_delayed <= 0;
        end else if (neuron_delay_valid) begin
            lif_tag_delayed <= neuron_tag_delayed;
            lif_channel_base_delayed <= neuron_channel_base_delayed;
            lif_currents_delayed <= neuron_currents_delayed;
        end
    end

    assign output_valid = &lif_valid_lanes;
    assign output_address = lif_tag_delayed;
    assign output_channel_base = lif_channel_base_delayed;
    assign output_current_flat = lif_currents_delayed;
    assign output_spike_flat = lif_spikes_flat;
    assign output_membrane_flat = lif_membranes_flat;
    assign output_range_error = |lif_range_errors;

    assign state_write_enable = output_valid && write_state_register;
    assign state_write_address = output_address;
    assign state_write_channel_base = output_channel_base;
    assign state_write_data_flat = output_membrane_flat;

    wire last_mac_position = mac_output_tag ==
        ((TAG_BITS'(height_register) << width_log2_register) - 1'b1);
    wire last_neuron_position = output_address ==
        (output_spatial_register - 1'b1);
    wire last_output_tile = output_tile_index ==
        (output_tiles_register - 1'b1);

    always @(posedge clk) begin
        if (rst) begin
            control_state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            width_log2_register <= 0;
            height_register <= 0;
            input_tiles_register <= 0;
            output_tiles_register <= 0;
            output_spatial_register <= 0;
            pool_enable_register <= 1'b0;
            use_previous_state_register <= 1'b0;
            write_state_register <= 1'b0;
            input_tile_index <= 0;
            output_tile_index <= 0;
            weight_tile_base <= 0;
            parameter_tile_base <= 0;
            load_parameters <= 1'b0;
            scan_issue_active <= 1'b0;
            scan_row <= 0;
            scan_column <= 0;
        end else begin
            done <= 1'b0;
            case (control_state)
                STATE_IDLE: begin
                    scan_issue_active <= 1'b0;
                    if (start) begin
                        width_log2_register <= config_width_log2;
                        height_register <= config_height;
                        input_tiles_register <= config_input_tiles;
                        output_tiles_register <= config_output_tiles;
                        output_spatial_register <= config_output_spatial;
                        pool_enable_register <= config_pool_enable;
                        use_previous_state_register
                            <= config_use_previous_state;
                        write_state_register <= config_write_state;
                        input_tile_index <= 0;
                        output_tile_index <= 0;
                        weight_tile_base <= config_weight_beat_base;
                        parameter_tile_base <= config_parameter_base;
                        load_parameters <= 1'b1;
                        busy <= 1'b1;
                        control_state <= STATE_LOAD;
                    end
                end

                STATE_LOAD: begin
                    if ((weight_loaded_count == 5'(WEIGHT_BEATS_PER_TILE))
                            && (!load_parameters
                                || (parameter_loaded_count
                                    == 4'(OUTPUT_PES)))) begin
                        scan_row <= 0;
                        scan_column <= 0;
                        scan_issue_active <= 1'b1;
                        control_state <= STATE_SCAN;
                    end
                end

                STATE_SCAN: begin
                    if (scan_issue_active) begin
                        if (scan_column == last_padded_column) begin
                            scan_column <= 0;
                            if (scan_row == last_padded_row) begin
                                scan_issue_active <= 1'b0;
                            end else begin
                                scan_row <= scan_row + 1'b1;
                            end
                        end else begin
                            scan_column <= scan_column + 1'b1;
                        end
                    end

                    if (mac_output_valid && last_mac_position
                            && !final_input_tile) begin
                        input_tile_index <= input_tile_index + 1'b1;
                        weight_tile_base <= weight_tile_base
                            + WEIGHT_ADDRESS_BITS'(WEIGHT_BEATS_PER_TILE);
                        load_parameters <= 1'b0;
                        scan_issue_active <= 1'b0;
                        control_state <= STATE_LOAD;
                    end

                    if (output_valid && last_neuron_position) begin
                        if (last_output_tile) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            scan_issue_active <= 1'b0;
                            control_state <= STATE_IDLE;
                        end else begin
                            input_tile_index <= 0;
                            output_tile_index <= output_tile_index + 1'b1;
                            weight_tile_base <= weight_tile_base
                                + WEIGHT_ADDRESS_BITS'(WEIGHT_BEATS_PER_TILE);
                            parameter_tile_base <= parameter_tile_base
                                + PARAMETER_ADDRESS_BITS'(OUTPUT_PES);
                            load_parameters <= 1'b1;
                            scan_issue_active <= 1'b0;
                            control_state <= STATE_LOAD;
                        end
                    end
                end

                default: control_state <= STATE_IDLE;
            endcase
        end
    end

    initial begin
        if (OUTPUT_PES != 8 || INPUT_LANES != 4 || KERNEL_TAPS != 9) begin
            $error("packed hardware image requires an 8x4x9 convolution core");
        end
        if (WEIGHT_BITS != 16 || WEIGHT_BEATS_PER_TILE != 18) begin
            $error("packed weight beat geometry is inconsistent");
        end
        if ((1 << TAG_BITS) < MAX_WIDTH*MAX_HEIGHT) begin
            $error("TAG_BITS cannot represent the largest input feature map");
        end
    end
endmodule

`default_nettype wire
