`timescale 1ns/1ps
`default_nettype none

// Replays cached pooled currents through the stateful LIF stage.  SL-HYB uses
// this engine for L3 at t1, avoiding repeated L1/L2/L3 convolution while
// preserving the exact L3 membrane transition.
module cached_current_lif_replay #(
    parameter integer OUTPUT_PES = 8,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6,
    parameter integer PARAMETER_ADDRESS_BITS = 9,
    parameter integer FRACTION_BITS_WIDTH = 5
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output reg                                       busy,
    output reg                                       done,
    input  wire [3:0]                                config_output_tiles,
    input  wire [TAG_BITS-1:0]                       config_output_spatial,
    input  wire [PARAMETER_ADDRESS_BITS-1:0]         config_parameter_base,
    output wire                                      parameter_read_enable,
    output wire [PARAMETER_ADDRESS_BITS-1:0]         parameter_read_address,
    input  wire [FRACTION_BITS_WIDTH-1:0]            parameter_fraction_data,
    output wire                                      cache_read_enable,
    output wire [TAG_BITS-1:0]                       cache_read_address,
    output wire [CHANNEL_BITS-1:0]                   cache_read_channel_base,
    input  wire [OUTPUT_PES*CURRENT_BITS-1:0]        cache_read_current_flat,
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
    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_PARAMETERS = 2'd1;
    localparam [1:0] STATE_REPLAY = 2'd2;
    localparam [1:0] STATE_DRAIN = 2'd3;

    reg [1:0] control_state;
    reg [3:0] output_tiles_register;
    reg [TAG_BITS-1:0] output_spatial_register;
    reg [PARAMETER_ADDRESS_BITS-1:0] parameter_group_base;
    reg [3:0] output_tile_index;
    reg [3:0] parameter_issue_count;
    reg [3:0] parameter_loaded_count;
    reg [2:0] parameter_response_index;
    reg parameter_response_valid;
    reg [FRACTION_BITS_WIDTH-1:0] fraction_registers [0:OUTPUT_PES-1];

    assign parameter_read_enable = (control_state == STATE_PARAMETERS)
        && (parameter_issue_count < 4'(OUTPUT_PES));
    assign parameter_read_address = parameter_group_base
        + PARAMETER_ADDRESS_BITS'(parameter_issue_count);

    reg [TAG_BITS-1:0] issue_address;
    assign cache_read_enable = control_state == STATE_REPLAY;
    assign cache_read_address = issue_address;
    assign cache_read_channel_base =
        CHANNEL_BITS'(output_tile_index * OUTPUT_PES);

    reg cache_response_valid;
    reg [TAG_BITS-1:0] cache_response_address;
    reg [CHANNEL_BITS-1:0] cache_response_channel_base;
    reg [OUTPUT_PES*CURRENT_BITS-1:0] current_wait_register;
    reg [TAG_BITS-1:0] state_wait_address;
    reg [CHANNEL_BITS-1:0] state_wait_channel_base;
    reg state_response_valid;

    assign state_read_enable = cache_response_valid;
    assign state_read_address = cache_response_address;
    assign state_read_channel_base = cache_response_channel_base;

    always @(posedge clk) begin
        if (rst) begin
            parameter_issue_count <= 0;
            parameter_loaded_count <= 0;
            parameter_response_index <= 0;
            parameter_response_valid <= 1'b0;
            cache_response_valid <= 1'b0;
            cache_response_address <= 0;
            cache_response_channel_base <= 0;
            current_wait_register <= 0;
            state_wait_address <= 0;
            state_wait_channel_base <= 0;
            state_response_valid <= 1'b0;
        end else begin
            parameter_response_valid <= parameter_read_enable;
            if (parameter_read_enable) begin
                parameter_response_index <= parameter_issue_count[2:0];
                parameter_issue_count <= parameter_issue_count + 1'b1;
            end
            if (parameter_response_valid) begin
                fraction_registers[parameter_response_index]
                    <= parameter_fraction_data;
                parameter_loaded_count <= parameter_loaded_count + 1'b1;
            end
            if (control_state != STATE_PARAMETERS) begin
                parameter_issue_count <= 0;
                parameter_loaded_count <= 0;
                parameter_response_valid <= 1'b0;
            end

            cache_response_valid <= cache_read_enable;
            if (cache_read_enable) begin
                cache_response_address <= issue_address;
                cache_response_channel_base <=
                    CHANNEL_BITS'(output_tile_index * OUTPUT_PES);
            end
            state_response_valid <= cache_response_valid;
            if (cache_response_valid) begin
                current_wait_register <= cache_read_current_flat;
                state_wait_address <= cache_response_address;
                state_wait_channel_base <= cache_response_channel_base;
            end
        end
    end

    wire [OUTPUT_PES-1:0] lif_valid_lanes;
    wire [OUTPUT_PES-1:0] lif_range_errors;
    wire [OUTPUT_PES*SPIKE_BITS-1:0] lif_spikes_flat;
    wire [OUTPUT_PES*STATE_BITS-1:0] lif_membranes_flat;
    genvar neuron_lane;
    generate
        for (neuron_lane = 0; neuron_lane < OUTPUT_PES;
                neuron_lane = neuron_lane + 1) begin : gen_cached_lif
            lif_multispike #(
                .CURRENT_BITS(CURRENT_BITS),
                .STATE_BITS(STATE_BITS),
                .SPIKE_BITS(SPIKE_BITS),
                .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
            ) neuron (
                .clk(clk),
                .rst(rst),
                .in_valid(state_response_valid),
                .current_integer($signed(current_wait_register[
                    neuron_lane*CURRENT_BITS +: CURRENT_BITS
                ])),
                .previous_membrane_integer($signed(state_read_data_flat[
                    neuron_lane*STATE_BITS +: STATE_BITS
                ])),
                .fraction_bits(fraction_registers[neuron_lane]),
                .out_valid(lif_valid_lanes[neuron_lane]),
                .spike_integer(lif_spikes_flat[
                    neuron_lane*SPIKE_BITS +: SPIKE_BITS
                ]),
                .membrane_integer(lif_membranes_flat[
                    neuron_lane*STATE_BITS +: STATE_BITS
                ]),
                .range_error(lif_range_errors[neuron_lane])
            );
        end
    endgenerate

    reg [TAG_BITS-1:0] output_address_register;
    reg [CHANNEL_BITS-1:0] output_channel_base_register;
    reg [OUTPUT_PES*CURRENT_BITS-1:0] output_current_register;
    always @(posedge clk) begin
        if (rst) begin
            output_address_register <= 0;
            output_channel_base_register <= 0;
            output_current_register <= 0;
        end else if (state_response_valid) begin
            output_address_register <= state_wait_address;
            output_channel_base_register <= state_wait_channel_base;
            output_current_register <= current_wait_register;
        end
    end

    assign output_valid = &lif_valid_lanes;
    assign output_address = output_address_register;
    assign output_channel_base = output_channel_base_register;
    assign output_current_flat = output_current_register;
    assign output_spike_flat = lif_spikes_flat;
    assign output_membrane_flat = lif_membranes_flat;
    assign output_range_error = |lif_range_errors;
    assign state_write_enable = output_valid;
    assign state_write_address = output_address;
    assign state_write_channel_base = output_channel_base;
    assign state_write_data_flat = output_membrane_flat;

    wire last_output = output_valid
        && (output_address == output_spatial_register - 1'b1);
    wire last_tile = output_tile_index == output_tiles_register - 1'b1;
    always @(posedge clk) begin
        if (rst) begin
            control_state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            output_tiles_register <= 0;
            output_spatial_register <= 0;
            parameter_group_base <= 0;
            output_tile_index <= 0;
            issue_address <= 0;
        end else begin
            done <= 1'b0;
            case (control_state)
                STATE_IDLE: begin
                    if (start) begin
                        output_tiles_register <= config_output_tiles;
                        output_spatial_register <= config_output_spatial;
                        parameter_group_base <= config_parameter_base;
                        output_tile_index <= 0;
                        issue_address <= 0;
                        busy <= 1'b1;
                        control_state <= STATE_PARAMETERS;
                    end
                end
                STATE_PARAMETERS: begin
                    if (parameter_loaded_count == 4'(OUTPUT_PES)) begin
                        issue_address <= 0;
                        control_state <= STATE_REPLAY;
                    end
                end
                STATE_REPLAY: begin
                    if (issue_address == output_spatial_register - 1'b1) begin
                        control_state <= STATE_DRAIN;
                    end else begin
                        issue_address <= issue_address + 1'b1;
                    end
                end
                STATE_DRAIN: begin
                    if (last_output) begin
                        if (last_tile) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            control_state <= STATE_IDLE;
                        end else begin
                            output_tile_index <= output_tile_index + 1'b1;
                            parameter_group_base <= parameter_group_base
                                + PARAMETER_ADDRESS_BITS'(OUTPUT_PES);
                            control_state <= STATE_PARAMETERS;
                        end
                    end
                end
                default: control_state <= STATE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
