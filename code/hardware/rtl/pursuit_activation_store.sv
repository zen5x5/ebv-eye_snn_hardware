`timescale 1ns/1ps
`default_nettype none

// Banked activation storage for the fixed pursuit topology.
// source 0 is the two-channel U9 event frame; sources 1/2 are U8 ping/pong
// feature buffers.  Eight feature banks allow an aligned eight-channel write
// and an aligned four-channel convolution read every cycle.
module pursuit_activation_store #(
    parameter integer MAX_INPUT_SPATIAL = 12288,
    parameter integer MAX_FEATURE_BANK_DEPTH = 3072,
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6
) (
    input  wire                      clk,
    input  wire                      host_input_write_enable,
    input  wire                      host_input_write_channel,
    input  wire [TAG_BITS-1:0]       host_input_write_address,
    input  wire [8:0]                host_input_write_data,

    input  wire                      conv_read_enable,
    input  wire [1:0]                conv_read_source,
    input  wire [TAG_BITS-1:0]       conv_read_address,
    input  wire [CHANNEL_BITS-1:0]   conv_read_channel_base,
    input  wire [TAG_BITS-1:0]       conv_read_spatial,
    output wire [35:0]               conv_read_data_flat,

    input  wire                      conv_write_enable,
    input  wire                      conv_write_buffer,
    input  wire [TAG_BITS-1:0]       conv_write_address,
    input  wire [CHANNEL_BITS-1:0]   conv_write_channel_base,
    input  wire [TAG_BITS-1:0]       conv_write_spatial,
    input  wire [63:0]               conv_write_spike_flat,

    input  wire                      dense_read_enable,
    input  wire                      dense_read_buffer,
    input  wire [8:0]                dense_read_flat_address,
    output wire [7:0]                dense_read_data
);
    (* ram_style = "block" *) reg [8:0] input_memory_0
        [0:MAX_INPUT_SPATIAL-1];
    (* ram_style = "block" *) reg [8:0] input_memory_1
        [0:MAX_INPUT_SPATIAL-1];
    function automatic [11:0] group_spatial_offset;
        input [2:0] group;
        input [TAG_BITS-1:0] spatial;
        begin
            case (spatial)
                14'd3072: group_spatial_offset =
                    (12'(group) << 11) + (12'(group) << 10);
                14'd768: group_spatial_offset =
                    (12'(group) << 9) + (12'(group) << 8);
                14'd192: group_spatial_offset =
                    (12'(group) << 7) + (12'(group) << 6);
                14'd48: group_spatial_offset =
                    (12'(group) << 5) + (12'(group) << 4);
                default: group_spatial_offset =
                    (12'(group) << 3) + (12'(group) << 2);
            endcase
        end
    endfunction

    wire [2:0] conv_write_group = conv_write_channel_base[5:3];
    wire [11:0] conv_write_feature_address = 12'(
        group_spatial_offset(conv_write_group, conv_write_spatial)
        + conv_write_address
    );
    wire [2:0] conv_read_group = conv_read_channel_base[5:3];
    wire conv_read_upper_banks = conv_read_channel_base[2];
    wire [11:0] conv_read_feature_address = 12'(
        group_spatial_offset(conv_read_group, conv_read_spatial)
        + conv_read_address
    );
    wire [5:0] dense_channel = 6'(dense_read_flat_address / 12);
    wire [3:0] dense_position = 4'(dense_read_flat_address % 12);
    wire [2:0] dense_bank = dense_channel[2:0];
    wire [11:0] dense_feature_address = 12'(
        dense_channel[5:3] * 12 + dense_position
    );
    wire conv_feature_read_enable = conv_read_enable
        && (conv_read_source != 0);
    wire feature_read_enable = conv_feature_read_enable || dense_read_enable;
    wire [11:0] feature_read_address = conv_feature_read_enable
        ? conv_read_feature_address : dense_feature_address;
    wire [63:0] feature_a_read_flat;
    wire [63:0] feature_b_read_flat;

    reg [1:0] conv_read_source_delayed;
    reg conv_read_upper_banks_delayed;
    reg conv_input_base_zero_delayed;
    reg [8:0] input_memory_0_read;
    reg [8:0] input_memory_1_read;
    reg [2:0] dense_bank_delayed;
    reg dense_buffer_delayed;

    wire [31:0] conv_feature_a_selected = conv_read_upper_banks_delayed
        ? feature_a_read_flat[63:32] : feature_a_read_flat[31:0];
    wire [31:0] conv_feature_b_selected = conv_read_upper_banks_delayed
        ? feature_b_read_flat[63:32] : feature_b_read_flat[31:0];
    wire [31:0] conv_feature_selected = (conv_read_source_delayed == 1)
        ? conv_feature_a_selected : conv_feature_b_selected;
    assign conv_read_data_flat = (conv_read_source_delayed == 0)
        ? (conv_input_base_zero_delayed
            ? {18'd0, input_memory_1_read, input_memory_0_read}
            : 36'd0)
        : {
            1'b0, conv_feature_selected[31:24],
            1'b0, conv_feature_selected[23:16],
            1'b0, conv_feature_selected[15:8],
            1'b0, conv_feature_selected[7:0]
        };

    function automatic [7:0] select_dense_bank;
        input [63:0] values;
        input [2:0] bank;
        begin
            case (bank)
                0: select_dense_bank = values[7:0];
                1: select_dense_bank = values[15:8];
                2: select_dense_bank = values[23:16];
                3: select_dense_bank = values[31:24];
                4: select_dense_bank = values[39:32];
                5: select_dense_bank = values[47:40];
                6: select_dense_bank = values[55:48];
                default: select_dense_bank = values[63:56];
            endcase
        end
    endfunction
    assign dense_read_data = dense_buffer_delayed
        ? select_dense_bank(feature_b_read_flat, dense_bank_delayed)
        : select_dense_bank(feature_a_read_flat, dense_bank_delayed);

    genvar feature_lane;
    generate
        for (feature_lane = 0; feature_lane < 8;
                feature_lane = feature_lane + 1) begin : gen_feature_bank
            (* ram_style = "block" *) reg [7:0] memory_a
                [0:MAX_FEATURE_BANK_DEPTH-1];
            (* ram_style = "block" *) reg [7:0] memory_b
                [0:MAX_FEATURE_BANK_DEPTH-1];
            reg [7:0] read_a;
            reg [7:0] read_b;
            assign feature_a_read_flat[feature_lane*8 +: 8] = read_a;
            assign feature_b_read_flat[feature_lane*8 +: 8] = read_b;
            always @(posedge clk) begin
                if (feature_read_enable) begin
                    read_a <= memory_a[feature_read_address];
                    read_b <= memory_b[feature_read_address];
                end
                if (conv_write_enable
                        && (conv_write_channel_base[2:0] == 0)) begin
                    if (conv_write_buffer) begin
                        memory_b[conv_write_feature_address]
                            <= conv_write_spike_flat[
                                feature_lane*8 +: 8
                            ];
                    end else begin
                        memory_a[conv_write_feature_address]
                            <= conv_write_spike_flat[
                                feature_lane*8 +: 8
                            ];
                    end
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (host_input_write_enable) begin
            if (host_input_write_channel) begin
                input_memory_1[host_input_write_address]
                    <= host_input_write_data;
            end else begin
                input_memory_0[host_input_write_address]
                    <= host_input_write_data;
            end
        end

        if (conv_read_enable) begin
            conv_read_source_delayed <= conv_read_source;
            conv_read_upper_banks_delayed <= conv_read_upper_banks;
            conv_input_base_zero_delayed <= conv_read_channel_base == 0;
            if (conv_read_source == 0) begin
                input_memory_0_read <= input_memory_0[conv_read_address];
                input_memory_1_read <= input_memory_1[conv_read_address];
            end
        end

        if (dense_read_enable) begin
            dense_bank_delayed <= dense_bank;
            dense_buffer_delayed <= dense_read_buffer;
        end
    end

    initial begin
        if (MAX_INPUT_SPATIAL < 12288 || MAX_FEATURE_BANK_DEPTH < 3072) begin
            $error("pursuit activation memories are undersized");
        end
    end
endmodule

`default_nettype wire
