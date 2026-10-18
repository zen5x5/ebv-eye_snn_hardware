`timescale 1ns/1ps
`default_nettype none

// Four-bank-set activation store for the layer-major two-time-step schedule.
// Each time stream owns an A/B ping-pong buffer.  Eight U8 banks support one
// aligned eight-channel write or one aligned four-channel convolution read.
module pursuit_state_depth_feature_store #(
    parameter integer T0_A_DEPTH = 3072,
    parameter integer T0_B_DEPTH = 1536,
    parameter integer T1_A_DEPTH = 3072,
    parameter integer T1_B_DEPTH = 1536,
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6
) (
    input  wire                      clk,

    input  wire                      conv_read_enable,
    input  wire                      conv_read_time,
    input  wire                      conv_read_buffer,
    input  wire [TAG_BITS-1:0]       conv_read_address,
    input  wire [CHANNEL_BITS-1:0]   conv_read_channel_base,
    input  wire [TAG_BITS-1:0]       conv_read_spatial,
    output wire [35:0]               conv_read_data_flat,

    input  wire                      conv_write_enable,
    input  wire                      conv_write_time,
    input  wire                      conv_write_buffer,
    input  wire [TAG_BITS-1:0]       conv_write_address,
    input  wire [CHANNEL_BITS-1:0]   conv_write_channel_base,
    input  wire [TAG_BITS-1:0]       conv_write_spatial,
    input  wire [63:0]               conv_write_spike_flat,

    input  wire                      dense_read_enable,
    input  wire                      dense_read_time,
    input  wire                      dense_read_buffer,
    input  wire [8:0]                dense_read_flat_address,
    output wire [7:0]                dense_read_data
);
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

    wire [2:0] write_group = conv_write_channel_base[5:3];
    wire [11:0] write_address = 12'(
        group_spatial_offset(write_group, conv_write_spatial)
        + conv_write_address
    );
    wire [2:0] read_group = conv_read_channel_base[5:3];
    wire read_upper_banks = conv_read_channel_base[2];
    wire [11:0] conv_feature_address = 12'(
        group_spatial_offset(read_group, conv_read_spatial)
        + conv_read_address
    );

    wire [5:0] dense_channel = 6'(dense_read_flat_address / 12);
    wire [3:0] dense_position = 4'(dense_read_flat_address % 12);
    wire [2:0] dense_bank = dense_channel[2:0];
    wire [11:0] dense_feature_address = 12'(
        dense_channel[5:3] * 12 + dense_position
    );

    wire memory_read_enable = conv_read_enable || dense_read_enable;
    wire memory_read_time = conv_read_enable ? conv_read_time : dense_read_time;
    wire memory_read_buffer = conv_read_enable ? conv_read_buffer : dense_read_buffer;
    wire [11:0] memory_read_address = conv_read_enable
        ? conv_feature_address : dense_feature_address;
    wire [63:0] read_t0_a_flat;
    wire [63:0] read_t0_b_flat;
    wire [63:0] read_t1_a_flat;
    wire [63:0] read_t1_b_flat;

    reg read_time_delayed;
    reg read_buffer_delayed;
    reg read_upper_banks_delayed;
    reg [2:0] dense_bank_delayed;
    reg read_alignment_delayed;

    wire [63:0] selected_read_flat = read_time_delayed
        ? (read_buffer_delayed ? read_t1_b_flat : read_t1_a_flat)
        : (read_buffer_delayed ? read_t0_b_flat : read_t0_a_flat);
    wire [31:0] selected_conv_flat = read_upper_banks_delayed
        ? selected_read_flat[63:32] : selected_read_flat[31:0];
    wire [35:0] selected_conv_data = {
        1'b0, selected_conv_flat[31:24],
        1'b0, selected_conv_flat[23:16],
        1'b0, selected_conv_flat[15:8],
        1'b0, selected_conv_flat[7:0]
    };
    assign conv_read_data_flat = read_alignment_delayed
        ? selected_conv_data : 36'd0;

    function automatic [7:0] select_bank;
        input [63:0] values;
        input [2:0] bank;
        begin
            case (bank)
                0: select_bank = values[7:0];
                1: select_bank = values[15:8];
                2: select_bank = values[23:16];
                3: select_bank = values[31:24];
                4: select_bank = values[39:32];
                5: select_bank = values[47:40];
                6: select_bank = values[55:48];
                default: select_bank = values[63:56];
            endcase
        end
    endfunction
    assign dense_read_data = select_bank(selected_read_flat, dense_bank_delayed);

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : gen_bank
            (* ram_style = "block" *) reg [7:0] memory_t0_a
                [0:T0_A_DEPTH-1];
            (* ram_style = "block" *) reg [7:0] memory_t0_b
                [0:T0_B_DEPTH-1];
            (* ram_style = "block" *) reg [7:0] memory_t1_a
                [0:T1_A_DEPTH-1];
            (* ram_style = "block" *) reg [7:0] memory_t1_b
                [0:T1_B_DEPTH-1];
            reg [7:0] read_t0_a;
            reg [7:0] read_t0_b;
            reg [7:0] read_t1_a;
            reg [7:0] read_t1_b;
            assign read_t0_a_flat[lane*8 +: 8] = read_t0_a;
            assign read_t0_b_flat[lane*8 +: 8] = read_t0_b;
            assign read_t1_a_flat[lane*8 +: 8] = read_t1_a;
            assign read_t1_b_flat[lane*8 +: 8] = read_t1_b;
            always @(posedge clk) begin
                if (memory_read_enable && !memory_read_time && !memory_read_buffer)
                    read_t0_a <= memory_t0_a[$clog2(T0_A_DEPTH)'(memory_read_address)];
                if (memory_read_enable && !memory_read_time && memory_read_buffer)
                    read_t0_b <= memory_t0_b[$clog2(T0_B_DEPTH)'(memory_read_address)];
                if (memory_read_enable && memory_read_time && !memory_read_buffer)
                    read_t1_a <= memory_t1_a[$clog2(T1_A_DEPTH)'(memory_read_address)];
                if (memory_read_enable && memory_read_time && memory_read_buffer)
                    read_t1_b <= memory_t1_b[$clog2(T1_B_DEPTH)'(memory_read_address)];
                if (conv_write_enable
                        && (conv_write_channel_base[2:0] == 0)) begin
                    if (!conv_write_time && !conv_write_buffer) begin
                        memory_t0_a[$clog2(T0_A_DEPTH)'(write_address)]
                            <= conv_write_spike_flat[lane*8 +: 8];
                    end else if (!conv_write_time && conv_write_buffer) begin
                        memory_t0_b[$clog2(T0_B_DEPTH)'(write_address)]
                            <= conv_write_spike_flat[lane*8 +: 8];
                    end else if (conv_write_time && !conv_write_buffer) begin
                        memory_t1_a[$clog2(T1_A_DEPTH)'(write_address)]
                            <= conv_write_spike_flat[lane*8 +: 8];
                    end else begin
                        memory_t1_b[$clog2(T1_B_DEPTH)'(write_address)]
                            <= conv_write_spike_flat[lane*8 +: 8];
                    end
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (conv_read_enable) begin
            read_time_delayed <= conv_read_time;
            read_buffer_delayed <= conv_read_buffer;
            read_upper_banks_delayed <= read_upper_banks;
            read_alignment_delayed <= conv_read_channel_base[1:0] == 0;
        end else if (dense_read_enable) begin
            read_time_delayed <= dense_read_time;
            read_buffer_delayed <= dense_read_buffer;
            dense_bank_delayed <= dense_bank;
        end
    end

endmodule

`default_nettype wire
