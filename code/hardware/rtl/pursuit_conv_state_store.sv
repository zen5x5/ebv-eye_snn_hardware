`timescale 1ns/1ps
`default_nettype none

// Eight-bank S16 membrane store.  A lean build holds L3--L8 (1008 words per
// bank); FULL_STATE_CAPABLE also reserves L1/L2 and uses 5616 words per bank.
module pursuit_conv_state_store #(
    parameter integer FULL_STATE_CAPABLE = 0,
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6,
    parameter integer STATE_BITS = 16
) (
    input  wire                                      clk,
    input  wire                                      read_enable,
    input  wire [12:0]                               layer_bank_base,
    input  wire [TAG_BITS-1:0]                       layer_spatial,
    input  wire [TAG_BITS-1:0]                       read_address,
    input  wire [CHANNEL_BITS-1:0]                   read_channel_base,
    output reg  [8*STATE_BITS-1:0]                   read_data_flat,
    input  wire                                      write_enable,
    input  wire [TAG_BITS-1:0]                       write_address,
    input  wire [CHANNEL_BITS-1:0]                   write_channel_base,
    input  wire [8*STATE_BITS-1:0]                   write_data_flat
);
    localparam integer STATE_DEPTH = (FULL_STATE_CAPABLE != 0) ? 5616 : 1008;
    localparam integer STATE_ADDRESS_BITS = $clog2(STATE_DEPTH);

    function automatic [12:0] group_spatial_offset;
        input [2:0] group;
        input [TAG_BITS-1:0] spatial;
        begin
            case (spatial)
                14'd3072: group_spatial_offset =
                    (13'(group) << 11) + (13'(group) << 10);
                14'd768: group_spatial_offset =
                    (13'(group) << 9) + (13'(group) << 8);
                14'd192: group_spatial_offset =
                    (13'(group) << 7) + (13'(group) << 6);
                14'd48: group_spatial_offset =
                    (13'(group) << 5) + (13'(group) << 4);
                default: group_spatial_offset =
                    (13'(group) << 3) + (13'(group) << 2);
            endcase
        end
    endfunction
    wire [2:0] read_group = read_channel_base[5:3];
    wire [STATE_ADDRESS_BITS-1:0] read_bank_address = STATE_ADDRESS_BITS'(
        layer_bank_base + group_spatial_offset(read_group, layer_spatial)
        + read_address
    );
    wire [2:0] write_group = write_channel_base[5:3];
    wire [STATE_ADDRESS_BITS-1:0] write_bank_address = STATE_ADDRESS_BITS'(
        layer_bank_base + group_spatial_offset(write_group, layer_spatial)
        + write_address
    );

    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : gen_state_bank
            (* ram_style = "block" *) reg signed [STATE_BITS-1:0] memory
                [0:STATE_DEPTH-1];
            always @(posedge clk) begin
                if (read_enable && (read_channel_base[2:0] == 0)) begin
                    read_data_flat[lane*STATE_BITS +: STATE_BITS]
                        <= memory[read_bank_address];
                end
                if (write_enable && (write_channel_base[2:0] == 0)) begin
                    memory[write_bank_address]
                        <= write_data_flat[lane*STATE_BITS +: STATE_BITS];
                end
            end
        end
    endgenerate

    initial begin
        if (FULL_STATE_CAPABLE != 0 && FULL_STATE_CAPABLE != 1) begin
            $error("FULL_STATE_CAPABLE must be zero or one");
        end
    end
endmodule

`default_nettype wire
