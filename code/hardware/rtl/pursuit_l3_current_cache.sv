`timescale 1ns/1ps
`default_nettype none

// 24 channels x 12 x 16 positions x S24 = 13.5 KiB, arranged as eight banks.
module pursuit_l3_current_cache #(
    parameter integer TAG_BITS = 14,
    parameter integer CHANNEL_BITS = 6,
    parameter integer CURRENT_BITS = 24
) (
    input  wire                                      clk,
    input  wire                                      write_enable,
    input  wire [TAG_BITS-1:0]                       write_address,
    input  wire [CHANNEL_BITS-1:0]                   write_channel_base,
    input  wire [8*CURRENT_BITS-1:0]                 write_current_flat,
    input  wire                                      read_enable,
    input  wire [TAG_BITS-1:0]                       read_address,
    input  wire [CHANNEL_BITS-1:0]                   read_channel_base,
    output reg  [8*CURRENT_BITS-1:0]                 read_current_flat
);
    localparam integer L3_SPATIAL = 192;
    localparam integer BANK_DEPTH = 576;
    wire [9:0] write_bank_address = 10'(
        write_channel_base[5:3] * L3_SPATIAL + write_address
    );
    wire [9:0] read_bank_address = 10'(
        read_channel_base[5:3] * L3_SPATIAL + read_address
    );
    genvar lane;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : gen_current_bank
            (* ram_style = "block" *) reg signed [CURRENT_BITS-1:0] memory
                [0:BANK_DEPTH-1];
            always @(posedge clk) begin
                if (write_enable && (write_channel_base[2:0] == 0)) begin
                    memory[write_bank_address]
                        <= write_current_flat[
                            lane*CURRENT_BITS +: CURRENT_BITS
                        ];
                end
                if (read_enable && (read_channel_base[2:0] == 0)) begin
                    read_current_flat[lane*CURRENT_BITS +: CURRENT_BITS]
                        <= memory[read_bank_address];
                end
            end
        end
    endgenerate
endmodule

`default_nettype wire
