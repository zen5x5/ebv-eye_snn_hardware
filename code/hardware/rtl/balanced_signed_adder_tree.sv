`timescale 1ns/1ps
`default_nettype none

// Combinational balanced tree used by the arithmetic microkernels. The first
// full convolution core may insert registers between tree levels after Vivado
// timing identifies the required pipeline depth.
module balanced_signed_adder_tree #(
    parameter integer INPUTS = 2,
    parameter integer WIDTH = 40
) (
    input  wire [INPUTS*WIDTH-1:0] values_flat,
    output wire signed [WIDTH-1:0] sum
);
    generate
        if (INPUTS == 1) begin : gen_leaf
            assign sum = $signed(values_flat[0 +: WIDTH]);
        end else begin : gen_branch
            localparam integer LEFT_INPUTS = INPUTS / 2;
            localparam integer RIGHT_INPUTS = INPUTS - LEFT_INPUTS;
            wire signed [WIDTH-1:0] left_sum;
            wire signed [WIDTH-1:0] right_sum;
            balanced_signed_adder_tree #(
                .INPUTS(LEFT_INPUTS),
                .WIDTH(WIDTH)
            ) left_tree (
                .values_flat(values_flat[0 +: LEFT_INPUTS*WIDTH]),
                .sum(left_sum)
            );
            balanced_signed_adder_tree #(
                .INPUTS(RIGHT_INPUTS),
                .WIDTH(WIDTH)
            ) right_tree (
                .values_flat(
                    values_flat[LEFT_INPUTS*WIDTH +: RIGHT_INPUTS*WIDTH]
                ),
                .sum(right_sum)
            );
            assign sum = left_sum + right_sum;
        end
    endgenerate

    initial begin
        if (INPUTS <= 0 || WIDTH <= 0) begin
            $error("INPUTS and WIDTH must be positive");
        end
    end
endmodule

`default_nettype wire
