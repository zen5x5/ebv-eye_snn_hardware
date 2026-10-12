`timescale 1ns/1ps
`default_nettype none

// One binary shift term per enabled lane and cycle.  A scheduler decomposes a
// count n into popcount(n) terms, so this implementation does not hide nine
// parallel adders behind each "multiplier-free" lane.  Independent terms can
// still be distributed over different lanes in the same cycle.
module synapse_shift_add_array #(
    parameter integer OUTPUT_PES = 8,
    parameter integer LANES = 18,
    parameter integer WEIGHT_BITS = 16,
    parameter integer SHIFT_BITS = 4,
    parameter integer ACCUMULATOR_BITS = 40
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [LANES-1:0]                         term_enables,
    input  wire [LANES*SHIFT_BITS-1:0]              shift_amounts_flat,
    input  wire [OUTPUT_PES*LANES*WEIGHT_BITS-1:0]  weights_flat,
    input  wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_in_flat,
    output reg                                       out_valid,
    output reg  [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_out_flat
);
    wire [OUTPUT_PES*LANES*ACCUMULATOR_BITS-1:0] contributions_flat;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] reductions_flat;

    genvar output_index;
    genvar lane_index;
    generate
        for (output_index = 0; output_index < OUTPUT_PES; output_index = output_index + 1) begin : gen_output
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin : gen_lane
                wire signed [WEIGHT_BITS-1:0] weight;
                wire signed [ACCUMULATOR_BITS-1:0] extended_weight;
                wire [SHIFT_BITS-1:0] shift_amount;
                wire signed [ACCUMULATOR_BITS-1:0] shifted_weight;

                assign weight = weights_flat[
                    (output_index*LANES + lane_index)*WEIGHT_BITS +: WEIGHT_BITS
                ];
                assign shift_amount = shift_amounts_flat[
                    lane_index*SHIFT_BITS +: SHIFT_BITS
                ];
                assign extended_weight = {
                    {(ACCUMULATOR_BITS-WEIGHT_BITS){weight[WEIGHT_BITS-1]}},
                    weight
                };
                assign shifted_weight = extended_weight <<< shift_amount;
                assign contributions_flat[
                    (output_index*LANES + lane_index)*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                ] = term_enables[lane_index]
                    ? shifted_weight
                    : {ACCUMULATOR_BITS{1'b0}};
            end
            balanced_signed_adder_tree #(
                .INPUTS(LANES),
                .WIDTH(ACCUMULATOR_BITS)
            ) reduction_tree (
                .values_flat(contributions_flat[
                    output_index*LANES*ACCUMULATOR_BITS +: LANES*ACCUMULATOR_BITS
                ]),
                .sum(reductions_flat[
                    output_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                ])
            );
        end
    endgenerate

    integer output_counter;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            accumulators_out_flat <= {OUTPUT_PES*ACCUMULATOR_BITS{1'b0}};
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                for (output_counter = 0; output_counter < OUTPUT_PES; output_counter = output_counter + 1) begin
                    accumulators_out_flat[
                        output_counter*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                    ] <= $signed(accumulators_in_flat[
                        output_counter*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                    ]) + $signed(reductions_flat[
                        output_counter*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                    ]);
                end
            end
        end
    end

    initial begin
        if (ACCUMULATOR_BITS < WEIGHT_BITS + (1 << SHIFT_BITS) - 1) begin
            $error("ACCUMULATOR_BITS must cover the largest shifted weight");
        end
    end
endmodule

`default_nettype wire
