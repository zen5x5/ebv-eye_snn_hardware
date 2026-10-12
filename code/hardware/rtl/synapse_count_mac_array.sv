`timescale 1ns/1ps
`default_nettype none

// Eight output PEs with LANES parallel count-times-weight products per PE.
// Counts are broadcast to every output PE; weights and accumulators are private.
// The accumulator is deliberately wide and never saturates inside the dot product.
module synapse_count_mac_array #(
    parameter integer OUTPUT_PES = 8,
    parameter integer LANES = 18,
    parameter integer WEIGHT_BITS = 16,
    parameter integer COUNT_BITS = 9,
    parameter integer ACCUMULATOR_BITS = 40
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [LANES*COUNT_BITS-1:0]              counts_flat,
    input  wire [OUTPUT_PES*LANES*WEIGHT_BITS-1:0]  weights_flat,
    input  wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_in_flat,
    output reg                                       out_valid,
    output reg  [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_out_flat
);
    localparam integer PRODUCT_BITS = WEIGHT_BITS + COUNT_BITS + 1;
    wire [OUTPUT_PES*LANES*ACCUMULATOR_BITS-1:0] contributions_flat;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] reductions_flat;

    genvar output_index;
    genvar lane_index;
    generate
        for (output_index = 0; output_index < OUTPUT_PES; output_index = output_index + 1) begin : gen_output
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin : gen_lane
                wire signed [WEIGHT_BITS-1:0] weight;
                wire signed [COUNT_BITS:0] positive_count;
                wire signed [PRODUCT_BITS-1:0] product;
                wire signed [ACCUMULATOR_BITS-1:0] extended_product;

                assign weight = weights_flat[
                    (output_index*LANES + lane_index)*WEIGHT_BITS +: WEIGHT_BITS
                ];
                assign positive_count = $signed({1'b0, counts_flat[
                    lane_index*COUNT_BITS +: COUNT_BITS
                ]});
                assign product = weight * positive_count;
                assign extended_product = {
                    {(ACCUMULATOR_BITS-PRODUCT_BITS){product[PRODUCT_BITS-1]}},
                    product
                };
                assign contributions_flat[
                    (output_index*LANES + lane_index)*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                ] = extended_product;
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
        if (ACCUMULATOR_BITS < PRODUCT_BITS) begin
            $error("ACCUMULATOR_BITS must cover a signed count-times-weight product");
        end
    end
endmodule

`default_nettype wire
