`timescale 1ns/1ps
`default_nettype none

// Throughput-oriented count MAC array for the regular convolution core.
// Products are registered and every adder-tree level is pipelined.  The input
// partial sum and caller-defined address tag travel through matching delay
// registers, allowing one independent output position to be issued per cycle.
module pipelined_count_mac_array #(
    parameter integer OUTPUT_PES = 8,
    parameter integer LANES = 36,
    parameter integer WEIGHT_BITS = 16,
    parameter integer COUNT_BITS = 9,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer TAG_BITS = 16
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [TAG_BITS-1:0]                       tag_in,
    input  wire [LANES*COUNT_BITS-1:0]              counts_flat,
    input  wire [OUTPUT_PES*LANES*WEIGHT_BITS-1:0]  weights_flat,
    input  wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_in_flat,
    output reg                                       out_valid,
    output reg  [TAG_BITS-1:0]                       tag_out,
    output reg  [OUTPUT_PES*ACCUMULATOR_BITS-1:0]   accumulators_out_flat
);
    localparam integer PRODUCT_BITS = WEIGHT_BITS + COUNT_BITS + 1;
    localparam integer TREE_LEVELS = (LANES <= 1) ? 0 : $clog2(LANES);
    localparam integer PIPELINE_STAGES = TREE_LEVELS + 1;

    reg signed [ACCUMULATOR_BITS-1:0] tree_pipeline
        [0:OUTPUT_PES-1][0:TREE_LEVELS][0:LANES-1];
    reg [OUTPUT_PES*ACCUMULATOR_BITS-1:0] accumulator_pipeline
        [0:TREE_LEVELS];
    reg [TAG_BITS-1:0] tag_pipeline [0:TREE_LEVELS];
    reg [TREE_LEVELS:0] valid_pipeline;

    genvar output_index;
    genvar lane_index;
    generate
        for (output_index = 0; output_index < OUTPUT_PES; output_index = output_index + 1) begin : gen_products
            for (lane_index = 0; lane_index < LANES; lane_index = lane_index + 1) begin : gen_product_lane
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

                always @(posedge clk) begin
                    if (rst || !in_valid) begin
                        tree_pipeline[output_index][0][lane_index]
                            <= {ACCUMULATOR_BITS{1'b0}};
                    end else begin
                        tree_pipeline[output_index][0][lane_index]
                            <= extended_product;
                    end
                end
            end
        end
    endgenerate

    genvar level_index;
    genvar node_index;
    generate
        for (level_index = 1; level_index <= TREE_LEVELS; level_index = level_index + 1) begin : gen_tree_level
            localparam integer PREVIOUS_ACTIVE =
                (LANES + (1 << (level_index-1)) - 1) >> (level_index-1);
            localparam integer ACTIVE =
                (LANES + (1 << level_index) - 1) >> level_index;
            for (output_index = 0; output_index < OUTPUT_PES; output_index = output_index + 1) begin : gen_tree_output
                for (node_index = 0; node_index < LANES; node_index = node_index + 1) begin : gen_tree_node
                    if (node_index < ACTIVE) begin : gen_active_node
                        if (2*node_index + 1 < PREVIOUS_ACTIVE) begin : gen_pair
                            always @(posedge clk) begin
                                if (rst) begin
                                    tree_pipeline[output_index][level_index][node_index]
                                        <= {ACCUMULATOR_BITS{1'b0}};
                                end else begin
                                    tree_pipeline[output_index][level_index][node_index]
                                        <= tree_pipeline[output_index][level_index-1][2*node_index]
                                        + tree_pipeline[output_index][level_index-1][2*node_index+1];
                                end
                            end
                        end else begin : gen_unpaired
                            always @(posedge clk) begin
                                if (rst) begin
                                    tree_pipeline[output_index][level_index][node_index]
                                        <= {ACCUMULATOR_BITS{1'b0}};
                                end else begin
                                    tree_pipeline[output_index][level_index][node_index]
                                        <= tree_pipeline[output_index][level_index-1][2*node_index];
                                end
                            end
                        end
                    end
                end
            end
        end
    endgenerate

    integer delay_index;
    integer output_counter;
    always @(posedge clk) begin
        if (rst) begin
            valid_pipeline <= {(TREE_LEVELS+1){1'b0}};
            out_valid <= 1'b0;
            tag_out <= {TAG_BITS{1'b0}};
            accumulators_out_flat <= {OUTPUT_PES*ACCUMULATOR_BITS{1'b0}};
            for (delay_index = 0; delay_index <= TREE_LEVELS; delay_index = delay_index + 1) begin
                accumulator_pipeline[delay_index]
                    <= {OUTPUT_PES*ACCUMULATOR_BITS{1'b0}};
                tag_pipeline[delay_index] <= {TAG_BITS{1'b0}};
            end
        end else begin
            valid_pipeline[0] <= in_valid;
            accumulator_pipeline[0] <= accumulators_in_flat;
            tag_pipeline[0] <= tag_in;
            for (delay_index = 1; delay_index <= TREE_LEVELS; delay_index = delay_index + 1) begin
                valid_pipeline[delay_index] <= valid_pipeline[delay_index-1];
                accumulator_pipeline[delay_index]
                    <= accumulator_pipeline[delay_index-1];
                tag_pipeline[delay_index] <= tag_pipeline[delay_index-1];
            end

            out_valid <= valid_pipeline[TREE_LEVELS];
            tag_out <= tag_pipeline[TREE_LEVELS];
            if (valid_pipeline[TREE_LEVELS]) begin
                for (output_counter = 0; output_counter < OUTPUT_PES; output_counter = output_counter + 1) begin
                    accumulators_out_flat[
                        output_counter*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                    ] <= $signed(accumulator_pipeline[TREE_LEVELS][
                        output_counter*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                    ]) + tree_pipeline[output_counter][TREE_LEVELS][0];
                end
            end
        end
    end

    initial begin
        if (OUTPUT_PES <= 0 || LANES <= 0 || TAG_BITS <= 0) begin
            $error("parallelism and tag width must be positive");
        end
        if (ACCUMULATOR_BITS < PRODUCT_BITS) begin
            $error("ACCUMULATOR_BITS must cover a signed count-times-weight product");
        end
        if (PIPELINE_STAGES > 32) begin
            $error("unexpectedly deep reduction pipeline");
        end
    end
endmodule

`default_nettype wire
