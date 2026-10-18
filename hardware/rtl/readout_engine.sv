`timescale 1ns/1ps
`default_nettype none

// Two-output integer readout.  The source is the U9 sum of the two H2 spike
// vectors.  Outputs are the exact S64 dot products before division by
// 2^(readout_fraction_bits+1); software or a lightweight formatter can apply
// that final power-of-two normalization without changing the regression value.
module readout_engine #(
    parameter integer INPUT_BITS = 9,
    parameter integer WEIGHT_BITS = 16,
    parameter integer OUTPUTS = 2,
    parameter integer OUTPUT_ACCUMULATOR_BITS = 64,
    parameter integer INPUT_ADDRESS_BITS = 4,
    parameter integer WEIGHT_ADDRESS_BITS = 11
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output reg                                       busy,
    output reg                                       done,
    input  wire [INPUT_ADDRESS_BITS:0]               config_input_features,
    input  wire [WEIGHT_ADDRESS_BITS-1:0]            config_weight_beat_base,
    output wire                                      input_read_enable,
    output wire [INPUT_ADDRESS_BITS-1:0]             input_read_address,
    input  wire [INPUT_BITS-1:0]                     input_read_data,
    output wire                                      weight_read_enable,
    output wire [WEIGHT_ADDRESS_BITS-1:0]            weight_read_address,
    input  wire [OUTPUTS*WEIGHT_BITS-1:0]            weight_read_data,
    output reg                                       output_valid,
    output reg  [OUTPUTS*OUTPUT_ACCUMULATOR_BITS-1:0]
                                                    output_integer_flat
);
    reg [INPUT_ADDRESS_BITS:0] input_features_register;
    reg [WEIGHT_ADDRESS_BITS-1:0] weight_base_register;
    reg [INPUT_ADDRESS_BITS:0] issue_count;
    reg response_valid;
    reg [INPUT_ADDRESS_BITS:0] response_index;
    reg signed [OUTPUT_ACCUMULATOR_BITS-1:0] accumulators [0:OUTPUTS-1];

    wire request_valid = busy && (issue_count < input_features_register);
    assign input_read_enable = request_valid;
    assign input_read_address = issue_count[INPUT_ADDRESS_BITS-1:0];
    assign weight_read_enable = request_valid;
    assign weight_read_address = weight_base_register
        + WEIGHT_ADDRESS_BITS'(issue_count);

    wire signed [INPUT_BITS:0] positive_input =
        $signed({1'b0, input_read_data});
    wire [OUTPUTS*OUTPUT_ACCUMULATOR_BITS-1:0] next_accumulators_flat;
    genvar output_lane;
    generate
        for (output_lane = 0; output_lane < OUTPUTS;
                output_lane = output_lane + 1) begin : gen_readout_arithmetic
            wire signed [WEIGHT_BITS-1:0] weight_value;
            wire signed [WEIGHT_BITS+INPUT_BITS:0] product_value;
            wire signed [OUTPUT_ACCUMULATOR_BITS-1:0] product_extended;
            assign weight_value = $signed(weight_read_data[
                output_lane*WEIGHT_BITS +: WEIGHT_BITS
            ]);
            assign product_value = weight_value * positive_input;
            assign product_extended = {
                {(OUTPUT_ACCUMULATOR_BITS-(WEIGHT_BITS+INPUT_BITS+1)){
                    product_value[WEIGHT_BITS+INPUT_BITS]
                }},
                product_value
            };
            assign next_accumulators_flat[
                output_lane*OUTPUT_ACCUMULATOR_BITS
                +: OUTPUT_ACCUMULATOR_BITS
            ] = accumulators[output_lane] + product_extended;
        end
    endgenerate

    integer sequential_output;
    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            done <= 1'b0;
            output_valid <= 1'b0;
            output_integer_flat <= 0;
            input_features_register <= 0;
            weight_base_register <= 0;
            issue_count <= 0;
            response_valid <= 1'b0;
            response_index <= 0;
            for (sequential_output = 0; sequential_output < OUTPUTS;
                    sequential_output = sequential_output + 1) begin
                accumulators[sequential_output] <= 0;
            end
        end else begin
            done <= 1'b0;
            output_valid <= 1'b0;
            response_valid <= request_valid;
            if (request_valid) begin
                response_index <= issue_count;
                issue_count <= issue_count + 1'b1;
            end
            if (response_valid) begin
                for (sequential_output = 0; sequential_output < OUTPUTS;
                        sequential_output = sequential_output + 1) begin
                    accumulators[sequential_output]
                        <= next_accumulators_flat[
                            sequential_output*OUTPUT_ACCUMULATOR_BITS
                            +: OUTPUT_ACCUMULATOR_BITS
                        ];
                end
                if (response_index == input_features_register - 1'b1) begin
                    output_integer_flat <= next_accumulators_flat;
                    output_valid <= 1'b1;
                    busy <= 1'b0;
                    done <= 1'b1;
                end
            end
            if (start && !busy) begin
                input_features_register <= config_input_features;
                weight_base_register <= config_weight_beat_base;
                issue_count <= 0;
                response_valid <= 1'b0;
                busy <= 1'b1;
                for (sequential_output = 0; sequential_output < OUTPUTS;
                        sequential_output = sequential_output + 1) begin
                    accumulators[sequential_output] <= 0;
                end
            end
        end
    end

    initial begin
        if (OUTPUTS != 2 || WEIGHT_BITS != 16) begin
            $error("packed pursuit readout requires two S16 output weights");
        end
    end
endmodule

`default_nettype wire
