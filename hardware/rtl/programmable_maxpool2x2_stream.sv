`timescale 1ns/1ps
`default_nettype none

// Width-programmable signed 2x2 max pool.  Widths must be powers of two.
module programmable_maxpool2x2_stream #(
    parameter integer OUTPUT_PES = 8,
    parameter integer MAX_WIDTH = 128,
    parameter integer CURRENT_BITS = 24,
    parameter integer TAG_BITS = 14,
    parameter integer WIDTH_LOG2_BITS = 4
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [WIDTH_LOG2_BITS-1:0]                width_log2,
    input  wire [TAG_BITS-1:0]                       tag_in,
    input  wire [OUTPUT_PES*CURRENT_BITS-1:0]        currents_flat,
    output reg                                       out_valid,
    output reg  [TAG_BITS-1:0]                       tag_out,
    output reg  [OUTPUT_PES*CURRENT_BITS-1:0]        pooled_currents_flat
);
    localparam integer COLUMN_ADDRESS_BITS = $clog2(MAX_WIDTH);
    localparam integer VECTOR_BITS = OUTPUT_PES * CURRENT_BITS;

    wire [TAG_BITS-1:0] write_column_mask =
        (TAG_BITS'(1) << width_log2) - TAG_BITS'(1);
    wire [COLUMN_ADDRESS_BITS-1:0] write_column =
        COLUMN_ADDRESS_BITS'(tag_in & write_column_mask);

    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] previous_row
        [0:MAX_WIDTH-1];
    reg [VECTOR_BITS-1:0] previous_row_delayed;
    reg [VECTOR_BITS-1:0] currents_delayed;
    reg [TAG_BITS-1:0] tag_delayed;
    reg [WIDTH_LOG2_BITS-1:0] width_log2_delayed;
    reg read_valid;

    wire [TAG_BITS-1:0] delayed_column_mask =
        (TAG_BITS'(1) << width_log2_delayed) - TAG_BITS'(1);
    wire [TAG_BITS-1:0] input_column_wide =
        tag_delayed & delayed_column_mask;
    wire [TAG_BITS-1:0] input_row = tag_delayed >> width_log2_delayed;
    wire [TAG_BITS-1:0] pooled_tag =
        ((input_row >> 1) << (width_log2_delayed - 1'b1))
        | (input_column_wide >> 1);

    reg signed [CURRENT_BITS-1:0] current_left [0:OUTPUT_PES-1];
    reg signed [CURRENT_BITS-1:0] previous_left [0:OUTPUT_PES-1];
    wire signed [CURRENT_BITS-1:0] current_value [0:OUTPUT_PES-1];
    wire signed [CURRENT_BITS-1:0] previous_value [0:OUTPUT_PES-1];
    wire signed [CURRENT_BITS-1:0] maximum_top [0:OUTPUT_PES-1];
    wire signed [CURRENT_BITS-1:0] maximum_bottom [0:OUTPUT_PES-1];
    wire signed [CURRENT_BITS-1:0] maximum_all [0:OUTPUT_PES-1];

    genvar output_index;
    generate
        for (output_index = 0; output_index < OUTPUT_PES;
                output_index = output_index + 1) begin : gen_pool_values
            assign current_value[output_index] = $signed(currents_delayed[
                output_index*CURRENT_BITS +: CURRENT_BITS
            ]);
            assign previous_value[output_index] = $signed(previous_row_delayed[
                output_index*CURRENT_BITS +: CURRENT_BITS
            ]);
            assign maximum_top[output_index] =
                (previous_left[output_index] > previous_value[output_index])
                ? previous_left[output_index] : previous_value[output_index];
            assign maximum_bottom[output_index] =
                (current_left[output_index] > current_value[output_index])
                ? current_left[output_index] : current_value[output_index];
            assign maximum_all[output_index] =
                (maximum_top[output_index] > maximum_bottom[output_index])
                ? maximum_top[output_index] : maximum_bottom[output_index];
        end
    endgenerate

    integer output_counter;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            tag_out <= {TAG_BITS{1'b0}};
            pooled_currents_flat <= {VECTOR_BITS{1'b0}};
            previous_row_delayed <= {VECTOR_BITS{1'b0}};
            currents_delayed <= {VECTOR_BITS{1'b0}};
            tag_delayed <= {TAG_BITS{1'b0}};
            width_log2_delayed <= {WIDTH_LOG2_BITS{1'b0}};
            read_valid <= 1'b0;
            for (output_counter = 0; output_counter < OUTPUT_PES;
                    output_counter = output_counter + 1) begin
                current_left[output_counter] <= {CURRENT_BITS{1'b0}};
                previous_left[output_counter] <= {CURRENT_BITS{1'b0}};
            end
        end else begin
            out_valid <= 1'b0;
            read_valid <= in_valid;
            if (in_valid) begin
                previous_row_delayed <= previous_row[write_column];
                previous_row[write_column] <= currents_flat;
                currents_delayed <= currents_flat;
                tag_delayed <= tag_in;
                width_log2_delayed <= width_log2;
            end
            if (read_valid) begin
                out_valid <= input_row[0] && input_column_wide[0];
                tag_out <= pooled_tag;
                for (output_counter = 0; output_counter < OUTPUT_PES;
                        output_counter = output_counter + 1) begin
                    current_left[output_counter] <= current_value[output_counter];
                    previous_left[output_counter]
                        <= previous_value[output_counter];
                    pooled_currents_flat[
                        output_counter*CURRENT_BITS +: CURRENT_BITS
                    ] <= maximum_all[output_counter];
                end
            end
        end
    end

    initial begin
        if (OUTPUT_PES <= 0 || MAX_WIDTH <= 1) begin
            $error("invalid programmable max-pool parameters");
        end
    end
endmodule

`default_nettype wire
