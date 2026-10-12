`timescale 1ns/1ps
`default_nettype none

// One-pixel-per-cycle signed 2x2 max pool for raster-ordered vectors.
// All OUTPUT_PES channels for one spatial coordinate arrive together.
module maxpool2x2_stream #(
    parameter integer OUTPUT_PES = 8,
    parameter integer WIDTH = 128,
    parameter integer HEIGHT = 96,
    parameter integer CURRENT_BITS = 24,
    parameter integer INPUT_TAG_BITS = 14,
    parameter integer OUTPUT_TAG_BITS = 12
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [INPUT_TAG_BITS-1:0]                 tag_in,
    input  wire [OUTPUT_PES*CURRENT_BITS-1:0]        currents_flat,
    output reg                                       out_valid,
    output reg  [OUTPUT_TAG_BITS-1:0]                tag_out,
    output reg  [OUTPUT_PES*CURRENT_BITS-1:0]        pooled_currents_flat
);
    localparam integer COLUMN_BITS = $clog2(WIDTH);
    localparam integer ROW_BITS = INPUT_TAG_BITS - COLUMN_BITS;
    localparam integer VECTOR_BITS = OUTPUT_PES * CURRENT_BITS;

    wire [COLUMN_BITS-1:0] write_column = tag_in[COLUMN_BITS-1:0];

    // A single wide synchronous read-first memory gives one read and one write
    // per raster position.  Packing all channels together avoids eight narrow
    // BRAM banks and preserves one vector per cycle throughput.
    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] previous_row
        [0:WIDTH-1];
    reg [VECTOR_BITS-1:0] previous_row_delayed;
    reg [VECTOR_BITS-1:0] currents_delayed;
    reg [INPUT_TAG_BITS-1:0] tag_delayed;
    reg read_valid;

    wire [COLUMN_BITS-1:0] input_column =
        tag_delayed[COLUMN_BITS-1:0];
    wire [ROW_BITS-1:0] input_row =
        tag_delayed[INPUT_TAG_BITS-1:COLUMN_BITS];
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
            assign previous_value[output_index] =
                $signed(previous_row_delayed[
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
            tag_out <= {OUTPUT_TAG_BITS{1'b0}};
            pooled_currents_flat <= {OUTPUT_PES*CURRENT_BITS{1'b0}};
            previous_row_delayed <= {VECTOR_BITS{1'b0}};
            currents_delayed <= {VECTOR_BITS{1'b0}};
            tag_delayed <= {INPUT_TAG_BITS{1'b0}};
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
            end
            if (read_valid) begin
                out_valid <= input_row[0] && input_column[0];
                tag_out <= {
                    input_row[ROW_BITS-1:1],
                    input_column[COLUMN_BITS-1:1]
                };
                for (output_counter = 0; output_counter < OUTPUT_PES;
                        output_counter = output_counter + 1) begin
                    current_left[output_counter] <= current_value[output_counter];
                    previous_left[output_counter] <= previous_value[output_counter];
                    pooled_currents_flat[
                        output_counter*CURRENT_BITS +: CURRENT_BITS
                    ] <= maximum_all[output_counter];
                end
            end
        end
    end

    initial begin
        if (OUTPUT_PES <= 0 || WIDTH <= 0 || HEIGHT <= 0) begin
            $error("invalid max-pool parameters");
        end
        if ((WIDTH % 2) != 0 || (HEIGHT % 2) != 0) begin
            $error("2x2 pooling requires even spatial dimensions");
        end
        if ((1 << COLUMN_BITS) != WIDTH) begin
            $error("current raster tag decoder requires power-of-two WIDTH");
        end
        if ((1 << OUTPUT_TAG_BITS) < (WIDTH/2)*(HEIGHT/2)) begin
            $error("OUTPUT_TAG_BITS cannot represent pooled coordinates");
        end
        if (OUTPUT_TAG_BITS != (ROW_BITS-1) + (COLUMN_BITS-1)) begin
            $error("pooled raster tag width does not match packed row/column");
        end
    end
endmodule

`default_nettype wire
