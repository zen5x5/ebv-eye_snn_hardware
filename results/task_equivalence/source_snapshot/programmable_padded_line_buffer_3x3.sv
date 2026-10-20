`timescale 1ns/1ps
`default_nettype none

// Reusable 3x3 line buffer for power-of-two feature-map widths up to MAX_WIDTH.
// The controller streams an explicitly zero-padded raster.  All CHANNELS input
// channels arrive together, and the two delayed rows use synchronous BRAM.
module programmable_padded_line_buffer_3x3 #(
    parameter integer CHANNELS = 4,
    parameter integer MAX_WIDTH = 128,
    parameter integer MAX_HEIGHT = 96,
    parameter integer PIXEL_BITS = 9,
    parameter integer TAG_BITS = 14,
    parameter integer WIDTH_LOG2_BITS = 4
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [WIDTH_LOG2_BITS-1:0]                width_log2,
    input  wire [$clog2(MAX_HEIGHT+2)-1:0]           padded_row,
    input  wire [$clog2(MAX_WIDTH+2)-1:0]            padded_column,
    input  wire [CHANNELS*PIXEL_BITS-1:0]            pixels_flat,
    output reg                                       out_valid,
    output reg  [TAG_BITS-1:0]                       tag_out,
    output reg  [CHANNELS*9*PIXEL_BITS-1:0]          window_flat
);
    localparam integer PADDED_WIDTH = MAX_WIDTH + 2;
    localparam integer VECTOR_BITS = CHANNELS * PIXEL_BITS;
    localparam integer PADDED_ROW_BITS = $clog2(MAX_HEIGHT + 2);
    localparam integer PADDED_COLUMN_BITS = $clog2(MAX_WIDTH + 2);

    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] one_row_memory
        [0:PADDED_WIDTH-1];
    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] two_row_memory
        [0:PADDED_WIDTH-1];
    reg [VECTOR_BITS-1:0] one_row_read;
    reg [VECTOR_BITS-1:0] two_row_read;
    reg [VECTOR_BITS-1:0] pixels_delayed;
    reg [PADDED_ROW_BITS-1:0] row_delayed;
    reg [PADDED_COLUMN_BITS-1:0] column_delayed;
    reg [WIDTH_LOG2_BITS-1:0] width_log2_delayed;
    reg read_valid;

    reg [VECTOR_BITS-1:0] top_left;
    reg [VECTOR_BITS-1:0] top_middle;
    reg [VECTOR_BITS-1:0] middle_left;
    reg [VECTOR_BITS-1:0] middle_middle;
    reg [VECTOR_BITS-1:0] bottom_left;
    reg [VECTOR_BITS-1:0] bottom_middle;

    wire [TAG_BITS-1:0] unpadded_row =
        TAG_BITS'(row_delayed) - TAG_BITS'(2);
    wire [TAG_BITS-1:0] unpadded_column =
        TAG_BITS'(column_delayed) - TAG_BITS'(2);
    wire [TAG_BITS-1:0] calculated_tag =
        (unpadded_row << width_log2_delayed) | unpadded_column;

    integer channel_index;
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            tag_out <= {TAG_BITS{1'b0}};
            window_flat <= {CHANNELS*9*PIXEL_BITS{1'b0}};
            one_row_read <= {VECTOR_BITS{1'b0}};
            two_row_read <= {VECTOR_BITS{1'b0}};
            pixels_delayed <= {VECTOR_BITS{1'b0}};
            row_delayed <= {PADDED_ROW_BITS{1'b0}};
            column_delayed <= {PADDED_COLUMN_BITS{1'b0}};
            width_log2_delayed <= {WIDTH_LOG2_BITS{1'b0}};
            read_valid <= 1'b0;
            top_left <= {VECTOR_BITS{1'b0}};
            top_middle <= {VECTOR_BITS{1'b0}};
            middle_left <= {VECTOR_BITS{1'b0}};
            middle_middle <= {VECTOR_BITS{1'b0}};
            bottom_left <= {VECTOR_BITS{1'b0}};
            bottom_middle <= {VECTOR_BITS{1'b0}};
        end else begin
            out_valid <= 1'b0;
            read_valid <= in_valid;
            if (in_valid) begin
                one_row_read <= one_row_memory[padded_column];
                two_row_read <= two_row_memory[padded_column];
                one_row_memory[padded_column] <= pixels_flat;
                pixels_delayed <= pixels_flat;
                row_delayed <= padded_row;
                column_delayed <= padded_column;
                width_log2_delayed <= width_log2;
            end
            if (read_valid) begin
                two_row_memory[column_delayed] <= one_row_read;
                out_valid <= (row_delayed >= 2) && (column_delayed >= 2);
                tag_out <= calculated_tag;
                for (channel_index = 0; channel_index < CHANNELS;
                        channel_index = channel_index + 1) begin
                    window_flat[(channel_index*9 + 0)*PIXEL_BITS +: PIXEL_BITS]
                        <= top_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 1)*PIXEL_BITS +: PIXEL_BITS]
                        <= top_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 2)*PIXEL_BITS +: PIXEL_BITS]
                        <= two_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 3)*PIXEL_BITS +: PIXEL_BITS]
                        <= middle_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 4)*PIXEL_BITS +: PIXEL_BITS]
                        <= middle_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 5)*PIXEL_BITS +: PIXEL_BITS]
                        <= one_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 6)*PIXEL_BITS +: PIXEL_BITS]
                        <= bottom_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 7)*PIXEL_BITS +: PIXEL_BITS]
                        <= bottom_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    window_flat[(channel_index*9 + 8)*PIXEL_BITS +: PIXEL_BITS]
                        <= pixels_delayed[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];

                    if (column_delayed == 0) begin
                        top_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= {PIXEL_BITS{1'b0}};
                        top_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= two_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        middle_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= {PIXEL_BITS{1'b0}};
                        middle_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= one_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        bottom_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= {PIXEL_BITS{1'b0}};
                        bottom_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= pixels_delayed[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    end else begin
                        top_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= top_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        top_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= two_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        middle_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= middle_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        middle_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= one_row_read[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        bottom_left[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= bottom_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                        bottom_middle[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ] <= pixels_delayed[
                            channel_index*PIXEL_BITS +: PIXEL_BITS
                        ];
                    end
                end
            end
        end
    end

    initial begin
        if (CHANNELS <= 0 || MAX_WIDTH <= 0 || MAX_HEIGHT <= 0) begin
            $error("invalid programmable line-buffer parameters");
        end
        if ((1 << TAG_BITS) < MAX_WIDTH*MAX_HEIGHT) begin
            $error("TAG_BITS cannot represent the largest feature map");
        end
    end
endmodule

`default_nettype wire
