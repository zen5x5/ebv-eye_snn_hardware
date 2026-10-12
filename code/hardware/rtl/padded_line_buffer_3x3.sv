`timescale 1ns/1ps
`default_nettype none

// Streaming 3x3 window generator for an explicitly padded raster.
//
// The caller sends one pixel from every channel for each coordinate of a
// (HEIGHT+2) x (WIDTH+2) raster.  Border pixels are zero.  For original
// coordinate (y,x), the corresponding window is emitted when padded
// coordinate (y+2,x+2) arrives.  Windows use channel-major, then row-major
// 3x3 lane order, matching PyTorch OIHW cross-correlation weights.
module padded_line_buffer_3x3 #(
    parameter integer CHANNELS = 2,
    parameter integer WIDTH = 128,
    parameter integer HEIGHT = 96,
    parameter integer PIXEL_BITS = 9,
    parameter integer TAG_BITS = 14
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      in_valid,
    input  wire [$clog2(HEIGHT+2)-1:0]               padded_row,
    input  wire [$clog2(WIDTH+2)-1:0]                padded_column,
    input  wire [CHANNELS*PIXEL_BITS-1:0]            pixels_flat,
    output reg                                       out_valid,
    output reg  [TAG_BITS-1:0]                       tag_out,
    output reg  [CHANNELS*9*PIXEL_BITS-1:0]          window_flat
);
    localparam integer PADDED_WIDTH = WIDTH + 2;
    localparam integer COLUMN_BITS = $clog2(WIDTH);
    localparam integer OUTPUT_ROW_BITS = TAG_BITS - COLUMN_BITS;
    localparam integer VECTOR_BITS = CHANNELS * PIXEL_BITS;
    localparam integer PADDED_ROW_BITS = $clog2(HEIGHT + 2);
    localparam integer PADDED_COLUMN_BITS = $clog2(WIDTH + 2);

    // Both previous rows use synchronous memories.  The second row delay is
    // written one cycle later with the value read from the first delay.  This
    // permits BRAM inference while sustaining one padded coordinate per cycle.
    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] one_row_memory
        [0:PADDED_WIDTH-1];
    (* ram_style = "block" *) reg [VECTOR_BITS-1:0] two_row_memory
        [0:PADDED_WIDTH-1];
    reg [VECTOR_BITS-1:0] one_row_read;
    reg [VECTOR_BITS-1:0] two_row_read;
    reg [VECTOR_BITS-1:0] pixels_delayed;
    reg [PADDED_ROW_BITS-1:0] row_delayed;
    reg [PADDED_COLUMN_BITS-1:0] column_delayed;
    reg read_valid;

    wire [OUTPUT_ROW_BITS-1:0] unpadded_row =
        OUTPUT_ROW_BITS'(row_delayed) - OUTPUT_ROW_BITS'(2);
    wire [COLUMN_BITS-1:0] unpadded_column =
        COLUMN_BITS'(column_delayed) - COLUMN_BITS'(2);

    reg [VECTOR_BITS-1:0] top_left;
    reg [VECTOR_BITS-1:0] top_middle;
    reg [VECTOR_BITS-1:0] middle_left;
    reg [VECTOR_BITS-1:0] middle_middle;
    reg [VECTOR_BITS-1:0] bottom_left;
    reg [VECTOR_BITS-1:0] bottom_middle;

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
            end
            if (read_valid) begin
                two_row_memory[column_delayed] <= one_row_read;
                out_valid <= (row_delayed >= 2) && (column_delayed >= 2);
                tag_out <= {unpadded_row, unpadded_column};

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
        if (CHANNELS <= 0 || WIDTH <= 0 || HEIGHT <= 0 || PIXEL_BITS <= 0) begin
            $error("invalid line-buffer parameters");
        end
        if ((1 << TAG_BITS) < WIDTH*HEIGHT) begin
            $error("TAG_BITS cannot represent every output coordinate");
        end
        if ((1 << COLUMN_BITS) != WIDTH) begin
            $error("raster tag packing requires power-of-two WIDTH");
        end
    end
endmodule

`default_nettype wire
