`timescale 1ns/1ps
`default_nettype none

// Pursuit Sparse-LM-A accelerator.
//
// L1 consumes the sorted 400-event list and L2 consumes only nonzero L1
// packets.  L2 is materialized once because both model time steps are
// identical and stateless.  One shared 8x4x9 (288-DSP) convolution engine then
// executes L3--L8 layer-major.  L3 computes its current once and replays that
// current through the second LIF update; L4--L8 load each weight tile once and
// evaluate t0/t1 before advancing.  H1/H2 likewise share sixteen DSPs across
// their adjacent time steps.
module pursuit_sparse_layer_major_accelerator_top (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    output reg                       busy,
    output reg                       done,
    output reg                       result_valid,
    output reg                       range_error_seen,
    output wire                      protocol_error,

    input  wire                      event_valid,
    output wire                      event_ready,
    input  wire                      event_polarity,
    input  wire [6:0]                event_row,
    input  wire [6:0]                event_column,
    input  wire                      events_end,

    input  wire                      host_l1_weight_write_enable,
    input  wire [2:0]                host_l1_weight_write_output_channel,
    input  wire                      host_l1_weight_write_input_channel,
    input  wire [3:0]                host_l1_weight_write_kernel_index,
    input  wire signed [15:0]        host_l1_weight_write_data,
    input  wire                      host_l1_channel_write_enable,
    input  wire [2:0]                host_l1_channel_write_channel,
    input  wire signed [23:0]        host_l1_constant_write_data,
    input  wire [4:0]                host_l1_fraction_bits_write_data,

    input  wire                      host_l2_weight_write_enable,
    input  wire [3:0]                host_l2_weight_write_output_channel,
    input  wire [2:0]                host_l2_weight_write_input_channel,
    input  wire [3:0]                host_l2_weight_write_kernel_index,
    input  wire signed [15:0]        host_l2_weight_write_data,
    input  wire                      host_l2_channel_write_enable,
    input  wire [3:0]                host_l2_channel_write_channel,
    input  wire signed [23:0]        host_l2_constant_write_data,
    input  wire [4:0]                host_l2_fraction_bits_write_data,

    input  wire                      host_conv_weight_write_enable,
    input  wire [12:0]               host_conv_weight_write_address,
    input  wire [255:0]              host_conv_weight_write_data,
    input  wire                      host_linear_weight_write_enable,
    input  wire [10:0]               host_linear_weight_write_address,
    input  wire [255:0]              host_linear_weight_write_data,
    input  wire                      host_parameter_write_enable,
    input  wire [8:0]                host_parameter_write_address,
    input  wire signed [23:0]        host_parameter_constant_data,
    input  wire [4:0]                host_parameter_fraction_data,
    input  wire                      host_readout_fraction_write_enable,
    input  wire                      host_readout_fraction_write_address,
    input  wire [4:0]                host_readout_fraction_write_data,

    output wire [127:0]              result_integer_flat,
    output wire [9:0]                result_fraction_bits_flat,
    output reg  [31:0]               inference_cycles,
    output reg  [31:0]               frontend_cycles,
    output reg  [31:0]               convolution_cycles,
    output reg  [31:0]               dense_cycles,
    output reg  [31:0]               conv_weight_read_beats,
    output reg  [31:0]               linear_weight_read_beats,
    output reg  [31:0]               readout_weight_read_beats,
    output reg  [31:0]               feature_read_requests,
    output reg  [31:0]               feature_write_transactions,
    output wire [31:0]               accepted_event_count,
    output wire [31:0]               l1_emitted_packet_count,
    output wire [31:0]               l2_accepted_packet_count,

    output wire                      trace_valid,
    output wire                      trace_time_index,
    output wire [3:0]                trace_layer,
    output wire [13:0]               trace_address,
    output wire [6:0]                trace_channel_base,
    output wire [383:0]              trace_current_flat,
    output wire [127:0]              trace_spike_flat,
    output wire [255:0]              trace_membrane_flat
);
    localparam [2:0] PHASE_IDLE = 3'd0;
    localparam [2:0] PHASE_FRONTEND = 3'd1;
    localparam [2:0] PHASE_CONV = 3'd2;
    localparam [2:0] PHASE_H1 = 3'd3;
    localparam [2:0] PHASE_H2 = 3'd4;
    localparam [2:0] PHASE_READOUT = 3'd5;

    (* ram_style = "block" *) reg [255:0] convolution_weight_memory
        [0:4157];
    (* ram_style = "block" *) reg [255:0] linear_weight_memory [0:1991];
    reg signed [23:0] parameter_constant_memory [0:319];
    reg [4:0] parameter_fraction_memory [0:319];
    reg [4:0] readout_fraction_memory [0:1];

    reg [2:0] phase;
    reg [2:0] layer_index;
    reg dense_layer;
    reg frontend_start;
    reg convolution_start;
    reg linear_start;
    reg readout_start;

    // --------------------------- sparse L1/L2 ---------------------------
    wire frontend_busy;
    wire frontend_done;
    wire frontend_dense_read_enable;
    wire [9:0] frontend_dense_read_address;
    wire [3:0] frontend_dense_read_channel_base;
    wire frontend_dense_read_valid;
    wire [31:0] frontend_dense_read_data_flat;
    wire frontend_protocol_error;
    wire l1_trace_valid;
    wire [11:0] l1_trace_address;
    wire [191:0] l1_trace_current_flat;
    wire [63:0] l1_trace_spike_flat;
    wire [127:0] l1_trace_membrane_flat;
    wire l1_trace_range_error;
    wire l2_trace_valid;
    wire [9:0] l2_trace_address;
    wire [383:0] l2_trace_current_flat;
    wire [127:0] l2_trace_spike_flat;
    wire [255:0] l2_trace_membrane_flat;
    wire l2_trace_range_error;
    wire [31:0] frontend_cycle_count;
    wire [31:0] unused_l1_cycle_count;
    wire [31:0] unused_l1_core_cycle_count;
    wire [31:0] unused_l2_cycle_count;
    wire [31:0] unused_l2_dense_write_count;
    wire unused_l1_packet_valid;
    wire unused_l1_packet_ready;
    wire [11:0] unused_l1_packet_address;
    wire [2:0] unused_l1_packet_channel;
    wire [7:0] unused_l1_packet_count;
    wire unused_l1_protocol_error;
    wire unused_l2_protocol_error;
    wire unused_frontend_monitor = &{
        1'b0,
        unused_l1_packet_valid,
        unused_l1_packet_ready,
        unused_l1_packet_address,
        unused_l1_packet_channel,
        unused_l1_packet_count,
        unused_l1_protocol_error,
        unused_l2_protocol_error,
        unused_l1_cycle_count,
        unused_l1_core_cycle_count,
        unused_l2_cycle_count,
        unused_l2_dense_write_count
    };

    pursuit_sparse_frontend sparse_frontend (
        .clk(clk), .rst(rst), .start(frontend_start),
        .busy(frontend_busy), .done(frontend_done),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_polarity(event_polarity), .event_row(event_row),
        .event_column(event_column), .events_end(events_end),
        .l1_weight_write_enable(host_l1_weight_write_enable && !busy),
        .l1_weight_write_output_channel(
            host_l1_weight_write_output_channel
        ),
        .l1_weight_write_input_channel(
            host_l1_weight_write_input_channel
        ),
        .l1_weight_write_kernel_index(host_l1_weight_write_kernel_index),
        .l1_weight_write_data(host_l1_weight_write_data),
        .l1_channel_write_enable(host_l1_channel_write_enable && !busy),
        .l1_channel_write_channel(host_l1_channel_write_channel),
        .l1_constant_write_data(host_l1_constant_write_data),
        .l1_fraction_bits_write_data(host_l1_fraction_bits_write_data),
        .l2_weight_write_enable(host_l2_weight_write_enable && !busy),
        .l2_weight_write_output_channel(
            host_l2_weight_write_output_channel
        ),
        .l2_weight_write_input_channel(
            host_l2_weight_write_input_channel
        ),
        .l2_weight_write_kernel_index(host_l2_weight_write_kernel_index),
        .l2_weight_write_data(host_l2_weight_write_data),
        .l2_channel_write_enable(host_l2_channel_write_enable && !busy),
        .l2_channel_write_channel(host_l2_channel_write_channel),
        .l2_constant_write_data(host_l2_constant_write_data),
        .l2_fraction_bits_write_data(host_l2_fraction_bits_write_data),
        .dense_read_enable(frontend_dense_read_enable),
        .dense_read_address(frontend_dense_read_address),
        .dense_read_channel_base(frontend_dense_read_channel_base),
        .dense_read_valid(frontend_dense_read_valid),
        .dense_read_data_flat(frontend_dense_read_data_flat),
        .l1_trace_valid(l1_trace_valid),
        .l1_trace_address(l1_trace_address),
        .l1_trace_current_flat(l1_trace_current_flat),
        .l1_trace_spike_flat(l1_trace_spike_flat),
        .l1_trace_membrane_flat(l1_trace_membrane_flat),
        .l1_trace_range_error(l1_trace_range_error),
        .l2_trace_valid(l2_trace_valid),
        .l2_trace_address(l2_trace_address),
        .l2_trace_current_flat(l2_trace_current_flat),
        .l2_trace_spike_flat(l2_trace_spike_flat),
        .l2_trace_membrane_flat(l2_trace_membrane_flat),
        .l2_trace_range_error(l2_trace_range_error),
        .l1_packet_valid(unused_l1_packet_valid),
        .l1_packet_ready(unused_l1_packet_ready),
        .l1_packet_address(unused_l1_packet_address),
        .l1_packet_channel(unused_l1_packet_channel),
        .l1_packet_count(unused_l1_packet_count),
        .l1_protocol_error(unused_l1_protocol_error),
        .l2_protocol_error(unused_l2_protocol_error),
        .protocol_error(frontend_protocol_error),
        .cycle_count(frontend_cycle_count),
        .l1_cycle_count(unused_l1_cycle_count),
        .l1_core_cycle_count(unused_l1_core_cycle_count),
        .l2_cycle_count(unused_l2_cycle_count),
        .accepted_event_count(accepted_event_count),
        .l1_emitted_packet_count(l1_emitted_packet_count),
        .l2_accepted_packet_count(l2_accepted_packet_count),
        .l2_dense_write_count(unused_l2_dense_write_count)
    );
    assign protocol_error = frontend_protocol_error
        | (1'b0 & unused_frontend_monitor);

    // -------------------------- layer descriptor -------------------------
    reg [3:0] layer_width_log2;
    reg [6:0] layer_height;
    reg [3:0] layer_input_tiles;
    reg [3:0] layer_output_tiles;
    reg [13:0] layer_input_spatial;
    reg [13:0] layer_output_spatial;
    reg layer_pool_enable;
    reg layer_reuse_current;
    reg [12:0] layer_weight_base;
    reg [8:0] layer_parameter_base;
    reg layer_input_buffer;
    reg layer_output_buffer;

    always @* begin
        layer_width_log2 = 0;
        layer_height = 0;
        layer_input_tiles = 0;
        layer_output_tiles = 0;
        layer_input_spatial = 0;
        layer_output_spatial = 0;
        layer_pool_enable = 1'b0;
        layer_reuse_current = 1'b0;
        layer_weight_base = 0;
        layer_parameter_base = 0;
        layer_input_buffer = 1'b0;
        layer_output_buffer = 1'b0;
        case (layer_index)
            3'd2: begin
                layer_width_log2 = 5; layer_height = 24;
                layer_input_tiles = 4; layer_output_tiles = 3;
                layer_input_spatial = 768; layer_output_spatial = 192;
                layer_pool_enable = 1'b1; layer_reuse_current = 1'b1;
                layer_weight_base = 90; layer_parameter_base = 24;
                layer_output_buffer = 1'b0;
            end
            3'd3: begin
                layer_width_log2 = 4; layer_height = 12;
                layer_input_tiles = 6; layer_output_tiles = 4;
                layer_input_spatial = 192; layer_output_spatial = 48;
                layer_pool_enable = 1'b1; layer_weight_base = 306;
                layer_parameter_base = 48;
                layer_input_buffer = 1'b0; layer_output_buffer = 1'b1;
            end
            3'd4: begin
                layer_width_log2 = 3; layer_height = 6;
                layer_input_tiles = 8; layer_output_tiles = 5;
                layer_input_spatial = 48; layer_output_spatial = 12;
                layer_pool_enable = 1'b1; layer_weight_base = 738;
                layer_parameter_base = 80;
                layer_input_buffer = 1'b1; layer_output_buffer = 1'b0;
            end
            3'd5: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 1458; layer_parameter_base = 120;
                layer_input_buffer = 1'b0; layer_output_buffer = 1'b1;
            end
            3'd6: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 2358; layer_parameter_base = 160;
                layer_input_buffer = 1'b1; layer_output_buffer = 1'b0;
            end
            3'd7: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 3258; layer_parameter_base = 200;
                layer_input_buffer = 1'b0; layer_output_buffer = 1'b1;
            end
            default: begin end
        endcase
    end

    // ---------------------- shared paired convolution --------------------
    wire convolution_busy;
    wire convolution_done;
    wire conv_weight_read_enable;
    wire [12:0] conv_weight_read_address;
    reg [255:0] conv_weight_read_data;
    wire conv_parameter_read_enable;
    wire [8:0] conv_parameter_read_address;
    wire conv_input_read_enable;
    wire conv_input_read_time;
    wire [13:0] conv_input_read_address;
    wire [5:0] conv_input_read_channel_base;
    wire [35:0] feature_conv_read_data_flat;
    wire [35:0] conv_input_read_data_flat = (layer_index == 2)
        ? {1'b0, frontend_dense_read_data_flat[31:24],
           1'b0, frontend_dense_read_data_flat[23:16],
           1'b0, frontend_dense_read_data_flat[15:8],
           1'b0, frontend_dense_read_data_flat[7:0]}
        : feature_conv_read_data_flat;
    wire conv_output_valid;
    wire conv_output_time;
    wire [13:0] conv_output_address;
    wire [5:0] conv_output_channel_base;
    wire [191:0] conv_output_current_flat;
    wire [63:0] conv_output_spike_flat;
    wire [127:0] conv_output_membrane_flat;
    wire conv_output_range_error;
    reg signed [23:0] shared_parameter_constant_data;
    reg [4:0] shared_parameter_fraction_data;

    assign frontend_dense_read_enable = conv_input_read_enable
        && (layer_index == 2);
    assign frontend_dense_read_address = conv_input_read_address[9:0];
    assign frontend_dense_read_channel_base =
        conv_input_read_channel_base[3:0];

    paired_conv3x3_layer_engine convolution_engine (
        .clk(clk), .rst(rst), .start(convolution_start),
        .busy(convolution_busy), .done(convolution_done),
        .config_width_log2(layer_width_log2),
        .config_height(layer_height),
        .config_input_tiles(layer_input_tiles),
        .config_output_tiles(layer_output_tiles),
        .config_output_spatial(layer_output_spatial),
        .config_pool_enable(layer_pool_enable),
        .config_reuse_current(layer_reuse_current),
        .config_weight_beat_base(layer_weight_base),
        .config_parameter_base(layer_parameter_base),
        .weight_read_enable(conv_weight_read_enable),
        .weight_read_address(conv_weight_read_address),
        .weight_read_data(conv_weight_read_data),
        .parameter_read_enable(conv_parameter_read_enable),
        .parameter_read_address(conv_parameter_read_address),
        .parameter_constant_data(shared_parameter_constant_data),
        .parameter_fraction_data(shared_parameter_fraction_data),
        .input_read_enable(conv_input_read_enable),
        .input_read_time(conv_input_read_time),
        .input_read_address(conv_input_read_address),
        .input_read_channel_base(conv_input_read_channel_base),
        .input_read_data_flat(conv_input_read_data_flat),
        .output_valid(conv_output_valid),
        .output_time(conv_output_time),
        .output_address(conv_output_address),
        .output_channel_base(conv_output_channel_base),
        .output_current_flat(conv_output_current_flat),
        .output_spike_flat(conv_output_spike_flat),
        .output_membrane_flat(conv_output_membrane_flat),
        .output_range_error(conv_output_range_error)
    );

    wire feature_dense_read_enable;
    wire feature_dense_read_time;
    wire [8:0] feature_dense_read_address;
    wire [7:0] feature_dense_read_data;
    pursuit_paired_feature_store feature_store (
        .clk(clk),
        .conv_read_enable(conv_input_read_enable && (layer_index >= 3)),
        .conv_read_time(conv_input_read_time),
        .conv_read_buffer(layer_input_buffer),
        .conv_read_address(conv_input_read_address),
        .conv_read_channel_base(conv_input_read_channel_base),
        .conv_read_spatial(layer_input_spatial),
        .conv_read_data_flat(feature_conv_read_data_flat),
        .conv_write_enable(conv_output_valid),
        .conv_write_time(conv_output_time),
        .conv_write_buffer(layer_output_buffer),
        .conv_write_address(conv_output_address),
        .conv_write_channel_base(conv_output_channel_base),
        .conv_write_spatial(layer_output_spatial),
        .conv_write_spike_flat(conv_output_spike_flat),
        .dense_read_enable(feature_dense_read_enable),
        .dense_read_time(feature_dense_read_time),
        .dense_read_buffer(1'b1),
        .dense_read_flat_address(feature_dense_read_address),
        .dense_read_data(feature_dense_read_data)
    );

    // ------------------------- paired H1 and H2 --------------------------
    wire linear_busy;
    wire linear_done;
    wire linear_weight_read_enable;
    wire [10:0] linear_weight_read_address;
    reg [255:0] shared_linear_weight_read_data;
    wire linear_input_read_enable;
    wire linear_input_read_time;
    wire [8:0] linear_input_read_address;
    reg [7:0] h2_input_read_data;
    wire [7:0] linear_input_read_data = dense_layer
        ? h2_input_read_data : feature_dense_read_data;
    wire linear_parameter_read_enable;
    wire [8:0] linear_parameter_read_address;
    wire linear_output_valid;
    wire linear_output_time;
    wire [6:0] linear_output_channel_base;
    wire [383:0] linear_output_current_flat;
    wire [127:0] linear_output_spike_flat;
    wire [255:0] linear_output_membrane_flat;
    wire linear_output_range_error;

    assign feature_dense_read_enable = linear_input_read_enable && !dense_layer;
    assign feature_dense_read_time = linear_input_read_time;
    assign feature_dense_read_address = linear_input_read_address;

    wire [9:0] dense_input_features = dense_layer ? 10'd60 : 10'd480;
    wire [2:0] dense_output_groups = dense_layer ? 3'd1 : 3'd4;
    wire [10:0] dense_weight_base = dense_layer ? 11'd1920 : 11'd0;
    wire [8:0] dense_parameter_base = dense_layer ? 9'd304 : 9'd240;

    paired_linear_lif_layer_engine linear_engine (
        .clk(clk), .rst(rst), .start(linear_start),
        .busy(linear_busy), .done(linear_done),
        .config_input_features(dense_input_features),
        .config_output_groups(dense_output_groups),
        .config_weight_beat_base(dense_weight_base),
        .config_parameter_base(dense_parameter_base),
        .weight_read_enable(linear_weight_read_enable),
        .weight_read_address(linear_weight_read_address),
        .weight_read_data(shared_linear_weight_read_data),
        .input_read_enable(linear_input_read_enable),
        .input_read_time(linear_input_read_time),
        .input_read_address(linear_input_read_address),
        .input_read_data(linear_input_read_data),
        .parameter_read_enable(linear_parameter_read_enable),
        .parameter_read_address(linear_parameter_read_address),
        .parameter_constant_data(shared_parameter_constant_data),
        .parameter_fraction_data(shared_parameter_fraction_data),
        .output_valid(linear_output_valid),
        .output_time(linear_output_time),
        .output_channel_base(linear_output_channel_base),
        .output_current_flat(linear_output_current_flat),
        .output_spike_flat(linear_output_spike_flat),
        .output_membrane_flat(linear_output_membrane_flat),
        .output_range_error(linear_output_range_error)
    );

    reg [7:0] h1_t0_spike [0:63];
    reg [7:0] h1_t1_spike [0:63];
    reg [7:0] h2_t0_spike [0:15];
    reg [8:0] temporal_sum [0:15];
    integer dense_lane;
    always @(posedge clk) begin
        if (linear_input_read_enable && dense_layer) begin
            h2_input_read_data <= linear_input_read_time
                ? h1_t1_spike[linear_input_read_address[5:0]]
                : h1_t0_spike[linear_input_read_address[5:0]];
        end
        if (linear_output_valid) begin
            for (dense_lane = 0; dense_lane < 16;
                    dense_lane = dense_lane + 1) begin
                if (!dense_layer) begin
                    if (linear_output_time) begin
                        h1_t1_spike[6'(linear_output_channel_base
                                + 7'(dense_lane))]
                            <= linear_output_spike_flat[
                                dense_lane*8 +: 8
                            ];
                    end else begin
                        h1_t0_spike[6'(linear_output_channel_base
                                + 7'(dense_lane))]
                            <= linear_output_spike_flat[
                                dense_lane*8 +: 8
                            ];
                    end
                end else if (!linear_output_time) begin
                    h2_t0_spike[4'(linear_output_channel_base
                            + 7'(dense_lane))]
                        <= linear_output_spike_flat[dense_lane*8 +: 8];
                end else begin
                    temporal_sum[4'(linear_output_channel_base
                            + 7'(dense_lane))]
                        <= {1'b0, h2_t0_spike[4'(
                            linear_output_channel_base + 7'(dense_lane)
                        )]} + {1'b0, linear_output_spike_flat[
                            dense_lane*8 +: 8
                        ]};
                end
            end
        end
    end

    // ------------------------------- readout ------------------------------
    wire readout_busy;
    wire readout_done;
    wire readout_input_read_enable;
    wire [3:0] readout_input_read_address;
    reg [8:0] readout_input_read_data;
    wire readout_weight_read_enable;
    wire [10:0] readout_weight_read_address;
    wire readout_output_valid;
    wire [127:0] readout_output_integer_flat;

    readout_engine output_engine (
        .clk(clk), .rst(rst), .start(readout_start),
        .busy(readout_busy), .done(readout_done),
        .config_input_features(5'd12),
        .config_weight_beat_base(11'd1980),
        .input_read_enable(readout_input_read_enable),
        .input_read_address(readout_input_read_address),
        .input_read_data(readout_input_read_data),
        .weight_read_enable(readout_weight_read_enable),
        .weight_read_address(readout_weight_read_address),
        .weight_read_data(shared_linear_weight_read_data[31:0]),
        .output_valid(readout_output_valid),
        .output_integer_flat(readout_output_integer_flat)
    );
    always @(posedge clk) begin
        if (readout_input_read_enable) begin
            readout_input_read_data
                <= temporal_sum[readout_input_read_address];
        end
    end
    assign result_integer_flat = readout_output_integer_flat;
    assign result_fraction_bits_flat = {
        readout_fraction_memory[1], readout_fraction_memory[0]
    };

    // ---------------------- memory arbitration/programming ----------------
    wire shared_parameter_read_enable = conv_parameter_read_enable
        || linear_parameter_read_enable;
    wire [8:0] shared_parameter_read_address = conv_parameter_read_enable
        ? conv_parameter_read_address : linear_parameter_read_address;
    wire shared_linear_weight_read_enable = linear_weight_read_enable
        || readout_weight_read_enable;
    wire [10:0] shared_linear_weight_read_address =
        linear_weight_read_enable ? linear_weight_read_address
                                  : readout_weight_read_address;

    always @(posedge clk) begin
        if (host_conv_weight_write_enable && !busy) begin
            convolution_weight_memory[host_conv_weight_write_address]
                <= host_conv_weight_write_data;
        end
        if (conv_weight_read_enable) begin
            conv_weight_read_data
                <= convolution_weight_memory[conv_weight_read_address];
        end
        if (host_linear_weight_write_enable && !busy) begin
            linear_weight_memory[host_linear_weight_write_address]
                <= host_linear_weight_write_data;
        end
        if (shared_linear_weight_read_enable) begin
            shared_linear_weight_read_data <= linear_weight_memory[
                shared_linear_weight_read_address
            ];
        end
        if (host_parameter_write_enable && !busy) begin
            parameter_constant_memory[host_parameter_write_address]
                <= host_parameter_constant_data;
            parameter_fraction_memory[host_parameter_write_address]
                <= host_parameter_fraction_data;
        end
        if (shared_parameter_read_enable) begin
            shared_parameter_constant_data <= parameter_constant_memory[
                shared_parameter_read_address
            ];
            shared_parameter_fraction_data <= parameter_fraction_memory[
                shared_parameter_read_address
            ];
        end
        if (host_readout_fraction_write_enable && !busy) begin
            readout_fraction_memory[host_readout_fraction_write_address]
                <= host_readout_fraction_write_data;
        end
    end

    // ------------------------------- trace --------------------------------
    wire frontend_trace_valid = l2_trace_valid || l1_trace_valid;
    wire trace_from_linear = (phase == PHASE_H1) || (phase == PHASE_H2);
    assign trace_valid = (phase == PHASE_FRONTEND) ? frontend_trace_valid
        : (trace_from_linear ? linear_output_valid : conv_output_valid);
    assign trace_time_index = trace_from_linear ? linear_output_time
        : ((phase == PHASE_CONV) ? conv_output_time : 1'b0);
    assign trace_layer = (phase == PHASE_FRONTEND)
        ? (l2_trace_valid ? 4'd1 : 4'd0)
        : (trace_from_linear ? (dense_layer ? 4'd9 : 4'd8)
                             : {1'b0, layer_index});
    assign trace_address = (phase == PHASE_FRONTEND)
        ? (l2_trace_valid ? {4'd0, l2_trace_address}
                          : {2'd0, l1_trace_address})
        : (trace_from_linear ? 14'd0 : conv_output_address);
    assign trace_channel_base = (phase == PHASE_FRONTEND) ? 7'd0
        : (trace_from_linear ? linear_output_channel_base
                             : {1'b0, conv_output_channel_base});
    assign trace_current_flat = (phase == PHASE_FRONTEND)
        ? (l2_trace_valid ? l2_trace_current_flat
                          : {192'd0, l1_trace_current_flat})
        : (trace_from_linear ? linear_output_current_flat
                             : {192'd0, conv_output_current_flat});
    assign trace_spike_flat = (phase == PHASE_FRONTEND)
        ? (l2_trace_valid ? l2_trace_spike_flat
                          : {64'd0, l1_trace_spike_flat})
        : (trace_from_linear ? linear_output_spike_flat
                             : {64'd0, conv_output_spike_flat});
    assign trace_membrane_flat = (phase == PHASE_FRONTEND)
        ? (l2_trace_valid ? l2_trace_membrane_flat
                          : {128'd0, l1_trace_membrane_flat})
        : (trace_from_linear ? linear_output_membrane_flat
                             : {128'd0, conv_output_membrane_flat});

    // -------------------------- controller/counters -----------------------
    wire engines_any_busy = frontend_busy || convolution_busy || linear_busy
        || readout_busy;
    always @(posedge clk) begin
        if (rst) begin
            phase <= PHASE_IDLE;
            layer_index <= 3'd2;
            dense_layer <= 1'b0;
            frontend_start <= 1'b0;
            convolution_start <= 1'b0;
            linear_start <= 1'b0;
            readout_start <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;
            range_error_seen <= 1'b0;
            inference_cycles <= 0;
            frontend_cycles <= 0;
            convolution_cycles <= 0;
            dense_cycles <= 0;
            conv_weight_read_beats <= 0;
            linear_weight_read_beats <= 0;
            readout_weight_read_beats <= 0;
            feature_read_requests <= 0;
            feature_write_transactions <= 0;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            frontend_start <= 1'b0;
            convolution_start <= 1'b0;
            linear_start <= 1'b0;
            readout_start <= 1'b0;

            if (busy) begin
                inference_cycles <= inference_cycles + 1'b1;
                if (phase == PHASE_FRONTEND) begin
                    frontend_cycles <= frontend_cycles + 1'b1;
                end else if (phase == PHASE_CONV) begin
                    convolution_cycles <= convolution_cycles + 1'b1;
                end else if ((phase == PHASE_H1) || (phase == PHASE_H2)) begin
                    dense_cycles <= dense_cycles + 1'b1;
                end
            end
            if (conv_weight_read_enable) begin
                conv_weight_read_beats <= conv_weight_read_beats + 1'b1;
            end
            if (linear_weight_read_enable) begin
                linear_weight_read_beats <= linear_weight_read_beats + 1'b1;
            end
            if (readout_weight_read_enable) begin
                readout_weight_read_beats <= readout_weight_read_beats + 1'b1;
            end
            if ((conv_input_read_enable && (layer_index >= 3))
                    || frontend_dense_read_valid) begin
                feature_read_requests <= feature_read_requests + 1'b1;
            end
            if (conv_output_valid) begin
                feature_write_transactions
                    <= feature_write_transactions + 1'b1;
            end
            if (l1_trace_range_error || l2_trace_range_error
                    || (conv_output_valid && conv_output_range_error)
                    || (linear_output_valid && linear_output_range_error)) begin
                range_error_seen <= 1'b1;
            end

            case (phase)
                PHASE_IDLE: begin
                    if (start && !busy && !engines_any_busy) begin
                        busy <= 1'b1;
                        range_error_seen <= 1'b0;
                        inference_cycles <= 0;
                        frontend_cycles <= 0;
                        convolution_cycles <= 0;
                        dense_cycles <= 0;
                        conv_weight_read_beats <= 0;
                        linear_weight_read_beats <= 0;
                        readout_weight_read_beats <= 0;
                        feature_read_requests <= 0;
                        feature_write_transactions <= 0;
                        layer_index <= 3'd2;
                        frontend_start <= 1'b1;
                        phase <= PHASE_FRONTEND;
                    end
                end
                PHASE_FRONTEND: begin
                    if (frontend_done) begin
                        frontend_cycles <= frontend_cycle_count;
                        layer_index <= 3'd2;
                        convolution_start <= 1'b1;
                        phase <= PHASE_CONV;
                    end
                end
                PHASE_CONV: begin
                    if (convolution_done) begin
                        if (layer_index == 3'd7) begin
                            dense_layer <= 1'b0;
                            linear_start <= 1'b1;
                            phase <= PHASE_H1;
                        end else begin
                            layer_index <= layer_index + 1'b1;
                            convolution_start <= 1'b1;
                        end
                    end
                end
                PHASE_H1: begin
                    if (linear_done) begin
                        dense_layer <= 1'b1;
                        linear_start <= 1'b1;
                        phase <= PHASE_H2;
                    end
                end
                PHASE_H2: begin
                    if (linear_done) begin
                        readout_start <= 1'b1;
                        phase <= PHASE_READOUT;
                    end
                end
                PHASE_READOUT: begin
                    if (readout_done && readout_output_valid) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        result_valid <= 1'b1;
                        phase <= PHASE_IDLE;
                    end
                end
                default: phase <= PHASE_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
