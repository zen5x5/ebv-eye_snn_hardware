`timescale 1ns/1ps
`default_nettype none

// Complete fixed-topology pursuit SNN accelerator core.
//
// DATAFLOW_MODE_SL_TM  : recompute all layers at t0/t1; L1/L2 stateless.
// DATAFLOW_MODE_SL_HYB : cache pooled L3 current at t0 and replay L3 LIF at t1.
// DATAFLOW_MODE_FS_TM  : recompute all layers with full cross-bin state;
//                        requires FULL_STATE_CAPABLE=1 and full-state weights.
//
// Host programming ports are deliberately protocol-neutral.  An AXI wrapper
// can map them to BRAM writes/registers without changing the compute core.
module pursuit_snn_accelerator_top #(
    parameter integer FULL_STATE_CAPABLE = 0
) (
    input  wire                      clk,
    input  wire                      rst,
    input  wire                      start,
    input  wire [1:0]                dataflow_mode,
    output reg                       busy,
    output reg                       done,
    output reg                       result_valid,
    output reg                       config_error,
    output reg                       range_error_seen,

    input  wire                      host_input_write_enable,
    input  wire                      host_input_write_channel,
    input  wire [13:0]               host_input_write_address,
    input  wire [8:0]                host_input_write_data,
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
    output reg  [31:0]               convolution_cycles,
    output reg  [31:0]               dense_cycles,
    output reg  [31:0]               replay_cycles,
    output wire                      trace_valid,
    output wire                      trace_is_dense,
    output wire                      trace_time_index,
    output wire [2:0]                trace_layer,
    output wire [13:0]               trace_address,
    output wire [6:0]                trace_channel_base,
    output wire [383:0]              trace_current_flat,
    output wire [127:0]              trace_spike_flat,
    output wire [255:0]              trace_membrane_flat
);
    localparam [1:0] DATAFLOW_MODE_SL_TM = 2'd0;
    localparam [1:0] DATAFLOW_MODE_SL_HYB = 2'd1;
    localparam [1:0] DATAFLOW_MODE_FS_TM = 2'd2;

    localparam [2:0] PHASE_IDLE = 3'd0;
    localparam [2:0] PHASE_CONV = 3'd1;
    localparam [2:0] PHASE_H1 = 3'd2;
    localparam [2:0] PHASE_H2 = 3'd3;
    localparam [2:0] PHASE_REPLAY = 3'd4;
    localparam [2:0] PHASE_READOUT = 3'd5;

    (* ram_style = "block" *) reg [255:0] convolution_weight_memory
        [0:4157];
    (* ram_style = "block" *) reg [255:0] linear_weight_memory [0:1991];
    reg signed [23:0] parameter_constant_memory [0:319];
    reg [4:0] parameter_fraction_memory [0:319];
    reg [4:0] readout_fraction_memory [0:1];

    reg [2:0] phase;
    reg [1:0] mode_register;
    reg time_index;
    reg [2:0] layer_index;
    reg dense_layer;
    reg conv_start;
    reg linear_start;
    reg replay_start;
    reg readout_start;

    // --------------------------- layer descriptor -------------------------
    reg [3:0] layer_width_log2;
    reg [6:0] layer_height;
    reg [3:0] layer_input_tiles;
    reg [3:0] layer_output_tiles;
    reg [13:0] layer_input_spatial;
    reg [13:0] layer_output_spatial;
    reg layer_pool_enable;
    reg [12:0] layer_weight_base;
    reg [8:0] layer_parameter_base;
    reg [1:0] layer_input_source;
    reg layer_output_buffer;
    reg [12:0] layer_state_bank_base;

    always @* begin
        layer_width_log2 = 0;
        layer_height = 0;
        layer_input_tiles = 0;
        layer_output_tiles = 0;
        layer_input_spatial = 0;
        layer_output_spatial = 0;
        layer_pool_enable = 1'b0;
        layer_weight_base = 0;
        layer_parameter_base = 0;
        layer_input_source = 0;
        layer_output_buffer = 1'b0;
        layer_state_bank_base = 0;
        case (layer_index)
            3'd0: begin
                layer_width_log2 = 7; layer_height = 96;
                layer_input_tiles = 1; layer_output_tiles = 1;
                layer_input_spatial = 12288; layer_output_spatial = 3072;
                layer_pool_enable = 1'b1; layer_weight_base = 0;
                layer_parameter_base = 0; layer_input_source = 0;
                layer_output_buffer = 1'b0;
                layer_state_bank_base = 0;
            end
            3'd1: begin
                layer_width_log2 = 6; layer_height = 48;
                layer_input_tiles = 2; layer_output_tiles = 2;
                layer_input_spatial = 3072; layer_output_spatial = 768;
                layer_pool_enable = 1'b1; layer_weight_base = 18;
                layer_parameter_base = 8; layer_input_source = 1;
                layer_output_buffer = 1'b1;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 3072 : 0;
            end
            3'd2: begin
                layer_width_log2 = 5; layer_height = 24;
                layer_input_tiles = 4; layer_output_tiles = 3;
                layer_input_spatial = 768; layer_output_spatial = 192;
                layer_pool_enable = 1'b1; layer_weight_base = 90;
                layer_parameter_base = 24; layer_input_source = 2;
                layer_output_buffer = 1'b0;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 4608 : 0;
            end
            3'd3: begin
                layer_width_log2 = 4; layer_height = 12;
                layer_input_tiles = 6; layer_output_tiles = 4;
                layer_input_spatial = 192; layer_output_spatial = 48;
                layer_pool_enable = 1'b1; layer_weight_base = 306;
                layer_parameter_base = 48; layer_input_source = 1;
                layer_output_buffer = 1'b1;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 5184 : 576;
            end
            3'd4: begin
                layer_width_log2 = 3; layer_height = 6;
                layer_input_tiles = 8; layer_output_tiles = 5;
                layer_input_spatial = 48; layer_output_spatial = 12;
                layer_pool_enable = 1'b1; layer_weight_base = 738;
                layer_parameter_base = 80; layer_input_source = 2;
                layer_output_buffer = 1'b0;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 5376 : 768;
            end
            3'd5: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 1458; layer_parameter_base = 120;
                layer_input_source = 1; layer_output_buffer = 1'b1;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 5436 : 828;
            end
            3'd6: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 2358; layer_parameter_base = 160;
                layer_input_source = 2; layer_output_buffer = 1'b0;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 5496 : 888;
            end
            3'd7: begin
                layer_width_log2 = 2; layer_height = 3;
                layer_input_tiles = 10; layer_output_tiles = 5;
                layer_input_spatial = 12; layer_output_spatial = 12;
                layer_weight_base = 3258; layer_parameter_base = 200;
                layer_input_source = 1; layer_output_buffer = 1'b1;
                layer_state_bank_base = (FULL_STATE_CAPABLE != 0) ? 5556 : 948;
            end
        endcase
    end

    wire early_state_enabled = mode_register == DATAFLOW_MODE_FS_TM;
    wire conv_use_previous_state = time_index
        && (early_state_enabled || (layer_index >= 2));
    wire conv_write_state = early_state_enabled || (layer_index >= 2);

    // ---------------------- convolution engine/memories -------------------
    wire conv_busy;
    wire conv_done;
    wire conv_weight_read_enable;
    wire [12:0] conv_weight_read_address;
    reg [255:0] conv_weight_read_data;
    wire conv_parameter_read_enable;
    wire [8:0] conv_parameter_read_address;
    wire conv_input_read_enable;
    wire [13:0] conv_input_read_address;
    wire [5:0] conv_input_read_channel_base;
    wire [35:0] conv_input_read_data_flat;
    wire conv_state_read_enable;
    wire [13:0] conv_state_read_address;
    wire [5:0] conv_state_read_channel_base;
    wire conv_state_write_enable;
    wire [13:0] conv_state_write_address;
    wire [5:0] conv_state_write_channel_base;
    wire [127:0] conv_state_write_data_flat;
    wire [127:0] state_store_read_data_flat;
    wire conv_output_valid;
    wire [13:0] conv_output_address;
    wire [5:0] conv_output_channel_base;
    wire [191:0] conv_output_current_flat;
    wire [63:0] conv_output_spike_flat;
    wire [127:0] conv_output_membrane_flat;
    wire conv_output_range_error;

    reg signed [23:0] shared_parameter_constant_data;
    reg [4:0] shared_parameter_fraction_data;

    conv3x3_layer_engine convolution_engine (
        .clk(clk), .rst(rst), .start(conv_start),
        .busy(conv_busy), .done(conv_done),
        .config_width_log2(layer_width_log2),
        .config_height(layer_height),
        .config_input_tiles(layer_input_tiles),
        .config_output_tiles(layer_output_tiles),
        .config_output_spatial(layer_output_spatial),
        .config_pool_enable(layer_pool_enable),
        .config_use_previous_state(conv_use_previous_state),
        .config_write_state(conv_write_state),
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
        .input_read_address(conv_input_read_address),
        .input_read_channel_base(conv_input_read_channel_base),
        .input_read_data_flat(conv_input_read_data_flat),
        .state_read_enable(conv_state_read_enable),
        .state_read_address(conv_state_read_address),
        .state_read_channel_base(conv_state_read_channel_base),
        .state_read_data_flat(state_store_read_data_flat),
        .state_write_enable(conv_state_write_enable),
        .state_write_address(conv_state_write_address),
        .state_write_channel_base(conv_state_write_channel_base),
        .state_write_data_flat(conv_state_write_data_flat),
        .output_valid(conv_output_valid),
        .output_address(conv_output_address),
        .output_channel_base(conv_output_channel_base),
        .output_current_flat(conv_output_current_flat),
        .output_spike_flat(conv_output_spike_flat),
        .output_membrane_flat(conv_output_membrane_flat),
        .output_range_error(conv_output_range_error)
    );

    // --------------------------- cached L3 replay -------------------------
    wire replay_busy;
    wire replay_done;
    wire replay_parameter_read_enable;
    wire [8:0] replay_parameter_read_address;
    wire cache_read_enable;
    wire [13:0] cache_read_address;
    wire [5:0] cache_read_channel_base;
    wire [191:0] cache_read_current_flat;
    wire replay_state_read_enable;
    wire [13:0] replay_state_read_address;
    wire [5:0] replay_state_read_channel_base;
    wire replay_state_write_enable;
    wire [13:0] replay_state_write_address;
    wire [5:0] replay_state_write_channel_base;
    wire [127:0] replay_state_write_data_flat;
    wire replay_output_valid;
    wire [13:0] replay_output_address;
    wire [5:0] replay_output_channel_base;
    wire [191:0] replay_output_current_flat;
    wire [63:0] replay_output_spike_flat;
    wire [127:0] replay_output_membrane_flat;
    wire replay_output_range_error;

    cached_current_lif_replay replay_engine (
        .clk(clk), .rst(rst), .start(replay_start),
        .busy(replay_busy), .done(replay_done),
        .config_output_tiles(4'd3), .config_output_spatial(14'd192),
        .config_parameter_base(9'd24),
        .parameter_read_enable(replay_parameter_read_enable),
        .parameter_read_address(replay_parameter_read_address),
        .parameter_fraction_data(shared_parameter_fraction_data),
        .cache_read_enable(cache_read_enable),
        .cache_read_address(cache_read_address),
        .cache_read_channel_base(cache_read_channel_base),
        .cache_read_current_flat(cache_read_current_flat),
        .state_read_enable(replay_state_read_enable),
        .state_read_address(replay_state_read_address),
        .state_read_channel_base(replay_state_read_channel_base),
        .state_read_data_flat(state_store_read_data_flat),
        .state_write_enable(replay_state_write_enable),
        .state_write_address(replay_state_write_address),
        .state_write_channel_base(replay_state_write_channel_base),
        .state_write_data_flat(replay_state_write_data_flat),
        .output_valid(replay_output_valid),
        .output_address(replay_output_address),
        .output_channel_base(replay_output_channel_base),
        .output_current_flat(replay_output_current_flat),
        .output_spike_flat(replay_output_spike_flat),
        .output_membrane_flat(replay_output_membrane_flat),
        .output_range_error(replay_output_range_error)
    );

    pursuit_l3_current_cache current_cache (
        .clk(clk),
        .write_enable(conv_output_valid && (layer_index == 2)
            && !time_index && (mode_register == DATAFLOW_MODE_SL_HYB)),
        .write_address(conv_output_address),
        .write_channel_base(conv_output_channel_base),
        .write_current_flat(conv_output_current_flat),
        .read_enable(cache_read_enable),
        .read_address(cache_read_address),
        .read_channel_base(cache_read_channel_base),
        .read_current_flat(cache_read_current_flat)
    );

    // ------------------------- activation ping/pong -----------------------
    wire activation_write_from_replay = phase == PHASE_REPLAY;
    wire activation_write_enable = activation_write_from_replay
        ? replay_output_valid : conv_output_valid;
    wire [13:0] activation_write_address = activation_write_from_replay
        ? replay_output_address : conv_output_address;
    wire [5:0] activation_write_channel_base = activation_write_from_replay
        ? replay_output_channel_base : conv_output_channel_base;
    wire [63:0] activation_write_spike_flat = activation_write_from_replay
        ? replay_output_spike_flat : conv_output_spike_flat;

    wire activation_dense_read_enable;
    wire [7:0] activation_dense_read_data;
    wire [8:0] linear_input_read_address;
    pursuit_activation_store activation_store (
        .clk(clk),
        .host_input_write_enable(host_input_write_enable && !busy),
        .host_input_write_channel(host_input_write_channel),
        .host_input_write_address(host_input_write_address),
        .host_input_write_data(host_input_write_data),
        .conv_read_enable(conv_input_read_enable),
        .conv_read_source(layer_input_source),
        .conv_read_address(conv_input_read_address),
        .conv_read_channel_base(conv_input_read_channel_base),
        .conv_read_spatial(layer_input_spatial),
        .conv_read_data_flat(conv_input_read_data_flat),
        .conv_write_enable(activation_write_enable),
        .conv_write_buffer(layer_output_buffer),
        .conv_write_address(activation_write_address),
        .conv_write_channel_base(activation_write_channel_base),
        .conv_write_spatial(layer_output_spatial),
        .conv_write_spike_flat(activation_write_spike_flat),
        .dense_read_enable(activation_dense_read_enable),
        .dense_read_buffer(1'b1),
        .dense_read_flat_address(linear_input_read_address),
        .dense_read_data(activation_dense_read_data)
    );

    // ---------------------------- state store -----------------------------
    wire state_access_from_replay = phase == PHASE_REPLAY;
    wire state_store_read_enable = state_access_from_replay
        ? replay_state_read_enable : conv_state_read_enable;
    wire [13:0] state_store_read_address = state_access_from_replay
        ? replay_state_read_address : conv_state_read_address;
    wire [5:0] state_store_read_channel_base = state_access_from_replay
        ? replay_state_read_channel_base : conv_state_read_channel_base;
    wire state_store_write_enable = state_access_from_replay
        ? replay_state_write_enable : conv_state_write_enable;
    wire [13:0] state_store_write_address = state_access_from_replay
        ? replay_state_write_address : conv_state_write_address;
    wire [5:0] state_store_write_channel_base = state_access_from_replay
        ? replay_state_write_channel_base : conv_state_write_channel_base;
    wire [127:0] state_store_write_data_flat = state_access_from_replay
        ? replay_state_write_data_flat : conv_state_write_data_flat;

    pursuit_conv_state_store #(
        .FULL_STATE_CAPABLE(FULL_STATE_CAPABLE)
    ) convolution_state_store (
        .clk(clk),
        .read_enable(state_store_read_enable),
        .layer_bank_base(layer_state_bank_base),
        .layer_spatial(layer_output_spatial),
        .read_address(state_store_read_address),
        .read_channel_base(state_store_read_channel_base),
        .read_data_flat(state_store_read_data_flat),
        .write_enable(state_store_write_enable),
        .write_address(state_store_write_address),
        .write_channel_base(state_store_write_channel_base),
        .write_data_flat(state_store_write_data_flat)
    );

    // ---------------------------- H1/H2 engine ----------------------------
    wire linear_busy;
    wire linear_done;
    wire linear_weight_read_enable;
    wire [10:0] linear_weight_read_address;
    reg [255:0] shared_linear_weight_read_data;
    wire linear_input_read_enable;
    reg [7:0] h2_input_read_data;
    wire [7:0] linear_input_read_data = dense_layer
        ? h2_input_read_data : activation_dense_read_data;
    assign activation_dense_read_enable = linear_input_read_enable && !dense_layer;
    wire linear_parameter_read_enable;
    wire [8:0] linear_parameter_read_address;
    wire linear_state_read_enable;
    wire [6:0] linear_state_read_channel_base;
    reg [255:0] linear_state_read_data_flat;
    wire linear_state_write_enable;
    wire [6:0] linear_state_write_channel_base;
    wire [255:0] linear_state_write_data_flat;
    wire linear_output_valid;
    wire [6:0] linear_output_channel_base;
    wire [383:0] linear_output_current_flat;
    wire [127:0] linear_output_spike_flat;
    wire [255:0] linear_output_membrane_flat;
    wire linear_output_range_error;

    wire [9:0] dense_input_features = dense_layer ? 10'd60 : 10'd480;
    wire [2:0] dense_output_groups = dense_layer ? 3'd1 : 3'd4;
    wire [10:0] dense_weight_base = dense_layer ? 11'd1920 : 11'd0;
    wire [8:0] dense_parameter_base = dense_layer ? 9'd304 : 9'd240;

    linear_lif_layer_engine linear_engine (
        .clk(clk), .rst(rst), .start(linear_start),
        .busy(linear_busy), .done(linear_done),
        .config_input_features(dense_input_features),
        .config_output_groups(dense_output_groups),
        .config_use_previous_state(time_index),
        .config_write_state(1'b1),
        .config_weight_beat_base(dense_weight_base),
        .config_parameter_base(dense_parameter_base),
        .weight_read_enable(linear_weight_read_enable),
        .weight_read_address(linear_weight_read_address),
        .weight_read_data(shared_linear_weight_read_data),
        .input_read_enable(linear_input_read_enable),
        .input_read_address(linear_input_read_address),
        .input_read_data(linear_input_read_data),
        .parameter_read_enable(linear_parameter_read_enable),
        .parameter_read_address(linear_parameter_read_address),
        .parameter_constant_data(shared_parameter_constant_data),
        .parameter_fraction_data(shared_parameter_fraction_data),
        .state_read_enable(linear_state_read_enable),
        .state_read_channel_base(linear_state_read_channel_base),
        .state_read_data_flat(linear_state_read_data_flat),
        .state_write_enable(linear_state_write_enable),
        .state_write_channel_base(linear_state_write_channel_base),
        .state_write_data_flat(linear_state_write_data_flat),
        .output_valid(linear_output_valid),
        .output_channel_base(linear_output_channel_base),
        .output_current_flat(linear_output_current_flat),
        .output_spike_flat(linear_output_spike_flat),
        .output_membrane_flat(linear_output_membrane_flat),
        .output_range_error(linear_output_range_error)
    );

    reg signed [15:0] h1_state [0:63];
    reg signed [15:0] h2_state [0:15];
    reg [7:0] h1_spike [0:63];
    reg [7:0] h2_t0_spike [0:15];
    reg [8:0] temporal_sum [0:15];
    integer dense_lane;
    always @(posedge clk) begin
        if (linear_input_read_enable && dense_layer) begin
            h2_input_read_data <= h1_spike[linear_input_read_address[5:0]];
        end
        if (linear_state_read_enable) begin
            for (dense_lane = 0; dense_lane < 16;
                    dense_lane = dense_lane + 1) begin
                if (dense_layer) begin
                    linear_state_read_data_flat[dense_lane*16 +: 16]
                        <= h2_state[4'(
                            linear_state_read_channel_base + 7'(dense_lane)
                        )];
                end else begin
                    linear_state_read_data_flat[dense_lane*16 +: 16]
                        <= h1_state[6'(
                            linear_state_read_channel_base + 7'(dense_lane)
                        )];
                end
            end
        end
        if (linear_state_write_enable) begin
            for (dense_lane = 0; dense_lane < 16;
                    dense_lane = dense_lane + 1) begin
                if (dense_layer) begin
                    h2_state[4'(linear_state_write_channel_base
                            + 7'(dense_lane))]
                        <= linear_state_write_data_flat[
                            dense_lane*16 +: 16
                        ];
                end else begin
                    h1_state[6'(linear_state_write_channel_base
                            + 7'(dense_lane))]
                        <= linear_state_write_data_flat[
                            dense_lane*16 +: 16
                        ];
                end
            end
        end
        if (linear_output_valid) begin
            for (dense_lane = 0; dense_lane < 16;
                    dense_lane = dense_lane + 1) begin
                if (dense_layer) begin
                    if (!time_index) begin
                        h2_t0_spike[4'(linear_output_channel_base
                                + 7'(dense_lane))]
                            <= linear_output_spike_flat[
                            dense_lane*8 +: 8
                        ];
                    end else begin
                        temporal_sum[4'(linear_output_channel_base
                                + 7'(dense_lane))]
                            <= {1'b0, h2_t0_spike[4'(
                                linear_output_channel_base + 7'(dense_lane)
                            )]} + {1'b0, linear_output_spike_flat[
                            dense_lane*8 +: 8
                        ]};
                    end
                end else begin
                    h1_spike[6'(linear_output_channel_base
                            + 7'(dense_lane))]
                        <= linear_output_spike_flat[dense_lane*8 +: 8];
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

    wire trace_from_linear = (phase == PHASE_H1) || (phase == PHASE_H2);
    wire trace_from_replay = phase == PHASE_REPLAY;
    assign trace_valid = trace_from_linear ? linear_output_valid
        : (trace_from_replay ? replay_output_valid : conv_output_valid);
    assign trace_is_dense = trace_from_linear;
    assign trace_time_index = time_index;
    assign trace_layer = trace_from_linear ? (dense_layer ? 3'd1 : 3'd0)
                                            : layer_index;
    assign trace_address = trace_from_linear ? 14'd0
        : (trace_from_replay ? replay_output_address : conv_output_address);
    assign trace_channel_base = trace_from_linear
        ? linear_output_channel_base
        : {1'b0, (trace_from_replay ? replay_output_channel_base
                                    : conv_output_channel_base)};
    assign trace_current_flat = trace_from_linear ? linear_output_current_flat
        : {192'd0, (trace_from_replay ? replay_output_current_flat
                                      : conv_output_current_flat)};
    assign trace_spike_flat = trace_from_linear ? linear_output_spike_flat
        : {64'd0, (trace_from_replay ? replay_output_spike_flat
                                     : conv_output_spike_flat)};
    assign trace_membrane_flat = trace_from_linear ? linear_output_membrane_flat
        : {128'd0, (trace_from_replay ? replay_output_membrane_flat
                                      : conv_output_membrane_flat)};

    wire engines_any_busy = conv_busy || linear_busy || replay_busy
        || readout_busy;

    // --------------------- shared parameter/weight ports ------------------
    wire shared_parameter_read_enable = conv_parameter_read_enable
        || linear_parameter_read_enable || replay_parameter_read_enable;
    wire [8:0] shared_parameter_read_address = conv_parameter_read_enable
        ? conv_parameter_read_address
        : (linear_parameter_read_enable ? linear_parameter_read_address
                                        : replay_parameter_read_address);
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

    // -------------------------- top-level schedule ------------------------
    always @(posedge clk) begin
        if (rst) begin
            phase <= PHASE_IDLE;
            mode_register <= DATAFLOW_MODE_SL_TM;
            time_index <= 1'b0;
            layer_index <= 0;
            dense_layer <= 1'b0;
            conv_start <= 1'b0;
            linear_start <= 1'b0;
            replay_start <= 1'b0;
            readout_start <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
            result_valid <= 1'b0;
            config_error <= 1'b0;
            range_error_seen <= 1'b0;
            inference_cycles <= 0;
            convolution_cycles <= 0;
            dense_cycles <= 0;
            replay_cycles <= 0;
        end else begin
            done <= 1'b0;
            result_valid <= 1'b0;
            conv_start <= 1'b0;
            linear_start <= 1'b0;
            replay_start <= 1'b0;
            readout_start <= 1'b0;

            if (busy) begin
                inference_cycles <= inference_cycles + 1'b1;
                if (phase == PHASE_CONV) begin
                    convolution_cycles <= convolution_cycles + 1'b1;
                end else if ((phase == PHASE_H1) || (phase == PHASE_H2)) begin
                    dense_cycles <= dense_cycles + 1'b1;
                end else if (phase == PHASE_REPLAY) begin
                    replay_cycles <= replay_cycles + 1'b1;
                end
            end
            if ((conv_output_valid && conv_output_range_error)
                    || (linear_output_valid && linear_output_range_error)
                    || (replay_output_valid && replay_output_range_error)) begin
                range_error_seen <= 1'b1;
            end

            case (phase)
                PHASE_IDLE: begin
                    if (start && !busy && !engines_any_busy) begin
                        config_error <= 1'b0;
                        range_error_seen <= 1'b0;
                        inference_cycles <= 0;
                        convolution_cycles <= 0;
                        dense_cycles <= 0;
                        replay_cycles <= 0;
                        if ((dataflow_mode == DATAFLOW_MODE_FS_TM)
                                && (FULL_STATE_CAPABLE == 0)) begin
                            config_error <= 1'b1;
                            done <= 1'b1;
                        end else if (dataflow_mode > DATAFLOW_MODE_FS_TM) begin
                            config_error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            mode_register <= dataflow_mode;
                            time_index <= 1'b0;
                            layer_index <= 0;
                            busy <= 1'b1;
                            conv_start <= 1'b1;
                            phase <= PHASE_CONV;
                        end
                    end
                end

                PHASE_CONV: begin
                    if (conv_done) begin
                        if (layer_index == 7) begin
                            dense_layer <= 1'b0;
                            linear_start <= 1'b1;
                            phase <= PHASE_H1;
                        end else begin
                            layer_index <= layer_index + 1'b1;
                            conv_start <= 1'b1;
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
                        if (!time_index) begin
                            time_index <= 1'b1;
                            if (mode_register == DATAFLOW_MODE_SL_HYB) begin
                                layer_index <= 2;
                                replay_start <= 1'b1;
                                phase <= PHASE_REPLAY;
                            end else begin
                                layer_index <= 0;
                                conv_start <= 1'b1;
                                phase <= PHASE_CONV;
                            end
                        end else begin
                            readout_start <= 1'b1;
                            phase <= PHASE_READOUT;
                        end
                    end
                end

                PHASE_REPLAY: begin
                    if (replay_done) begin
                        layer_index <= 3;
                        conv_start <= 1'b1;
                        phase <= PHASE_CONV;
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

    initial begin
        if (FULL_STATE_CAPABLE != 0 && FULL_STATE_CAPABLE != 1) begin
            $error("FULL_STATE_CAPABLE must be zero or one");
        end
    end
endmodule

`default_nettype wire
