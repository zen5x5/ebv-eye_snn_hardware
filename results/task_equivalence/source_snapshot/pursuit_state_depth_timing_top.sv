`timescale 1ns/1ps
`default_nettype none

// Physical timing evaluation harness. All host signals use the compute clock.
// This pin-level register bus is an evaluation interface, not an AUP-ZU3 pinout.
module pursuit_state_depth_timing_top #(
    parameter integer STATELESS_DEPTH = 2
) (
    input wire clk,
    input wire rst,
    input wire host_write_enable,
    input wire [7:0] host_address,
    input wire [31:0] host_write_data,
    output reg [31:0] host_read_data
);
    wire compute_clk;
    BUFGCE compute_clock_buffer (.I(clk), .CE(1'b1), .O(compute_clk));

    reg core_reset;
    reg write_enable;
    reg [7:0] address;
    reg [31:0] write_data;
    always @(posedge compute_clk) begin
        core_reset <= rst;
        write_enable <= host_write_enable;
        address <= host_address;
        write_data <= host_write_data;
    end

    reg [14:0] input_address;
    reg [12:0] weight_address;
    reg [255:0] weight_data;
    reg [8:0] parameter_address;
    reg [23:0] parameter_constant;
    always @(posedge compute_clk) begin
        if (write_enable) begin
            case (address)
                8'h01: input_address <= write_data[14:0];
                8'h03: weight_address <= write_data[12:0];
                8'h04: weight_data[0 +: 32] <= write_data;
                8'h05: weight_data[32 +: 32] <= write_data;
                8'h06: weight_data[64 +: 32] <= write_data;
                8'h07: weight_data[96 +: 32] <= write_data;
                8'h08: weight_data[128 +: 32] <= write_data;
                8'h09: weight_data[160 +: 32] <= write_data;
                8'h0a: weight_data[192 +: 32] <= write_data;
                8'h0b: weight_data[224 +: 32] <= write_data;
                8'h0d: parameter_address <= write_data[8:0];
                8'h0e: parameter_constant <= write_data[23:0];
                default: begin end
            endcase
        end
    end

    wire busy, done, result_valid, range_error_seen;
    wire [127:0] result_integer_flat;
    wire [9:0] result_fraction_bits_flat;
    wire [31:0] inference_cycles, frontend_cycles, convolution_cycles;
    wire [31:0] dense_cycles, conv_weight_read_beats, linear_weight_read_beats;
    wire [31:0] readout_weight_read_beats, feature_read_requests;
    wire [31:0] feature_write_transactions;
    wire trace_valid, trace_is_dense, trace_time_index;
    wire [2:0] trace_layer;
    wire [13:0] trace_address;
    wire [6:0] trace_channel_base;
    wire [383:0] trace_current_flat;
    wire [127:0] trace_spike_flat;
    wire [255:0] trace_membrane_flat;

    (* keep_hierarchy = "yes" *)
    pursuit_state_depth_accelerator_top #(.STATELESS_DEPTH(STATELESS_DEPTH)) core (
        .clk(compute_clk), .rst(core_reset),
        .start(write_enable && address == 8'h00 && write_data[0]),
        .busy(busy), .done(done), .result_valid(result_valid),
        .range_error_seen(range_error_seen),
        .host_input_write_enable(write_enable && address == 8'h02),
        .host_input_write_channel(input_address[14]),
        .host_input_write_address(input_address[13:0]),
        .host_input_write_data(write_data[8:0]),
        .host_conv_weight_write_enable(write_enable && address == 8'h0c && write_data[0]),
        .host_conv_weight_write_address(weight_address),
        .host_conv_weight_write_data(weight_data),
        .host_linear_weight_write_enable(write_enable && address == 8'h0c && write_data[1]),
        .host_linear_weight_write_address(weight_address[10:0]),
        .host_linear_weight_write_data(weight_data),
        .host_parameter_write_enable(write_enable && address == 8'h10),
        .host_parameter_write_address(parameter_address),
        .host_parameter_constant_data(parameter_constant),
        .host_parameter_fraction_data(write_data[4:0]),
        .host_readout_fraction_write_enable(write_enable && address == 8'h11),
        .host_readout_fraction_write_address(write_data[5]),
        .host_readout_fraction_write_data(write_data[4:0]),
        .result_integer_flat(result_integer_flat),
        .result_fraction_bits_flat(result_fraction_bits_flat),
        .inference_cycles(inference_cycles), .frontend_cycles(frontend_cycles),
        .convolution_cycles(convolution_cycles), .dense_cycles(dense_cycles),
        .conv_weight_read_beats(conv_weight_read_beats),
        .linear_weight_read_beats(linear_weight_read_beats),
        .readout_weight_read_beats(readout_weight_read_beats),
        .feature_read_requests(feature_read_requests),
        .feature_write_transactions(feature_write_transactions),
        .trace_valid(trace_valid), .trace_is_dense(trace_is_dense),
        .trace_time_index(trace_time_index), .trace_layer(trace_layer),
        .trace_address(trace_address), .trace_channel_base(trace_channel_base),
        .trace_current_flat(trace_current_flat), .trace_spike_flat(trace_spike_flat),
        .trace_membrane_flat(trace_membrane_flat)
    );

    reg completed;
    reg [127:0] result_snapshot;
    reg [9:0] fraction_snapshot;
    reg [799:0] trace_snapshot;
    always @(posedge compute_clk) begin
        if (core_reset) completed <= 1'b0;
        else if (write_enable && address == 8'h00 && write_data[0]) completed <= 1'b0;
        else if (done) completed <= 1'b1;
        if (result_valid) begin
            result_snapshot <= result_integer_flat;
            fraction_snapshot <= result_fraction_bits_flat;
        end
        if (trace_valid) begin
            trace_snapshot <= {6'b0, trace_is_dense, trace_time_index,
                trace_layer, trace_address, trace_channel_base,
                trace_membrane_flat, trace_spike_flat, trace_current_flat};
        end
        case (address)
            8'h00: host_read_data <= {28'b0, range_error_seen, result_valid, completed, busy};
            8'h20: host_read_data <= result_snapshot[0 +: 32];
            8'h21: host_read_data <= result_snapshot[32 +: 32];
            8'h22: host_read_data <= result_snapshot[64 +: 32];
            8'h23: host_read_data <= result_snapshot[96 +: 32];
            8'h24: host_read_data <= {22'b0, fraction_snapshot};
            8'h25: host_read_data <= inference_cycles;
            8'h26: host_read_data <= frontend_cycles;
            8'h27: host_read_data <= convolution_cycles;
            8'h28: host_read_data <= dense_cycles;
            8'h29: host_read_data <= conv_weight_read_beats;
            8'h2a: host_read_data <= linear_weight_read_beats;
            8'h2b: host_read_data <= readout_weight_read_beats;
            8'h2c: host_read_data <= feature_read_requests;
            8'h2d: host_read_data <= feature_write_transactions;
            default: host_read_data <= 32'b0;
        endcase
        if (address >= 8'h40 && address <= 8'h58)
            host_read_data <= trace_snapshot[(address - 8'h40)*32 +: 32];
    end
endmodule

`default_nettype wire
