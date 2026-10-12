`timescale 1ns/1ps
`default_nettype none

module tb_pursuit_sparse_time_major_accelerator_top;
    localparam integer TRACE_VALUES = 90000;
    localparam integer L1_OUTPUT_SPATIAL = 3072;
    localparam integer L2_OUTPUT_SPATIAL = 768;
    localparam integer L1_WEIGHT_COUNT = 8*2*9;
    localparam integer L2_WEIGHT_COUNT = 16*8*9;

    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire result_valid;
    wire range_error_seen;
    wire protocol_error;

    reg feed_active = 1'b0;
    integer feed_index = 0;
    reg [14:0] sorted_events [0:399];
    wire event_valid = feed_active && feed_index < 400;
    wire event_ready;
    wire [14:0] current_event = (feed_index < 400)
        ? sorted_events[feed_index] : 15'd0;
    wire event_polarity = current_event[0];
    wire [6:0] event_row = current_event[14:8];
    wire [6:0] event_column = current_event[7:1];
    reg events_end = 1'b0;

    reg host_l1_weight_write_enable = 1'b0;
    reg [2:0] host_l1_weight_write_output_channel = 0;
    reg host_l1_weight_write_input_channel = 0;
    reg [3:0] host_l1_weight_write_kernel_index = 0;
    reg signed [15:0] host_l1_weight_write_data = 0;
    reg host_l1_channel_write_enable = 1'b0;
    reg [2:0] host_l1_channel_write_channel = 0;
    reg signed [23:0] host_l1_constant_write_data = 0;
    reg [4:0] host_l1_fraction_bits_write_data = 0;
    reg host_l2_weight_write_enable = 1'b0;
    reg [3:0] host_l2_weight_write_output_channel = 0;
    reg [2:0] host_l2_weight_write_input_channel = 0;
    reg [3:0] host_l2_weight_write_kernel_index = 0;
    reg signed [15:0] host_l2_weight_write_data = 0;
    reg host_l2_channel_write_enable = 1'b0;
    reg [3:0] host_l2_channel_write_channel = 0;
    reg signed [23:0] host_l2_constant_write_data = 0;
    reg [4:0] host_l2_fraction_bits_write_data = 0;

    reg host_conv_weight_write_enable = 1'b0;
    reg [12:0] host_conv_weight_write_address = 0;
    reg [255:0] host_conv_weight_write_data = 0;
    reg host_linear_weight_write_enable = 1'b0;
    reg [10:0] host_linear_weight_write_address = 0;
    reg [255:0] host_linear_weight_write_data = 0;
    reg host_parameter_write_enable = 1'b0;
    reg [8:0] host_parameter_write_address = 0;
    reg signed [23:0] host_parameter_constant_data = 0;
    reg [4:0] host_parameter_fraction_data = 0;
    reg host_readout_fraction_write_enable = 1'b0;
    reg host_readout_fraction_write_address = 1'b0;
    reg [4:0] host_readout_fraction_write_data = 0;

    wire [127:0] result_integer_flat;
    wire [9:0] result_fraction_bits_flat;
    wire [31:0] inference_cycles;
    wire [31:0] frontend_cycles;
    wire [31:0] convolution_cycles;
    wire [31:0] dense_cycles;
    wire [31:0] conv_weight_read_beats;
    wire [31:0] linear_weight_read_beats;
    wire [31:0] replay_cycles;
    wire [31:0] state_read_transactions;
    wire [31:0] state_write_transactions;
    wire [31:0] accepted_event_count;
    wire [31:0] l1_emitted_packet_count;
    wire [31:0] l2_accepted_packet_count;
    wire trace_valid;
    wire trace_is_dense;
    wire trace_time_index;
    wire [2:0] trace_layer;
    wire [13:0] trace_address;
    wire [6:0] trace_channel_base;
    wire [383:0] trace_current_flat;
    wire [127:0] trace_spike_flat;
    wire [255:0] trace_membrane_flat;

    pursuit_sparse_time_major_accelerator_top dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .result_valid(result_valid), .range_error_seen(range_error_seen),
        .protocol_error(protocol_error),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_polarity(event_polarity), .event_row(event_row),
        .event_column(event_column), .events_end(events_end),
        .host_l1_weight_write_enable(host_l1_weight_write_enable),
        .host_l1_weight_write_output_channel(
            host_l1_weight_write_output_channel
        ),
        .host_l1_weight_write_input_channel(
            host_l1_weight_write_input_channel
        ),
        .host_l1_weight_write_kernel_index(
            host_l1_weight_write_kernel_index
        ),
        .host_l1_weight_write_data(host_l1_weight_write_data),
        .host_l1_channel_write_enable(host_l1_channel_write_enable),
        .host_l1_channel_write_channel(host_l1_channel_write_channel),
        .host_l1_constant_write_data(host_l1_constant_write_data),
        .host_l1_fraction_bits_write_data(
            host_l1_fraction_bits_write_data
        ),
        .host_l2_weight_write_enable(host_l2_weight_write_enable),
        .host_l2_weight_write_output_channel(
            host_l2_weight_write_output_channel
        ),
        .host_l2_weight_write_input_channel(
            host_l2_weight_write_input_channel
        ),
        .host_l2_weight_write_kernel_index(
            host_l2_weight_write_kernel_index
        ),
        .host_l2_weight_write_data(host_l2_weight_write_data),
        .host_l2_channel_write_enable(host_l2_channel_write_enable),
        .host_l2_channel_write_channel(host_l2_channel_write_channel),
        .host_l2_constant_write_data(host_l2_constant_write_data),
        .host_l2_fraction_bits_write_data(
            host_l2_fraction_bits_write_data
        ),
        .host_conv_weight_write_enable(host_conv_weight_write_enable),
        .host_conv_weight_write_address(host_conv_weight_write_address),
        .host_conv_weight_write_data(host_conv_weight_write_data),
        .host_linear_weight_write_enable(host_linear_weight_write_enable),
        .host_linear_weight_write_address(host_linear_weight_write_address),
        .host_linear_weight_write_data(host_linear_weight_write_data),
        .host_parameter_write_enable(host_parameter_write_enable),
        .host_parameter_write_address(host_parameter_write_address),
        .host_parameter_constant_data(host_parameter_constant_data),
        .host_parameter_fraction_data(host_parameter_fraction_data),
        .host_readout_fraction_write_enable(
            host_readout_fraction_write_enable
        ),
        .host_readout_fraction_write_address(
            host_readout_fraction_write_address
        ),
        .host_readout_fraction_write_data(
            host_readout_fraction_write_data
        ),
        .result_integer_flat(result_integer_flat),
        .result_fraction_bits_flat(result_fraction_bits_flat),
        .inference_cycles(inference_cycles),
        .frontend_cycles(frontend_cycles),
        .convolution_cycles(convolution_cycles),
        .dense_cycles(dense_cycles),
        .replay_cycles(replay_cycles),
        .conv_weight_read_beats(conv_weight_read_beats),
        .linear_weight_read_beats(linear_weight_read_beats),
        .state_read_transactions(state_read_transactions),
        .state_write_transactions(state_write_transactions),
        .accepted_event_count(accepted_event_count),
        .l1_emitted_packet_count(l1_emitted_packet_count),
        .l2_accepted_packet_count(l2_accepted_packet_count),
        .trace_valid(trace_valid),
        .trace_is_dense(trace_is_dense),
        .trace_time_index(trace_time_index),
        .trace_layer(trace_layer),
        .trace_address(trace_address),
        .trace_channel_base(trace_channel_base),
        .trace_current_flat(trace_current_flat),
        .trace_spike_flat(trace_spike_flat),
        .trace_membrane_flat(trace_membrane_flat)
    );

    reg signed [15:0] l1_weights [0:L1_WEIGHT_COUNT-1];
    reg signed [23:0] l1_constants [0:7];
    reg [4:0] l1_fractions [0:7];
    reg signed [15:0] l2_weights [0:L2_WEIGHT_COUNT-1];
    reg signed [23:0] l2_constants [0:15];
    reg [4:0] l2_fractions [0:15];
    reg [255:0] conv_weights [0:4157];
    reg [255:0] linear_weights [0:1991];
    reg signed [23:0] constants [0:319];
    reg [4:0] fractions [0:319];
    reg [4:0] readout_fractions [0:1];
    reg signed [23:0] expected_current [0:TRACE_VALUES-1];
    reg [7:0] expected_spike [0:TRACE_VALUES-1];
    reg signed [15:0] expected_membrane [0:TRACE_VALUES-1];
    reg signed [63:0] expected_result [0:1];

    string event_file;
    string l1_weight_file;
    string l1_constant_file;
    string l1_fraction_file;
    string l2_weight_file;
    string l2_constant_file;
    string l2_fraction_file;
    string conv_weight_file;
    string linear_weight_file;
    string constant_file;
    string fraction_file;
    string readout_fraction_file;
    string expected_current_file;
    string expected_spike_file;
    string expected_membrane_file;
    string expected_result_file;
    integer expected_l1_packets;
    integer timeout_cycles;

    function integer layer_offset;
        input [3:0] layer;
        begin
            case (layer)
                0: layer_offset = 0;
                1: layer_offset = 24576;
                2: layer_offset = 36864;
                3: layer_offset = 41472;
                4: layer_offset = 43008;
                5: layer_offset = 43488;
                6: layer_offset = 43968;
                7: layer_offset = 44448;
                8: layer_offset = 44928;
                default: layer_offset = 44988;
            endcase
        end
    endfunction

    function integer layer_spatial;
        input [3:0] layer;
        begin
            case (layer)
                0: layer_spatial = 3072;
                1: layer_spatial = 768;
                2: layer_spatial = 192;
                3: layer_spatial = 48;
                4, 5, 6, 7: layer_spatial = 12;
                default: layer_spatial = 1;
            endcase
        end
    endfunction

    function integer layer_channels;
        input [3:0] layer;
        begin
            case (layer)
                0: layer_channels = 8;
                1: layer_channels = 16;
                2: layer_channels = 24;
                3: layer_channels = 32;
                4, 5, 6, 7: layer_channels = 40;
                8: layer_channels = 60;
                default: layer_channels = 12;
            endcase
        end
    endfunction

    integer trace_transactions;
    integer trace_values_checked;
    integer l1_trace_transactions;
    integer l2_trace_transactions;
    integer backend_trace_transactions;
    integer frontend_trace_collisions;
    integer errors;
    integer report_errors;
    integer lane;
    integer valid_lanes;
    integer output_channel;
    integer logical_layer;
    integer expected_index;

    task automatic compare_value;
        input integer time_index;
        input integer layer;
        input integer address;
        input integer channel;
        input signed [23:0] actual_current;
        input [7:0] actual_spike;
        input signed [15:0] actual_membrane;
        begin
            expected_index = time_index*45000 + layer_offset(layer[3:0])
                + channel*layer_spatial(layer[3:0]) + address;
            trace_values_checked = trace_values_checked + 3;
            if (actual_current !== expected_current[expected_index]) begin
                if (report_errors < 20) $display(
                    "ERROR current t=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                    time_index, layer, channel, address, actual_current,
                    expected_current[expected_index]);
                report_errors = report_errors + 1;
                errors = errors + 1;
            end
            if (actual_spike !== expected_spike[expected_index]) begin
                if (report_errors < 20) $display(
                    "ERROR spike t=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                    time_index, layer, channel, address, actual_spike,
                    expected_spike[expected_index]);
                report_errors = report_errors + 1;
                errors = errors + 1;
            end
            if (actual_membrane !== expected_membrane[expected_index]) begin
                if (report_errors < 20) $display(
                    "ERROR membrane t=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                    time_index, layer, channel, address, actual_membrane,
                    expected_membrane[expected_index]);
                report_errors = report_errors + 1;
                errors = errors + 1;
            end
        end
    endtask

    always @(posedge clk) begin
        if (!rst && dut.l1_trace_valid) begin
            l1_trace_transactions = l1_trace_transactions + 1;
            trace_transactions = trace_transactions + 1;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                compare_value(
                    0, 0, dut.l1_trace_address, lane,
                    $signed(dut.l1_trace_current_flat[lane*24 +: 24]),
                    dut.l1_trace_spike_flat[lane*8 +: 8],
                    $signed(dut.l1_trace_membrane_flat[lane*16 +: 16])
                );
            end
        end
        if (!rst && dut.l2_trace_valid) begin
            l2_trace_transactions = l2_trace_transactions + 1;
            trace_transactions = trace_transactions + 1;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                compare_value(
                    0, 1, dut.l2_trace_address, lane,
                    $signed(dut.l2_trace_current_flat[lane*24 +: 24]),
                    dut.l2_trace_spike_flat[lane*8 +: 8],
                    $signed(dut.l2_trace_membrane_flat[lane*16 +: 16])
                );
            end
        end
        if (!rst && dut.l1_trace_valid && dut.l2_trace_valid) begin
            frontend_trace_collisions = frontend_trace_collisions + 1;
        end
        if (!rst && trace_valid && (trace_is_dense || trace_layer >= 2)) begin
            backend_trace_transactions = backend_trace_transactions + 1;
            trace_transactions = trace_transactions + 1;
            valid_lanes = trace_is_dense ? 16 : 8;
            logical_layer = trace_is_dense
                ? (trace_layer == 0 ? 8 : 9) : trace_layer;
            for (lane = 0; lane < valid_lanes; lane = lane + 1) begin
                output_channel = trace_channel_base + lane;
                if (output_channel < layer_channels(logical_layer)) begin
                    compare_value(
                        trace_time_index, logical_layer, trace_address,
                        output_channel,
                        $signed(trace_current_flat[lane*24 +: 24]),
                        trace_spike_flat[lane*8 +: 8],
                        $signed(trace_membrane_flat[lane*16 +: 16])
                    );
                end
            end
        end
        if (!rst && busy && inference_cycles > timeout_cycles) begin
            $fatal(1, "Sparse-TM timeout after %0d cycles",
                inference_cycles);
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            feed_index <= 0;
        end else if (event_valid && event_ready) begin
            feed_index <= feed_index + 1;
        end
    end

    integer ok;
    integer index;
    initial begin
        ok = 1;
        ok &= $value$plusargs("EVENT_FILE=%s", event_file);
        ok &= $value$plusargs("L1_WEIGHT_FILE=%s", l1_weight_file);
        ok &= $value$plusargs("L1_CONSTANT_FILE=%s", l1_constant_file);
        ok &= $value$plusargs("L1_FRACTION_FILE=%s", l1_fraction_file);
        ok &= $value$plusargs("L2_WEIGHT_FILE=%s", l2_weight_file);
        ok &= $value$plusargs("L2_CONSTANT_FILE=%s", l2_constant_file);
        ok &= $value$plusargs("L2_FRACTION_FILE=%s", l2_fraction_file);
        ok &= $value$plusargs("CONV_WEIGHT_FILE=%s", conv_weight_file);
        ok &= $value$plusargs("LINEAR_WEIGHT_FILE=%s", linear_weight_file);
        ok &= $value$plusargs("CONSTANT_FILE=%s", constant_file);
        ok &= $value$plusargs("FRACTION_FILE=%s", fraction_file);
        ok &= $value$plusargs(
            "READOUT_FRACTION_FILE=%s", readout_fraction_file
        );
        ok &= $value$plusargs(
            "EXPECTED_CURRENT_FILE=%s", expected_current_file
        );
        ok &= $value$plusargs(
            "EXPECTED_SPIKE_FILE=%s", expected_spike_file
        );
        ok &= $value$plusargs(
            "EXPECTED_MEMBRANE_FILE=%s", expected_membrane_file
        );
        ok &= $value$plusargs(
            "EXPECTED_RESULT_FILE=%s", expected_result_file
        );
        ok &= $value$plusargs("EXPECTED_L1_PACKETS=%d", expected_l1_packets);
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
            timeout_cycles = 200000;
        end
        if (!ok) $fatal(1, "missing required Sparse-TM replay plusarg");

        $readmemh(event_file, sorted_events);
        $readmemh(l1_weight_file, l1_weights);
        $readmemh(l1_constant_file, l1_constants);
        $readmemh(l1_fraction_file, l1_fractions);
        $readmemh(l2_weight_file, l2_weights);
        $readmemh(l2_constant_file, l2_constants);
        $readmemh(l2_fraction_file, l2_fractions);
        $readmemh(conv_weight_file, conv_weights);
        $readmemh(linear_weight_file, linear_weights);
        $readmemh(constant_file, constants);
        $readmemh(fraction_file, fractions);
        $readmemh(readout_fraction_file, readout_fractions);
        $readmemh(expected_current_file, expected_current);
        $readmemh(expected_spike_file, expected_spike);
        $readmemh(expected_membrane_file, expected_membrane);
        $readmemh(expected_result_file, expected_result);

        trace_transactions = 0;
        trace_values_checked = 0;
        l1_trace_transactions = 0;
        l2_trace_transactions = 0;
        backend_trace_transactions = 0;
        frontend_trace_collisions = 0;
        errors = 0;
        report_errors = 0;

        repeat (5) @(posedge clk);
        @(negedge clk); rst = 1'b0;

        for (index = 0; index < L1_WEIGHT_COUNT; index = index + 1) begin
            @(negedge clk);
            host_l1_weight_write_enable = 1'b1;
            host_l1_weight_write_output_channel = index / 18;
            host_l1_weight_write_input_channel = (index / 9) % 2;
            host_l1_weight_write_kernel_index = index % 9;
            host_l1_weight_write_data = l1_weights[index];
        end
        @(negedge clk); host_l1_weight_write_enable = 1'b0;
        for (index = 0; index < 8; index = index + 1) begin
            @(negedge clk);
            host_l1_channel_write_enable = 1'b1;
            host_l1_channel_write_channel = index;
            host_l1_constant_write_data = l1_constants[index];
            host_l1_fraction_bits_write_data = l1_fractions[index];
        end
        @(negedge clk); host_l1_channel_write_enable = 1'b0;

        for (index = 0; index < L2_WEIGHT_COUNT; index = index + 1) begin
            @(negedge clk);
            host_l2_weight_write_enable = 1'b1;
            host_l2_weight_write_output_channel = index / 72;
            host_l2_weight_write_input_channel = (index / 9) % 8;
            host_l2_weight_write_kernel_index = index % 9;
            host_l2_weight_write_data = l2_weights[index];
        end
        @(negedge clk); host_l2_weight_write_enable = 1'b0;
        for (index = 0; index < 16; index = index + 1) begin
            @(negedge clk);
            host_l2_channel_write_enable = 1'b1;
            host_l2_channel_write_channel = index;
            host_l2_constant_write_data = l2_constants[index];
            host_l2_fraction_bits_write_data = l2_fractions[index];
        end
        @(negedge clk); host_l2_channel_write_enable = 1'b0;

        for (index = 0; index < 4158; index = index + 1) begin
            @(negedge clk);
            host_conv_weight_write_enable = 1'b1;
            host_conv_weight_write_address = index;
            host_conv_weight_write_data = conv_weights[index];
        end
        @(negedge clk); host_conv_weight_write_enable = 1'b0;
        for (index = 0; index < 1992; index = index + 1) begin
            @(negedge clk);
            host_linear_weight_write_enable = 1'b1;
            host_linear_weight_write_address = index;
            host_linear_weight_write_data = linear_weights[index];
        end
        @(negedge clk); host_linear_weight_write_enable = 1'b0;
        for (index = 0; index < 320; index = index + 1) begin
            @(negedge clk);
            host_parameter_write_enable = 1'b1;
            host_parameter_write_address = index;
            host_parameter_constant_data = constants[index];
            host_parameter_fraction_data = fractions[index];
        end
        @(negedge clk); host_parameter_write_enable = 1'b0;
        for (index = 0; index < 2; index = index + 1) begin
            @(negedge clk);
            host_readout_fraction_write_enable = 1'b1;
            host_readout_fraction_write_address = index;
            host_readout_fraction_write_data = readout_fractions[index];
        end
        @(negedge clk); host_readout_fraction_write_enable = 1'b0;

        repeat (2) @(posedge clk);
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0; feed_active = 1'b1;
        wait (feed_index == 400);
        @(negedge clk); feed_active = 1'b0; events_end = 1'b1;
        @(negedge clk); events_end = 1'b0;

        wait(done);
        #1;
        if (!result_valid) begin
            $display("ERROR result_valid not asserted with done");
            errors = errors + 1;
        end
        if (protocol_error || range_error_seen) begin
            $display("ERROR status protocol=%0d range=%0d",
                protocol_error, range_error_seen);
            errors = errors + 1;
        end
        for (index = 0; index < 2; index = index + 1) begin
            if ($signed(result_integer_flat[index*64 +: 64])
                    !== expected_result[index]) begin
                $display("ERROR final output %0d got=%0d expected=%0d",
                    index, $signed(result_integer_flat[index*64 +: 64]),
                    expected_result[index]);
                errors = errors + 1;
            end
            if (result_fraction_bits_flat[index*5 +: 5]
                    !== readout_fractions[index]) begin
                $display("ERROR final fraction %0d", index);
                errors = errors + 1;
            end
        end
        if (accepted_event_count != 400
                || l1_emitted_packet_count != expected_l1_packets
                || l2_accepted_packet_count != expected_l1_packets) begin
            $display("ERROR sparse counts events=%0d L1=%0d L2=%0d expected_packets=%0d",
                accepted_event_count, l1_emitted_packet_count,
                l2_accepted_packet_count, expected_l1_packets);
            errors = errors + 1;
        end
        if (l1_trace_transactions != L1_OUTPUT_SPATIAL
                || l2_trace_transactions != L2_OUTPUT_SPATIAL
                || backend_trace_transactions != 2026
                || trace_transactions != 5866
                || trace_values_checked != 159408) begin
            $display("ERROR trace counts all=%0d values=%0d L1=%0d L2=%0d backend=%0d",
                trace_transactions, trace_values_checked,
                l1_trace_transactions, l2_trace_transactions,
                backend_trace_transactions);
            errors = errors + 1;
        end
        if (inference_cycles != 72431
                || frontend_cycles != 15259
                || convolution_cycles != 52350
                || dense_cycles != 4188
                || replay_cycles != 617) begin
            $display("ERROR cycle counts total=%0d front=%0d conv=%0d dense=%0d replay=%0d",
                inference_cycles, frontend_cycles, convolution_cycles,
                dense_cycles, replay_cycles);
            errors = errors + 1;
        end
        if (conv_weight_read_beats != 7920
                || linear_weight_read_beats != 3972
                || state_read_transactions != 1013
                || state_write_transactions != 2026) begin
            $display("ERROR access counts convW=%0d linearW=%0d stateR=%0d stateW=%0d",
                conv_weight_read_beats, linear_weight_read_beats,
                state_read_transactions, state_write_transactions);
            errors = errors + 1;
        end
        if (errors != 0) begin
            $fatal(1, "Sparse-TM replay failed errors=%0d", errors);
        end
        $display(
            "PASS: transactions=%0d values=%0d total_cycles=%0d frontend_cycles=%0d convolution_cycles=%0d dense_cycles=%0d replay_cycles=%0d conv_weight_reads=%0d linear_weight_reads=%0d state_reads=%0d state_writes=%0d events=%0d l1_packets=%0d l2_packets=%0d frontend_trace_collisions=%0d",
            trace_transactions, trace_values_checked, inference_cycles,
            frontend_cycles, convolution_cycles, dense_cycles,
            replay_cycles,
            conv_weight_read_beats, linear_weight_read_beats,
            state_read_transactions, state_write_transactions,
            accepted_event_count,
            l1_emitted_packet_count, l2_accepted_packet_count,
            frontend_trace_collisions
        );
        $finish;
    end
endmodule

`default_nettype wire
