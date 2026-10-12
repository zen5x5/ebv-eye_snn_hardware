`timescale 1ns/1ps
`default_nettype none

module tb_pursuit_snn_accelerator_top #(
    parameter integer FULL_STATE_CAPABLE = 0
);
    localparam integer TRACE_VALUES = 90000;
    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    reg [1:0] dataflow_mode;
    wire busy;
    wire done;
    wire result_valid;
    wire config_error;
    wire range_error_seen;

    reg host_input_write_enable = 1'b0;
    reg host_input_write_channel = 1'b0;
    reg [13:0] host_input_write_address = 0;
    reg [8:0] host_input_write_data = 0;
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
    wire [31:0] convolution_cycles;
    wire [31:0] dense_cycles;
    wire [31:0] replay_cycles;
    wire trace_valid;
    wire trace_is_dense;
    wire trace_time_index;
    wire [2:0] trace_layer;
    wire [13:0] trace_address;
    wire [6:0] trace_channel_base;
    wire [383:0] trace_current_flat;
    wire [127:0] trace_spike_flat;
    wire [255:0] trace_membrane_flat;

    pursuit_snn_accelerator_top #(
        .FULL_STATE_CAPABLE(FULL_STATE_CAPABLE)
    ) dut (
        .clk(clk), .rst(rst), .start(start), .dataflow_mode(dataflow_mode),
        .busy(busy), .done(done), .result_valid(result_valid),
        .config_error(config_error), .range_error_seen(range_error_seen),
        .host_input_write_enable(host_input_write_enable),
        .host_input_write_channel(host_input_write_channel),
        .host_input_write_address(host_input_write_address),
        .host_input_write_data(host_input_write_data),
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
        .host_readout_fraction_write_data(host_readout_fraction_write_data),
        .result_integer_flat(result_integer_flat),
        .result_fraction_bits_flat(result_fraction_bits_flat),
        .inference_cycles(inference_cycles),
        .convolution_cycles(convolution_cycles),
        .dense_cycles(dense_cycles), .replay_cycles(replay_cycles),
        .trace_valid(trace_valid), .trace_is_dense(trace_is_dense),
        .trace_time_index(trace_time_index), .trace_layer(trace_layer),
        .trace_address(trace_address),
        .trace_channel_base(trace_channel_base),
        .trace_current_flat(trace_current_flat),
        .trace_spike_flat(trace_spike_flat),
        .trace_membrane_flat(trace_membrane_flat)
    );

    reg [255:0] conv_weights [0:4157];
    reg [255:0] linear_weights [0:1991];
    reg signed [23:0] constants [0:319];
    reg [4:0] fractions [0:319];
    reg [4:0] readout_fractions [0:1];
    reg [8:0] input_frame [0:24575];
    reg signed [23:0] expected_current [0:TRACE_VALUES-1];
    reg [7:0] expected_spike [0:TRACE_VALUES-1];
    reg signed [15:0] expected_membrane [0:TRACE_VALUES-1];
    reg signed [63:0] expected_result [0:1];

    string conv_weight_file;
    string linear_weight_file;
    string constant_file;
    string fraction_file;
    string readout_fraction_file;
    string input_file;
    string expected_current_file;
    string expected_spike_file;
    string expected_membrane_file;
    string expected_result_file;
    integer requested_mode;

    function integer layer_offset;
        input [2:0] layer;
        begin
            case (layer)
                0: layer_offset = 0;
                1: layer_offset = 24576;
                2: layer_offset = 36864;
                3: layer_offset = 41472;
                4: layer_offset = 43008;
                5: layer_offset = 43488;
                6: layer_offset = 43968;
                default: layer_offset = 44448;
            endcase
        end
    endfunction

    function integer layer_spatial;
        input [2:0] layer;
        begin
            case (layer)
                0: layer_spatial = 3072;
                1: layer_spatial = 768;
                2: layer_spatial = 192;
                3: layer_spatial = 48;
                default: layer_spatial = 12;
            endcase
        end
    endfunction

    integer trace_transactions;
    integer trace_values_checked;
    integer errors;
    integer lane;
    integer valid_lanes;
    integer output_channel;
    integer expected_index;
    integer report_errors;
    always @(posedge clk) begin
        if (!rst && trace_valid) begin
            trace_transactions = trace_transactions + 1;
            valid_lanes = trace_is_dense ? 16 : 8;
            for (lane = 0; lane < valid_lanes; lane = lane + 1) begin
                output_channel = trace_channel_base + lane;
                if (!trace_is_dense) begin
                    expected_index = trace_time_index*45000
                        + layer_offset(trace_layer)
                        + output_channel*layer_spatial(trace_layer)
                        + trace_address;
                end else if (trace_layer == 0 && output_channel < 60) begin
                    expected_index = trace_time_index*45000 + 44928
                        + output_channel;
                end else if (trace_layer == 1 && output_channel < 12) begin
                    expected_index = trace_time_index*45000 + 44988
                        + output_channel;
                end else begin
                    expected_index = -1;
                end
                if (expected_index >= 0) begin
                    trace_values_checked = trace_values_checked + 3;
                    if ($signed(trace_current_flat[lane*24 +: 24])
                            !== expected_current[expected_index]) begin
                        if (report_errors < 20) $display(
                            "ERROR current t=%0d dense=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                            trace_time_index, trace_is_dense, trace_layer,
                            output_channel, trace_address,
                            $signed(trace_current_flat[lane*24 +: 24]),
                            expected_current[expected_index]);
                        report_errors = report_errors + 1;
                        errors = errors + 1;
                    end
                    if (trace_spike_flat[lane*8 +: 8]
                            !== expected_spike[expected_index]) begin
                        if (report_errors < 20) $display(
                            "ERROR spike t=%0d dense=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                            trace_time_index, trace_is_dense, trace_layer,
                            output_channel, trace_address,
                            trace_spike_flat[lane*8 +: 8],
                            expected_spike[expected_index]);
                        report_errors = report_errors + 1;
                        errors = errors + 1;
                    end
                    if ($signed(trace_membrane_flat[lane*16 +: 16])
                            !== expected_membrane[expected_index]) begin
                        if (report_errors < 20) $display(
                            "ERROR membrane t=%0d dense=%0d layer=%0d c=%0d a=%0d got=%0d expected=%0d",
                            trace_time_index, trace_is_dense, trace_layer,
                            output_channel, trace_address,
                            $signed(trace_membrane_flat[lane*16 +: 16]),
                            expected_membrane[expected_index]);
                        report_errors = report_errors + 1;
                        errors = errors + 1;
                    end
                end
            end
        end
        if (!rst && busy && inference_cycles > 200000) begin
            $fatal(1, "full accelerator timeout");
        end
    end

    integer ok;
    integer index;
    integer expected_transactions;
    integer expected_values_checked;
    initial begin
        ok = 1;
        ok &= $value$plusargs("CONV_WEIGHT_FILE=%s", conv_weight_file);
        ok &= $value$plusargs("LINEAR_WEIGHT_FILE=%s", linear_weight_file);
        ok &= $value$plusargs("CONSTANT_FILE=%s", constant_file);
        ok &= $value$plusargs("FRACTION_FILE=%s", fraction_file);
        ok &= $value$plusargs("READOUT_FRACTION_FILE=%s", readout_fraction_file);
        ok &= $value$plusargs("INPUT_FILE=%s", input_file);
        ok &= $value$plusargs("EXPECTED_CURRENT_FILE=%s", expected_current_file);
        ok &= $value$plusargs("EXPECTED_SPIKE_FILE=%s", expected_spike_file);
        ok &= $value$plusargs("EXPECTED_MEMBRANE_FILE=%s", expected_membrane_file);
        ok &= $value$plusargs("EXPECTED_RESULT_FILE=%s", expected_result_file);
        ok &= $value$plusargs("DATAFLOW_MODE=%d", requested_mode);
        if (!ok) $fatal(1, "missing required full-replay plusarg");
        if (requested_mode < 0 || requested_mode > 2) begin
            $fatal(1, "invalid dataflow mode");
        end
        if (requested_mode == 2 && FULL_STATE_CAPABLE == 0) begin
            $fatal(1, "FS-TM requires FULL_STATE_CAPABLE=1");
        end
        dataflow_mode = requested_mode[1:0];
        $readmemh(conv_weight_file, conv_weights);
        $readmemh(linear_weight_file, linear_weights);
        $readmemh(constant_file, constants);
        $readmemh(fraction_file, fractions);
        $readmemh(readout_fraction_file, readout_fractions);
        $readmemh(input_file, input_frame);
        $readmemh(expected_current_file, expected_current);
        $readmemh(expected_spike_file, expected_spike);
        $readmemh(expected_membrane_file, expected_membrane);
        $readmemh(expected_result_file, expected_result);
        trace_transactions = 0;
        trace_values_checked = 0;
        errors = 0;
        report_errors = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;

        for (index = 0; index < 4158; index = index + 1) begin
            @(negedge clk);
            host_conv_weight_write_enable = 1'b1;
            host_conv_weight_write_address = index[12:0];
            host_conv_weight_write_data = conv_weights[index];
        end
        @(negedge clk); host_conv_weight_write_enable = 1'b0;
        for (index = 0; index < 1992; index = index + 1) begin
            @(negedge clk);
            host_linear_weight_write_enable = 1'b1;
            host_linear_weight_write_address = index[10:0];
            host_linear_weight_write_data = linear_weights[index];
        end
        @(negedge clk); host_linear_weight_write_enable = 1'b0;
        for (index = 0; index < 320; index = index + 1) begin
            @(negedge clk);
            host_parameter_write_enable = 1'b1;
            host_parameter_write_address = index[8:0];
            host_parameter_constant_data = constants[index];
            host_parameter_fraction_data = fractions[index];
        end
        @(negedge clk); host_parameter_write_enable = 1'b0;
        for (index = 0; index < 2; index = index + 1) begin
            @(negedge clk);
            host_readout_fraction_write_enable = 1'b1;
            host_readout_fraction_write_address = index[0];
            host_readout_fraction_write_data = readout_fractions[index];
        end
        @(negedge clk); host_readout_fraction_write_enable = 1'b0;
        for (index = 0; index < 24576; index = index + 1) begin
            @(negedge clk);
            host_input_write_enable = 1'b1;
            host_input_write_channel = index >= 12288;
            host_input_write_address = index % 12288;
            host_input_write_data = input_frame[index];
        end
        @(negedge clk); host_input_write_enable = 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        #1;

        if (!result_valid) begin
            $display("ERROR result_valid not asserted with done");
            errors = errors + 1;
        end
        if (config_error || range_error_seen) begin
            $display("ERROR status config=%0d range=%0d",
                config_error, range_error_seen);
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
        expected_transactions = requested_mode == 1 ? 6634 : 11242;
        expected_values_checked = requested_mode == 1 ? 159408 : 270000;
        if (trace_transactions != expected_transactions) begin
            $display("ERROR transactions got=%0d expected=%0d",
                trace_transactions, expected_transactions);
            errors = errors + 1;
        end
        if (trace_values_checked != expected_values_checked) begin
            $display("ERROR values checked got=%0d expected=%0d",
                trace_values_checked, expected_values_checked);
            errors = errors + 1;
        end
        if (errors != 0) $fatal(1, "full replay failed errors=%0d", errors);
        $display(
            "PASS: mode=%0d transactions=%0d values=%0d total_cycles=%0d conv_cycles=%0d dense_cycles=%0d replay_cycles=%0d",
            requested_mode, trace_transactions, trace_values_checked,
            inference_cycles, convolution_cycles, dense_cycles, replay_cycles
        );
        $finish;
    end
endmodule

`default_nettype wire
