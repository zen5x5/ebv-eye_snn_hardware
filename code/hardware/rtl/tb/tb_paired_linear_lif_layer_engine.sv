`timescale 1ns/1ps
`default_nettype none

module tb_paired_linear_lif_layer_engine;
    localparam integer MAX_INPUTS = 480;
    localparam integer MAX_OUTPUTS = 64;
    localparam integer LINEAR_BEATS = 1992;
    localparam integer PARAMETERS = 320;

    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;

    integer input_features;
    integer output_features;
    integer output_groups;
    integer weight_base;
    integer parameter_base;
    integer timeout_cycles;

    wire weight_read_enable;
    wire [10:0] weight_read_address;
    reg [255:0] weight_read_data;
    wire input_read_enable;
    wire input_read_time;
    wire [8:0] input_read_address;
    reg [7:0] input_read_data;
    wire parameter_read_enable;
    wire [8:0] parameter_read_address;
    reg signed [23:0] parameter_constant_data;
    reg [4:0] parameter_fraction_data;
    wire output_valid;
    wire output_time;
    wire [6:0] output_channel_base;
    wire [383:0] output_current_flat;
    wire [127:0] output_spike_flat;
    wire [255:0] output_membrane_flat;
    wire output_range_error;

    reg [255:0] weight_memory [0:LINEAR_BEATS-1];
    reg signed [23:0] constant_memory [0:PARAMETERS-1];
    reg [4:0] fraction_memory [0:PARAMETERS-1];
    reg [7:0] input_memory_t0 [0:MAX_INPUTS-1];
    reg [7:0] input_memory_t1 [0:MAX_INPUTS-1];
    reg signed [23:0] expected_current_t0 [0:MAX_OUTPUTS-1];
    reg signed [23:0] expected_current_t1 [0:MAX_OUTPUTS-1];
    reg [7:0] expected_spike_t0 [0:MAX_OUTPUTS-1];
    reg [7:0] expected_spike_t1 [0:MAX_OUTPUTS-1];
    reg signed [15:0] expected_membrane_t0 [0:MAX_OUTPUTS-1];
    reg signed [15:0] expected_membrane_t1 [0:MAX_OUTPUTS-1];

    string weight_file;
    string constant_file;
    string fraction_file;
    string input_t0_file;
    string input_t1_file;
    string expected_current_t0_file;
    string expected_current_t1_file;
    string expected_spike_t0_file;
    string expected_spike_t1_file;
    string expected_membrane_t0_file;
    string expected_membrane_t1_file;

    paired_linear_lif_layer_engine dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .config_input_features(input_features[9:0]),
        .config_output_groups(output_groups[2:0]),
        .config_weight_beat_base(weight_base[10:0]),
        .config_parameter_base(parameter_base[8:0]),
        .weight_read_enable(weight_read_enable),
        .weight_read_address(weight_read_address),
        .weight_read_data(weight_read_data),
        .input_read_enable(input_read_enable),
        .input_read_time(input_read_time),
        .input_read_address(input_read_address),
        .input_read_data(input_read_data),
        .parameter_read_enable(parameter_read_enable),
        .parameter_read_address(parameter_read_address),
        .parameter_constant_data(parameter_constant_data),
        .parameter_fraction_data(parameter_fraction_data),
        .output_valid(output_valid),
        .output_time(output_time),
        .output_channel_base(output_channel_base),
        .output_current_flat(output_current_flat),
        .output_spike_flat(output_spike_flat),
        .output_membrane_flat(output_membrane_flat),
        .output_range_error(output_range_error)
    );

    always @(posedge clk) begin
        if (weight_read_enable) begin
            weight_read_data <= weight_memory[weight_read_address];
        end
        if (input_read_enable) begin
            input_read_data <= input_read_time
                ? input_memory_t1[input_read_address]
                : input_memory_t0[input_read_address];
        end
        if (parameter_read_enable) begin
            parameter_constant_data <= constant_memory[parameter_read_address];
            parameter_fraction_data <= fraction_memory[parameter_read_address];
        end
    end

    integer elapsed_cycles;
    integer inference_cycles;
    integer transactions;
    integer values_checked;
    integer weight_reads;
    integer input_reads;
    integer errors;
    integer lane;
    integer output_index;
    integer expected_group;
    integer expected_time;
    integer valid_output_lanes;
    always @(posedge clk) begin
        if (rst) begin
            elapsed_cycles <= 0;
            inference_cycles <= 0;
            transactions <= 0;
            values_checked <= 0;
            weight_reads <= 0;
            input_reads <= 0;
            errors <= 0;
        end else begin
            elapsed_cycles <= elapsed_cycles + 1;
            if (start) begin
                inference_cycles <= 0;
            end else if (busy) begin
                inference_cycles <= inference_cycles + 1;
            end
            if (weight_read_enable) weight_reads <= weight_reads + 1;
            if (input_read_enable) input_reads <= input_reads + 1;
            if (output_valid) begin
                expected_group = transactions / 2;
                expected_time = transactions % 2;
                valid_output_lanes = output_features - output_channel_base;
                if (valid_output_lanes > 16) valid_output_lanes = 16;
                transactions <= transactions + 1;
                values_checked <= values_checked + valid_output_lanes*3;
                if (output_channel_base !== expected_group*16) begin
                    $display("ERROR channel base got=%0d expected=%0d",
                        output_channel_base, expected_group*16);
                    errors <= errors + 1;
                end
                if (output_time !== expected_time[0]) begin
                    $display("ERROR time tag got=%0d expected=%0d group=%0d",
                        output_time, expected_time, expected_group);
                    errors <= errors + 1;
                end
                if (output_range_error) begin
                    $display("ERROR range t=%0d channel=%0d",
                        output_time, output_channel_base);
                    errors <= errors + 1;
                end
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    output_index = output_channel_base + lane;
                    if (output_index < output_features) begin
                        if (!output_time) begin
                            if ($signed(output_current_flat[lane*24 +: 24])
                                    !== expected_current_t0[output_index]) begin
                                $display("ERROR current t0 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    $signed(output_current_flat[lane*24 +: 24]),
                                    expected_current_t0[output_index]);
                                errors <= errors + 1;
                            end
                            if (output_spike_flat[lane*8 +: 8]
                                    !== expected_spike_t0[output_index]) begin
                                $display("ERROR spike t0 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    output_spike_flat[lane*8 +: 8],
                                    expected_spike_t0[output_index]);
                                errors <= errors + 1;
                            end
                            if ($signed(output_membrane_flat[lane*16 +: 16])
                                    !== expected_membrane_t0[output_index]) begin
                                $display("ERROR membrane t0 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    $signed(output_membrane_flat[lane*16 +: 16]),
                                    expected_membrane_t0[output_index]);
                                errors <= errors + 1;
                            end
                        end else begin
                            if ($signed(output_current_flat[lane*24 +: 24])
                                    !== expected_current_t1[output_index]) begin
                                $display("ERROR current t1 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    $signed(output_current_flat[lane*24 +: 24]),
                                    expected_current_t1[output_index]);
                                errors <= errors + 1;
                            end
                            if (output_spike_flat[lane*8 +: 8]
                                    !== expected_spike_t1[output_index]) begin
                                $display("ERROR spike t1 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    output_spike_flat[lane*8 +: 8],
                                    expected_spike_t1[output_index]);
                                errors <= errors + 1;
                            end
                            if ($signed(output_membrane_flat[lane*16 +: 16])
                                    !== expected_membrane_t1[output_index]) begin
                                $display("ERROR membrane t1 c=%0d got=%0d exp=%0d",
                                    output_index,
                                    $signed(output_membrane_flat[lane*16 +: 16]),
                                    expected_membrane_t1[output_index]);
                                errors <= errors + 1;
                            end
                        end
                    end
                end
            end
            if (elapsed_cycles > timeout_cycles) begin
                $fatal(1, "timeout after %0d cycles", elapsed_cycles);
            end
        end
    end

    integer argument_ok;
    integer memory_index;
    initial begin
        argument_ok = 1;
        argument_ok &= $value$plusargs("WEIGHT_FILE=%s", weight_file);
        argument_ok &= $value$plusargs("CONSTANT_FILE=%s", constant_file);
        argument_ok &= $value$plusargs("FRACTION_FILE=%s", fraction_file);
        argument_ok &= $value$plusargs("INPUT_T0_FILE=%s", input_t0_file);
        argument_ok &= $value$plusargs("INPUT_T1_FILE=%s", input_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_CURRENT_T0_FILE=%s", expected_current_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_CURRENT_T1_FILE=%s", expected_current_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_SPIKE_T0_FILE=%s", expected_spike_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_SPIKE_T1_FILE=%s", expected_spike_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_MEMBRANE_T0_FILE=%s", expected_membrane_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_MEMBRANE_T1_FILE=%s", expected_membrane_t1_file);
        argument_ok &= $value$plusargs("INPUT_FEATURES=%d", input_features);
        argument_ok &= $value$plusargs("OUTPUT_FEATURES=%d", output_features);
        argument_ok &= $value$plusargs("OUTPUT_GROUPS=%d", output_groups);
        argument_ok &= $value$plusargs("WEIGHT_BASE=%d", weight_base);
        argument_ok &= $value$plusargs("PARAMETER_BASE=%d", parameter_base);
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
            timeout_cycles = 10000;
        end
        if (!argument_ok) $fatal(1, "missing required replay plusarg");

        for (memory_index = 0; memory_index < MAX_INPUTS;
                memory_index = memory_index + 1) begin
            input_memory_t0[memory_index] = 0;
            input_memory_t1[memory_index] = 0;
        end
        for (memory_index = 0; memory_index < MAX_OUTPUTS;
                memory_index = memory_index + 1) begin
            expected_current_t0[memory_index] = 0;
            expected_current_t1[memory_index] = 0;
            expected_spike_t0[memory_index] = 0;
            expected_spike_t1[memory_index] = 0;
            expected_membrane_t0[memory_index] = 0;
            expected_membrane_t1[memory_index] = 0;
        end
        $readmemh(weight_file, weight_memory);
        $readmemh(constant_file, constant_memory);
        $readmemh(fraction_file, fraction_memory);
        $readmemh(input_t0_file, input_memory_t0, 0, input_features-1);
        $readmemh(input_t1_file, input_memory_t1, 0, input_features-1);
        $readmemh(expected_current_t0_file, expected_current_t0,
            0, output_features-1);
        $readmemh(expected_current_t1_file, expected_current_t1,
            0, output_features-1);
        $readmemh(expected_spike_t0_file, expected_spike_t0,
            0, output_features-1);
        $readmemh(expected_spike_t1_file, expected_spike_t1,
            0, output_features-1);
        $readmemh(expected_membrane_t0_file, expected_membrane_t0,
            0, output_features-1);
        $readmemh(expected_membrane_t1_file, expected_membrane_t1,
            0, output_features-1);

        weight_read_data = 0;
        input_read_data = 0;
        parameter_constant_data = 0;
        parameter_fraction_data = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        @(posedge clk);
        if (transactions != 2*output_groups) begin
            $fatal(1, "transaction count %0d expected %0d",
                transactions, 2*output_groups);
        end
        if (values_checked != 2*output_features*3) begin
            $fatal(1, "value count %0d expected %0d",
                values_checked, 2*output_features*3);
        end
        if (weight_reads != input_features*output_groups) begin
            $fatal(1, "weight reads %0d expected %0d",
                weight_reads, input_features*output_groups);
        end
        if (input_reads != 2*input_features*output_groups) begin
            $fatal(1, "input reads %0d expected %0d",
                input_reads, 2*input_features*output_groups);
        end
        if (errors != 0) begin
            $fatal(1, "paired golden comparison failed with %0d errors",
                errors);
        end
        $display("PASS: transactions=%0d values=%0d weight_reads=%0d input_reads=%0d inference_cycles=%0d",
            transactions, values_checked, weight_reads, input_reads,
            inference_cycles);
        $finish;
    end
endmodule

`default_nettype wire
