`timescale 1ns/1ps
`default_nettype none

module tb_linear_lif_layer_engine;
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
    integer use_previous_state;
    integer write_state;
    integer weight_base;
    integer parameter_base;
    integer timeout_cycles;

    wire weight_read_enable;
    wire [10:0] weight_read_address;
    reg [255:0] weight_read_data;
    wire input_read_enable;
    wire [8:0] input_read_address;
    reg [7:0] input_read_data;
    wire parameter_read_enable;
    wire [8:0] parameter_read_address;
    reg signed [23:0] parameter_constant_data;
    reg [4:0] parameter_fraction_data;
    wire state_read_enable;
    wire [6:0] state_read_channel_base;
    reg [255:0] state_read_data_flat;
    wire state_write_enable;
    wire [6:0] state_write_channel_base;
    wire [255:0] state_write_data_flat;
    wire output_valid;
    wire [6:0] output_channel_base;
    wire [383:0] output_current_flat;
    wire [127:0] output_spike_flat;
    wire [255:0] output_membrane_flat;
    wire output_range_error;

    reg [255:0] weight_memory [0:LINEAR_BEATS-1];
    reg signed [23:0] constant_memory [0:PARAMETERS-1];
    reg [4:0] fraction_memory [0:PARAMETERS-1];
    reg [7:0] input_memory [0:MAX_INPUTS-1];
    reg signed [15:0] state_memory [0:MAX_OUTPUTS-1];
    reg signed [23:0] expected_current [0:MAX_OUTPUTS-1];
    reg [7:0] expected_spike [0:MAX_OUTPUTS-1];
    reg signed [15:0] expected_membrane [0:MAX_OUTPUTS-1];

    string weight_file;
    string constant_file;
    string fraction_file;
    string input_file;
    string previous_state_file;
    string expected_current_file;
    string expected_spike_file;
    string expected_membrane_file;

    linear_lif_layer_engine dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .config_input_features(input_features[9:0]),
        .config_output_groups(output_groups[2:0]),
        .config_use_previous_state(use_previous_state[0]),
        .config_write_state(write_state[0]),
        .config_weight_beat_base(weight_base[10:0]),
        .config_parameter_base(parameter_base[8:0]),
        .weight_read_enable(weight_read_enable),
        .weight_read_address(weight_read_address),
        .weight_read_data(weight_read_data),
        .input_read_enable(input_read_enable),
        .input_read_address(input_read_address),
        .input_read_data(input_read_data),
        .parameter_read_enable(parameter_read_enable),
        .parameter_read_address(parameter_read_address),
        .parameter_constant_data(parameter_constant_data),
        .parameter_fraction_data(parameter_fraction_data),
        .state_read_enable(state_read_enable),
        .state_read_channel_base(state_read_channel_base),
        .state_read_data_flat(state_read_data_flat),
        .state_write_enable(state_write_enable),
        .state_write_channel_base(state_write_channel_base),
        .state_write_data_flat(state_write_data_flat),
        .output_valid(output_valid),
        .output_channel_base(output_channel_base),
        .output_current_flat(output_current_flat),
        .output_spike_flat(output_spike_flat),
        .output_membrane_flat(output_membrane_flat),
        .output_range_error(output_range_error)
    );

    integer lane;
    always @(posedge clk) begin
        if (weight_read_enable) begin
            weight_read_data <= weight_memory[weight_read_address];
        end
        if (input_read_enable) begin
            input_read_data <= input_memory[input_read_address];
        end
        if (parameter_read_enable) begin
            parameter_constant_data <= constant_memory[parameter_read_address];
            parameter_fraction_data <= fraction_memory[parameter_read_address];
        end
        if (state_read_enable) begin
            for (lane = 0; lane < 16; lane = lane + 1) begin
                state_read_data_flat[lane*16 +: 16]
                    <= state_memory[state_read_channel_base + lane];
            end
        end
        if (state_write_enable) begin
            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (state_write_channel_base + lane < output_features) begin
                    state_memory[state_write_channel_base + lane]
                        <= state_write_data_flat[lane*16 +: 16];
                end
            end
        end
    end

    integer cycles;
    integer transactions;
    integer errors;
    integer output_index;
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (output_valid) begin
                transactions <= transactions + 1;
                if (output_range_error) begin
                    $display("ERROR range_error channel=%0d", output_channel_base);
                    errors <= errors + 1;
                end
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    output_index = output_channel_base + lane;
                    if (output_index < output_features) begin
                        if ($signed(output_current_flat[lane*24 +: 24])
                                !== expected_current[output_index]) begin
                            $display("ERROR current c=%0d got=%0d expected=%0d",
                                output_index,
                                $signed(output_current_flat[lane*24 +: 24]),
                                expected_current[output_index]);
                            errors <= errors + 1;
                        end
                        if (output_spike_flat[lane*8 +: 8]
                                !== expected_spike[output_index]) begin
                            $display("ERROR spike c=%0d got=%0d expected=%0d",
                                output_index,
                                output_spike_flat[lane*8 +: 8],
                                expected_spike[output_index]);
                            errors <= errors + 1;
                        end
                        if ($signed(output_membrane_flat[lane*16 +: 16])
                                !== expected_membrane[output_index]) begin
                            $display("ERROR membrane c=%0d got=%0d expected=%0d",
                                output_index,
                                $signed(output_membrane_flat[lane*16 +: 16]),
                                expected_membrane[output_index]);
                            errors <= errors + 1;
                        end
                    end
                end
            end
            if (cycles > timeout_cycles) begin
                $fatal(1, "timeout after %0d cycles", cycles);
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
        argument_ok &= $value$plusargs("INPUT_FILE=%s", input_file);
        argument_ok &= $value$plusargs("PREVIOUS_STATE_FILE=%s", previous_state_file);
        argument_ok &= $value$plusargs("EXPECTED_CURRENT_FILE=%s", expected_current_file);
        argument_ok &= $value$plusargs("EXPECTED_SPIKE_FILE=%s", expected_spike_file);
        argument_ok &= $value$plusargs("EXPECTED_MEMBRANE_FILE=%s", expected_membrane_file);
        argument_ok &= $value$plusargs("INPUT_FEATURES=%d", input_features);
        argument_ok &= $value$plusargs("OUTPUT_FEATURES=%d", output_features);
        argument_ok &= $value$plusargs("OUTPUT_GROUPS=%d", output_groups);
        argument_ok &= $value$plusargs("USE_PREVIOUS_STATE=%d", use_previous_state);
        argument_ok &= $value$plusargs("WRITE_STATE=%d", write_state);
        argument_ok &= $value$plusargs("WEIGHT_BASE=%d", weight_base);
        argument_ok &= $value$plusargs("PARAMETER_BASE=%d", parameter_base);
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
            timeout_cycles = 10000;
        end
        if (!argument_ok) begin
            $fatal(1, "missing required replay plusarg");
        end
        for (memory_index = 0; memory_index < MAX_OUTPUTS;
                memory_index = memory_index + 1) begin
            state_memory[memory_index] = 0;
            expected_current[memory_index] = 0;
            expected_spike[memory_index] = 0;
            expected_membrane[memory_index] = 0;
        end
        $readmemh(weight_file, weight_memory);
        $readmemh(constant_file, constant_memory);
        $readmemh(fraction_file, fraction_memory);
        $readmemh(input_file, input_memory, 0, input_features-1);
        if (use_previous_state) begin
            $readmemh(previous_state_file, state_memory, 0, output_features-1);
        end
        $readmemh(expected_current_file, expected_current, 0, output_features-1);
        $readmemh(expected_spike_file, expected_spike, 0, output_features-1);
        $readmemh(expected_membrane_file, expected_membrane, 0, output_features-1);
        weight_read_data = 0;
        input_read_data = 0;
        parameter_constant_data = 0;
        parameter_fraction_data = 0;
        state_read_data_flat = 0;
        cycles = 0;
        transactions = 0;
        errors = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        @(posedge clk);
        if (transactions != output_groups) begin
            $fatal(1, "transaction count %0d expected %0d",
                transactions, output_groups);
        end
        if (errors != 0) begin
            $fatal(1, "golden comparison failed with %0d errors", errors);
        end
        $display("PASS: transactions=%0d values=%0d inference_cycles=%0d",
            transactions, output_features*3, cycles);
        $finish;
    end
endmodule

`default_nettype wire
