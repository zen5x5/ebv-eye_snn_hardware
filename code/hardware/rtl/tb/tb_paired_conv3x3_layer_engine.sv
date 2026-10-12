`timescale 1ns/1ps
`default_nettype none

module tb_paired_conv3x3_layer_engine;
    localparam integer MAX_ACTIVATIONS = 24576;
    localparam integer MAX_CONV_BEATS = 4158;
    localparam integer MAX_PARAMETERS = 320;

    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;

    integer width_log2;
    integer height;
    integer input_channels;
    integer output_channels;
    integer input_spatial;
    integer output_spatial;
    integer input_tiles;
    integer output_tiles;
    integer pool_enable;
    integer reuse_current;
    integer weight_base;
    integer parameter_base;
    integer timeout_cycles;

    wire weight_read_enable;
    wire [12:0] weight_read_address;
    reg [255:0] weight_read_data;
    wire parameter_read_enable;
    wire [8:0] parameter_read_address;
    reg signed [23:0] parameter_constant_data;
    reg [4:0] parameter_fraction_data;
    wire input_read_enable;
    wire input_read_time;
    wire [13:0] input_read_address;
    wire [5:0] input_read_channel_base;
    reg [35:0] input_read_data_flat;
    wire output_valid;
    wire output_time;
    wire [13:0] output_address;
    wire [5:0] output_channel_base;
    wire [191:0] output_current_flat;
    wire [63:0] output_spike_flat;
    wire [127:0] output_membrane_flat;
    wire output_range_error;

    reg [255:0] weight_memory [0:MAX_CONV_BEATS-1];
    reg signed [23:0] constant_memory [0:MAX_PARAMETERS-1];
    reg [4:0] fraction_memory [0:MAX_PARAMETERS-1];
    reg [8:0] input_memory_t0 [0:MAX_ACTIVATIONS-1];
    reg [8:0] input_memory_t1 [0:MAX_ACTIVATIONS-1];
    reg signed [23:0] expected_current_t0 [0:MAX_ACTIVATIONS-1];
    reg signed [23:0] expected_current_t1 [0:MAX_ACTIVATIONS-1];
    reg [7:0] expected_spike_t0 [0:MAX_ACTIVATIONS-1];
    reg [7:0] expected_spike_t1 [0:MAX_ACTIVATIONS-1];
    reg signed [15:0] expected_membrane_t0 [0:MAX_ACTIVATIONS-1];
    reg signed [15:0] expected_membrane_t1 [0:MAX_ACTIVATIONS-1];

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

    paired_conv3x3_layer_engine dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .config_width_log2(width_log2[3:0]),
        .config_height(height[6:0]),
        .config_input_tiles(input_tiles[3:0]),
        .config_output_tiles(output_tiles[3:0]),
        .config_output_spatial(output_spatial[13:0]),
        .config_pool_enable(pool_enable[0]),
        .config_reuse_current(reuse_current[0]),
        .config_weight_beat_base(weight_base[12:0]),
        .config_parameter_base(parameter_base[8:0]),
        .weight_read_enable(weight_read_enable),
        .weight_read_address(weight_read_address),
        .weight_read_data(weight_read_data),
        .parameter_read_enable(parameter_read_enable),
        .parameter_read_address(parameter_read_address),
        .parameter_constant_data(parameter_constant_data),
        .parameter_fraction_data(parameter_fraction_data),
        .input_read_enable(input_read_enable),
        .input_read_time(input_read_time),
        .input_read_address(input_read_address),
        .input_read_channel_base(input_read_channel_base),
        .input_read_data_flat(input_read_data_flat),
        .output_valid(output_valid),
        .output_time(output_time),
        .output_address(output_address),
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
        if (parameter_read_enable) begin
            parameter_constant_data <= constant_memory[parameter_read_address];
            parameter_fraction_data <= fraction_memory[parameter_read_address];
        end
        if (input_read_enable) begin
            for (lane = 0; lane < 4; lane = lane + 1) begin
                if (input_read_channel_base + lane < input_channels) begin
                    if (input_read_time) begin
                        input_read_data_flat[lane*9 +: 9]
                            <= input_memory_t1[
                                (input_read_channel_base + lane)*input_spatial
                                + input_read_address
                            ];
                    end else begin
                        input_read_data_flat[lane*9 +: 9]
                            <= input_memory_t0[
                                (input_read_channel_base + lane)*input_spatial
                                + input_read_address
                            ];
                    end
                end else begin
                    input_read_data_flat[lane*9 +: 9] <= 0;
                end
            end
        end
    end

    integer cycles;
    integer transactions;
    integer values_checked;
    integer errors;
    integer expected_index;
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (output_valid) begin
                transactions <= transactions + 1;
                if (output_range_error) begin
                    $display("ERROR range t=%0d c=%0d a=%0d",
                        output_time, output_channel_base, output_address);
                    errors <= errors + 1;
                end
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    expected_index =
                        (output_channel_base + lane)*output_spatial
                        + output_address;
                    values_checked <= values_checked + 3;
                    if (!output_time) begin
                        if ($signed(output_current_flat[lane*24 +: 24])
                                !== expected_current_t0[expected_index]) begin
                            $display("ERROR current t0 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                $signed(output_current_flat[lane*24 +: 24]),
                                expected_current_t0[expected_index]);
                            errors <= errors + 1;
                        end
                        if (output_spike_flat[lane*8 +: 8]
                                !== expected_spike_t0[expected_index]) begin
                            $display("ERROR spike t0 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                output_spike_flat[lane*8 +: 8],
                                expected_spike_t0[expected_index]);
                            errors <= errors + 1;
                        end
                        if ($signed(output_membrane_flat[lane*16 +: 16])
                                !== expected_membrane_t0[expected_index]) begin
                            $display("ERROR membrane t0 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                $signed(output_membrane_flat[lane*16 +: 16]),
                                expected_membrane_t0[expected_index]);
                            errors <= errors + 1;
                        end
                    end else begin
                        if ($signed(output_current_flat[lane*24 +: 24])
                                !== expected_current_t1[expected_index]) begin
                            $display("ERROR current t1 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                $signed(output_current_flat[lane*24 +: 24]),
                                expected_current_t1[expected_index]);
                            errors <= errors + 1;
                        end
                        if (output_spike_flat[lane*8 +: 8]
                                !== expected_spike_t1[expected_index]) begin
                            $display("ERROR spike t1 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                output_spike_flat[lane*8 +: 8],
                                expected_spike_t1[expected_index]);
                            errors <= errors + 1;
                        end
                        if ($signed(output_membrane_flat[lane*16 +: 16])
                                !== expected_membrane_t1[expected_index]) begin
                            $display("ERROR membrane t1 c=%0d a=%0d got=%0d exp=%0d",
                                output_channel_base + lane, output_address,
                                $signed(output_membrane_flat[lane*16 +: 16]),
                                expected_membrane_t1[expected_index]);
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
        argument_ok &= $value$plusargs("INPUT_T0_FILE=%s", input_t0_file);
        argument_ok &= $value$plusargs("INPUT_T1_FILE=%s", input_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_CURRENT_T0_FILE=%s", expected_current_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_CURRENT_T1_FILE=%s", expected_current_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_SPIKE_T0_FILE=%s", expected_spike_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_SPIKE_T1_FILE=%s", expected_spike_t1_file);
        argument_ok &= $value$plusargs("EXPECTED_MEMBRANE_T0_FILE=%s", expected_membrane_t0_file);
        argument_ok &= $value$plusargs("EXPECTED_MEMBRANE_T1_FILE=%s", expected_membrane_t1_file);
        argument_ok &= $value$plusargs("WIDTH_LOG2=%d", width_log2);
        argument_ok &= $value$plusargs("HEIGHT=%d", height);
        argument_ok &= $value$plusargs("INPUT_CHANNELS=%d", input_channels);
        argument_ok &= $value$plusargs("OUTPUT_CHANNELS=%d", output_channels);
        argument_ok &= $value$plusargs("INPUT_SPATIAL=%d", input_spatial);
        argument_ok &= $value$plusargs("OUTPUT_SPATIAL=%d", output_spatial);
        argument_ok &= $value$plusargs("INPUT_TILES=%d", input_tiles);
        argument_ok &= $value$plusargs("OUTPUT_TILES=%d", output_tiles);
        argument_ok &= $value$plusargs("POOL_ENABLE=%d", pool_enable);
        if (!$value$plusargs("REUSE_CURRENT=%d", reuse_current)) begin
            reuse_current = 0;
        end
        argument_ok &= $value$plusargs("WEIGHT_BASE=%d", weight_base);
        argument_ok &= $value$plusargs("PARAMETER_BASE=%d", parameter_base);
        if (!$value$plusargs("TIMEOUT_CYCLES=%d", timeout_cycles)) begin
            timeout_cycles = 100000;
        end
        if (!argument_ok) $fatal(1, "missing required replay plusarg");

        for (memory_index = 0; memory_index < MAX_ACTIVATIONS;
                memory_index = memory_index + 1) begin
            input_memory_t0[memory_index] = 0;
            input_memory_t1[memory_index] = 0;
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
        $readmemh(input_t0_file, input_memory_t0, 0,
            input_channels*input_spatial-1);
        $readmemh(input_t1_file, input_memory_t1, 0,
            input_channels*input_spatial-1);
        $readmemh(expected_current_t0_file, expected_current_t0, 0,
            output_channels*output_spatial-1);
        $readmemh(expected_current_t1_file, expected_current_t1, 0,
            output_channels*output_spatial-1);
        $readmemh(expected_spike_t0_file, expected_spike_t0, 0,
            output_channels*output_spatial-1);
        $readmemh(expected_spike_t1_file, expected_spike_t1, 0,
            output_channels*output_spatial-1);
        $readmemh(expected_membrane_t0_file, expected_membrane_t0, 0,
            output_channels*output_spatial-1);
        $readmemh(expected_membrane_t1_file, expected_membrane_t1, 0,
            output_channels*output_spatial-1);

        weight_read_data = 0;
        parameter_constant_data = 0;
        parameter_fraction_data = 0;
        input_read_data_flat = 0;
        cycles = 0;
        transactions = 0;
        values_checked = 0;
        errors = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        @(posedge clk);
        if (transactions != 2*output_tiles*output_spatial) begin
            $fatal(1, "transaction count %0d expected %0d",
                transactions, 2*output_tiles*output_spatial);
        end
        if (errors != 0) begin
            $fatal(1, "paired golden comparison failed with %0d errors", errors);
        end
        $display("PASS: transactions=%0d values=%0d inference_cycles=%0d",
            transactions, 2*output_channels*output_spatial*3, cycles);
        $finish;
    end
endmodule

`default_nettype wire
