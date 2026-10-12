`timescale 1ns/1ps
`default_nettype none

module tb_readout_engine;
    localparam integer LINEAR_BEATS = 1992;
    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire input_read_enable;
    wire [3:0] input_read_address;
    reg [8:0] input_read_data;
    wire weight_read_enable;
    wire [10:0] weight_read_address;
    reg [31:0] weight_read_data;
    wire output_valid;
    wire [127:0] output_integer_flat;

    reg [255:0] weight_memory [0:LINEAR_BEATS-1];
    reg [8:0] input_memory [0:11];
    reg signed [63:0] expected_output [0:1];
    string weight_file;
    string input_file;
    string expected_file;
    integer weight_base;

    readout_engine dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .config_input_features(5'd12),
        .config_weight_beat_base(weight_base[10:0]),
        .input_read_enable(input_read_enable),
        .input_read_address(input_read_address),
        .input_read_data(input_read_data),
        .weight_read_enable(weight_read_enable),
        .weight_read_address(weight_read_address),
        .weight_read_data(weight_read_data),
        .output_valid(output_valid),
        .output_integer_flat(output_integer_flat)
    );

    always @(posedge clk) begin
        if (input_read_enable) begin
            input_read_data <= input_memory[input_read_address];
        end
        if (weight_read_enable) begin
            weight_read_data <= weight_memory[weight_read_address][31:0];
        end
    end

    integer cycles;
    integer errors;
    integer output_index;
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (output_valid) begin
                for (output_index = 0; output_index < 2;
                        output_index = output_index + 1) begin
                    if ($signed(output_integer_flat[output_index*64 +: 64])
                            !== expected_output[output_index]) begin
                        $display("ERROR output=%0d got=%0d expected=%0d",
                            output_index,
                            $signed(output_integer_flat[
                                output_index*64 +: 64
                            ]),
                            expected_output[output_index]);
                        errors <= errors + 1;
                    end
                end
            end
            if (cycles > 1000) begin
                $fatal(1, "readout timeout");
            end
        end
    end

    integer argument_ok;
    initial begin
        argument_ok = 1;
        argument_ok &= $value$plusargs("WEIGHT_FILE=%s", weight_file);
        argument_ok &= $value$plusargs("INPUT_FILE=%s", input_file);
        argument_ok &= $value$plusargs("EXPECTED_FILE=%s", expected_file);
        argument_ok &= $value$plusargs("WEIGHT_BASE=%d", weight_base);
        if (!argument_ok) begin
            $fatal(1, "missing required replay plusarg");
        end
        $readmemh(weight_file, weight_memory);
        $readmemh(input_file, input_memory);
        $readmemh(expected_file, expected_output);
        input_read_data = 0;
        weight_read_data = 0;
        cycles = 0;
        errors = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        @(posedge clk);
        if (errors != 0) begin
            $fatal(1, "readout comparison failed with %0d errors", errors);
        end
        $display("PASS: values=2 inference_cycles=%0d", cycles);
        $finish;
    end
endmodule

`default_nettype wire
