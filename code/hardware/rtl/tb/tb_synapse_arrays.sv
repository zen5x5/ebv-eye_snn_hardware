`timescale 1ns/1ps
`default_nettype none

module tb_synapse_arrays;
    localparam integer OUTPUT_PES = 2;
    localparam integer LANES = 3;
    localparam integer WEIGHT_BITS = 16;
    localparam integer COUNT_BITS = 9;
    localparam integer SHIFT_BITS = 4;
    localparam integer ACCUMULATOR_BITS = 40;

    reg clk;
    reg rst;
    reg in_valid;
    reg [LANES*COUNT_BITS-1:0] counts_flat;
    reg [LANES-1:0] unit_enables;
    reg [LANES-1:0] term_enables;
    reg [LANES*SHIFT_BITS-1:0] shift_amounts_flat;
    reg [OUTPUT_PES*LANES*WEIGHT_BITS-1:0] weights_flat;
    reg [OUTPUT_PES*ACCUMULATOR_BITS-1:0] accumulators_in_flat;
    wire mac_valid;
    wire shift_valid;
    wire unary_valid;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] mac_out;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] shift_out;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] unary_out;

    always #5 clk = ~clk;

    synapse_count_mac_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .COUNT_BITS(COUNT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS)
    ) count_mac (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .counts_flat(counts_flat),
        .weights_flat(weights_flat),
        .accumulators_in_flat(accumulators_in_flat),
        .out_valid(mac_valid),
        .accumulators_out_flat(mac_out)
    );

    synapse_shift_add_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .SHIFT_BITS(SHIFT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS)
    ) shift_add (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .term_enables(term_enables),
        .shift_amounts_flat(shift_amounts_flat),
        .weights_flat(weights_flat),
        .accumulators_in_flat(accumulators_in_flat),
        .out_valid(shift_valid),
        .accumulators_out_flat(shift_out)
    );

    synapse_unary_add_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS)
    ) unary_add (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .unit_enables(unit_enables),
        .weights_flat(weights_flat),
        .accumulators_in_flat(accumulators_in_flat),
        .out_valid(unary_valid),
        .accumulators_out_flat(unary_out)
    );

    task set_count;
        input integer lane;
        input [COUNT_BITS-1:0] value;
        begin
            counts_flat[lane*COUNT_BITS +: COUNT_BITS] = value;
        end
    endtask

    task set_weight;
        input integer output_index;
        input integer lane;
        input signed [WEIGHT_BITS-1:0] value;
        begin
            weights_flat[
                (output_index*LANES + lane)*WEIGHT_BITS +: WEIGHT_BITS
            ] = value;
        end
    endtask

    task set_shift;
        input integer lane;
        input [SHIFT_BITS-1:0] value;
        begin
            shift_amounts_flat[lane*SHIFT_BITS +: SHIFT_BITS] = value;
        end
    endtask

    task set_accumulator;
        input integer output_index;
        input signed [ACCUMULATOR_BITS-1:0] value;
        begin
            accumulators_in_flat[
                output_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ] = value;
        end
    endtask

    task check_output;
        input [8*32-1:0] implementation;
        input [OUTPUT_PES*ACCUMULATOR_BITS-1:0] values;
        input signed [ACCUMULATOR_BITS-1:0] expected_zero;
        input signed [ACCUMULATOR_BITS-1:0] expected_one;
        reg signed [ACCUMULATOR_BITS-1:0] actual_zero;
        reg signed [ACCUMULATOR_BITS-1:0] actual_one;
        begin
            actual_zero = $signed(values[0 +: ACCUMULATOR_BITS]);
            actual_one = $signed(values[ACCUMULATOR_BITS +: ACCUMULATOR_BITS]);
            if (actual_zero !== expected_zero || actual_one !== expected_one) begin
                $display(
                    "FAIL %0s: got (%0d,%0d), expected (%0d,%0d)",
                    implementation,
                    actual_zero,
                    actual_one,
                    expected_zero,
                    expected_one
                );
                $fatal(1);
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        in_valid = 1'b0;
        counts_flat = 0;
        unit_enables = 0;
        term_enables = 0;
        shift_amounts_flat = 0;
        weights_flat = 0;
        accumulators_in_flat = 0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        set_count(0, 0);
        set_count(1, 3);
        set_count(2, 5);
        unit_enables = 3'b101;
        term_enables = 3'b110;
        set_shift(1, 0);
        set_shift(2, 0);
        set_weight(0, 0, -7);
        set_weight(0, 1, 11);
        set_weight(0, 2, -13);
        set_weight(1, 0, 32767);
        set_weight(1, 1, -32768);
        set_weight(1, 2, 1234);
        set_accumulator(0, 100);
        set_accumulator(1, -200);
        in_valid = 1'b1;

        @(posedge clk);
        #1;
        if (!mac_valid || !shift_valid || !unary_valid) begin
            $display("FAIL: valid pipeline");
            $fatal(1);
        end
        check_output("count_mac", mac_out, 68, -92334);
        check_output("shift_add_first", shift_out, 98, -31734);
        check_output("unary_add", unary_out, 80, 33801);

        @(negedge clk);
        term_enables = 3'b110;
        set_shift(1, 1);
        set_shift(2, 2);
        accumulators_in_flat = shift_out;
        unit_enables = 0;
        counts_flat = 0;
        @(posedge clk);
        #1;
        check_output("shift_add_second", shift_out, 68, -92334);

        @(negedge clk);
        in_valid = 1'b0;
        @(posedge clk);
        #1;
        if (mac_valid || shift_valid || unary_valid) begin
            $display("FAIL: valid did not clear");
            $fatal(1);
        end
        $display("PASS synapse arithmetic arrays");
        $finish;
    end
endmodule

`default_nettype wire
