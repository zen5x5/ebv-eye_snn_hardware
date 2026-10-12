`timescale 1ns/1ps
`default_nettype none

module tb_pipelined_count_mac_array;
    localparam integer OUTPUT_PES = 2;
    localparam integer LANES = 3;
    localparam integer WEIGHT_BITS = 16;
    localparam integer COUNT_BITS = 9;
    localparam integer ACCUMULATOR_BITS = 40;
    localparam integer TAG_BITS = 8;

    reg clk;
    reg rst;
    reg in_valid;
    reg [TAG_BITS-1:0] tag_in;
    reg [LANES*COUNT_BITS-1:0] counts_flat;
    reg [OUTPUT_PES*LANES*WEIGHT_BITS-1:0] weights_flat;
    reg [OUTPUT_PES*ACCUMULATOR_BITS-1:0] accumulators_in_flat;
    wire out_valid;
    wire [TAG_BITS-1:0] tag_out;
    wire [OUTPUT_PES*ACCUMULATOR_BITS-1:0] accumulators_out_flat;

    always #5 clk = ~clk;

    pipelined_count_mac_array #(
        .OUTPUT_PES(OUTPUT_PES),
        .LANES(LANES),
        .WEIGHT_BITS(WEIGHT_BITS),
        .COUNT_BITS(COUNT_BITS),
        .ACCUMULATOR_BITS(ACCUMULATOR_BITS),
        .TAG_BITS(TAG_BITS)
    ) dut (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .tag_in(tag_in),
        .counts_flat(counts_flat),
        .weights_flat(weights_flat),
        .accumulators_in_flat(accumulators_in_flat),
        .out_valid(out_valid),
        .tag_out(tag_out),
        .accumulators_out_flat(accumulators_out_flat)
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

    task set_accumulator;
        input integer output_index;
        input signed [ACCUMULATOR_BITS-1:0] value;
        begin
            accumulators_in_flat[
                output_index*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ] = value;
        end
    endtask

    task drive_first;
        begin
            tag_in = 8'h51;
            set_count(0, 0);
            set_count(1, 3);
            set_count(2, 5);
            set_accumulator(0, 100);
            set_accumulator(1, -200);
        end
    endtask

    task drive_second;
        begin
            tag_in = 8'h62;
            set_count(0, 1);
            set_count(1, 2);
            set_count(2, 0);
            set_accumulator(0, 10);
            set_accumulator(1, 20);
        end
    endtask

    integer outputs_seen;
    reg signed [ACCUMULATOR_BITS-1:0] actual_zero;
    reg signed [ACCUMULATOR_BITS-1:0] actual_one;
    initial begin
        clk = 1'b0;
        rst = 1'b1;
        in_valid = 1'b0;
        tag_in = 0;
        counts_flat = 0;
        weights_flat = 0;
        accumulators_in_flat = 0;
        outputs_seen = 0;
        set_weight(0, 0, -7);
        set_weight(0, 1, 11);
        set_weight(0, 2, -13);
        set_weight(1, 0, 32767);
        set_weight(1, 1, -32768);
        set_weight(1, 2, 1234);

        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        in_valid = 1'b1;
        drive_first();
        @(negedge clk);
        drive_second();
        @(negedge clk);
        in_valid = 1'b0;

        repeat (10) begin
            @(posedge clk);
            #1;
            if (out_valid) begin
                actual_zero = $signed(accumulators_out_flat[0 +: ACCUMULATOR_BITS]);
                actual_one = $signed(accumulators_out_flat[
                    ACCUMULATOR_BITS +: ACCUMULATOR_BITS
                ]);
                if (outputs_seen == 0) begin
                    if (tag_out !== 8'h51 || actual_zero !== 68 || actual_one !== -92334) begin
                        $display("FAIL pipelined first: tag=%0h values=(%0d,%0d)", tag_out, actual_zero, actual_one);
                        $fatal(1);
                    end
                end else if (outputs_seen == 1) begin
                    if (tag_out !== 8'h62 || actual_zero !== 25 || actual_one !== -32749) begin
                        $display("FAIL pipelined second: tag=%0h values=(%0d,%0d)", tag_out, actual_zero, actual_one);
                        $fatal(1);
                    end
                end else begin
                    $display("FAIL pipelined: unexpected extra output");
                    $fatal(1);
                end
                outputs_seen = outputs_seen + 1;
            end
        end
        if (outputs_seen !== 2) begin
            $display("FAIL pipelined: expected two outputs, saw %0d", outputs_seen);
            $fatal(1);
        end
        $display("PASS pipelined count MAC array");
        $finish;
    end
endmodule

`default_nettype wire
