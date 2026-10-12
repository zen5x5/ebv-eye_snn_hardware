`timescale 1ns/1ps
`default_nettype none

module tb_lif_multispike;
    localparam integer CURRENT_BITS = 24;
    localparam integer STATE_BITS = 16;
    localparam integer SPIKE_BITS = 8;

    reg clk;
    reg rst;
    reg in_valid;
    reg signed [CURRENT_BITS-1:0] current_integer;
    reg signed [STATE_BITS-1:0] previous_membrane_integer;
    reg [4:0] fraction_bits;
    wire out_valid;
    wire [SPIKE_BITS-1:0] spike_integer;
    wire signed [STATE_BITS-1:0] membrane_integer;
    wire range_error;

    always #5 clk = ~clk;

    lif_multispike dut (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .current_integer(current_integer),
        .previous_membrane_integer(previous_membrane_integer),
        .fraction_bits(fraction_bits),
        .out_valid(out_valid),
        .spike_integer(spike_integer),
        .membrane_integer(membrane_integer),
        .range_error(range_error)
    );

    task apply_and_check;
        input signed [CURRENT_BITS-1:0] current_value;
        input signed [STATE_BITS-1:0] previous_value;
        input [4:0] fraction_value;
        input [SPIKE_BITS-1:0] expected_spike;
        input signed [STATE_BITS-1:0] expected_membrane;
        input expected_error;
        begin
            @(negedge clk);
            current_integer = current_value;
            previous_membrane_integer = previous_value;
            fraction_bits = fraction_value;
            in_valid = 1'b1;
            @(posedge clk);
            #1;
            if (!out_valid
                || spike_integer !== expected_spike
                || membrane_integer !== expected_membrane
                || range_error !== expected_error) begin
                $display(
                    "FAIL LIF: I=%0d v=%0d f=%0d got (%0d,%0d,%0d) expected (%0d,%0d,%0d)",
                    current_value,
                    previous_value,
                    fraction_value,
                    spike_integer,
                    membrane_integer,
                    range_error,
                    expected_spike,
                    expected_membrane,
                    expected_error
                );
                $fatal(1);
            end
        end
    endtask

    initial begin
        clk = 1'b0;
        rst = 1'b1;
        in_valid = 1'b0;
        current_integer = 0;
        previous_membrane_integer = 0;
        fraction_bits = 0;
        repeat (2) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        apply_and_check(32768, 0, 15, 1, 0, 0);
        apply_and_check(98427, 0, 15, 3, 123, 0);
        apply_and_check(0, -1, 15, 0, -1, 0);
        apply_and_check(-10000, -32768, 15, 0, -32768, 0);
        apply_and_check(-50, 100, 12, 0, 25, 0);
        apply_and_check(1044580, 0, 12, 255, 100, 0);
        apply_and_check(1048576, 0, 12, 255, 4096, 1);

        @(negedge clk);
        in_valid = 1'b0;
        @(posedge clk);
        #1;
        if (out_valid) begin
            $display("FAIL LIF: valid did not clear");
            $fatal(1);
        end
        $display("PASS fixed-point LIF MultiSpike");
        $finish;
    end
endmodule

`default_nettype wire
