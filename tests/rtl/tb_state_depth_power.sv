`timescale 1ns/1ps
// Bus stimulus for the implemented netlist. Weights and inputs are runtime loaded.
module tb_state_depth_power;
    reg clk = 0;
    always #3.125 clk = ~clk;
    reg rst = 1;
    reg host_write_enable = 0;
    reg [7:0] host_address = 0;
    reg [31:0] host_write_data = 0;
    wire [31:0] host_read_data;
    pursuit_state_depth_timing_top dut (.*);

    reg [255:0] conv_weights [0:4157];
    reg [255:0] linear_weights [0:1991];
    reg [23:0] constants [0:319];
    reg [4:0] fractions [0:319];
    reg [4:0] readout_fractions [0:1];
    reg [8:0] input_frame [0:24575];
    reg [63:0] expected_result [0:1];
    reg power_active = 0;
    string image_directory, sample_directory;
    integer expected_cycles;
    integer i, word_index;
    reg [31:0] read_value;
    realtime power_start_time;
    integer active_clocks = 0;
    always @(posedge clk) begin
        if (power_active) begin
            active_clocks = active_clocks + 1;
            if (active_clocks % 16000 == 0)
                $display("POWER_PROGRESS clocks=%0d time_ns=%0.3f", active_clocks, $realtime);
        end
    end

    // The physical constraint specifies input arrival 1 ns after the clock edge.
    task bus_write(input [7:0] addr, input [31:0] value);
        @(posedge clk);
        #1;
        host_write_enable = 1;
        host_address = addr;
        host_write_data = value;
    endtask
    task bus_read(input [7:0] addr, output [31:0] value);
        @(posedge clk);
        #1;
        host_write_enable = 0;
        host_address = addr;
        repeat (2) @(posedge clk);
        // Read just before the following edge, after the constrained output delay.
        #6.249;
        value = host_read_data;
    endtask

    initial begin
        if (!$value$plusargs("IMAGE_DIRECTORY=%s", image_directory)) $fatal(1, "IMAGE_DIRECTORY required");
        if (!$value$plusargs("SAMPLE_DIRECTORY=%s", sample_directory)) $fatal(1, "SAMPLE_DIRECTORY required");
        if (!$value$plusargs("EXPECTED_CYCLES=%d", expected_cycles)) $fatal(1, "EXPECTED_CYCLES required");
        $readmemh({image_directory, "/conv_weights_tiled_256.mem"}, conv_weights);
        $readmemh({image_directory, "/linear_weights_input_major_256.mem"}, linear_weights);
        $readmemh({image_directory, "/layer_constants_s24.mem"}, constants);
        $readmemh({image_directory, "/layer_fraction_bits_u5.mem"}, fractions);
        $readmemh({image_directory, "/readout_fraction_bits_u5.mem"}, readout_fractions);
        $readmemh({sample_directory, "/input.mem"}, input_frame);
        $readmemh({sample_directory, "/result.mem"}, expected_result);
        wait(glbl.GSR == 0);
        repeat (5) @(posedge clk);
        #1; rst = 0;
        repeat (3) @(posedge clk);
        for (i = 0; i < 4158; i = i + 1) begin
            bus_write(8'h03, i);
            for (word_index = 0; word_index < 8; word_index = word_index + 1)
                bus_write(8'h04 + word_index, conv_weights[i][word_index*32 +: 32]);
            bus_write(8'h0c, 1);
        end
        for (i = 0; i < 1992; i = i + 1) begin
            bus_write(8'h03, i);
            for (word_index = 0; word_index < 8; word_index = word_index + 1)
                bus_write(8'h04 + word_index, linear_weights[i][word_index*32 +: 32]);
            bus_write(8'h0c, 2);
        end
        for (i = 0; i < 320; i = i + 1) begin
            bus_write(8'h0d, i);
            bus_write(8'h0e, constants[i]);
            bus_write(8'h10, fractions[i]);
        end
        for (i = 0; i < 2; i = i + 1)
            bus_write(8'h11, (i << 5) | readout_fractions[i]);
        for (i = 0; i < 24576; i = i + 1) begin
            bus_write(8'h01, ((i >= 12288) << 14) | (i % 12288));
            bus_write(8'h02, input_frame[i]);
        end
        bus_write(8'h00, 1);
        @(posedge clk); #1; host_write_enable = 0;
        // busy/done are retained top-level wires in the exported implemented netlist.
        wait(dut.busy === 1'b1);
        power_start_time = $realtime;
        power_active = 1;
        $display("POWER_START time_ns=%0.3f", $realtime);
        wait(dut.done === 1'b1);
        power_active = 0;
        $display("POWER_END time_ns=%0.3f duration_ns=%0.3f", $realtime, $realtime-power_start_time);
        repeat (3) @(posedge clk);
        for (i = 0; i < 4; i = i + 1) begin
            bus_read(8'h20 + i, read_value);
            if (read_value !== expected_result[i/2][(i%2)*32 +: 32])
                $fatal(1, "result mismatch word=%0d got=%h expected=%h", i, read_value, expected_result[i/2][(i%2)*32 +: 32]);
        end
        bus_read(8'h24, read_value);
        if (read_value[9:0] !== {readout_fractions[1], readout_fractions[0]}) $fatal(1, "fraction mismatch");
        bus_read(8'h25, read_value);
        if (read_value !== expected_cycles) $fatal(1, "cycle mismatch got=%0d expected=%0d", read_value, expected_cycles);
        bus_read(8'h00, read_value);
        if (read_value !== 32'h2) $fatal(1, "final status mismatch: %h", read_value);
        $display("PASS: post-route power replay core_cycles=%0d", expected_cycles);
        $finish;
    end
endmodule
