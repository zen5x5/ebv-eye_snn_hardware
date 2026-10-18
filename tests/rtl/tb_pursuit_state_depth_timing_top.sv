`timescale 1ns/1ps
`default_nettype none
module tb_pursuit_state_depth_timing_top #(parameter integer STATELESS_DEPTH = 2);
    reg clk = 0;
    always #3.125 clk = ~clk;
    reg rst = 1;
    reg host_write_enable = 0;
    reg [7:0] host_address = 0;
    reg [31:0] host_write_data = 0;
    wire [31:0] host_read_data;
    pursuit_state_depth_timing_top #(.STATELESS_DEPTH(STATELESS_DEPTH)) dut (.*);

    reg [255:0] conv_weights [0:4157];
    reg [255:0] linear_weights [0:1991];
    reg [23:0] constants [0:319];
    reg [4:0] fractions [0:319];
    reg [4:0] readout_fractions [0:1];
    reg [8:0] input_frame [0:24575];
    reg signed [23:0] expected_current [0:89999];
    reg [7:0] expected_spike [0:89999];
    reg signed [15:0] expected_membrane [0:89999];
    reg [63:0] expected_result [0:1];
    string image_directory, sample_directory;

    task bus_write(input [7:0] addr, input [31:0] value);
        @(negedge clk);
        host_write_enable = 1;
        host_address = addr;
        host_write_data = value;
    endtask
    task bus_read(input [7:0] addr, output [31:0] value);
        @(negedge clk);
        host_write_enable = 0;
        host_address = addr;
        repeat (2) @(posedge clk);
        #0.1;
        value = host_read_data;
    endtask

    function integer layer_offset(input [2:0] layer);
        case (layer)
            0: layer_offset = 0;
            1: layer_offset = 24576;
            2: layer_offset = 36864;
            3: layer_offset = 41472;
            4: layer_offset = 43008;
            5: layer_offset = 43488;
            6: layer_offset = 43968;
            7: layer_offset = 44448;
        endcase
    endfunction
    function integer layer_spatial(input [2:0] layer);
        case (layer)
            0: layer_spatial = 3072;
            1: layer_spatial = 768;
            2: layer_spatial = 192;
            3: layer_spatial = 48;
            default: layer_spatial = 12;
        endcase
    endfunction

    integer transactions = 0, values_checked = 0;
    integer lane, channel, expected_index;
    always @(posedge dut.compute_clk) begin
        if (!dut.core_reset && dut.trace_valid) begin
            transactions = transactions + 1;
            for (lane = 0; lane < (dut.trace_is_dense ? 16 : 8); lane = lane + 1) begin
                channel = dut.trace_channel_base + lane;
                if (!dut.trace_is_dense)
                    expected_index = dut.trace_time_index * 45000
                        + layer_offset(dut.trace_layer)
                        + channel * layer_spatial(dut.trace_layer) + dut.trace_address;
                else if (dut.trace_layer == 0 && channel < 60)
                    expected_index = dut.trace_time_index * 45000 + 44928 + channel;
                else if (dut.trace_layer == 1 && channel < 12)
                    expected_index = dut.trace_time_index * 45000 + 44988 + channel;
                else expected_index = -1;
                if (expected_index >= 0) begin
                    if ($signed(dut.trace_current_flat[lane*24 +: 24]) !== expected_current[expected_index])
                        $fatal(1, "current mismatch index=%0d", expected_index);
                    if (dut.trace_spike_flat[lane*8 +: 8] !== expected_spike[expected_index])
                        $fatal(1, "spike mismatch index=%0d", expected_index);
                    if ($signed(dut.trace_membrane_flat[lane*16 +: 16]) !== expected_membrane[expected_index])
                        $fatal(1, "membrane mismatch index=%0d", expected_index);
                    values_checked = values_checked + 3;
                end
            end
        end
    end


    integer layer_cycles [0:7];
    integer state_writes [0:7];
    integer state_reads [0:7];
    integer mac_issues [0:7];
    integer monitor_index;
    initial for (monitor_index=0; monitor_index<8; monitor_index=monitor_index+1) begin
        layer_cycles[monitor_index]=0;
        state_writes[monitor_index]=0;
        state_reads[monitor_index]=0;
        mac_issues[monitor_index]=0;
    end
    always @(posedge dut.compute_clk) begin
        if (!dut.core_reset && dut.core.busy && dut.core.phase==2) begin
            layer_cycles[dut.core.layer_index]=layer_cycles[dut.core.layer_index]+1;
            if (dut.core.convolution_engine.output_valid && !dut.core.convolution_engine.output_time && !dut.core.convolution_engine.stateless_register)
                state_writes[dut.core.layer_index]=state_writes[dut.core.layer_index]+1;
            if (dut.core.convolution_engine.neuron_current_valid && dut.core.convolution_engine.neuron_current_time)
                state_reads[dut.core.layer_index]=state_reads[dut.core.layer_index]+1;
            if (dut.core.convolution_engine.mac_issue_valid)
                mac_issues[dut.core.layer_index]=mac_issues[dut.core.layer_index]+1;
        end
    end

    integer i, word_index;
    reg [31:0] read_value;
    initial begin
        if (!$value$plusargs("IMAGE_DIRECTORY=%s", image_directory)) $fatal(1, "IMAGE_DIRECTORY required");
        if (!$value$plusargs("SAMPLE_DIRECTORY=%s", sample_directory)) $fatal(1, "SAMPLE_DIRECTORY required");
        $readmemh({image_directory, "/conv_weights_tiled_256.mem"}, conv_weights);
        $readmemh({image_directory, "/linear_weights_input_major_256.mem"}, linear_weights);
        $readmemh({image_directory, "/layer_constants_s24.mem"}, constants);
        $readmemh({image_directory, "/layer_fraction_bits_u5.mem"}, fractions);
        $readmemh({image_directory, "/readout_fraction_bits_u5.mem"}, readout_fractions);
        $readmemh({sample_directory, "/input.mem"}, input_frame);
        $readmemh({sample_directory, "/current.mem"}, expected_current);
        $readmemh({sample_directory, "/spike.mem"}, expected_spike);
        $readmemh({sample_directory, "/membrane.mem"}, expected_membrane);
        $readmemh({sample_directory, "/result.mem"}, expected_result);
        wait(glbl.GSR == 0);
        repeat (5) @(negedge clk);
        rst = 0;
        repeat (3) @(negedge clk);
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
        @(negedge clk);
        host_write_enable = 0;
        wait(dut.done);
        #0.1;
        if (!dut.result_valid || dut.range_error_seen) $fatal(1, "invalid final status");
        if (transactions != (11242 - (STATELESS_DEPTH >= 1 ? 3072 : 0) - (STATELESS_DEPTH >= 2 ? 1536 : 0) - (STATELESS_DEPTH >= 3 ? 576 : 0)) || values_checked != (270000 - (STATELESS_DEPTH >= 1 ? 73728 : 0) - (STATELESS_DEPTH >= 2 ? 36864 : 0) - (STATELESS_DEPTH >= 3 ? 13824 : 0))) $fatal(1, "trace count mismatch");
        repeat (2) @(posedge clk);
        for (i = 0; i < 4; i = i + 1) begin
            bus_read(8'h20 + i, read_value);
            if (read_value !== expected_result[i/2][(i%2)*32 +: 32])
                $fatal(1, "result bus read mismatch word=%0d", i);
        end
        bus_read(8'h24, read_value);
        if (read_value[9:0] !== {readout_fractions[1], readout_fractions[0]}) $fatal(1, "fraction bus mismatch");
        bus_read(8'h25, read_value);
        if (read_value !== dut.inference_cycles) $fatal(1, "cycle bus mismatch");
        bus_read(8'h00, read_value);
        if (read_value !== 32'h2) $fatal(1, "status bus mismatch: %h", read_value);
        $display("PASS: state-depth register-bus replay transactions=%0d values=%0d core_cycles=%0d", transactions, values_checked, dut.inference_cycles);
        for (monitor_index=0; monitor_index<8; monitor_index=monitor_index+1)
            $display("LAYER layer=%0d cycles=%0d state_write_groups=%0d state_read_groups=%0d mac_issue_groups=%0d",
                monitor_index+1,layer_cycles[monitor_index],state_writes[monitor_index],state_reads[monitor_index],mac_issues[monitor_index]);
        $finish;
    end
endmodule
`default_nettype wire
