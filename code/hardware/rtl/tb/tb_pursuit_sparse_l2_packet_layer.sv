`timescale 1ns/1ps
`default_nettype none

module tb_pursuit_sparse_l2_packet_layer;
    localparam integer INPUT_WIDTH = 64;
    localparam integer INPUT_HEIGHT = 48;
    localparam integer INPUT_PIXELS = INPUT_WIDTH*INPUT_HEIGHT;
    localparam integer INPUT_CHANNELS = 8;
    localparam integer OUTPUT_CHANNELS = 16;
    localparam integer OUTPUT_PIXELS = (INPUT_WIDTH/2)*(INPUT_HEIGHT/2);
    localparam integer WEIGHT_COUNT = OUTPUT_CHANNELS*INPUT_CHANNELS*9;
    localparam integer CURRENT_BITS = 24;
    localparam integer STATE_BITS = 16;
    localparam integer SPIKE_BITS = 8;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;

    reg sparse_valid = 1'b0;
    wire sparse_ready;
    reg [11:0] sparse_address = 0;
    reg [2:0] sparse_channel = 0;
    reg [7:0] sparse_count = 0;
    reg sparse_end = 1'b0;

    reg weight_write_enable = 1'b0;
    reg [3:0] weight_write_output_channel = 0;
    reg [2:0] weight_write_input_channel = 0;
    reg [3:0] weight_write_kernel_index = 0;
    reg signed [15:0] weight_write_data = 0;
    reg channel_write_enable = 1'b0;
    reg [3:0] channel_write_channel = 0;
    reg signed [23:0] constant_write_data = 0;
    reg [4:0] fraction_bits_write_data = 0;

    reg dense_read_enable = 1'b0;
    reg [9:0] dense_read_address = 0;
    reg [3:0] dense_read_channel_base = 0;
    wire dense_read_valid;
    wire [31:0] dense_read_data_flat;

    wire trace_valid;
    wire [9:0] trace_address;
    wire [OUTPUT_CHANNELS*CURRENT_BITS-1:0] trace_current_flat;
    wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0] trace_spike_flat;
    wire [OUTPUT_CHANNELS*STATE_BITS-1:0] trace_membrane_flat;
    wire trace_range_error;
    wire protocol_error;
    wire [31:0] cycle_count;
    wire [31:0] accepted_packet_count;
    wire [31:0] dense_write_count;

    pursuit_sparse_l2_packet_layer dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .sparse_valid(sparse_valid), .sparse_ready(sparse_ready),
        .sparse_address(sparse_address), .sparse_channel(sparse_channel),
        .sparse_count(sparse_count), .sparse_end(sparse_end),
        .weight_write_enable(weight_write_enable),
        .weight_write_output_channel(weight_write_output_channel),
        .weight_write_input_channel(weight_write_input_channel),
        .weight_write_kernel_index(weight_write_kernel_index),
        .weight_write_data(weight_write_data),
        .channel_write_enable(channel_write_enable),
        .channel_write_channel(channel_write_channel),
        .constant_write_data(constant_write_data),
        .fraction_bits_write_data(fraction_bits_write_data),
        .dense_read_enable(dense_read_enable),
        .dense_read_address(dense_read_address),
        .dense_read_channel_base(dense_read_channel_base),
        .dense_read_valid(dense_read_valid),
        .dense_read_data_flat(dense_read_data_flat),
        .trace_valid(trace_valid), .trace_address(trace_address),
        .trace_current_flat(trace_current_flat),
        .trace_spike_flat(trace_spike_flat),
        .trace_membrane_flat(trace_membrane_flat),
        .trace_range_error(trace_range_error),
        .protocol_error(protocol_error), .cycle_count(cycle_count),
        .accepted_packet_count(accepted_packet_count),
        .dense_write_count(dense_write_count)
    );

    reg [7:0] input_spike [0:INPUT_CHANNELS*INPUT_PIXELS-1];
    reg [15:0] weight_memory [0:WEIGHT_COUNT-1];
    reg [23:0] constant_memory [0:OUTPUT_CHANNELS-1];
    reg [7:0] fraction_memory [0:OUTPUT_CHANNELS-1];
    reg [23:0] golden_current [0:OUTPUT_CHANNELS*OUTPUT_PIXELS-1];
    reg [7:0] golden_spike [0:OUTPUT_CHANNELS*OUTPUT_PIXELS-1];
    reg [15:0] golden_membrane [0:OUTPUT_CHANNELS*OUTPUT_PIXELS-1];
    string golden_directory;
    integer index;
    integer trace_channel_index;
    integer stimulus_channel_index;
    integer address_index;
    integer input_nonzero;

    initial begin
        if (!$value$plusargs("GOLDEN_DIR=%s", golden_directory)) begin
            $fatal(1, "GOLDEN_DIR plusarg is required");
        end
        $readmemh($sformatf("%s/mem/vector_t0_L1_spike_integer.mem",
            golden_directory), input_spike);
        $readmemh($sformatf("%s/mem/parameter_L2_weight_integer.mem",
            golden_directory), weight_memory);
        $readmemh($sformatf("%s/mem/parameter_L2_constant_integer.mem",
            golden_directory), constant_memory);
        $readmemh($sformatf("%s/mem/parameter_L2_current_and_state_fraction_bits.mem",
            golden_directory), fraction_memory);
        $readmemh($sformatf("%s/mem/vector_t0_L2_current_integer.mem",
            golden_directory), golden_current);
        $readmemh($sformatf("%s/mem/vector_t0_L2_spike_integer.mem",
            golden_directory), golden_spike);
        $readmemh($sformatf("%s/mem/vector_t0_L2_membrane_integer.mem",
            golden_directory), golden_membrane);
        input_nonzero = 0;
        for (index = 0; index < INPUT_CHANNELS*INPUT_PIXELS;
                index = index + 1) begin
            if (input_spike[index] != 0) input_nonzero = input_nonzero + 1;
        end
    end

    integer run_mode = 0;
    integer trace_seen = 0;
    integer errors_seen = 0;
    integer expected_index;
    integer signed default_current;
    integer default_spike;
    integer signed default_membrane;

    task automatic compute_default;
        input integer output_channel;
        begin
            default_current = $signed(constant_memory[output_channel]);
            if (default_current > 8388607) default_current = 8388607;
            if (default_current < -8388608) default_current = -8388608;
            if (default_current > 0)
                default_spike = default_current
                    >>> fraction_memory[output_channel];
            else
                default_spike = 0;
            if (default_spike > 255) default_spike = 255;
            default_membrane = default_current
                - (default_spike <<< fraction_memory[output_channel]);
            if (default_membrane
                    < -(1 <<< fraction_memory[output_channel])) begin
                default_membrane = -(1 <<< fraction_memory[output_channel]);
            end
        end
    endtask

    always @(posedge clk) begin
        if (trace_valid) begin
            if (trace_address !== trace_seen[9:0]) begin
                $display("L2 trace address mismatch got=%0d expected=%0d",
                    trace_address, trace_seen);
                errors_seen = errors_seen + 1;
            end
            if (trace_range_error) begin
                $display("L2 unexpected range error at %0d", trace_address);
                errors_seen = errors_seen + 1;
            end
            for (trace_channel_index = 0;
                    trace_channel_index < OUTPUT_CHANNELS;
                    trace_channel_index = trace_channel_index + 1) begin
                if (run_mode == 0) begin
                    expected_index = trace_channel_index*OUTPUT_PIXELS
                        + trace_address;
                    if (trace_current_flat[trace_channel_index*24 +: 24]
                            !== golden_current[expected_index]) begin
                        if (errors_seen < 20) $display(
                            "L2 current mismatch a=%0d c=%0d got=%0d exp=%0d",
                            trace_address, trace_channel_index,
                            $signed(trace_current_flat[trace_channel_index*24 +: 24]),
                            $signed(golden_current[expected_index]));
                        errors_seen = errors_seen + 1;
                    end
                    if (trace_spike_flat[trace_channel_index*8 +: 8]
                            !== golden_spike[expected_index]) begin
                        if (errors_seen < 20) $display(
                            "L2 spike mismatch a=%0d c=%0d got=%0d exp=%0d",
                            trace_address, trace_channel_index,
                            trace_spike_flat[trace_channel_index*8 +: 8],
                            golden_spike[expected_index]);
                        errors_seen = errors_seen + 1;
                    end
                    if (trace_membrane_flat[trace_channel_index*16 +: 16]
                            !== golden_membrane[expected_index]) begin
                        if (errors_seen < 20) $display(
                            "L2 membrane mismatch a=%0d c=%0d got=%0d exp=%0d",
                            trace_address, trace_channel_index,
                            $signed(trace_membrane_flat[trace_channel_index*16 +: 16]),
                            $signed(golden_membrane[expected_index]));
                        errors_seen = errors_seen + 1;
                    end
                end else begin
                    compute_default(trace_channel_index);
                    if ($signed(trace_current_flat[trace_channel_index*24 +: 24])
                            != default_current
                            || trace_spike_flat[trace_channel_index*8 +: 8]
                                != default_spike
                            || $signed(trace_membrane_flat[
                                trace_channel_index*16 +: 16
                            ]) != default_membrane) begin
                        if (errors_seen < 20) $display(
                            "L2 default mismatch a=%0d c=%0d current=%0d/%0d spike=%0d/%0d membrane=%0d/%0d",
                            trace_address, trace_channel_index,
                            $signed(trace_current_flat[trace_channel_index*24 +: 24]),
                            default_current,
                            trace_spike_flat[trace_channel_index*8 +: 8],
                            default_spike,
                            $signed(trace_membrane_flat[trace_channel_index*16 +: 16]),
                            default_membrane);
                        errors_seen = errors_seen + 1;
                    end
                end
            end
            trace_seen = trace_seen + 1;
        end
    end

    task automatic send_packet;
        input integer packet_channel;
        input integer packet_address;
        input integer packet_count;
        integer accepted_before;
        begin
            @(negedge clk);
            accepted_before = accepted_packet_count;
            sparse_valid = 1'b1;
            sparse_address = packet_address;
            sparse_channel = packet_channel;
            sparse_count = packet_count;
            while (accepted_packet_count == accepted_before) begin
                @(negedge clk);
            end
            sparse_valid = 1'b0;
        end
    endtask

    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic pulse_end;
        begin
            @(negedge clk);
            sparse_valid = 1'b0;
            sparse_end = 1'b1;
            @(negedge clk);
            sparse_end = 1'b0;
        end
    endtask

    integer timeout_cycles;
    integer first_run_cycles;
    integer first_run_writes;
    integer base_index;
    integer read_lane;
    initial begin
        repeat (5) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        for (index = 0; index < WEIGHT_COUNT; index = index + 1) begin
            @(negedge clk);
            weight_write_enable = 1'b1;
            weight_write_output_channel = index / (INPUT_CHANNELS*9);
            weight_write_input_channel = (index / 9) % INPUT_CHANNELS;
            weight_write_kernel_index = index % 9;
            weight_write_data = weight_memory[index];
        end
        @(negedge clk);
        weight_write_enable = 1'b0;
        for (index = 0; index < OUTPUT_CHANNELS; index = index + 1) begin
            @(negedge clk);
            channel_write_enable = 1'b1;
            channel_write_channel = index;
            constant_write_data = constant_memory[index];
            fraction_bits_write_data = fraction_memory[index][4:0];
        end
        @(negedge clk);
        channel_write_enable = 1'b0;

        run_mode = 0;
        trace_seen = 0;
        pulse_start();
        for (address_index = 0; address_index < INPUT_PIXELS;
                address_index = address_index + 1) begin
            for (stimulus_channel_index = 0;
                    stimulus_channel_index < INPUT_CHANNELS;
                    stimulus_channel_index = stimulus_channel_index + 1) begin
                expected_index = stimulus_channel_index*INPUT_PIXELS + address_index;
                if (input_spike[expected_index] != 0) begin
                    send_packet(stimulus_channel_index, address_index,
                        input_spike[expected_index]);
                end
            end
        end
        pulse_end();
        timeout_cycles = 0;
        while (!done && timeout_cycles < 50000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!done) $fatal(1, "L2 sparse timeout");
        @(posedge clk);
        if (protocol_error) $fatal(1, "L2 protocol_error asserted");
        if (trace_seen != OUTPUT_PIXELS)
            $fatal(1, "L2 trace count got=%0d expected=%0d",
                trace_seen, OUTPUT_PIXELS);
        if (accepted_packet_count != input_nonzero)
            $fatal(1, "L2 packet count got=%0d expected=%0d",
                accepted_packet_count, input_nonzero);
        if (dense_write_count != OUTPUT_PIXELS)
            $fatal(1, "L2 dense writes got=%0d expected=%0d",
                dense_write_count, OUTPUT_PIXELS);
        first_run_cycles = cycle_count;
        first_run_writes = dense_write_count;

        // Verify the synchronous 4-lane L3 interface after done.  The memory
        // must remain readable and channel-major golden data is repacked here.
        for (address_index = 0; address_index < OUTPUT_PIXELS;
                address_index = address_index + 1) begin
            for (base_index = 0; base_index < OUTPUT_CHANNELS;
                    base_index = base_index + 4) begin
                @(negedge clk);
                dense_read_enable = 1'b1;
                dense_read_address = address_index;
                dense_read_channel_base = base_index;
                @(negedge clk);
                if (!dense_read_valid) begin
                    $display("L2 dense read_valid missing a=%0d base=%0d",
                        address_index, base_index);
                    errors_seen = errors_seen + 1;
                end
                for (read_lane = 0; read_lane < 4;
                        read_lane = read_lane + 1) begin
                    expected_index = (base_index + read_lane)*OUTPUT_PIXELS
                        + address_index;
                    if (dense_read_data_flat[read_lane*8 +: 8]
                            !== golden_spike[expected_index]) begin
                        if (errors_seen < 20) $display(
                            "L2 dense read mismatch a=%0d c=%0d got=%0d exp=%0d",
                            address_index, base_index + read_lane,
                            dense_read_data_flat[read_lane*8 +: 8],
                            golden_spike[expected_index]);
                        errors_seen = errors_seen + 1;
                    end
                end
            end
        end
        @(negedge clk);
        dense_read_enable = 1'b0;

        // A second empty run proves that bias-only defaults overwrite every
        // dense location and that no stale accumulator/RAM value leaks across
        // samples.
        run_mode = 1;
        trace_seen = 0;
        pulse_start();
        pulse_end();
        timeout_cycles = 0;
        while (!done && timeout_cycles < 10000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!done) $fatal(1, "L2 empty-stream timeout");
        @(posedge clk);
        if (trace_seen != OUTPUT_PIXELS)
            $fatal(1, "L2 empty trace count got=%0d expected=%0d",
                trace_seen, OUTPUT_PIXELS);
        if (accepted_packet_count != 0)
            $fatal(1, "L2 empty stream accepted %0d packets",
                accepted_packet_count);
        if (dense_write_count != OUTPUT_PIXELS)
            $fatal(1, "L2 empty dense writes got=%0d expected=%0d",
                dense_write_count, OUTPUT_PIXELS);
        if (errors_seen != 0)
            $fatal(1, "L2 sparse golden/default comparison failed errors=%0d",
                errors_seen);
        $display("tb_pursuit_sparse_l2_packet_layer PASS packets=%0d first_cycles=%0d first_writes=%0d empty_cycles=%0d",
            input_nonzero, first_run_cycles, first_run_writes, cycle_count);
        $finish;
    end
endmodule

`default_nettype wire
