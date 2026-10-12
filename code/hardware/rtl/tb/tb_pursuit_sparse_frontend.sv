`timescale 1ns/1ps
`default_nettype none

module tb_pursuit_sparse_frontend;
    localparam integer L1_INPUT_PIXELS = 96*128;
    localparam integer L1_OUTPUT_PIXELS = 48*64;
    localparam integer L2_OUTPUT_PIXELS = 24*32;
    localparam integer L1_WEIGHT_COUNT = 8*2*9;
    localparam integer L2_WEIGHT_COUNT = 16*8*9;

    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;

    reg feed_active = 1'b0;
    integer feed_index = 0;
    reg event_list_polarity [0:399];
    reg [6:0] event_list_row [0:399];
    reg [6:0] event_list_column [0:399];
    wire event_valid = feed_active && feed_index < 400;
    wire event_ready;
    wire event_polarity = event_list_polarity[feed_index];
    wire [6:0] event_row = event_list_row[feed_index];
    wire [6:0] event_column = event_list_column[feed_index];
    reg events_end = 1'b0;

    reg l1_weight_write_enable = 1'b0;
    reg [2:0] l1_weight_write_output_channel = 0;
    reg l1_weight_write_input_channel = 0;
    reg [3:0] l1_weight_write_kernel_index = 0;
    reg signed [15:0] l1_weight_write_data = 0;
    reg l1_channel_write_enable = 0;
    reg [2:0] l1_channel_write_channel = 0;
    reg signed [23:0] l1_constant_write_data = 0;
    reg [4:0] l1_fraction_bits_write_data = 0;
    reg l2_weight_write_enable = 1'b0;
    reg [3:0] l2_weight_write_output_channel = 0;
    reg [2:0] l2_weight_write_input_channel = 0;
    reg [3:0] l2_weight_write_kernel_index = 0;
    reg signed [15:0] l2_weight_write_data = 0;
    reg l2_channel_write_enable = 0;
    reg [3:0] l2_channel_write_channel = 0;
    reg signed [23:0] l2_constant_write_data = 0;
    reg [4:0] l2_fraction_bits_write_data = 0;

    reg dense_read_enable = 0;
    reg [9:0] dense_read_address = 0;
    reg [3:0] dense_read_channel_base = 0;
    wire dense_read_valid;
    wire [31:0] dense_read_data_flat;

    wire l1_trace_valid;
    wire [11:0] l1_trace_address;
    wire [8*24-1:0] l1_trace_current_flat;
    wire [8*8-1:0] l1_trace_spike_flat;
    wire [8*16-1:0] l1_trace_membrane_flat;
    wire l1_trace_range_error;
    wire l2_trace_valid;
    wire [9:0] l2_trace_address;
    wire [16*24-1:0] l2_trace_current_flat;
    wire [16*8-1:0] l2_trace_spike_flat;
    wire [16*16-1:0] l2_trace_membrane_flat;
    wire l2_trace_range_error;
    wire l1_packet_valid;
    wire l1_packet_ready;
    wire [11:0] l1_packet_address;
    wire [2:0] l1_packet_channel;
    wire [7:0] l1_packet_count;
    wire l1_protocol_error;
    wire l2_protocol_error;
    wire protocol_error;
    wire [31:0] cycle_count;
    wire [31:0] l1_cycle_count;
    wire [31:0] l1_core_cycle_count;
    wire [31:0] l2_cycle_count;
    wire [31:0] accepted_event_count;
    wire [31:0] l1_emitted_packet_count;
    wire [31:0] l2_accepted_packet_count;
    wire [31:0] l2_dense_write_count;

    pursuit_sparse_frontend dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_polarity(event_polarity), .event_row(event_row),
        .event_column(event_column), .events_end(events_end),
        .l1_weight_write_enable(l1_weight_write_enable),
        .l1_weight_write_output_channel(l1_weight_write_output_channel),
        .l1_weight_write_input_channel(l1_weight_write_input_channel),
        .l1_weight_write_kernel_index(l1_weight_write_kernel_index),
        .l1_weight_write_data(l1_weight_write_data),
        .l1_channel_write_enable(l1_channel_write_enable),
        .l1_channel_write_channel(l1_channel_write_channel),
        .l1_constant_write_data(l1_constant_write_data),
        .l1_fraction_bits_write_data(l1_fraction_bits_write_data),
        .l2_weight_write_enable(l2_weight_write_enable),
        .l2_weight_write_output_channel(l2_weight_write_output_channel),
        .l2_weight_write_input_channel(l2_weight_write_input_channel),
        .l2_weight_write_kernel_index(l2_weight_write_kernel_index),
        .l2_weight_write_data(l2_weight_write_data),
        .l2_channel_write_enable(l2_channel_write_enable),
        .l2_channel_write_channel(l2_channel_write_channel),
        .l2_constant_write_data(l2_constant_write_data),
        .l2_fraction_bits_write_data(l2_fraction_bits_write_data),
        .dense_read_enable(dense_read_enable),
        .dense_read_address(dense_read_address),
        .dense_read_channel_base(dense_read_channel_base),
        .dense_read_valid(dense_read_valid),
        .dense_read_data_flat(dense_read_data_flat),
        .l1_trace_valid(l1_trace_valid),
        .l1_trace_address(l1_trace_address),
        .l1_trace_current_flat(l1_trace_current_flat),
        .l1_trace_spike_flat(l1_trace_spike_flat),
        .l1_trace_membrane_flat(l1_trace_membrane_flat),
        .l1_trace_range_error(l1_trace_range_error),
        .l2_trace_valid(l2_trace_valid),
        .l2_trace_address(l2_trace_address),
        .l2_trace_current_flat(l2_trace_current_flat),
        .l2_trace_spike_flat(l2_trace_spike_flat),
        .l2_trace_membrane_flat(l2_trace_membrane_flat),
        .l2_trace_range_error(l2_trace_range_error),
        .l1_packet_valid(l1_packet_valid),
        .l1_packet_ready(l1_packet_ready),
        .l1_packet_address(l1_packet_address),
        .l1_packet_channel(l1_packet_channel),
        .l1_packet_count(l1_packet_count),
        .l1_protocol_error(l1_protocol_error),
        .l2_protocol_error(l2_protocol_error),
        .protocol_error(protocol_error),
        .cycle_count(cycle_count),
        .l1_cycle_count(l1_cycle_count),
        .l1_core_cycle_count(l1_core_cycle_count),
        .l2_cycle_count(l2_cycle_count),
        .accepted_event_count(accepted_event_count),
        .l1_emitted_packet_count(l1_emitted_packet_count),
        .l2_accepted_packet_count(l2_accepted_packet_count),
        .l2_dense_write_count(l2_dense_write_count)
    );

    reg [8:0] input_memory [0:2*L1_INPUT_PIXELS-1];
    reg [15:0] l1_weight_memory [0:L1_WEIGHT_COUNT-1];
    reg [23:0] l1_constant_memory [0:7];
    reg [7:0] l1_fraction_memory [0:7];
    reg [15:0] l2_weight_memory [0:L2_WEIGHT_COUNT-1];
    reg [23:0] l2_constant_memory [0:15];
    reg [7:0] l2_fraction_memory [0:15];
    reg [23:0] golden_l1_current [0:8*L1_OUTPUT_PIXELS-1];
    reg [7:0] golden_l1_spike [0:8*L1_OUTPUT_PIXELS-1];
    reg [15:0] golden_l1_membrane [0:8*L1_OUTPUT_PIXELS-1];
    reg [23:0] golden_l2_current [0:16*L2_OUTPUT_PIXELS-1];
    reg [7:0] golden_l2_spike [0:16*L2_OUTPUT_PIXELS-1];
    reg [15:0] golden_l2_membrane [0:16*L2_OUTPUT_PIXELS-1];
    string golden_directory;
    integer init_index;
    integer init_address;
    integer init_channel;
    integer init_repeat;
    integer built_events;

    initial begin
        if (!$value$plusargs("GOLDEN_DIR=%s", golden_directory))
            $fatal(1, "GOLDEN_DIR plusarg is required");
        $readmemh($sformatf("%s/mem/vector_input_event_count_integer.mem",
            golden_directory), input_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_weight_integer.mem",
            golden_directory), l1_weight_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_constant_integer.mem",
            golden_directory), l1_constant_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_current_and_state_fraction_bits.mem",
            golden_directory), l1_fraction_memory);
        $readmemh($sformatf("%s/mem/parameter_L2_weight_integer.mem",
            golden_directory), l2_weight_memory);
        $readmemh($sformatf("%s/mem/parameter_L2_constant_integer.mem",
            golden_directory), l2_constant_memory);
        $readmemh($sformatf("%s/mem/parameter_L2_current_and_state_fraction_bits.mem",
            golden_directory), l2_fraction_memory);
        $readmemh($sformatf("%s/mem/vector_t0_L1_current_integer.mem",
            golden_directory), golden_l1_current);
        $readmemh($sformatf("%s/mem/vector_t0_L1_spike_integer.mem",
            golden_directory), golden_l1_spike);
        $readmemh($sformatf("%s/mem/vector_t0_L1_membrane_integer.mem",
            golden_directory), golden_l1_membrane);
        $readmemh($sformatf("%s/mem/vector_t0_L2_current_integer.mem",
            golden_directory), golden_l2_current);
        $readmemh($sformatf("%s/mem/vector_t0_L2_spike_integer.mem",
            golden_directory), golden_l2_spike);
        $readmemh($sformatf("%s/mem/vector_t0_L2_membrane_integer.mem",
            golden_directory), golden_l2_membrane);
        built_events = 0;
        for (init_address = 0; init_address < L1_INPUT_PIXELS;
                init_address = init_address + 1) begin
            for (init_channel = 0; init_channel < 2;
                    init_channel = init_channel + 1) begin
                for (init_repeat = 0;
                        init_repeat < input_memory[
                            init_channel*L1_INPUT_PIXELS + init_address
                        ]; init_repeat = init_repeat + 1) begin
                    if (built_events >= 400)
                        $fatal(1, "input contains more than 400 events");
                    event_list_polarity[built_events] = init_channel[0];
                    event_list_row[built_events] = init_address / 128;
                    event_list_column[built_events] = init_address % 128;
                    built_events = built_events + 1;
                end
            end
        end
        if (built_events != 400)
            $fatal(1, "input contains %0d events", built_events);
    end

    always @(posedge clk) begin
        if (rst) begin
            feed_index <= 0;
        end else if (event_valid && event_ready) begin
            feed_index <= feed_index + 1;
        end
    end

    integer l1_trace_seen = 0;
    integer l2_trace_seen = 0;
    integer packet_seen = 0;
    integer errors_seen = 0;
    integer monitor_channel;
    integer monitor_expected;
    always @(posedge clk) begin
        if (l1_trace_valid) begin
            if (l1_trace_address !== l1_trace_seen[11:0]
                    || l1_trace_range_error) errors_seen = errors_seen + 1;
            for (monitor_channel = 0; monitor_channel < 8;
                    monitor_channel = monitor_channel + 1) begin
                monitor_expected = monitor_channel*L1_OUTPUT_PIXELS
                    + l1_trace_address;
                if (l1_trace_current_flat[monitor_channel*24 +: 24]
                        !== golden_l1_current[monitor_expected])
                    errors_seen = errors_seen + 1;
                if (l1_trace_spike_flat[monitor_channel*8 +: 8]
                        !== golden_l1_spike[monitor_expected])
                    errors_seen = errors_seen + 1;
                if (l1_trace_membrane_flat[monitor_channel*16 +: 16]
                        !== golden_l1_membrane[monitor_expected])
                    errors_seen = errors_seen + 1;
            end
            l1_trace_seen = l1_trace_seen + 1;
        end
        if (l1_packet_valid && l1_packet_ready) begin
            monitor_expected = l1_packet_channel*L1_OUTPUT_PIXELS
                + l1_packet_address;
            if (l1_packet_count !== golden_l1_spike[monitor_expected]
                    || l1_packet_count == 0) errors_seen = errors_seen + 1;
            packet_seen = packet_seen + 1;
        end
        if (l2_trace_valid) begin
            if (l2_trace_address !== l2_trace_seen[9:0]
                    || l2_trace_range_error) errors_seen = errors_seen + 1;
            for (monitor_channel = 0; monitor_channel < 16;
                    monitor_channel = monitor_channel + 1) begin
                monitor_expected = monitor_channel*L2_OUTPUT_PIXELS
                    + l2_trace_address;
                if (l2_trace_current_flat[monitor_channel*24 +: 24]
                        !== golden_l2_current[monitor_expected])
                    errors_seen = errors_seen + 1;
                if (l2_trace_spike_flat[monitor_channel*8 +: 8]
                        !== golden_l2_spike[monitor_expected])
                    errors_seen = errors_seen + 1;
                if (l2_trace_membrane_flat[monitor_channel*16 +: 16]
                        !== golden_l2_membrane[monitor_expected])
                    errors_seen = errors_seen + 1;
            end
            l2_trace_seen = l2_trace_seen + 1;
        end
    end

    integer run_index;
    integer timeout_cycles;
    integer read_address_index;
    integer read_base;
    integer read_lane;
    initial begin
        repeat (5) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;
        for (run_index = 0; run_index < L1_WEIGHT_COUNT;
                run_index = run_index + 1) begin
            @(negedge clk);
            l1_weight_write_enable = 1'b1;
            l1_weight_write_output_channel = run_index / 18;
            l1_weight_write_input_channel = (run_index / 9) % 2;
            l1_weight_write_kernel_index = run_index % 9;
            l1_weight_write_data = l1_weight_memory[run_index];
        end
        @(negedge clk); l1_weight_write_enable = 1'b0;
        for (run_index = 0; run_index < 8; run_index = run_index + 1) begin
            @(negedge clk);
            l1_channel_write_enable = 1'b1;
            l1_channel_write_channel = run_index;
            l1_constant_write_data = l1_constant_memory[run_index];
            l1_fraction_bits_write_data = l1_fraction_memory[run_index][4:0];
        end
        @(negedge clk); l1_channel_write_enable = 1'b0;
        for (run_index = 0; run_index < L2_WEIGHT_COUNT;
                run_index = run_index + 1) begin
            @(negedge clk);
            l2_weight_write_enable = 1'b1;
            l2_weight_write_output_channel = run_index / 72;
            l2_weight_write_input_channel = (run_index / 9) % 8;
            l2_weight_write_kernel_index = run_index % 9;
            l2_weight_write_data = l2_weight_memory[run_index];
        end
        @(negedge clk); l2_weight_write_enable = 1'b0;
        for (run_index = 0; run_index < 16; run_index = run_index + 1) begin
            @(negedge clk);
            l2_channel_write_enable = 1'b1;
            l2_channel_write_channel = run_index;
            l2_constant_write_data = l2_constant_memory[run_index];
            l2_fraction_bits_write_data = l2_fraction_memory[run_index][4:0];
        end
        @(negedge clk); l2_channel_write_enable = 1'b0;

        start = 1'b1;
        @(negedge clk); start = 1'b0; feed_active = 1'b1;
        wait (feed_index == 400);
        @(negedge clk); feed_active = 1'b0; events_end = 1'b1;
        @(negedge clk); events_end = 1'b0;

        timeout_cycles = 0;
        while (!done && timeout_cycles < 30000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!done) $fatal(1, "integrated sparse front-end timeout");
        @(posedge clk);
        if (protocol_error || l1_protocol_error || l2_protocol_error)
            $fatal(1, "integrated sparse front-end protocol error");
        if (accepted_event_count != 400
                || l1_trace_seen != L1_OUTPUT_PIXELS
                || l2_trace_seen != L2_OUTPUT_PIXELS
                || packet_seen != 2575
                || l1_emitted_packet_count != packet_seen
                || l2_accepted_packet_count != packet_seen
                || l2_dense_write_count != L2_OUTPUT_PIXELS)
            $fatal(1, "integrated count mismatch events=%0d l1trace=%0d packets=%0d/%0d/%0d l2trace=%0d writes=%0d",
                accepted_event_count, l1_trace_seen, packet_seen,
                l1_emitted_packet_count, l2_accepted_packet_count,
                l2_trace_seen, l2_dense_write_count);

        for (read_address_index = 0;
                read_address_index < L2_OUTPUT_PIXELS;
                read_address_index = read_address_index + 1) begin
            for (read_base = 0; read_base < 16; read_base = read_base + 4) begin
                @(negedge clk);
                dense_read_enable = 1'b1;
                dense_read_address = read_address_index;
                dense_read_channel_base = read_base;
                @(negedge clk);
                if (!dense_read_valid) errors_seen = errors_seen + 1;
                for (read_lane = 0; read_lane < 4;
                        read_lane = read_lane + 1) begin
                    monitor_expected = (read_base + read_lane)*L2_OUTPUT_PIXELS
                        + read_address_index;
                    if (dense_read_data_flat[read_lane*8 +: 8]
                            !== golden_l2_spike[monitor_expected])
                        errors_seen = errors_seen + 1;
                end
            end
        end
        @(negedge clk); dense_read_enable = 1'b0;
        if (errors_seen != 0)
            $fatal(1, "integrated sparse front-end golden errors=%0d",
                errors_seen);
        $display("tb_pursuit_sparse_frontend PASS events=%0d packets=%0d L1_traces=%0d L2_traces=%0d cycles=%0d l1=%0d l1_core=%0d l2=%0d",
            accepted_event_count, packet_seen, l1_trace_seen, l2_trace_seen,
            cycle_count, l1_cycle_count, l1_core_cycle_count, l2_cycle_count);
        $finish;
    end
endmodule

`default_nettype wire
