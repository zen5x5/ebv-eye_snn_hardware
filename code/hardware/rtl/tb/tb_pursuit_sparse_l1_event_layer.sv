`timescale 1ns/1ps
`default_nettype none

module tb_pursuit_sparse_l1_event_layer;
    localparam integer INPUT_WIDTH = 128;
    localparam integer INPUT_HEIGHT = 96;
    localparam integer INPUT_PIXELS = INPUT_WIDTH*INPUT_HEIGHT;
    localparam integer INPUT_CHANNELS = 2;
    localparam integer OUTPUT_CHANNELS = 8;
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

    reg event_valid = 1'b0;
    wire event_ready;
    reg event_polarity = 1'b0;
    reg [6:0] event_row = 0;
    reg [6:0] event_column = 0;
    reg events_end = 1'b0;

    reg weight_write_enable = 1'b0;
    reg [2:0] weight_write_output_channel = 0;
    reg weight_write_input_channel = 1'b0;
    reg [3:0] weight_write_kernel_index = 0;
    reg signed [15:0] weight_write_data = 0;
    reg channel_write_enable = 1'b0;
    reg [2:0] channel_write_channel = 0;
    reg signed [23:0] constant_write_data = 0;
    reg [4:0] fraction_bits_write_data = 0;

    wire sparse_valid;
    reg sparse_ready = 1'b1;
    wire [11:0] sparse_address;
    wire [2:0] sparse_channel;
    wire [7:0] sparse_count;
    wire sparse_end;

    wire trace_valid;
    wire [11:0] trace_address;
    wire [OUTPUT_CHANNELS*CURRENT_BITS-1:0] trace_current_flat;
    wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0] trace_spike_flat;
    wire [OUTPUT_CHANNELS*STATE_BITS-1:0] trace_membrane_flat;
    wire trace_range_error;
    wire protocol_error;
    wire [31:0] cycle_count;
    wire [31:0] core_cycle_count;
    wire [31:0] accepted_event_count;
    wire [31:0] emitted_packet_count;

    pursuit_sparse_l1_event_layer dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .event_valid(event_valid), .event_ready(event_ready),
        .event_polarity(event_polarity), .event_row(event_row),
        .event_column(event_column), .events_end(events_end),
        .weight_write_enable(weight_write_enable),
        .weight_write_output_channel(weight_write_output_channel),
        .weight_write_input_channel(weight_write_input_channel),
        .weight_write_kernel_index(weight_write_kernel_index),
        .weight_write_data(weight_write_data),
        .channel_write_enable(channel_write_enable),
        .channel_write_channel(channel_write_channel),
        .constant_write_data(constant_write_data),
        .fraction_bits_write_data(fraction_bits_write_data),
        .sparse_valid(sparse_valid), .sparse_ready(sparse_ready),
        .sparse_address(sparse_address), .sparse_channel(sparse_channel),
        .sparse_count(sparse_count), .sparse_end(sparse_end),
        .trace_valid(trace_valid), .trace_address(trace_address),
        .trace_current_flat(trace_current_flat),
        .trace_spike_flat(trace_spike_flat),
        .trace_membrane_flat(trace_membrane_flat),
        .trace_range_error(trace_range_error),
        .protocol_error(protocol_error), .cycle_count(cycle_count),
        .core_cycle_count(core_cycle_count),
        .accepted_event_count(accepted_event_count),
        .emitted_packet_count(emitted_packet_count)
    );

    reg [8:0] input_memory [0:INPUT_CHANNELS*INPUT_PIXELS-1];
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
    integer repeat_index;
    integer event_total;
    integer expected_nonzero;

    initial begin
        if (!$value$plusargs("GOLDEN_DIR=%s", golden_directory)) begin
            $fatal(1, "GOLDEN_DIR plusarg is required");
        end
        $readmemh($sformatf("%s/mem/vector_input_event_count_integer.mem",
            golden_directory), input_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_weight_integer.mem",
            golden_directory), weight_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_constant_integer.mem",
            golden_directory), constant_memory);
        $readmemh($sformatf("%s/mem/parameter_L1_current_and_state_fraction_bits.mem",
            golden_directory), fraction_memory);
        $readmemh($sformatf("%s/mem/vector_t0_L1_current_integer.mem",
            golden_directory), golden_current);
        $readmemh($sformatf("%s/mem/vector_t0_L1_spike_integer.mem",
            golden_directory), golden_spike);
        $readmemh($sformatf("%s/mem/vector_t0_L1_membrane_integer.mem",
            golden_directory), golden_membrane);
        event_total = 0;
        for (index = 0; index < INPUT_CHANNELS*INPUT_PIXELS;
                index = index + 1) begin
            event_total = event_total + input_memory[index];
        end
        expected_nonzero = 0;
        for (index = 0; index < OUTPUT_CHANNELS*OUTPUT_PIXELS;
                index = index + 1) begin
            if (golden_spike[index] != 0) begin
                expected_nonzero = expected_nonzero + 1;
            end
        end
        if (event_total != 400) begin
            $fatal(1, "golden input has %0d events, expected 400", event_total);
        end
    end

    integer trace_seen = 0;
    integer sparse_seen = 0;
    integer errors_seen = 0;
    integer expected_index;
    integer expected_sparse_address = 0;
    integer expected_sparse_channel = 0;
    integer send_task_count = 0;
    integer state_idle_cycles = 0;
    integer state_clear_cycles = 0;
    integer state_wait_cycles = 0;
    integer state_update_cycles = 0;
    integer state_finalize_cycles = 0;
    integer state_drain_cycles = 0;

    task automatic advance_expected_sparse;
        integer found;
        begin
            found = 0;
            while (!found && expected_sparse_address < OUTPUT_PIXELS) begin
                expected_index = expected_sparse_channel*OUTPUT_PIXELS
                    + expected_sparse_address;
                if (golden_spike[expected_index] != 0) begin
                    found = 1;
                end else if (expected_sparse_channel == OUTPUT_CHANNELS-1) begin
                    expected_sparse_channel = 0;
                    expected_sparse_address = expected_sparse_address + 1;
                end else begin
                    expected_sparse_channel = expected_sparse_channel + 1;
                end
            end
        end
    endtask

    always @(posedge clk) begin
        if (dut.core.busy) begin
            case (dut.core.control_state)
                0: state_idle_cycles = state_idle_cycles + 1;
                1: state_clear_cycles = state_clear_cycles + 1;
                2: state_wait_cycles = state_wait_cycles + 1;
                3: state_update_cycles = state_update_cycles + 1;
                4: state_finalize_cycles = state_finalize_cycles + 1;
                5: state_drain_cycles = state_drain_cycles + 1;
            endcase
        end
        if (trace_valid) begin
            if (trace_address !== trace_seen[11:0]) begin
                $display("L1 trace address mismatch got=%0d expected=%0d",
                    trace_address, trace_seen);
                errors_seen = errors_seen + 1;
            end
            if (trace_range_error) begin
                $display("L1 unexpected range error at %0d", trace_address);
                errors_seen = errors_seen + 1;
            end
            for (trace_channel_index = 0;
                    trace_channel_index < OUTPUT_CHANNELS;
                    trace_channel_index = trace_channel_index + 1) begin
                expected_index = trace_channel_index*OUTPUT_PIXELS + trace_address;
                if (trace_current_flat[trace_channel_index*24 +: 24]
                        !== golden_current[expected_index]) begin
                    if (errors_seen < 20) $display(
                        "L1 current mismatch a=%0d c=%0d got=%0d exp=%0d",
                        trace_address, trace_channel_index,
                        $signed(trace_current_flat[trace_channel_index*24 +: 24]),
                        $signed(golden_current[expected_index]));
                    errors_seen = errors_seen + 1;
                end
                if (trace_spike_flat[trace_channel_index*8 +: 8]
                        !== golden_spike[expected_index]) begin
                    if (errors_seen < 20) $display(
                        "L1 spike mismatch a=%0d c=%0d got=%0d exp=%0d",
                        trace_address, trace_channel_index,
                        trace_spike_flat[trace_channel_index*8 +: 8],
                        golden_spike[expected_index]);
                    errors_seen = errors_seen + 1;
                end
                if (trace_membrane_flat[trace_channel_index*16 +: 16]
                        !== golden_membrane[expected_index]) begin
                    if (errors_seen < 20) $display(
                        "L1 membrane mismatch a=%0d c=%0d got=%0d exp=%0d",
                        trace_address, trace_channel_index,
                        $signed(trace_membrane_flat[trace_channel_index*16 +: 16]),
                        $signed(golden_membrane[expected_index]));
                    errors_seen = errors_seen + 1;
                end
            end
            trace_seen = trace_seen + 1;
        end
        if (sparse_valid && sparse_ready) begin
            advance_expected_sparse();
            expected_index = expected_sparse_channel*OUTPUT_PIXELS
                + expected_sparse_address;
            if (sparse_address !== expected_sparse_address[11:0]
                    || sparse_channel !== expected_sparse_channel[2:0]
                    || sparse_count !== golden_spike[expected_index]) begin
                if (errors_seen < 20) $display(
                    "L1 sparse mismatch got=(%0d,%0d,%0d) exp=(%0d,%0d,%0d)",
                    sparse_address, sparse_channel, sparse_count,
                    expected_sparse_address, expected_sparse_channel,
                    golden_spike[expected_index]);
                errors_seen = errors_seen + 1;
            end
            sparse_seen = sparse_seen + 1;
            if (expected_sparse_channel == OUTPUT_CHANNELS-1) begin
                expected_sparse_channel = 0;
                expected_sparse_address = expected_sparse_address + 1;
            end else begin
                expected_sparse_channel = expected_sparse_channel + 1;
            end
        end
    end

    task automatic send_event;
        input integer packet_channel;
        input integer packet_address;
        integer accepted_before;
        begin
            send_task_count = send_task_count + 1;
            @(negedge clk);
            accepted_before = accepted_event_count;
            event_valid = 1'b1;
            event_polarity = packet_channel[0];
            event_row = packet_address / INPUT_WIDTH;
            event_column = packet_address % INPUT_WIDTH;
            while (accepted_event_count == accepted_before) begin
                @(negedge clk);
            end
            event_valid = 1'b0;
        end
    endtask

    integer timeout_cycles;
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
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        for (address_index = 0; address_index < INPUT_PIXELS;
                address_index = address_index + 1) begin
            for (stimulus_channel_index = 0;
                    stimulus_channel_index < INPUT_CHANNELS;
                    stimulus_channel_index = stimulus_channel_index + 1) begin
                for (repeat_index = 0;
                        repeat_index
                            < input_memory[stimulus_channel_index*INPUT_PIXELS
                                + address_index];
                        repeat_index = repeat_index + 1) begin
                    send_event(stimulus_channel_index, address_index);
                end
            end
        end
        @(negedge clk);
        event_valid = 1'b0;
        events_end = 1'b1;
        @(negedge clk);
        events_end = 1'b0;

        timeout_cycles = 0;
        while (!done && timeout_cycles < 30000) begin
            @(posedge clk);
            timeout_cycles = timeout_cycles + 1;
        end
        if (!done) $fatal(1, "L1 sparse timeout");
        @(posedge clk);
        if (protocol_error) $fatal(1, "L1 protocol_error asserted");
        $display("L1 DEBUG counters sent=%0d accepted=%0d emitted=%0d trace=%0d sparse=%0d expected_sparse=%0d",
            send_task_count,
            accepted_event_count, emitted_packet_count, trace_seen,
            sparse_seen, expected_nonzero);
        if (trace_seen != OUTPUT_PIXELS)
            $fatal(1, "L1 trace count got=%0d expected=%0d",
                trace_seen, OUTPUT_PIXELS);
        if (sparse_seen != expected_nonzero)
            $fatal(1, "L1 sparse count got=%0d expected=%0d",
                sparse_seen, expected_nonzero);
        if (accepted_event_count != 400)
            $fatal(1, "L1 accepted events got=%0d", accepted_event_count);
        if (emitted_packet_count != expected_nonzero)
            $fatal(1, "L1 emitted count got=%0d expected=%0d",
                emitted_packet_count, expected_nonzero);
        if (errors_seen != 0)
            $fatal(1, "L1 sparse golden comparison failed errors=%0d",
                errors_seen);
        $display("tb_pursuit_sparse_l1_event_layer PASS traces=%0d packets=%0d core_cycles=%0d wrapper_cycles=%0d",
            trace_seen, sparse_seen, core_cycle_count, cycle_count);
        $display("L1 state cycles idle=%0d clear=%0d wait=%0d update=%0d finalize=%0d drain=%0d",
            state_idle_cycles, state_clear_cycles, state_wait_cycles,
            state_update_cycles, state_finalize_cycles, state_drain_cycles);
        $finish;
    end
endmodule

`default_nettype wire
