`timescale 1ns/1ps
`default_nettype none

module tb_l1_streaming_layer;
    localparam integer INPUT_WIDTH = 128;
    localparam integer INPUT_HEIGHT = 96;
    localparam integer INPUT_PIXELS = INPUT_WIDTH * INPUT_HEIGHT;
    localparam integer OUTPUT_WIDTH = INPUT_WIDTH / 2;
    localparam integer OUTPUT_HEIGHT = INPUT_HEIGHT / 2;
    localparam integer OUTPUT_PIXELS = OUTPUT_WIDTH * OUTPUT_HEIGHT;
    localparam integer OUTPUT_PES = 8;
    localparam integer WEIGHT_COUNT = OUTPUT_PES * 2 * 9;
    localparam integer COUNT_BITS = 9;
    localparam integer WEIGHT_BITS = 16;
    localparam integer CURRENT_BITS = 24;
    localparam integer STATE_BITS = 16;
    localparam integer SPIKE_BITS = 8;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;

    wire input_read_enable;
    wire [$clog2(INPUT_PIXELS)-1:0] input_read_address;
    reg [2*COUNT_BITS-1:0] input_read_data_flat = 0;

    reg weight_write_enable = 1'b0;
    reg [$clog2(WEIGHT_COUNT)-1:0] weight_write_address = 0;
    reg signed [WEIGHT_BITS-1:0] weight_write_data = 0;
    reg channel_write_enable = 1'b0;
    reg [$clog2(OUTPUT_PES)-1:0] channel_write_address = 0;
    reg signed [CURRENT_BITS-1:0] constant_write_data = 0;
    reg [4:0] fraction_bits_write_data = 0;

    wire output_valid;
    wire [$clog2(OUTPUT_PIXELS)-1:0] output_address;
    wire [OUTPUT_PES*CURRENT_BITS-1:0] output_current_flat;
    wire [OUTPUT_PES*SPIKE_BITS-1:0] output_spike_flat;
    wire [OUTPUT_PES*STATE_BITS-1:0] output_membrane_flat;
    wire output_range_error;

    reg [COUNT_BITS-1:0] input_memory [0:2*INPUT_PIXELS-1];
    reg [WEIGHT_BITS-1:0] weight_memory [0:WEIGHT_COUNT-1];
    reg [CURRENT_BITS-1:0] constant_memory [0:OUTPUT_PES-1];
    reg [7:0] weight_fraction_memory [0:OUTPUT_PES-1];
    reg [7:0] current_fraction_memory [0:OUTPUT_PES-1];
    reg [CURRENT_BITS-1:0] golden_current [0:OUTPUT_PES*OUTPUT_PIXELS-1];
    reg [SPIKE_BITS-1:0] golden_spike [0:OUTPUT_PES*OUTPUT_PIXELS-1];
    reg [STATE_BITS-1:0] golden_membrane [0:OUTPUT_PES*OUTPUT_PIXELS-1];

    string golden_directory;
    integer index;
    initial begin
        if (!$value$plusargs("GOLDEN_DIR=%s", golden_directory)) begin
            $fatal(1, "GOLDEN_DIR plusarg is required");
        end
        $readmemh(
            $sformatf("%s/mem/vector_input_event_count_integer.mem",
                golden_directory),
            input_memory
        );
        $readmemh(
            $sformatf("%s/mem/parameter_L1_weight_integer.mem",
                golden_directory),
            weight_memory
        );
        $readmemh(
            $sformatf("%s/mem/parameter_L1_constant_integer.mem",
                golden_directory),
            constant_memory
        );
        $readmemh(
            $sformatf("%s/mem/parameter_L1_weight_fraction_bits.mem",
                golden_directory),
            weight_fraction_memory
        );
        $readmemh(
            $sformatf("%s/mem/parameter_L1_current_and_state_fraction_bits.mem",
                golden_directory),
            current_fraction_memory
        );
        $readmemh(
            $sformatf("%s/mem/vector_t0_L1_current_integer.mem",
                golden_directory),
            golden_current
        );
        $readmemh(
            $sformatf("%s/mem/vector_t0_L1_spike_integer.mem",
                golden_directory),
            golden_spike
        );
        $readmemh(
            $sformatf("%s/mem/vector_t0_L1_membrane_integer.mem",
                golden_directory),
            golden_membrane
        );
        for (index = 0; index < OUTPUT_PES; index = index + 1) begin
            if (weight_fraction_memory[index]
                    !== current_fraction_memory[index]) begin
                $fatal(1,
                    "L1 product shift is not zero for output channel %0d",
                    index);
            end
        end
    end

    always @(posedge clk) begin
        if (input_read_enable) begin
            input_read_data_flat <= {
                input_memory[INPUT_PIXELS + input_read_address],
                input_memory[input_read_address]
            };
        end
    end

    l1_streaming_layer dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .busy(busy),
        .done(done),
        .input_read_enable(input_read_enable),
        .input_read_address(input_read_address),
        .input_read_data_flat(input_read_data_flat),
        .weight_write_enable(weight_write_enable),
        .weight_write_address(weight_write_address),
        .weight_write_data(weight_write_data),
        .channel_write_enable(channel_write_enable),
        .channel_write_address(channel_write_address),
        .constant_write_data(constant_write_data),
        .fraction_bits_write_data(fraction_bits_write_data),
        .output_valid(output_valid),
        .output_address(output_address),
        .output_current_flat(output_current_flat),
        .output_spike_flat(output_spike_flat),
        .output_membrane_flat(output_membrane_flat),
        .output_range_error(output_range_error)
    );

    integer outputs_seen = 0;
    integer errors_seen = 0;
    integer channel_index;
    integer expected_index;
    always @(posedge clk) begin
        if (output_valid) begin
            if (output_address !== outputs_seen[$clog2(OUTPUT_PIXELS)-1:0]) begin
                $display("address mismatch: got %0d expected %0d",
                    output_address, outputs_seen);
                errors_seen = errors_seen + 1;
            end
            if (output_range_error) begin
                $display("unexpected LIF range error at output %0d",
                    output_address);
                errors_seen = errors_seen + 1;
            end
            for (channel_index = 0; channel_index < OUTPUT_PES;
                    channel_index = channel_index + 1) begin
                expected_index = channel_index*OUTPUT_PIXELS + output_address;
                if (output_current_flat[
                        channel_index*CURRENT_BITS +: CURRENT_BITS
                    ] !== golden_current[expected_index]) begin
                    if (errors_seen < 20) begin
                        $display(
                            "current mismatch addr=%0d channel=%0d got=%0d expected=%0d",
                            output_address,
                            channel_index,
                            $signed(output_current_flat[
                                channel_index*CURRENT_BITS +: CURRENT_BITS
                            ]),
                            $signed(golden_current[expected_index])
                        );
                    end
                    errors_seen = errors_seen + 1;
                end
                if (output_spike_flat[
                        channel_index*SPIKE_BITS +: SPIKE_BITS
                    ] !== golden_spike[expected_index]) begin
                    if (errors_seen < 20) begin
                        $display(
                            "spike mismatch addr=%0d channel=%0d got=%0d expected=%0d",
                            output_address,
                            channel_index,
                            output_spike_flat[
                                channel_index*SPIKE_BITS +: SPIKE_BITS
                            ],
                            golden_spike[expected_index]
                        );
                    end
                    errors_seen = errors_seen + 1;
                end
                if (output_membrane_flat[
                        channel_index*STATE_BITS +: STATE_BITS
                    ] !== golden_membrane[expected_index]) begin
                    if (errors_seen < 20) begin
                        $display(
                            "membrane mismatch addr=%0d channel=%0d got=%0d expected=%0d",
                            output_address,
                            channel_index,
                            $signed(output_membrane_flat[
                                channel_index*STATE_BITS +: STATE_BITS
                            ]),
                            $signed(golden_membrane[expected_index])
                        );
                    end
                    errors_seen = errors_seen + 1;
                end
            end
            outputs_seen = outputs_seen + 1;
        end
    end

    integer cycle_count;
    initial begin
        repeat (5) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        for (index = 0; index < WEIGHT_COUNT; index = index + 1) begin
            @(negedge clk);
            weight_write_enable = 1'b1;
            weight_write_address = index;
            weight_write_data = weight_memory[index];
        end
        @(negedge clk);
        weight_write_enable = 1'b0;

        for (index = 0; index < OUTPUT_PES; index = index + 1) begin
            @(negedge clk);
            channel_write_enable = 1'b1;
            channel_write_address = index;
            constant_write_data = constant_memory[index];
            fraction_bits_write_data = current_fraction_memory[index][4:0];
        end
        @(negedge clk);
        channel_write_enable = 1'b0;

        @(negedge clk);
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        cycle_count = 0;
        while (!done && cycle_count < 20000) begin
            @(posedge clk);
            cycle_count = cycle_count + 1;
        end
        if (!done) begin
            $fatal(1, "timeout waiting for L1 completion");
        end
        @(posedge clk);
        if (outputs_seen != OUTPUT_PIXELS) begin
            $fatal(1, "output count mismatch: got %0d expected %0d",
                outputs_seen, OUTPUT_PIXELS);
        end
        if (errors_seen != 0) begin
            $fatal(1, "L1 golden comparison failed with %0d errors",
                errors_seen);
        end
        if (busy) begin
            $fatal(1, "busy remained asserted after done");
        end
        $display(
            "tb_l1_streaming_layer PASS: outputs=%0d inference_cycles=%0d",
            outputs_seen,
            cycle_count
        );
        $finish;
    end
endmodule

`default_nettype wire
