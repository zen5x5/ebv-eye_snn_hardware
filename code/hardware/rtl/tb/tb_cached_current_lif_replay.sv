`timescale 1ns/1ps
`default_nettype none

module tb_cached_current_lif_replay;
    localparam integer VALUES = 4608;
    reg clk = 1'b0;
    always #2.5 clk = ~clk;
    reg rst = 1'b1;
    reg start = 1'b0;
    wire busy;
    wire done;
    wire parameter_read_enable;
    wire [8:0] parameter_read_address;
    reg [4:0] parameter_fraction_data;
    wire cache_read_enable;
    wire [13:0] cache_read_address;
    wire [5:0] cache_read_channel_base;
    reg [191:0] cache_read_current_flat;
    wire state_read_enable;
    wire [13:0] state_read_address;
    wire [5:0] state_read_channel_base;
    reg [127:0] state_read_data_flat;
    wire state_write_enable;
    wire [13:0] state_write_address;
    wire [5:0] state_write_channel_base;
    wire [127:0] state_write_data_flat;
    wire output_valid;
    wire [13:0] output_address;
    wire [5:0] output_channel_base;
    wire [191:0] output_current_flat;
    wire [63:0] output_spike_flat;
    wire [127:0] output_membrane_flat;
    wire output_range_error;

    reg [4:0] fraction_memory [0:319];
    reg signed [23:0] current_memory [0:VALUES-1];
    reg signed [15:0] state_memory [0:VALUES-1];
    reg signed [23:0] expected_current [0:VALUES-1];
    reg [7:0] expected_spike [0:VALUES-1];
    reg signed [15:0] expected_membrane [0:VALUES-1];
    string fraction_file;
    string current_file;
    string state_file;
    string expected_current_file;
    string expected_spike_file;
    string expected_membrane_file;

    cached_current_lif_replay dut (
        .clk(clk), .rst(rst), .start(start), .busy(busy), .done(done),
        .config_output_tiles(4'd3),
        .config_output_spatial(14'd192),
        .config_parameter_base(9'd24),
        .parameter_read_enable(parameter_read_enable),
        .parameter_read_address(parameter_read_address),
        .parameter_fraction_data(parameter_fraction_data),
        .cache_read_enable(cache_read_enable),
        .cache_read_address(cache_read_address),
        .cache_read_channel_base(cache_read_channel_base),
        .cache_read_current_flat(cache_read_current_flat),
        .state_read_enable(state_read_enable),
        .state_read_address(state_read_address),
        .state_read_channel_base(state_read_channel_base),
        .state_read_data_flat(state_read_data_flat),
        .state_write_enable(state_write_enable),
        .state_write_address(state_write_address),
        .state_write_channel_base(state_write_channel_base),
        .state_write_data_flat(state_write_data_flat),
        .output_valid(output_valid),
        .output_address(output_address),
        .output_channel_base(output_channel_base),
        .output_current_flat(output_current_flat),
        .output_spike_flat(output_spike_flat),
        .output_membrane_flat(output_membrane_flat),
        .output_range_error(output_range_error)
    );

    integer lane;
    integer index;
    always @(posedge clk) begin
        if (parameter_read_enable) begin
            parameter_fraction_data <= fraction_memory[parameter_read_address];
        end
        if (cache_read_enable) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                cache_read_current_flat[lane*24 +: 24]
                    <= current_memory[
                        (cache_read_channel_base + lane)*192
                        + cache_read_address
                    ];
            end
        end
        if (state_read_enable) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                state_read_data_flat[lane*16 +: 16]
                    <= state_memory[
                        (state_read_channel_base + lane)*192
                        + state_read_address
                    ];
            end
        end
        if (state_write_enable) begin
            for (lane = 0; lane < 8; lane = lane + 1) begin
                state_memory[
                    (state_write_channel_base + lane)*192
                    + state_write_address
                ] <= state_write_data_flat[lane*16 +: 16];
            end
        end
    end

    integer cycles;
    integer transactions;
    integer errors;
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (output_valid) begin
                transactions <= transactions + 1;
                if (output_range_error) errors <= errors + 1;
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    index = (output_channel_base + lane)*192 + output_address;
                    if ($signed(output_current_flat[lane*24 +: 24])
                            !== expected_current[index]) errors <= errors + 1;
                    if (output_spike_flat[lane*8 +: 8]
                            !== expected_spike[index]) errors <= errors + 1;
                    if ($signed(output_membrane_flat[lane*16 +: 16])
                            !== expected_membrane[index]) errors <= errors + 1;
                end
            end
            if (cycles > 5000) $fatal(1, "cached replay timeout");
        end
    end

    integer ok;
    initial begin
        ok = 1;
        ok &= $value$plusargs("FRACTION_FILE=%s", fraction_file);
        ok &= $value$plusargs("CURRENT_FILE=%s", current_file);
        ok &= $value$plusargs("STATE_FILE=%s", state_file);
        ok &= $value$plusargs("EXPECTED_CURRENT_FILE=%s", expected_current_file);
        ok &= $value$plusargs("EXPECTED_SPIKE_FILE=%s", expected_spike_file);
        ok &= $value$plusargs("EXPECTED_MEMBRANE_FILE=%s", expected_membrane_file);
        if (!ok) $fatal(1, "missing required replay plusarg");
        $readmemh(fraction_file, fraction_memory);
        $readmemh(current_file, current_memory);
        $readmemh(state_file, state_memory);
        $readmemh(expected_current_file, expected_current);
        $readmemh(expected_spike_file, expected_spike);
        $readmemh(expected_membrane_file, expected_membrane);
        parameter_fraction_data = 0;
        cache_read_current_flat = 0;
        state_read_data_flat = 0;
        cycles = 0;
        transactions = 0;
        errors = 0;
        repeat (5) @(posedge clk);
        rst <= 1'b0;
        repeat (2) @(posedge clk);
        start <= 1'b1;
        @(posedge clk);
        start <= 1'b0;
        wait(done);
        @(posedge clk);
        if (transactions != 576) $fatal(1, "wrong transaction count");
        if (errors != 0) $fatal(1, "cached replay mismatches=%0d", errors);
        $display("PASS: transactions=%0d values=13824 inference_cycles=%0d",
            transactions, cycles);
        $finish;
    end
endmodule

`default_nettype wire
