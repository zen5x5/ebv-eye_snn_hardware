`timescale 1ns/1ps
`default_nettype none

// Width extensions below are deliberate conversions from compact coordinate
// ports to integer address arithmetic.  Golden-vector tests cover every
// arithmetic output at the deployed dimensions.
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off UNUSEDSIGNAL */
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off DECLFILENAME */

// One physical single-write-port distributed-RAM bank.  Keeping the memory in
// its own instance makes the nine-bank structure explicit to synthesis; a
// flattened multi-write array cannot be legally inferred as LUT RAM.
module sparse_accumulator_vector_bank #(
    parameter integer DATA_BITS = 320,
    parameter integer DEPTH = 43,
    parameter integer ADDRESS_BITS = 6
) (
    input  wire                         clk,
    input  wire [ADDRESS_BITS-1:0]      address,
    output wire [DATA_BITS-1:0]         read_data,
    input  wire                         write_enable,
    input  wire [DATA_BITS-1:0]         write_data
);
    (* ram_style = "distributed" *) reg [DATA_BITS-1:0] memory [0:DEPTH-1];

    assign read_data = memory[address];

    always @(posedge clk) begin
        if (write_enable) begin
            memory[address] <= write_data;
        end
    end
endmodule

// One output-channel/tap-lane weight bank.  Parallel compute lanes each read
// a statically selected bank, avoiding replication of a flat multi-read-port
// weight array.  The independent host write address is a conventional 1R1W
// distributed-RAM interface; parameter loading and inference never overlap.
module sparse_weight_bank #(
    parameter integer DATA_BITS = 16,
    parameter integer DEPTH = 24,
    parameter integer ADDRESS_BITS = 5
) (
    input  wire                         clk,
    input  wire [ADDRESS_BITS-1:0]      read_address,
    output wire signed [DATA_BITS-1:0]  read_data,
    input  wire                         write_enable,
    input  wire [ADDRESS_BITS-1:0]      write_address,
    input  wire signed [DATA_BITS-1:0]  write_data
);
    (* ram_style = "distributed" *) reg signed [DATA_BITS-1:0]
        memory [0:DEPTH-1];

    assign read_data = memory[read_address];

    always @(posedge clk) begin
        if (write_enable) begin
            memory[write_address] <= write_data;
        end
    end
endmodule

// Event-driven same-pad 3x3 convolution followed by 2x2 max-pool and a
// stateless MultiSpike neuron.  Input packets must be ordered by nondecreasing
// row.  Packets inside one row may use any column/channel order.
//
// Only three convolution rows are live.  Even and odd columns are separate
// banks, so a complete horizontal pool pair is read in one cycle.  A row is
// cleared while it is finalized; no full feature-map clear or epoch tag is
// needed.  Pooling is performed on the S40 dot product.  The folded S24
// constant is added once after pooling, followed by the S24 saturation.
//
// UNIT_INPUT=1 implements a unit-event add-only datapath.  UNIT_INPUT=0 uses
// one unsigned-count x signed-weight multiplier per output channel.
module sparse_conv3x3_pool_lif_core #(
    parameter integer INPUT_CHANNELS = 2,
    parameter integer OUTPUT_CHANNELS = 8,
    parameter integer INPUT_WIDTH = 128,
    parameter integer INPUT_HEIGHT = 96,
    parameter integer COUNT_BITS = 1,
    parameter integer UNIT_INPUT = 1,
    // Legal values divide the nine kernel taps.  L1 uses 9 (72 parallel
    // add lanes); L2 uses 3 (48 parallel count-times-weight lanes).
    parameter integer KERNEL_TAPS_PER_CYCLE = 9,
    parameter integer WEIGHT_BITS = 16,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer FRACTION_BITS_WIDTH = 5
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output reg                                       busy,
    output reg                                       done,

    input  wire                                      input_valid,
    output wire                                      input_ready,
    input  wire [((INPUT_CHANNELS <= 1) ? 1 : $clog2(INPUT_CHANNELS))-1:0]
                                                    input_channel,
    input  wire [((INPUT_HEIGHT <= 1) ? 1 : $clog2(INPUT_HEIGHT))-1:0]
                                                    input_row,
    input  wire [((INPUT_WIDTH <= 1) ? 1 : $clog2(INPUT_WIDTH))-1:0]
                                                    input_column,
    input  wire [COUNT_BITS-1:0]                    input_count,
    // May pulse in any busy cycle after the final packet has been accepted.
    // It is latched internally, which also permits an empty packet stream.
    input  wire                                      input_end,

    input  wire                                      weight_write_enable,
    input  wire [((OUTPUT_CHANNELS <= 1) ? 1 : $clog2(OUTPUT_CHANNELS))-1:0]
                                                    weight_write_output_channel,
    input  wire [((INPUT_CHANNELS <= 1) ? 1 : $clog2(INPUT_CHANNELS))-1:0]
                                                    weight_write_input_channel,
    input  wire [3:0]                               weight_write_kernel_index,
    input  wire signed [WEIGHT_BITS-1:0]            weight_write_data,
    input  wire                                      channel_write_enable,
    input  wire [((OUTPUT_CHANNELS <= 1) ? 1 : $clog2(OUTPUT_CHANNELS))-1:0]
                                                    channel_write_channel,
    input  wire signed [CURRENT_BITS-1:0]           constant_write_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]           fraction_bits_write_data,

    output wire                                      output_valid,
    input  wire                                      output_ready,
    output wire [$clog2((INPUT_WIDTH/2)*(INPUT_HEIGHT/2))-1:0]
                                                    output_address,
    output wire [OUTPUT_CHANNELS*CURRENT_BITS-1:0]  output_current_flat,
    output wire [OUTPUT_CHANNELS*SPIKE_BITS-1:0]    output_spike_flat,
    output wire [OUTPUT_CHANNELS*STATE_BITS-1:0]    output_membrane_flat,
    output wire                                      output_range_error,

    output reg                                       protocol_error,
    output reg  [31:0]                              cycle_count,
    output reg  [31:0]                              accepted_packet_count
);
    localparam integer HALF_WIDTH = INPUT_WIDTH / 2;
    localparam integer HALF_HEIGHT = INPUT_HEIGHT / 2;
    localparam integer OUTPUT_PIXELS = HALF_WIDTH * HALF_HEIGHT;
    localparam integer INPUT_CHANNEL_BITS =
        (INPUT_CHANNELS <= 1) ? 1 : $clog2(INPUT_CHANNELS);
    localparam integer OUTPUT_CHANNEL_BITS =
        (OUTPUT_CHANNELS <= 1) ? 1 : $clog2(OUTPUT_CHANNELS);
    localparam integer INPUT_ROW_BITS =
        (INPUT_HEIGHT <= 1) ? 1 : $clog2(INPUT_HEIGHT);
    localparam integer INPUT_COLUMN_BITS =
        (INPUT_WIDTH <= 1) ? 1 : $clog2(INPUT_WIDTH);
    localparam integer CURSOR_BITS = $clog2(INPUT_HEIGHT + 1);
    localparam integer PAIR_COLUMN_BITS =
        (HALF_WIDTH <= 1) ? 1 : $clog2(HALF_WIDTH);
    localparam integer OUTPUT_ADDRESS_BITS = $clog2(OUTPUT_PIXELS);
    localparam integer VECTOR_ACCUMULATOR_BITS =
        OUTPUT_CHANNELS * ACCUMULATOR_BITS;
    // The three live rows and column modulo three form nine independent
    // banks.  All taps of one event therefore update distinct banks.
    localparam integer COLUMN_BANK_DEPTH = (INPUT_WIDTH + 2) / 3;
    localparam integer ACCUMULATOR_BANKS = 9;
    localparam integer COLUMN_BANK_ADDRESS_BITS =
        (COLUMN_BANK_DEPTH <= 1) ? 1 : $clog2(COLUMN_BANK_DEPTH);
    // One bank is assigned to every tap processed in parallel.  Its address
    // stores input_channel-major groups of taps.  Thus L1 has 8x9 depth-2
    // banks and L2 has 16x3 depth-24 banks.
    localparam integer WEIGHT_BANKS_PER_OUTPUT = KERNEL_TAPS_PER_CYCLE;
    localparam integer WEIGHT_GROUPS = 9 / WEIGHT_BANKS_PER_OUTPUT;
    localparam integer WEIGHT_BANK_DEPTH = INPUT_CHANNELS * WEIGHT_GROUPS;
    localparam integer WEIGHT_BANK_ADDRESS_BITS =
        (WEIGHT_BANK_DEPTH <= 1) ? 1 : $clog2(WEIGHT_BANK_DEPTH);
    localparam integer WEIGHT_TOTAL_BANKS =
        OUTPUT_CHANNELS * WEIGHT_BANKS_PER_OUTPUT;
    localparam integer PRODUCT_BITS = WEIGHT_BITS + COUNT_BITS + 1;
    localparam integer NEURON_WORK_BITS = CURRENT_BITS + 2;
    localparam integer NEURON_RESULT_BITS =
        1 + STATE_BITS + SPIKE_BITS + CURRENT_BITS;

    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_CLEAR = 3'd1;
    localparam [2:0] STATE_WAIT_PACKET = 3'd2;
    localparam [2:0] STATE_UPDATE = 3'd3;
    localparam [2:0] STATE_FINALIZE = 3'd4;
    localparam [2:0] STATE_FINALIZE_DRAIN = 3'd5;

    localparam signed [ACCUMULATOR_BITS-1:0] CURRENT_MAXIMUM = {
        {(ACCUMULATOR_BITS-CURRENT_BITS){1'b0}},
        1'b0,
        {(CURRENT_BITS-1){1'b1}}
    };
    localparam signed [ACCUMULATOR_BITS-1:0] CURRENT_MINIMUM = {
        {(ACCUMULATOR_BITS-CURRENT_BITS){1'b1}},
        1'b1,
        {(CURRENT_BITS-1){1'b0}}
    };
    localparam signed [NEURON_WORK_BITS-1:0] STATE_MINIMUM =
        -(1 <<< (STATE_BITS-1));
    localparam signed [NEURON_WORK_BITS-1:0] STATE_MAXIMUM =
        (1 <<< (STATE_BITS-1)) - 1;
    localparam [SPIKE_BITS-1:0] SPIKE_MAXIMUM = {SPIKE_BITS{1'b1}};
    localparam [FRACTION_BITS_WIDTH-1:0] MAX_FRACTION_BITS =
        FRACTION_BITS_WIDTH'(STATE_BITS - 1);

    // Fixed-width divide-by-three used only when an input packet is accepted.
    // The restoring network has INPUT_COLUMN_BITS stages of a three-bit
    // compare/subtract; it does not infer a word-sized divider or modulo unit.
    function automatic [INPUT_COLUMN_BITS+1:0] column_divmod_three;
        input [INPUT_COLUMN_BITS-1:0] value;
        integer bit_index;
        reg [INPUT_COLUMN_BITS-1:0] quotient;
        reg [1:0] remainder;
        reg [2:0] trial;
        begin
            quotient = {INPUT_COLUMN_BITS{1'b0}};
            remainder = 2'd0;
            trial = 3'd0;
            for (bit_index = INPUT_COLUMN_BITS-1;
                    bit_index >= 0; bit_index = bit_index - 1) begin
                trial = {remainder, value[bit_index]};
                if (trial >= 3'd3) begin
                    quotient[bit_index] = 1'b1;
                    remainder = trial - 3'd3;
                end else begin
                    quotient[bit_index] = 1'b0;
                    remainder = trial[1:0];
                end
            end
            column_divmod_three = {quotient, remainder};
        end
    endfunction

    function automatic [1:0] increment_modulo_three;
        input [1:0] value;
        begin
            case (value)
                2'd0: increment_modulo_three = 2'd1;
                2'd1: increment_modulo_three = 2'd2;
                default: increment_modulo_three = 2'd0;
            endcase
        end
    endfunction

    function automatic [1:0] decrement_modulo_three;
        input [1:0] value;
        begin
            case (value)
                2'd0: decrement_modulo_three = 2'd2;
                2'd1: decrement_modulo_three = 2'd0;
                default: decrement_modulo_three = 2'd1;
            endcase
        end
    endfunction

    // For a physical bank remainder b, solve
    //   b = packet_remainder + 1 - kernel_component (mod 3).
    function automatic [1:0] inverse_kernel_component;
        input [1:0] packet_remainder;
        input [1:0] bank_remainder;
        begin
            case ({packet_remainder, bank_remainder})
                4'b0000: inverse_kernel_component = 2'd1;
                4'b0001: inverse_kernel_component = 2'd0;
                4'b0010: inverse_kernel_component = 2'd2;
                4'b0100: inverse_kernel_component = 2'd2;
                4'b0101: inverse_kernel_component = 2'd1;
                4'b0110: inverse_kernel_component = 2'd0;
                4'b1000: inverse_kernel_component = 2'd0;
                4'b1001: inverse_kernel_component = 2'd2;
                4'b1010: inverse_kernel_component = 2'd1;
                default: inverse_kernel_component = 2'd0;
            endcase
        end
    endfunction

    function automatic [1:0] tap_row_component;
        input [3:0] tap_index;
        begin
            case (tap_index)
                4'd0, 4'd1, 4'd2: tap_row_component = 2'd0;
                4'd3, 4'd4, 4'd5: tap_row_component = 2'd1;
                default: tap_row_component = 2'd2;
            endcase
        end
    endfunction

    function automatic [1:0] tap_column_component;
        input [3:0] tap_index;
        begin
            case (tap_index)
                4'd0, 4'd3, 4'd6: tap_column_component = 2'd0;
                4'd1, 4'd4, 4'd7: tap_column_component = 2'd1;
                default: tap_column_component = 2'd2;
            endcase
        end
    endfunction

    function automatic [3:0] three_by_three_index;
        input [1:0] row_component;
        input [1:0] column_component;
        begin
            case (row_component)
                2'd0: three_by_three_index = {2'b00, column_component};
                2'd1: three_by_three_index = 4'd3 + column_component;
                default: three_by_three_index = 4'd6 + column_component;
            endcase
        end
    endfunction

    reg [2:0] control_state;
    reg [COLUMN_BANK_ADDRESS_BITS-1:0] clear_index;
    reg [CURSOR_BITS-1:0] input_row_cursor;
    reg [1:0] input_row_cursor_modulo_three;
    reg end_seen;

    reg [INPUT_CHANNEL_BITS-1:0] packet_channel;
    reg [INPUT_ROW_BITS-1:0] packet_row;
    reg [INPUT_COLUMN_BITS-1:0] packet_column;
    reg [COLUMN_BANK_ADDRESS_BITS-1:0] packet_column_quotient;
    reg [1:0] packet_column_remainder;
    reg [1:0] packet_row_remainder;
    reg [COUNT_BITS-1:0] packet_count;
    reg [3:0] tap_base;

    reg [INPUT_ROW_BITS-1:0] finalize_output_row;
    reg [1:0] finalize_output_row_remainder;
    reg [PAIR_COLUMN_BITS-1:0] finalize_pair_column;
    reg [1:0] finalize_even_column_bank;
    reg [1:0] finalize_odd_column_bank;
    reg [COLUMN_BANK_ADDRESS_BITS-1:0] finalize_even_column_index;
    reg [COLUMN_BANK_ADDRESS_BITS-1:0] finalize_odd_column_index;
    reg finalize_is_last_row;

    reg output_valid_register;
    reg [OUTPUT_ADDRESS_BITS-1:0] output_address_register;
    reg [OUTPUT_ADDRESS_BITS-1:0] next_output_address;
    reg [OUTPUT_CHANNELS*CURRENT_BITS-1:0]
        output_current_register;
    reg [OUTPUT_CHANNELS*SPIKE_BITS-1:0]
        output_spike_register;
    reg [OUTPUT_CHANNELS*STATE_BITS-1:0]
        output_membrane_register;
    reg output_range_error_register;

    reg signed [CURRENT_BITS-1:0]
        constant_registers [0:OUTPUT_CHANNELS-1];
    reg [FRACTION_BITS_WIDTH-1:0]
        fraction_bits_registers [0:OUTPUT_CHANNELS-1];

    // Three live rows x three column-modulo banks.  Each instance has exactly
    // one asynchronous read address and one synchronous write port.  The tap
    // grouping guarantees at most one update per physical bank per cycle.
    reg [COLUMN_BANK_ADDRESS_BITS-1:0]
        accumulator_bank_address [0:ACCUMULATOR_BANKS-1];
    wire [VECTOR_ACCUMULATOR_BITS-1:0]
        accumulator_bank_read_data [0:ACCUMULATOR_BANKS-1];
    reg accumulator_bank_write_enable [0:ACCUMULATOR_BANKS-1];
    reg [VECTOR_ACCUMULATOR_BITS-1:0]
        accumulator_bank_write_data [0:ACCUMULATOR_BANKS-1];

    genvar accumulator_bank_generate_index;
    generate
        for (accumulator_bank_generate_index = 0;
                accumulator_bank_generate_index < ACCUMULATOR_BANKS;
                accumulator_bank_generate_index =
                    accumulator_bank_generate_index + 1) begin : banks
            sparse_accumulator_vector_bank #(
                .DATA_BITS(VECTOR_ACCUMULATOR_BITS),
                .DEPTH(COLUMN_BANK_DEPTH),
                .ADDRESS_BITS(COLUMN_BANK_ADDRESS_BITS)
            ) bank (
                .clk(clk),
                .address(accumulator_bank_address[
                    accumulator_bank_generate_index
                ]),
                .read_data(accumulator_bank_read_data[
                    accumulator_bank_generate_index
                ]),
                .write_enable(accumulator_bank_write_enable[
                    accumulator_bank_generate_index
                ]),
                .write_data(accumulator_bank_write_data[
                    accumulator_bank_generate_index
                ])
            );
        end
    endgenerate

    (* ram_style = "distributed" *) reg [VECTOR_ACCUMULATOR_BITS-1:0]
        horizontal_pool_memory [0:HALF_WIDTH-1];

    reg [3:0] weight_write_bank_selector;
    reg [3:0] weight_write_group;
    reg [WEIGHT_BANK_ADDRESS_BITS-1:0] weight_write_bank_address;
    reg [3:0] weight_read_group;
    reg [WEIGHT_BANK_ADDRESS_BITS-1:0] weight_read_bank_address;
    wire signed [WEIGHT_BITS-1:0]
        weight_bank_read_data [0:WEIGHT_TOTAL_BANKS-1];

    // Decode the host's output-major/input-major/kernel-major interface into
    // the physical tap banks without synthesizing a variable divider.
    always_comb begin
        weight_write_bank_selector = 4'd0;
        weight_write_group = 4'd0;
        weight_write_bank_address = {WEIGHT_BANK_ADDRESS_BITS{1'b0}};
        if (WEIGHT_BANKS_PER_OUTPUT == 9) begin
            weight_write_bank_selector = weight_write_kernel_index;
            weight_write_bank_address = weight_write_input_channel;
        end else if (WEIGHT_BANKS_PER_OUTPUT == 3) begin
            case (weight_write_kernel_index)
                4'd0: begin
                    weight_write_bank_selector = 4'd0;
                    weight_write_group = 4'd0;
                end
                4'd1: begin
                    weight_write_bank_selector = 4'd1;
                    weight_write_group = 4'd0;
                end
                4'd2: begin
                    weight_write_bank_selector = 4'd2;
                    weight_write_group = 4'd0;
                end
                4'd3: begin
                    weight_write_bank_selector = 4'd0;
                    weight_write_group = 4'd1;
                end
                4'd4: begin
                    weight_write_bank_selector = 4'd1;
                    weight_write_group = 4'd1;
                end
                4'd5: begin
                    weight_write_bank_selector = 4'd2;
                    weight_write_group = 4'd1;
                end
                4'd6: begin
                    weight_write_bank_selector = 4'd0;
                    weight_write_group = 4'd2;
                end
                4'd7: begin
                    weight_write_bank_selector = 4'd1;
                    weight_write_group = 4'd2;
                end
                default: begin
                    weight_write_bank_selector = 4'd2;
                    weight_write_group = 4'd2;
                end
            endcase
            weight_write_bank_address =
                ($unsigned(weight_write_input_channel) << 1)
                + $unsigned(weight_write_input_channel)
                + weight_write_group;
        end else begin
            weight_write_bank_address =
                ($unsigned(weight_write_input_channel) << 3)
                + $unsigned(weight_write_input_channel)
                + weight_write_kernel_index;
        end
    end

    // All parallel lanes read the same input-channel/tap-group address; lane
    // j is hard-wired to bank j for every output channel.
    always_comb begin
        weight_read_group = 4'd0;
        weight_read_bank_address = {WEIGHT_BANK_ADDRESS_BITS{1'b0}};
        if (WEIGHT_BANKS_PER_OUTPUT == 9) begin
            weight_read_bank_address = packet_channel;
        end else if (WEIGHT_BANKS_PER_OUTPUT == 3) begin
            case (tap_base)
                4'd0: weight_read_group = 4'd0;
                4'd3: weight_read_group = 4'd1;
                default: weight_read_group = 4'd2;
            endcase
            weight_read_bank_address =
                ($unsigned(packet_channel) << 1)
                + $unsigned(packet_channel)
                + weight_read_group;
        end else begin
            weight_read_bank_address =
                ($unsigned(packet_channel) << 3)
                + $unsigned(packet_channel)
                + tap_base;
        end
    end

    genvar weight_output_generate_index;
    genvar weight_bank_generate_index;
    generate
        for (weight_output_generate_index = 0;
                weight_output_generate_index < OUTPUT_CHANNELS;
                weight_output_generate_index =
                    weight_output_generate_index + 1) begin : weight_outputs
            for (weight_bank_generate_index = 0;
                    weight_bank_generate_index < WEIGHT_BANKS_PER_OUTPUT;
                    weight_bank_generate_index =
                        weight_bank_generate_index + 1) begin : weight_banks
                localparam integer WEIGHT_FLAT_BANK =
                    weight_output_generate_index*WEIGHT_BANKS_PER_OUTPUT
                    + weight_bank_generate_index;
                wire weight_bank_write_enable = weight_write_enable
                    && weight_write_kernel_index < 9
                    && weight_write_output_channel < OUTPUT_CHANNELS
                    && weight_write_input_channel < INPUT_CHANNELS
                    && weight_write_output_channel
                        == weight_output_generate_index
                    && weight_write_bank_selector
                        == weight_bank_generate_index;

                sparse_weight_bank #(
                    .DATA_BITS(WEIGHT_BITS),
                    .DEPTH(WEIGHT_BANK_DEPTH),
                    .ADDRESS_BITS(WEIGHT_BANK_ADDRESS_BITS)
                ) bank (
                    .clk(clk),
                    .read_address(weight_read_bank_address),
                    .read_data(weight_bank_read_data[WEIGHT_FLAT_BANK]),
                    .write_enable(weight_bank_write_enable),
                    .write_address(weight_write_bank_address),
                    .write_data(weight_write_data)
                );
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (channel_write_enable
                && channel_write_channel < OUTPUT_CHANNELS) begin
            constant_registers[channel_write_channel]
                <= constant_write_data;
            fraction_bits_registers[channel_write_channel]
                <= fraction_bits_write_data;
        end
    end

    wire final_update_cycle = control_state == STATE_UPDATE
        && (tap_base + KERNEL_TAPS_PER_CYCLE >= 9);
    assign input_ready = busy
        && (control_state == STATE_WAIT_PACKET || final_update_cycle)
        && !end_seen && !input_end
        && input_row == input_row_cursor;

    assign output_valid = output_valid_register;
    assign output_address = output_address_register;
    assign output_current_flat = output_current_register;
    assign output_spike_flat = output_spike_register;
    assign output_membrane_flat = output_membrane_register;
    assign output_range_error = output_range_error_register;

    wire [INPUT_COLUMN_BITS+1:0] input_column_divmod_result =
        column_divmod_three(input_column);
    wire [INPUT_COLUMN_BITS-1:0] input_column_quotient =
        input_column_divmod_result[INPUT_COLUMN_BITS+1:2];
    wire [1:0] input_column_remainder =
        input_column_divmod_result[1:0];
    wire [1:0] current_tap_row = tap_row_component(tap_base);
    wire [1:0] current_tap_column = tap_column_component(tap_base);

    // Bank-centric scatter update.  For physical bank (br,bc), solve the
    // unique kernel tap that lands there.  Every accumulator bank reads and
    // writes only its own S40 vector; the only crossbar selects an S16 weight.
    wire update_bank_active [0:ACCUMULATOR_BANKS-1];
    wire [COLUMN_BANK_ADDRESS_BITS-1:0]
        update_bank_column_index [0:ACCUMULATOR_BANKS-1];
    wire [1:0] update_bank_kernel_row [0:ACCUMULATOR_BANKS-1];
    wire [1:0] update_bank_kernel_column [0:ACCUMULATOR_BANKS-1];
    wire [3:0] update_bank_weight_lane [0:ACCUMULATOR_BANKS-1];
    wire [VECTOR_ACCUMULATOR_BITS-1:0]
        update_contribution_word [0:ACCUMULATOR_BANKS-1];
    reg [VECTOR_ACCUMULATOR_BITS-1:0]
        update_new_bank_word [0:ACCUMULATOR_BANKS-1];

    genvar update_bank_row_generate_index;
    genvar update_bank_column_generate_index;
    generate
        for (update_bank_row_generate_index = 0;
                update_bank_row_generate_index < 3;
                update_bank_row_generate_index =
                    update_bank_row_generate_index + 1) begin : update_rows
            for (update_bank_column_generate_index = 0;
                    update_bank_column_generate_index < 3;
                    update_bank_column_generate_index =
                        update_bank_column_generate_index + 1) begin : update_columns
                localparam integer UPDATE_FLAT_BANK =
                    update_bank_row_generate_index*3
                    + update_bank_column_generate_index;
                wire [1:0] kernel_row = inverse_kernel_component(
                    packet_row_remainder,
                    2'(update_bank_row_generate_index)
                );
                wire [1:0] kernel_column = inverse_kernel_component(
                    packet_column_remainder,
                    2'(update_bank_column_generate_index)
                );
                wire row_in_bounds = (kernel_row == 2'd0)
                    ? {1'b0, packet_row}
                        < (INPUT_ROW_BITS+1)'(INPUT_HEIGHT-1)
                    : ((kernel_row == 2'd1)
                        ? {1'b0, packet_row}
                            < (INPUT_ROW_BITS+1)'(INPUT_HEIGHT)
                        : (packet_row != {INPUT_ROW_BITS{1'b0}}
                            && {1'b0, packet_row}
                                < (INPUT_ROW_BITS+1)'(INPUT_HEIGHT)));
                wire column_in_bounds = (kernel_column == 2'd0)
                    ? {1'b0, packet_column}
                        < (INPUT_COLUMN_BITS+1)'(INPUT_WIDTH-1)
                    : ((kernel_column == 2'd1)
                        ? {1'b0, packet_column}
                            < (INPUT_COLUMN_BITS+1)'(INPUT_WIDTH)
                        : (packet_column != {INPUT_COLUMN_BITS{1'b0}}
                            && {1'b0, packet_column}
                                < (INPUT_COLUMN_BITS+1)'(INPUT_WIDTH)));
                wire tap_selected = (WEIGHT_BANKS_PER_OUTPUT == 9)
                    ? 1'b1
                    : ((WEIGHT_BANKS_PER_OUTPUT == 3)
                        ? kernel_row == current_tap_row
                        : (kernel_row == current_tap_row
                            && kernel_column == current_tap_column));

                assign update_bank_kernel_row[UPDATE_FLAT_BANK] = kernel_row;
                assign update_bank_kernel_column[UPDATE_FLAT_BANK] =
                    kernel_column;
                assign update_bank_active[UPDATE_FLAT_BANK] =
                    row_in_bounds && column_in_bounds && tap_selected;
                assign update_bank_column_index[UPDATE_FLAT_BANK] =
                    (kernel_column == 2'd0)
                    ? packet_column_quotient
                        + ((packet_column_remainder == 2'd2) ? 1'b1 : 1'b0)
                    : ((kernel_column == 2'd1)
                        ? packet_column_quotient
                        : ((packet_column_remainder == 2'd0
                                && packet_column
                                    != {INPUT_COLUMN_BITS{1'b0}})
                            ? packet_column_quotient - 1'b1
                            : packet_column_quotient));
                assign update_bank_weight_lane[UPDATE_FLAT_BANK] =
                    (WEIGHT_BANKS_PER_OUTPUT == 9)
                    ? three_by_three_index(kernel_row, kernel_column)
                    : ((WEIGHT_BANKS_PER_OUTPUT == 3)
                        ? {2'b00, kernel_column} : 4'd0);
            end
        end
    endgenerate

    // The deployed L1 instance evaluates nine add-only tap words.  The L2
    // instance evaluates three count-times-weight words (48 multipliers) and
    // reuses each column word across the three row banks, only one of which
    // is write-enabled in a given tap group.
    generate
        if (WEIGHT_BANKS_PER_OUTPUT == 9) begin : nine_tap_contributions
            reg [VECTOR_ACCUMULATOR_BITS-1:0]
                contribution_register [0:ACCUMULATOR_BANKS-1];
            integer contribution_bank_index;
            integer contribution_channel_index;
            reg signed [WEIGHT_BITS-1:0] contribution_weight;
            reg signed [PRODUCT_BITS-1:0] contribution_product;
            reg signed [ACCUMULATOR_BITS-1:0] contribution_value;

            always @* begin
                contribution_weight = {WEIGHT_BITS{1'b0}};
                contribution_product = {PRODUCT_BITS{1'b0}};
                contribution_value = {ACCUMULATOR_BITS{1'b0}};
                for (contribution_bank_index = 0;
                        contribution_bank_index < ACCUMULATOR_BANKS;
                        contribution_bank_index =
                            contribution_bank_index + 1) begin
                    contribution_register[contribution_bank_index] =
                        {VECTOR_ACCUMULATOR_BITS{1'b0}};
                    for (contribution_channel_index = 0;
                            contribution_channel_index < OUTPUT_CHANNELS;
                            contribution_channel_index =
                                contribution_channel_index + 1) begin
                        case (update_bank_weight_lane[
                                contribution_bank_index
                            ])
                            4'd0: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 0]);
                            4'd1: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 1]);
                            4'd2: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 2]);
                            4'd3: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 3]);
                            4'd4: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 4]);
                            4'd5: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 5]);
                            4'd6: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 6]);
                            4'd7: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 7]);
                            default: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*9 + 8]);
                        endcase
                        if (UNIT_INPUT != 0) begin
                            contribution_product = {PRODUCT_BITS{1'b0}};
                            contribution_value = {
                                {(ACCUMULATOR_BITS-WEIGHT_BITS){
                                    contribution_weight[WEIGHT_BITS-1]
                                }},
                                contribution_weight
                            };
                        end else begin
                            contribution_product =
                                $signed({1'b0, packet_count})
                                * $signed(contribution_weight);
                            contribution_value = {
                                {(ACCUMULATOR_BITS-PRODUCT_BITS){
                                    contribution_product[PRODUCT_BITS-1]
                                }},
                                contribution_product
                            };
                        end
                        contribution_register[contribution_bank_index][
                            contribution_channel_index*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ] = contribution_value;
                    end
                end
            end

            for (update_bank_row_generate_index = 0;
                    update_bank_row_generate_index < ACCUMULATOR_BANKS;
                    update_bank_row_generate_index =
                        update_bank_row_generate_index + 1) begin : connect
                assign update_contribution_word[
                    update_bank_row_generate_index
                ] = contribution_register[update_bank_row_generate_index];
            end
        end else if (WEIGHT_BANKS_PER_OUTPUT == 3) begin : three_tap_contributions
            reg [VECTOR_ACCUMULATOR_BITS-1:0]
                column_contribution_register [0:2];
            integer contribution_column_index;
            integer contribution_channel_index;
            reg signed [WEIGHT_BITS-1:0] contribution_weight;
            reg signed [PRODUCT_BITS-1:0] contribution_product;
            reg signed [ACCUMULATOR_BITS-1:0] contribution_value;

            always @* begin
                contribution_weight = {WEIGHT_BITS{1'b0}};
                contribution_product = {PRODUCT_BITS{1'b0}};
                contribution_value = {ACCUMULATOR_BITS{1'b0}};
                for (contribution_column_index = 0;
                        contribution_column_index < 3;
                        contribution_column_index =
                            contribution_column_index + 1) begin
                    column_contribution_register[
                        contribution_column_index
                    ] = {VECTOR_ACCUMULATOR_BITS{1'b0}};
                    for (contribution_channel_index = 0;
                            contribution_channel_index < OUTPUT_CHANNELS;
                            contribution_channel_index =
                                contribution_channel_index + 1) begin
                        case (update_bank_kernel_column[
                                contribution_column_index
                            ])
                            2'd0: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*3 + 0]);
                            2'd1: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*3 + 1]);
                            default: contribution_weight = $signed(
                                weight_bank_read_data[
                                    contribution_channel_index*3 + 2]);
                        endcase
                        if (UNIT_INPUT != 0) begin
                            contribution_product = {PRODUCT_BITS{1'b0}};
                            contribution_value = {
                                {(ACCUMULATOR_BITS-WEIGHT_BITS){
                                    contribution_weight[WEIGHT_BITS-1]
                                }},
                                contribution_weight
                            };
                        end else begin
                            contribution_product =
                                $signed({1'b0, packet_count})
                                * $signed(contribution_weight);
                            contribution_value = {
                                {(ACCUMULATOR_BITS-PRODUCT_BITS){
                                    contribution_product[PRODUCT_BITS-1]
                                }},
                                contribution_product
                            };
                        end
                        column_contribution_register[
                            contribution_column_index
                        ][
                            contribution_channel_index*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ] = contribution_value;
                    end
                end
            end

            assign update_contribution_word[0] = column_contribution_register[0];
            assign update_contribution_word[1] = column_contribution_register[1];
            assign update_contribution_word[2] = column_contribution_register[2];
            assign update_contribution_word[3] = column_contribution_register[0];
            assign update_contribution_word[4] = column_contribution_register[1];
            assign update_contribution_word[5] = column_contribution_register[2];
            assign update_contribution_word[6] = column_contribution_register[0];
            assign update_contribution_word[7] = column_contribution_register[1];
            assign update_contribution_word[8] = column_contribution_register[2];
        end else begin : one_tap_contribution
            reg [VECTOR_ACCUMULATOR_BITS-1:0] contribution_register;
            integer contribution_channel_index;
            reg signed [WEIGHT_BITS-1:0] contribution_weight;
            reg signed [PRODUCT_BITS-1:0] contribution_product;
            reg signed [ACCUMULATOR_BITS-1:0] contribution_value;

            always @* begin
                contribution_register = {VECTOR_ACCUMULATOR_BITS{1'b0}};
                contribution_weight = {WEIGHT_BITS{1'b0}};
                contribution_product = {PRODUCT_BITS{1'b0}};
                contribution_value = {ACCUMULATOR_BITS{1'b0}};
                for (contribution_channel_index = 0;
                        contribution_channel_index < OUTPUT_CHANNELS;
                        contribution_channel_index =
                            contribution_channel_index + 1) begin
                    contribution_weight = $signed(weight_bank_read_data[
                        contribution_channel_index]);
                    if (UNIT_INPUT != 0) begin
                        contribution_product = {PRODUCT_BITS{1'b0}};
                        contribution_value = {
                            {(ACCUMULATOR_BITS-WEIGHT_BITS){
                                contribution_weight[WEIGHT_BITS-1]
                            }}, contribution_weight
                        };
                    end else begin
                        contribution_product =
                            $signed({1'b0, packet_count})
                            * $signed(contribution_weight);
                        contribution_value = {
                            {(ACCUMULATOR_BITS-PRODUCT_BITS){
                                contribution_product[PRODUCT_BITS-1]
                            }}, contribution_product
                        };
                    end
                    contribution_register[
                        contribution_channel_index*ACCUMULATOR_BITS
                        +: ACCUMULATOR_BITS
                    ] = contribution_value;
                end
            end

            for (update_bank_row_generate_index = 0;
                    update_bank_row_generate_index < ACCUMULATOR_BANKS;
                    update_bank_row_generate_index =
                        update_bank_row_generate_index + 1) begin : connect
                assign update_contribution_word[
                    update_bank_row_generate_index
                ] = contribution_register;
            end
        end
    endgenerate

    // L2 activates exactly one physical row of three column banks per tap
    // group.  Select that row first and share one 16-channel S40 adder vector
    // per column.  The generated branch is constant per instance, so L1 keeps
    // its nine independent 8-channel add-only vectors without a wide mux.
    generate
        if (WEIGHT_BANKS_PER_OUTPUT == 3) begin : shared_l2_update_adders
            wire [1:0] active_bank_row = inverse_kernel_component(
                packet_row_remainder,
                current_tap_row
            );
            reg [VECTOR_ACCUMULATOR_BITS-1:0]
                selected_old_word [0:2];
            reg [VECTOR_ACCUMULATOR_BITS-1:0]
                updated_column_word [0:2];
            integer update_column_index;
            integer update_channel_index;
            reg signed [ACCUMULATOR_BITS-1:0] old_accumulator;
            reg signed [ACCUMULATOR_BITS-1:0] contribution;

            always @* begin
                old_accumulator = {ACCUMULATOR_BITS{1'b0}};
                contribution = {ACCUMULATOR_BITS{1'b0}};
                for (update_column_index = 0;
                        update_column_index < 3;
                        update_column_index = update_column_index + 1) begin
                    case (active_bank_row)
                        2'd0: selected_old_word[update_column_index] =
                            accumulator_bank_read_data[
                                update_column_index];
                        2'd1: selected_old_word[update_column_index] =
                            accumulator_bank_read_data[
                                3 + update_column_index];
                        default: selected_old_word[update_column_index] =
                            accumulator_bank_read_data[
                                6 + update_column_index];
                    endcase
                    updated_column_word[update_column_index] =
                        selected_old_word[update_column_index];
                    for (update_channel_index = 0;
                            update_channel_index < OUTPUT_CHANNELS;
                            update_channel_index =
                                update_channel_index + 1) begin
                        old_accumulator = $signed(
                            selected_old_word[update_column_index][
                                update_channel_index*ACCUMULATOR_BITS
                                +: ACCUMULATOR_BITS
                            ]
                        );
                        contribution = $signed(
                            update_contribution_word[update_column_index][
                                update_channel_index*ACCUMULATOR_BITS
                                +: ACCUMULATOR_BITS
                            ]
                        );
                        updated_column_word[update_column_index][
                            update_channel_index*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ] = old_accumulator + contribution;
                    end
                end

                // Only the active row is write-enabled.  Replicating the
                // three computed words here is wiring, not extra adders.
                update_new_bank_word[0] = updated_column_word[0];
                update_new_bank_word[1] = updated_column_word[1];
                update_new_bank_word[2] = updated_column_word[2];
                update_new_bank_word[3] = updated_column_word[0];
                update_new_bank_word[4] = updated_column_word[1];
                update_new_bank_word[5] = updated_column_word[2];
                update_new_bank_word[6] = updated_column_word[0];
                update_new_bank_word[7] = updated_column_word[1];
                update_new_bank_word[8] = updated_column_word[2];
            end
        end else begin : independent_update_adders
            integer update_physical_bank_index;
            integer update_physical_channel_index;
            reg signed [ACCUMULATOR_BITS-1:0] old_accumulator;
            reg signed [ACCUMULATOR_BITS-1:0] contribution;

            always @* begin
                old_accumulator = {ACCUMULATOR_BITS{1'b0}};
                contribution = {ACCUMULATOR_BITS{1'b0}};
                for (update_physical_bank_index = 0;
                        update_physical_bank_index < ACCUMULATOR_BANKS;
                        update_physical_bank_index =
                            update_physical_bank_index + 1) begin
                    update_new_bank_word[update_physical_bank_index] =
                        accumulator_bank_read_data[
                            update_physical_bank_index];
                    for (update_physical_channel_index = 0;
                            update_physical_channel_index < OUTPUT_CHANNELS;
                            update_physical_channel_index =
                                update_physical_channel_index + 1) begin
                        old_accumulator = $signed(
                            accumulator_bank_read_data[
                                update_physical_bank_index][
                                update_physical_channel_index
                                    * ACCUMULATOR_BITS
                                +: ACCUMULATOR_BITS
                            ]
                        );
                        contribution = $signed(
                            update_contribution_word[
                                update_physical_bank_index][
                                update_physical_channel_index
                                    * ACCUMULATOR_BITS
                                +: ACCUMULATOR_BITS
                            ]
                        );
                        update_new_bank_word[update_physical_bank_index][
                            update_physical_channel_index*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ] = old_accumulator + contribution;
                    end
                end
            end
        end
    endgenerate

    reg [3:0] finalize_even_bank_index;
    reg [3:0] finalize_odd_bank_index;
    reg [VECTOR_ACCUMULATOR_BITS-1:0] finalize_even_word;
    reg [VECTOR_ACCUMULATOR_BITS-1:0] finalize_odd_word;
    reg [VECTOR_ACCUMULATOR_BITS-1:0] finalize_horizontal_word;
    reg [VECTOR_ACCUMULATOR_BITS-1:0] finalize_pooled_word;
    integer finalize_channel_index;
    reg signed [ACCUMULATOR_BITS-1:0] finalize_even_value;
    reg signed [ACCUMULATOR_BITS-1:0] finalize_odd_value;
    reg signed [ACCUMULATOR_BITS-1:0] finalize_horizontal_value;
    reg signed [ACCUMULATOR_BITS-1:0] finalize_previous_value;

    always @* begin
        finalize_even_bank_index = three_by_three_index(
            finalize_output_row_remainder,
            finalize_even_column_bank
        );
        finalize_odd_bank_index = three_by_three_index(
            finalize_output_row_remainder,
            finalize_odd_column_bank
        );
        finalize_even_word =
            accumulator_bank_read_data[finalize_even_bank_index];
        finalize_odd_word =
            accumulator_bank_read_data[finalize_odd_bank_index];
        finalize_horizontal_word = {VECTOR_ACCUMULATOR_BITS{1'b0}};
        finalize_pooled_word = {VECTOR_ACCUMULATOR_BITS{1'b0}};
        for (finalize_channel_index = 0;
                finalize_channel_index < OUTPUT_CHANNELS;
                finalize_channel_index = finalize_channel_index + 1) begin
            finalize_even_value = $signed(finalize_even_word[
                finalize_channel_index*ACCUMULATOR_BITS
                +: ACCUMULATOR_BITS
            ]);
            finalize_odd_value = $signed(finalize_odd_word[
                finalize_channel_index*ACCUMULATOR_BITS
                +: ACCUMULATOR_BITS
            ]);
            finalize_horizontal_value =
                (finalize_even_value >= finalize_odd_value)
                ? finalize_even_value : finalize_odd_value;
            finalize_horizontal_word[
                finalize_channel_index*ACCUMULATOR_BITS
                +: ACCUMULATOR_BITS
            ] = finalize_horizontal_value;
            finalize_previous_value = $signed(horizontal_pool_memory[
                finalize_pair_column
            ][
                finalize_channel_index*ACCUMULATOR_BITS
                +: ACCUMULATOR_BITS
            ]);
            finalize_pooled_word[
                finalize_channel_index*ACCUMULATOR_BITS
                +: ACCUMULATOR_BITS
            ] = (finalize_output_row[0]
                    && finalize_previous_value > finalize_horizontal_value)
                ? finalize_previous_value : finalize_horizontal_value;
        end
    end

    integer bank_control_index;
    always @* begin
        for (bank_control_index = 0;
                bank_control_index < ACCUMULATOR_BANKS;
                bank_control_index = bank_control_index + 1) begin
            accumulator_bank_address[bank_control_index]
                = {COLUMN_BANK_ADDRESS_BITS{1'b0}};
            accumulator_bank_write_enable[bank_control_index] = 1'b0;
            accumulator_bank_write_data[bank_control_index]
                = {VECTOR_ACCUMULATOR_BITS{1'b0}};
        end
        case (control_state)
            STATE_CLEAR: begin
                for (bank_control_index = 0;
                        bank_control_index < ACCUMULATOR_BANKS;
                        bank_control_index = bank_control_index + 1) begin
                    accumulator_bank_address[bank_control_index] = clear_index;
                    accumulator_bank_write_enable[bank_control_index] = 1'b1;
                end
            end
            STATE_UPDATE: begin
                for (bank_control_index = 0;
                        bank_control_index < ACCUMULATOR_BANKS;
                        bank_control_index = bank_control_index + 1) begin
                    accumulator_bank_address[bank_control_index] =
                        update_bank_column_index[bank_control_index];
                    accumulator_bank_write_enable[bank_control_index] =
                        update_bank_active[bank_control_index];
                    accumulator_bank_write_data[bank_control_index] =
                        update_new_bank_word[bank_control_index];
                end
            end
            STATE_FINALIZE: begin
                for (bank_control_index = 0;
                        bank_control_index < ACCUMULATOR_BANKS;
                        bank_control_index = bank_control_index + 1) begin
                    if (bank_control_index == finalize_even_bank_index) begin
                        accumulator_bank_address[bank_control_index] =
                            finalize_even_column_index;
                        if (!finalize_output_row[0]
                                || !output_valid_register || output_ready) begin
                            accumulator_bank_write_enable[
                                bank_control_index
                            ] = 1'b1;
                        end
                    end else if (bank_control_index
                            == finalize_odd_bank_index) begin
                        accumulator_bank_address[bank_control_index] =
                            finalize_odd_column_index;
                        if (!finalize_output_row[0]
                                || !output_valid_register || output_ready) begin
                            accumulator_bank_write_enable[
                                bank_control_index
                            ] = 1'b1;
                        end
                    end
                end
            end
            default: begin end
        endcase
    end

    function automatic [NEURON_RESULT_BITS-1:0] stateless_neuron;
        input signed [ACCUMULATOR_BITS-1:0] pooled_dot;
        input signed [CURRENT_BITS-1:0] folded_constant;
        input [FRACTION_BITS_WIDTH-1:0] fraction_bits;
        reg signed [ACCUMULATOR_BITS-1:0] constant_extended;
        reg signed [ACCUMULATOR_BITS-1:0] biased_dot;
        reg signed [CURRENT_BITS-1:0] clamped_current;
        reg signed [NEURON_WORK_BITS-1:0] pre_reset;
        reg signed [NEURON_WORK_BITS-1:0] threshold_integer;
        reg signed [NEURON_WORK_BITS-1:0] unsaturated_spike;
        reg spike_overflow;
        reg [SPIKE_BITS-1:0] clamped_spike;
        reg signed [NEURON_WORK_BITS-1:0] reset_amount;
        reg signed [NEURON_WORK_BITS-1:0] membrane_before_floor;
        reg signed [NEURON_WORK_BITS-1:0] membrane_after_floor;
        reg state_overflow;
        reg signed [STATE_BITS-1:0] clamped_membrane;
        reg range_error;
        begin
            constant_extended = {
                {(ACCUMULATOR_BITS-CURRENT_BITS){
                    folded_constant[CURRENT_BITS-1]
                }},
                folded_constant
            };
            biased_dot = pooled_dot + constant_extended;
            clamped_current = (biased_dot > CURRENT_MAXIMUM)
                ? CURRENT_MAXIMUM[CURRENT_BITS-1:0]
                : ((biased_dot < CURRENT_MINIMUM)
                    ? CURRENT_MINIMUM[CURRENT_BITS-1:0]
                    : biased_dot[CURRENT_BITS-1:0]);
            pre_reset = {
                {(NEURON_WORK_BITS-CURRENT_BITS){
                    clamped_current[CURRENT_BITS-1]
                }},
                clamped_current
            };
            threshold_integer =
                $signed({{(NEURON_WORK_BITS-1){1'b0}}, 1'b1})
                <<< fraction_bits;
            unsaturated_spike = (pre_reset > 0)
                ? (pre_reset >>> fraction_bits) : 0;
            spike_overflow = unsaturated_spike > $signed({
                {(NEURON_WORK_BITS-SPIKE_BITS){1'b0}}, SPIKE_MAXIMUM
            });
            clamped_spike = spike_overflow
                ? SPIKE_MAXIMUM
                : unsaturated_spike[SPIKE_BITS-1:0];
            reset_amount = $signed({
                {(NEURON_WORK_BITS-SPIKE_BITS){1'b0}}, clamped_spike
            }) <<< fraction_bits;
            membrane_before_floor = pre_reset - reset_amount;
            membrane_after_floor =
                (membrane_before_floor < -threshold_integer)
                ? -threshold_integer : membrane_before_floor;
            state_overflow = (membrane_after_floor < STATE_MINIMUM)
                || (membrane_after_floor > STATE_MAXIMUM);
            clamped_membrane = (membrane_after_floor < STATE_MINIMUM)
                ? STATE_MINIMUM[STATE_BITS-1:0]
                : ((membrane_after_floor > STATE_MAXIMUM)
                    ? STATE_MAXIMUM[STATE_BITS-1:0]
                    : membrane_after_floor[STATE_BITS-1:0]);
            range_error = (fraction_bits > MAX_FRACTION_BITS)
                || spike_overflow || state_overflow;
            stateless_neuron = {
                range_error,
                clamped_membrane,
                clamped_spike,
                clamped_current
            };
        end
    endfunction

    reg [OUTPUT_CHANNELS*CURRENT_BITS-1:0]
        finalize_current_flat;
    reg [OUTPUT_CHANNELS*SPIKE_BITS-1:0]
        finalize_spike_flat;
    reg [OUTPUT_CHANNELS*STATE_BITS-1:0]
        finalize_membrane_flat;
    reg finalize_range_error;
    reg [NEURON_RESULT_BITS-1:0] neuron_result;

    always @* begin
        finalize_current_flat = {OUTPUT_CHANNELS*CURRENT_BITS{1'b0}};
        finalize_spike_flat = {OUTPUT_CHANNELS*SPIKE_BITS{1'b0}};
        finalize_membrane_flat = {OUTPUT_CHANNELS*STATE_BITS{1'b0}};
        finalize_range_error = 1'b0;
        neuron_result = {NEURON_RESULT_BITS{1'b0}};
        for (finalize_channel_index = 0;
                finalize_channel_index < OUTPUT_CHANNELS;
                finalize_channel_index = finalize_channel_index + 1) begin
            neuron_result = stateless_neuron(
                $signed(finalize_pooled_word[
                    finalize_channel_index*ACCUMULATOR_BITS
                    +: ACCUMULATOR_BITS
                ]),
                constant_registers[finalize_channel_index],
                fraction_bits_registers[finalize_channel_index]
            );
            finalize_current_flat[
                finalize_channel_index*CURRENT_BITS +: CURRENT_BITS
            ] = neuron_result[0 +: CURRENT_BITS];
            finalize_spike_flat[
                finalize_channel_index*SPIKE_BITS +: SPIKE_BITS
            ] = neuron_result[CURRENT_BITS +: SPIKE_BITS];
            finalize_membrane_flat[
                finalize_channel_index*STATE_BITS +: STATE_BITS
            ] = neuron_result[
                CURRENT_BITS + SPIKE_BITS +: STATE_BITS
            ];
            finalize_range_error = finalize_range_error
                || neuron_result[NEURON_RESULT_BITS-1];
        end
    end

    task automatic advance_finalize_columns;
        begin
            finalize_pair_column <= finalize_pair_column + 1'b1;
            case (finalize_even_column_bank)
                2'd0: finalize_even_column_bank <= 2'd2;
                2'd1: begin
                    finalize_even_column_bank <= 2'd0;
                    finalize_even_column_index
                        <= finalize_even_column_index + 1'b1;
                end
                default: begin
                    finalize_even_column_bank <= 2'd1;
                    finalize_even_column_index
                        <= finalize_even_column_index + 1'b1;
                end
            endcase
            case (finalize_odd_column_bank)
                2'd0: finalize_odd_column_bank <= 2'd2;
                2'd1: begin
                    finalize_odd_column_bank <= 2'd0;
                    finalize_odd_column_index
                        <= finalize_odd_column_index + 1'b1;
                end
                default: begin
                    finalize_odd_column_bank <= 2'd1;
                    finalize_odd_column_index
                        <= finalize_odd_column_index + 1'b1;
                end
            endcase
        end
    endtask

    task automatic complete_finalized_row;
        begin
            if (finalize_is_last_row) begin
                busy <= 1'b0;
                done <= 1'b1;
                control_state <= STATE_IDLE;
            end else begin
                input_row_cursor <= input_row_cursor + 1'b1;
                input_row_cursor_modulo_three <=
                    increment_modulo_three(input_row_cursor_modulo_three);
                control_state <= STATE_WAIT_PACKET;
            end
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            control_state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            clear_index <= {COLUMN_BANK_ADDRESS_BITS{1'b0}};
            input_row_cursor <= {CURSOR_BITS{1'b0}};
            input_row_cursor_modulo_three <= 2'd0;
            end_seen <= 1'b0;
            packet_channel <= {INPUT_CHANNEL_BITS{1'b0}};
            packet_row <= {INPUT_ROW_BITS{1'b0}};
            packet_column <= {INPUT_COLUMN_BITS{1'b0}};
            packet_column_quotient <=
                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
            packet_column_remainder <= 2'd0;
            packet_row_remainder <= 2'd0;
            packet_count <= {COUNT_BITS{1'b0}};
            tap_base <= 4'd0;
            finalize_output_row <= {INPUT_ROW_BITS{1'b0}};
            finalize_output_row_remainder <= 2'd0;
            finalize_pair_column <= {PAIR_COLUMN_BITS{1'b0}};
            finalize_even_column_bank <= 2'd0;
            finalize_odd_column_bank <= 2'd1;
            finalize_even_column_index <=
                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
            finalize_odd_column_index <=
                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
            finalize_is_last_row <= 1'b0;
            output_valid_register <= 1'b0;
            output_address_register <= {OUTPUT_ADDRESS_BITS{1'b0}};
            next_output_address <= {OUTPUT_ADDRESS_BITS{1'b0}};
            output_current_register <=
                {OUTPUT_CHANNELS*CURRENT_BITS{1'b0}};
            output_spike_register <=
                {OUTPUT_CHANNELS*SPIKE_BITS{1'b0}};
            output_membrane_register <=
                {OUTPUT_CHANNELS*STATE_BITS{1'b0}};
            output_range_error_register <= 1'b0;
            protocol_error <= 1'b0;
            cycle_count <= 32'd0;
            accepted_packet_count <= 32'd0;
        end else begin
            done <= 1'b0;
            if (busy) begin
                cycle_count <= cycle_count + 1'b1;
            end
            if (busy && input_end) begin
                end_seen <= 1'b1;
            end
            if (output_valid_register && output_ready) begin
                output_valid_register <= 1'b0;
            end

            case (control_state)
                STATE_IDLE: begin
                    if (start) begin
                        busy <= 1'b1;
                        end_seen <= input_end;
                        clear_index <= {COLUMN_BANK_ADDRESS_BITS{1'b0}};
                        input_row_cursor <= {CURSOR_BITS{1'b0}};
                        input_row_cursor_modulo_three <= 2'd0;
                        output_valid_register <= 1'b0;
                        next_output_address <= {OUTPUT_ADDRESS_BITS{1'b0}};
                        protocol_error <= 1'b0;
                        cycle_count <= 32'd0;
                        accepted_packet_count <= 32'd0;
                        control_state <= STATE_CLEAR;
                    end
                end

                STATE_CLEAR: begin
                    if (clear_index == COLUMN_BANK_DEPTH-1) begin
                        control_state <= STATE_WAIT_PACKET;
                    end else begin
                        clear_index <= clear_index + 1'b1;
                    end
                end

                STATE_WAIT_PACKET: begin
                    if (input_valid
                            && input_row < input_row_cursor) begin
                        protocol_error <= 1'b1;
                    end
                    if (input_valid
                            && input_row == input_row_cursor
                            && !end_seen) begin
                        if (input_channel >= INPUT_CHANNELS
                                || input_column >= INPUT_WIDTH
                                || (!UNIT_INPUT && input_count == 0)) begin
                            protocol_error <= 1'b1;
                        end
                        packet_channel <= input_channel;
                        packet_row <= input_row;
                        packet_column <= input_column;
                        packet_column_quotient <= input_column_quotient;
                        packet_column_remainder <= input_column_remainder;
                        packet_row_remainder <=
                            input_row_cursor_modulo_three;
                        packet_count <= UNIT_INPUT
                            ? {{(COUNT_BITS-1){1'b0}}, 1'b1}
                            : input_count;
                        tap_base <= 4'd0;
                        accepted_packet_count
                            <= accepted_packet_count + 1'b1;
                        control_state <= STATE_UPDATE;
                    end else if ((input_valid
                                    && input_row > input_row_cursor)
                                || end_seen || input_end) begin
                        if (input_row_cursor == 0) begin
                            input_row_cursor <= input_row_cursor + 1'b1;
                            input_row_cursor_modulo_three <=
                                increment_modulo_three(
                                    input_row_cursor_modulo_three
                                );
                        end else if (input_row_cursor < INPUT_HEIGHT) begin
                            finalize_output_row <=
                                input_row_cursor[INPUT_ROW_BITS-1:0] - 1'b1;
                            finalize_output_row_remainder <=
                                decrement_modulo_three(
                                    input_row_cursor_modulo_three
                                );
                            finalize_pair_column <=
                                {PAIR_COLUMN_BITS{1'b0}};
                            finalize_even_column_bank <= 2'd0;
                            finalize_odd_column_bank <= 2'd1;
                            finalize_even_column_index <=
                                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
                            finalize_odd_column_index <=
                                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
                            finalize_is_last_row <= 1'b0;
                            control_state <= STATE_FINALIZE;
                        end else begin
                            finalize_output_row <= INPUT_ROW_BITS'(INPUT_HEIGHT-1);
                            finalize_output_row_remainder <=
                                decrement_modulo_three(
                                    input_row_cursor_modulo_three
                                );
                            finalize_pair_column <=
                                {PAIR_COLUMN_BITS{1'b0}};
                            finalize_even_column_bank <= 2'd0;
                            finalize_odd_column_bank <= 2'd1;
                            finalize_even_column_index <=
                                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
                            finalize_odd_column_index <=
                                {COLUMN_BANK_ADDRESS_BITS{1'b0}};
                            finalize_is_last_row <= 1'b1;
                            control_state <= STATE_FINALIZE;
                        end
                    end
                end

                STATE_UPDATE: begin
                    if (tap_base + KERNEL_TAPS_PER_CYCLE >= 9) begin
                        if (input_valid
                                && input_row == input_row_cursor
                                && !end_seen && !input_end) begin
                            if (input_channel >= INPUT_CHANNELS
                                    || input_column >= INPUT_WIDTH
                                    || (!UNIT_INPUT && input_count == 0)) begin
                                protocol_error <= 1'b1;
                            end
                            packet_channel <= input_channel;
                            packet_row <= input_row;
                            packet_column <= input_column;
                            packet_column_quotient <=
                                input_column_quotient;
                            packet_column_remainder <=
                                input_column_remainder;
                            packet_row_remainder <=
                                input_row_cursor_modulo_three;
                            packet_count <= UNIT_INPUT
                                ? {{(COUNT_BITS-1){1'b0}}, 1'b1}
                                : input_count;
                            tap_base <= 4'd0;
                            accepted_packet_count
                                <= accepted_packet_count + 1'b1;
                        end else begin
                            control_state <= STATE_WAIT_PACKET;
                        end
                    end else begin
                        tap_base <= tap_base + KERNEL_TAPS_PER_CYCLE;
                    end
                end

                STATE_FINALIZE: begin
                    if (!finalize_output_row[0]) begin
                        horizontal_pool_memory[finalize_pair_column]
                            <= finalize_horizontal_word;
                        if (finalize_pair_column == HALF_WIDTH-1) begin
                            complete_finalized_row();
                        end else begin
                            advance_finalize_columns();
                        end
                    end else if (!output_valid_register || output_ready) begin
                        output_valid_register <= 1'b1;
                        output_address_register <= next_output_address;
                        next_output_address <= next_output_address + 1'b1;
                        output_current_register <= finalize_current_flat;
                        output_spike_register <= finalize_spike_flat;
                        output_membrane_register <= finalize_membrane_flat;
                        output_range_error_register
                            <= finalize_range_error;
                        if (finalize_pair_column == HALF_WIDTH-1) begin
                            control_state <= STATE_FINALIZE_DRAIN;
                        end else begin
                            advance_finalize_columns();
                        end
                    end
                end

                STATE_FINALIZE_DRAIN: begin
                    if (!output_valid_register
                            || (output_valid_register && output_ready)) begin
                        complete_finalized_row();
                    end
                end

                default: begin
                    control_state <= STATE_IDLE;
                    busy <= 1'b0;
                    protocol_error <= 1'b1;
                end
            endcase
        end
    end

    initial begin
        if (INPUT_WIDTH <= 0 || INPUT_HEIGHT <= 0
                || INPUT_WIDTH % 2 != 0 || INPUT_HEIGHT % 2 != 0) begin
            $error("sparse core requires positive even spatial dimensions");
        end
        if (OUTPUT_CHANNELS <= 0 || INPUT_CHANNELS <= 0) begin
            $error("sparse core channel counts must be positive");
        end
        if (ACCUMULATOR_BITS < CURRENT_BITS
                || CURRENT_BITS < STATE_BITS) begin
            $error("invalid sparse core arithmetic widths");
        end
        if (UNIT_INPUT != 0 && UNIT_INPUT != 1) begin
            $error("UNIT_INPUT must be zero or one");
        end
        if (KERNEL_TAPS_PER_CYCLE <= 0
                || KERNEL_TAPS_PER_CYCLE > 9
                || (9 % KERNEL_TAPS_PER_CYCLE) != 0) begin
            $error("KERNEL_TAPS_PER_CYCLE must be 1, 3, or 9");
        end
    end
endmodule

/* verilator lint_on UNUSEDPARAM */
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on DECLFILENAME */
/* verilator lint_on WIDTHTRUNC */
/* verilator lint_on WIDTHEXPAND */

`default_nettype wire
