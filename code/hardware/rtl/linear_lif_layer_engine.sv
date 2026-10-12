`timescale 1ns/1ps
`default_nettype none

// Shared H1/H2 engine.  Sixteen output neurons accumulate in parallel while
// one input spike and one 256-bit weight beat stream each cycle.  The packed
// image is input-major: address = base + input*output_groups + output_group.
module linear_lif_layer_engine #(
    parameter integer OUTPUT_LANES = 16,
    parameter integer INPUT_BITS = 8,
    parameter integer WEIGHT_BITS = 16,
    parameter integer ACCUMULATOR_BITS = 40,
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer INPUT_ADDRESS_BITS = 9,
    parameter integer WEIGHT_ADDRESS_BITS = 11,
    parameter integer PARAMETER_ADDRESS_BITS = 9,
    parameter integer CHANNEL_BITS = 7,
    parameter integer FRACTION_BITS_WIDTH = 5
) (
    input  wire                                      clk,
    input  wire                                      rst,
    input  wire                                      start,
    output reg                                       busy,
    output reg                                       done,

    input  wire [INPUT_ADDRESS_BITS:0]               config_input_features,
    input  wire [2:0]                                config_output_groups,
    input  wire                                      config_use_previous_state,
    input  wire                                      config_write_state,
    input  wire [WEIGHT_ADDRESS_BITS-1:0]            config_weight_beat_base,
    input  wire [PARAMETER_ADDRESS_BITS-1:0]         config_parameter_base,

    output wire                                      weight_read_enable,
    output wire [WEIGHT_ADDRESS_BITS-1:0]            weight_read_address,
    input  wire [OUTPUT_LANES*WEIGHT_BITS-1:0]       weight_read_data,
    output wire                                      input_read_enable,
    output wire [INPUT_ADDRESS_BITS-1:0]             input_read_address,
    input  wire [INPUT_BITS-1:0]                     input_read_data,

    output wire                                      parameter_read_enable,
    output wire [PARAMETER_ADDRESS_BITS-1:0]         parameter_read_address,
    input  wire signed [CURRENT_BITS-1:0]            parameter_constant_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]            parameter_fraction_data,

    output wire                                      state_read_enable,
    output wire [CHANNEL_BITS-1:0]                   state_read_channel_base,
    input  wire [OUTPUT_LANES*STATE_BITS-1:0]        state_read_data_flat,
    output wire                                      state_write_enable,
    output wire [CHANNEL_BITS-1:0]                   state_write_channel_base,
    output wire [OUTPUT_LANES*STATE_BITS-1:0]        state_write_data_flat,

    output wire                                      output_valid,
    output wire [CHANNEL_BITS-1:0]                   output_channel_base,
    output wire [OUTPUT_LANES*CURRENT_BITS-1:0]      output_current_flat,
    output wire [OUTPUT_LANES*SPIKE_BITS-1:0]        output_spike_flat,
    output wire [OUTPUT_LANES*STATE_BITS-1:0]        output_membrane_flat,
    output wire                                      output_range_error
);
    localparam [1:0] STATE_IDLE = 2'd0;
    localparam [1:0] STATE_PARAMETERS = 2'd1;
    localparam [1:0] STATE_ACCUMULATE = 2'd2;
    localparam [1:0] STATE_NEURON_WAIT = 2'd3;

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

    function automatic signed [CURRENT_BITS-1:0] clamp_current;
        input signed [ACCUMULATOR_BITS-1:0] value;
        begin
            if (value > CURRENT_MAXIMUM) begin
                clamp_current = CURRENT_MAXIMUM[CURRENT_BITS-1:0];
            end else if (value < CURRENT_MINIMUM) begin
                clamp_current = CURRENT_MINIMUM[CURRENT_BITS-1:0];
            end else begin
                clamp_current = value[CURRENT_BITS-1:0];
            end
        end
    endfunction

    reg [1:0] control_state;
    reg [INPUT_ADDRESS_BITS:0] input_features_register;
    reg [2:0] output_groups_register;
    reg use_previous_state_register;
    reg write_state_register;
    reg [WEIGHT_ADDRESS_BITS-1:0] weight_base_register;
    reg [PARAMETER_ADDRESS_BITS-1:0] parameter_group_base;
    reg [2:0] output_group_index;

    reg [4:0] parameter_issue_count;
    reg [4:0] parameter_loaded_count;
    reg [3:0] parameter_response_index;
    reg parameter_response_valid;
    reg signed [CURRENT_BITS-1:0] constant_registers [0:OUTPUT_LANES-1];
    reg [FRACTION_BITS_WIDTH-1:0] fraction_registers [0:OUTPUT_LANES-1];

    assign parameter_read_enable = (control_state == STATE_PARAMETERS)
        && (parameter_issue_count < 5'(OUTPUT_LANES));
    assign parameter_read_address = parameter_group_base
        + PARAMETER_ADDRESS_BITS'(parameter_issue_count);

    reg [INPUT_ADDRESS_BITS:0] input_issue_count;
    reg response_valid;
    reg [INPUT_ADDRESS_BITS:0] response_index;
    wire request_valid = (control_state == STATE_ACCUMULATE)
        && (input_issue_count < input_features_register);
    assign input_read_enable = request_valid;
    assign input_read_address = input_issue_count[INPUT_ADDRESS_BITS-1:0];
    assign weight_read_enable = request_valid;
    wire [WEIGHT_ADDRESS_BITS-1:0] input_weight_offset =
        (output_groups_register == 4)
        ? WEIGHT_ADDRESS_BITS'(input_issue_count << 2)
        : WEIGHT_ADDRESS_BITS'(input_issue_count);
    assign weight_read_address = weight_base_register
        + input_weight_offset + WEIGHT_ADDRESS_BITS'(output_group_index);

    reg signed [ACCUMULATOR_BITS-1:0] accumulators [0:OUTPUT_LANES-1];
    wire signed [INPUT_BITS:0] positive_input = $signed({1'b0, input_read_data});
    wire [OUTPUT_LANES*ACCUMULATOR_BITS-1:0] next_accumulators_flat;
    wire [OUTPUT_LANES*CURRENT_BITS-1:0] completed_currents_flat;

    genvar arithmetic_lane;
    generate
        for (arithmetic_lane = 0; arithmetic_lane < OUTPUT_LANES;
                arithmetic_lane = arithmetic_lane + 1) begin : gen_arithmetic
            wire signed [WEIGHT_BITS-1:0] weight_value;
            wire signed [WEIGHT_BITS+INPUT_BITS:0] product_value;
            wire signed [ACCUMULATOR_BITS-1:0] product_extended;
            wire signed [ACCUMULATOR_BITS-1:0] next_accumulator;
            wire signed [ACCUMULATOR_BITS-1:0] constant_extended;
            wire signed [ACCUMULATOR_BITS-1:0] complete_current;
            assign weight_value = $signed(weight_read_data[
                arithmetic_lane*WEIGHT_BITS +: WEIGHT_BITS
            ]);
            assign product_value = weight_value * positive_input;
            assign product_extended = {
                {(ACCUMULATOR_BITS-(WEIGHT_BITS+INPUT_BITS+1)){
                    product_value[WEIGHT_BITS+INPUT_BITS]
                }},
                product_value
            };
            assign next_accumulator = accumulators[arithmetic_lane]
                + product_extended;
            assign next_accumulators_flat[
                arithmetic_lane*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ] = next_accumulator;
            assign constant_extended = {
                {(ACCUMULATOR_BITS-CURRENT_BITS){
                    constant_registers[arithmetic_lane][CURRENT_BITS-1]
                }},
                constant_registers[arithmetic_lane]
            };
            assign complete_current = next_accumulator + constant_extended;
            assign completed_currents_flat[
                arithmetic_lane*CURRENT_BITS +: CURRENT_BITS
            ] = clamp_current(complete_current);
        end
    endgenerate

    reg current_valid;
    reg [OUTPUT_LANES*CURRENT_BITS-1:0] current_registers;
    reg [CHANNEL_BITS-1:0] current_channel_base;

    integer sequential_lane;
    always @(posedge clk) begin
        if (rst) begin
            parameter_issue_count <= 0;
            parameter_loaded_count <= 0;
            parameter_response_index <= 0;
            parameter_response_valid <= 1'b0;
            input_issue_count <= 0;
            response_valid <= 1'b0;
            response_index <= 0;
            current_valid <= 1'b0;
            current_registers <= 0;
            current_channel_base <= 0;
            for (sequential_lane = 0; sequential_lane < OUTPUT_LANES;
                    sequential_lane = sequential_lane + 1) begin
                accumulators[sequential_lane] <= 0;
            end
        end else begin
            parameter_response_valid <= parameter_read_enable;
            if (parameter_read_enable) begin
                parameter_response_index <= parameter_issue_count[3:0];
                parameter_issue_count <= parameter_issue_count + 1'b1;
            end
            if (parameter_response_valid) begin
                constant_registers[parameter_response_index]
                    <= parameter_constant_data;
                fraction_registers[parameter_response_index]
                    <= parameter_fraction_data;
                parameter_loaded_count <= parameter_loaded_count + 1'b1;
            end
            if (control_state != STATE_PARAMETERS) begin
                parameter_issue_count <= 0;
                parameter_loaded_count <= 0;
                parameter_response_valid <= 1'b0;
            end

            response_valid <= request_valid;
            if (request_valid) begin
                response_index <= input_issue_count;
                input_issue_count <= input_issue_count + 1'b1;
            end
            current_valid <= 1'b0;
            if (response_valid) begin
                for (sequential_lane = 0;
                        sequential_lane < OUTPUT_LANES;
                        sequential_lane = sequential_lane + 1) begin
                    accumulators[sequential_lane]
                        <= next_accumulators_flat[
                            sequential_lane*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ];
                end
                if (response_index == input_features_register - 1'b1) begin
                    current_valid <= 1'b1;
                    current_registers <= completed_currents_flat;
                    current_channel_base <=
                        CHANNEL_BITS'(output_group_index * OUTPUT_LANES);
                end
            end
            if (control_state != STATE_ACCUMULATE) begin
                input_issue_count <= 0;
                response_valid <= 1'b0;
            end
            if ((control_state == STATE_PARAMETERS)
                    && (parameter_loaded_count == 5'(OUTPUT_LANES))) begin
                for (sequential_lane = 0;
                        sequential_lane < OUTPUT_LANES;
                        sequential_lane = sequential_lane + 1) begin
                    accumulators[sequential_lane] <= 0;
                end
            end
        end
    end

    assign state_read_enable = current_valid && use_previous_state_register;
    assign state_read_channel_base = current_channel_base;

    reg neuron_valid_delayed;
    reg [OUTPUT_LANES*CURRENT_BITS-1:0] neuron_currents_delayed;
    reg [CHANNEL_BITS-1:0] neuron_channel_base_delayed;
    always @(posedge clk) begin
        if (rst) begin
            neuron_valid_delayed <= 1'b0;
            neuron_currents_delayed <= 0;
            neuron_channel_base_delayed <= 0;
        end else begin
            neuron_valid_delayed <= current_valid;
            if (current_valid) begin
                neuron_currents_delayed <= current_registers;
                neuron_channel_base_delayed <= current_channel_base;
            end
        end
    end

    wire [OUTPUT_LANES-1:0] lif_valid_lanes;
    wire [OUTPUT_LANES-1:0] lif_range_errors;
    wire [OUTPUT_LANES*SPIKE_BITS-1:0] lif_spikes_flat;
    wire [OUTPUT_LANES*STATE_BITS-1:0] lif_membranes_flat;
    genvar neuron_lane;
    generate
        for (neuron_lane = 0; neuron_lane < OUTPUT_LANES;
                neuron_lane = neuron_lane + 1) begin : gen_neurons
            lif_multispike #(
                .CURRENT_BITS(CURRENT_BITS),
                .STATE_BITS(STATE_BITS),
                .SPIKE_BITS(SPIKE_BITS),
                .FRACTION_BITS_WIDTH(FRACTION_BITS_WIDTH)
            ) neuron (
                .clk(clk),
                .rst(rst),
                .in_valid(neuron_valid_delayed),
                .current_integer($signed(neuron_currents_delayed[
                    neuron_lane*CURRENT_BITS +: CURRENT_BITS
                ])),
                .previous_membrane_integer(use_previous_state_register
                    ? $signed(state_read_data_flat[
                        neuron_lane*STATE_BITS +: STATE_BITS
                    ])
                    : {STATE_BITS{1'b0}}),
                .fraction_bits(fraction_registers[neuron_lane]),
                .out_valid(lif_valid_lanes[neuron_lane]),
                .spike_integer(lif_spikes_flat[
                    neuron_lane*SPIKE_BITS +: SPIKE_BITS
                ]),
                .membrane_integer(lif_membranes_flat[
                    neuron_lane*STATE_BITS +: STATE_BITS
                ]),
                .range_error(lif_range_errors[neuron_lane])
            );
        end
    endgenerate

    reg [CHANNEL_BITS-1:0] output_channel_base_register;
    reg [OUTPUT_LANES*CURRENT_BITS-1:0] output_currents_register;
    always @(posedge clk) begin
        if (rst) begin
            output_channel_base_register <= 0;
            output_currents_register <= 0;
        end else if (neuron_valid_delayed) begin
            output_channel_base_register <= neuron_channel_base_delayed;
            output_currents_register <= neuron_currents_delayed;
        end
    end

    assign output_valid = &lif_valid_lanes;
    assign output_channel_base = output_channel_base_register;
    assign output_current_flat = output_currents_register;
    assign output_spike_flat = lif_spikes_flat;
    assign output_membrane_flat = lif_membranes_flat;
    assign output_range_error = |lif_range_errors;
    assign state_write_enable = output_valid && write_state_register;
    assign state_write_channel_base = output_channel_base;
    assign state_write_data_flat = output_membrane_flat;

    wire last_output_group = output_group_index
        == (output_groups_register - 1'b1);
    always @(posedge clk) begin
        if (rst) begin
            control_state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            input_features_register <= 0;
            output_groups_register <= 0;
            use_previous_state_register <= 1'b0;
            write_state_register <= 1'b0;
            weight_base_register <= 0;
            parameter_group_base <= 0;
            output_group_index <= 0;
        end else begin
            done <= 1'b0;
            case (control_state)
                STATE_IDLE: begin
                    if (start) begin
                        input_features_register <= config_input_features;
                        output_groups_register <= config_output_groups;
                        use_previous_state_register
                            <= config_use_previous_state;
                        write_state_register <= config_write_state;
                        weight_base_register <= config_weight_beat_base;
                        parameter_group_base <= config_parameter_base;
                        output_group_index <= 0;
                        busy <= 1'b1;
                        control_state <= STATE_PARAMETERS;
                    end
                end
                STATE_PARAMETERS: begin
                    if (parameter_loaded_count == 5'(OUTPUT_LANES)) begin
                        control_state <= STATE_ACCUMULATE;
                    end
                end
                STATE_ACCUMULATE: begin
                    if (current_valid) begin
                        control_state <= STATE_NEURON_WAIT;
                    end
                end
                STATE_NEURON_WAIT: begin
                    if (output_valid) begin
                        if (last_output_group) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            control_state <= STATE_IDLE;
                        end else begin
                            output_group_index <= output_group_index + 1'b1;
                            parameter_group_base <= parameter_group_base
                                + PARAMETER_ADDRESS_BITS'(OUTPUT_LANES);
                            control_state <= STATE_PARAMETERS;
                        end
                    end
                end
                default: control_state <= STATE_IDLE;
            endcase
        end
    end

    initial begin
        if (OUTPUT_LANES != 16 || WEIGHT_BITS != 16) begin
            $error("packed linear image requires sixteen S16 lanes");
        end
        if (ACCUMULATOR_BITS < WEIGHT_BITS + INPUT_BITS + 1) begin
            $error("linear accumulator is too narrow");
        end
    end
endmodule

`default_nettype wire
