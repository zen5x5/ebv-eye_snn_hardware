`timescale 1ns/1ps
`default_nettype none

// Paired-time H1/H2 engine.
//
// Sixteen S16 multipliers are shared by the two time steps.  For every input
// feature, one packed 256-bit weight beat is fetched once.  The engine first
// accumulates input t0, retains that beat for one cycle, and then accumulates
// input t1.  Separate S40 accumulators preserve the two dot products.  Once a
// complete output group is available, the same sixteen LIF neurons process t0
// followed immediately by t1; the registered t0 membrane is wired directly
// into the t1 update.
//
// The packed weight layout is input-major:
//   address = base + input_index*output_groups + output_group.
// External weight, input, and parameter memories have one-cycle read latency.
module paired_linear_lif_layer_engine #(
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
    input  wire [WEIGHT_ADDRESS_BITS-1:0]            config_weight_beat_base,
    input  wire [PARAMETER_ADDRESS_BITS-1:0]         config_parameter_base,

    output wire                                      weight_read_enable,
    output wire [WEIGHT_ADDRESS_BITS-1:0]            weight_read_address,
    input  wire [OUTPUT_LANES*WEIGHT_BITS-1:0]       weight_read_data,

    output wire                                      input_read_enable,
    output wire                                      input_read_time,
    output wire [INPUT_ADDRESS_BITS-1:0]             input_read_address,
    input  wire [INPUT_BITS-1:0]                     input_read_data,

    output wire                                      parameter_read_enable,
    output wire [PARAMETER_ADDRESS_BITS-1:0]         parameter_read_address,
    input  wire signed [CURRENT_BITS-1:0]            parameter_constant_data,
    input  wire [FRACTION_BITS_WIDTH-1:0]            parameter_fraction_data,

    output wire                                      output_valid,
    output wire                                      output_time,
    output wire [CHANNEL_BITS-1:0]                   output_channel_base,
    output wire [OUTPUT_LANES*CURRENT_BITS-1:0]      output_current_flat,
    output wire [OUTPUT_LANES*SPIKE_BITS-1:0]        output_spike_flat,
    output wire [OUTPUT_LANES*STATE_BITS-1:0]        output_membrane_flat,
    output wire                                      output_range_error
);
    localparam [2:0] STATE_IDLE = 3'd0;
    localparam [2:0] STATE_PARAMETERS = 3'd1;
    localparam [2:0] STATE_INITIAL_REQUEST = 3'd2;
    localparam [2:0] STATE_ACCUMULATE_T0 = 3'd3;
    localparam [2:0] STATE_ACCUMULATE_T1 = 3'd4;
    localparam [2:0] STATE_LIF_T0 = 3'd5;
    localparam [2:0] STATE_LIF_T1 = 3'd6;
    localparam [2:0] STATE_LIF_WAIT = 3'd7;

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

    reg [2:0] control_state;
    reg [INPUT_ADDRESS_BITS:0] input_features_register;
    reg [2:0] output_groups_register;
    reg [WEIGHT_ADDRESS_BITS-1:0] weight_base_register;
    reg [PARAMETER_ADDRESS_BITS-1:0] parameter_group_base;
    reg [2:0] output_group_index;
    reg [INPUT_ADDRESS_BITS:0] input_index;

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

    integer parameter_lane;
    always @(posedge clk) begin
        if (rst) begin
            parameter_issue_count <= 0;
            parameter_loaded_count <= 0;
            parameter_response_index <= 0;
            parameter_response_valid <= 1'b0;
            for (parameter_lane = 0; parameter_lane < OUTPUT_LANES;
                    parameter_lane = parameter_lane + 1) begin
                constant_registers[parameter_lane] <= 0;
                fraction_registers[parameter_lane] <= 0;
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
        end
    end

    wire initial_pair_request =
        (control_state == STATE_INITIAL_REQUEST);
    wire following_pair_request =
        (control_state == STATE_ACCUMULATE_T1)
        && (input_index + 1'b1 < input_features_register);
    wire t1_input_request = (control_state == STATE_ACCUMULATE_T0);
    wire [INPUT_ADDRESS_BITS:0] requested_input_index =
        following_pair_request ? (input_index + 1'b1) : input_index;

    assign weight_read_enable = initial_pair_request
        || following_pair_request;
    wire [WEIGHT_ADDRESS_BITS-1:0] requested_weight_input_offset =
        (output_groups_register == 3'd4)
        ? WEIGHT_ADDRESS_BITS'(requested_input_index << 2)
        : WEIGHT_ADDRESS_BITS'(requested_input_index);
    assign weight_read_address = weight_base_register
        + requested_weight_input_offset
        + WEIGHT_ADDRESS_BITS'(output_group_index);

    assign input_read_enable = initial_pair_request
        || following_pair_request || t1_input_request;
    assign input_read_time = t1_input_request;
    assign input_read_address = t1_input_request
        ? input_index[INPUT_ADDRESS_BITS-1:0]
        : requested_input_index[INPUT_ADDRESS_BITS-1:0];

    reg [OUTPUT_LANES*WEIGHT_BITS-1:0] retained_weight_beat;
    reg signed [ACCUMULATOR_BITS-1:0] accumulators_t0
        [0:OUTPUT_LANES-1];
    reg signed [ACCUMULATOR_BITS-1:0] accumulators_t1
        [0:OUTPUT_LANES-1];

    wire [OUTPUT_LANES*WEIGHT_BITS-1:0] active_weight_beat =
        (control_state == STATE_ACCUMULATE_T0)
        ? weight_read_data : retained_weight_beat;
    wire signed [INPUT_BITS:0] positive_input =
        $signed({1'b0, input_read_data});
    wire [OUTPUT_LANES*ACCUMULATOR_BITS-1:0] next_accumulators_flat;
    wire [OUTPUT_LANES*CURRENT_BITS-1:0] completed_currents_flat;
    wire accumulate_t1 = (control_state == STATE_ACCUMULATE_T1);
    wire current_for_t1 = (control_state == STATE_LIF_T1);

    genvar arithmetic_lane;
    generate
        for (arithmetic_lane = 0; arithmetic_lane < OUTPUT_LANES;
                arithmetic_lane = arithmetic_lane + 1) begin : gen_arithmetic
            wire signed [WEIGHT_BITS-1:0] active_weight;
            wire signed [WEIGHT_BITS+INPUT_BITS:0] product_value;
            wire signed [ACCUMULATOR_BITS-1:0] product_extended;
            wire signed [ACCUMULATOR_BITS-1:0] active_accumulator;
            wire signed [ACCUMULATOR_BITS-1:0] next_accumulator;
            wire signed [ACCUMULATOR_BITS-1:0] current_accumulator;
            wire signed [ACCUMULATOR_BITS-1:0] constant_extended;
            wire signed [ACCUMULATOR_BITS-1:0] complete_current;

            assign active_weight = $signed(active_weight_beat[
                arithmetic_lane*WEIGHT_BITS +: WEIGHT_BITS
            ]);
            assign product_value = active_weight * positive_input;
            assign product_extended = {
                {(ACCUMULATOR_BITS-(WEIGHT_BITS+INPUT_BITS+1)){
                    product_value[WEIGHT_BITS+INPUT_BITS]
                }},
                product_value
            };
            assign active_accumulator = accumulate_t1
                ? accumulators_t1[arithmetic_lane]
                : accumulators_t0[arithmetic_lane];
            assign next_accumulator = active_accumulator + product_extended;
            assign next_accumulators_flat[
                arithmetic_lane*ACCUMULATOR_BITS +: ACCUMULATOR_BITS
            ] = next_accumulator;

            assign constant_extended = {
                {(ACCUMULATOR_BITS-CURRENT_BITS){
                    constant_registers[arithmetic_lane][CURRENT_BITS-1]
                }},
                constant_registers[arithmetic_lane]
            };
            assign current_accumulator = current_for_t1
                ? accumulators_t1[arithmetic_lane]
                : accumulators_t0[arithmetic_lane];
            assign complete_current = current_accumulator + constant_extended;
            assign completed_currents_flat[
                arithmetic_lane*CURRENT_BITS +: CURRENT_BITS
            ] = clamp_current(complete_current);
        end
    endgenerate

    wire advance_output_group;

    integer accumulator_lane;
    always @(posedge clk) begin
        if (rst) begin
            retained_weight_beat <= 0;
            for (accumulator_lane = 0; accumulator_lane < OUTPUT_LANES;
                    accumulator_lane = accumulator_lane + 1) begin
                accumulators_t0[accumulator_lane] <= 0;
                accumulators_t1[accumulator_lane] <= 0;
            end
        end else begin
            if ((control_state == STATE_IDLE && start)
                    || advance_output_group) begin
                retained_weight_beat <= 0;
                for (accumulator_lane = 0;
                        accumulator_lane < OUTPUT_LANES;
                        accumulator_lane = accumulator_lane + 1) begin
                    accumulators_t0[accumulator_lane] <= 0;
                    accumulators_t1[accumulator_lane] <= 0;
                end
            end else if (control_state == STATE_ACCUMULATE_T0) begin
                retained_weight_beat <= weight_read_data;
                for (accumulator_lane = 0;
                        accumulator_lane < OUTPUT_LANES;
                        accumulator_lane = accumulator_lane + 1) begin
                    accumulators_t0[accumulator_lane]
                        <= next_accumulators_flat[
                            accumulator_lane*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ];
                end
            end else if (control_state == STATE_ACCUMULATE_T1) begin
                for (accumulator_lane = 0;
                        accumulator_lane < OUTPUT_LANES;
                        accumulator_lane = accumulator_lane + 1) begin
                    accumulators_t1[accumulator_lane]
                        <= next_accumulators_flat[
                            accumulator_lane*ACCUMULATOR_BITS
                            +: ACCUMULATOR_BITS
                        ];
                end
            end
        end
    end

    wire lif_input_valid = (control_state == STATE_LIF_T0)
        || (control_state == STATE_LIF_T1);
    wire lif_input_time = (control_state == STATE_LIF_T1);
    wire [OUTPUT_LANES*CURRENT_BITS-1:0] lif_input_currents =
        completed_currents_flat;
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
                .in_valid(lif_input_valid),
                .current_integer($signed(lif_input_currents[
                    neuron_lane*CURRENT_BITS +: CURRENT_BITS
                ])),
                .previous_membrane_integer(lif_input_time
                    ? $signed(lif_membranes_flat[
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

    reg output_time_register;
    reg [CHANNEL_BITS-1:0] output_channel_base_register;
    reg [OUTPUT_LANES*CURRENT_BITS-1:0] output_current_register;
    always @(posedge clk) begin
        if (rst) begin
            output_time_register <= 1'b0;
            output_channel_base_register <= 0;
            output_current_register <= 0;
        end else if (lif_input_valid) begin
            output_time_register <= lif_input_time;
            output_channel_base_register
                <= CHANNEL_BITS'(output_group_index * OUTPUT_LANES);
            output_current_register <= lif_input_currents;
        end
    end

    assign output_valid = &lif_valid_lanes;
    assign output_time = output_time_register;
    assign output_channel_base = output_channel_base_register;
    assign output_current_flat = output_current_register;
    assign output_spike_flat = lif_spikes_flat;
    assign output_membrane_flat = lif_membranes_flat;
    assign output_range_error = |lif_range_errors;

    wire last_output_group = output_group_index
        == output_groups_register - 1'b1;
    assign advance_output_group = (control_state == STATE_LIF_WAIT)
        && output_valid && !last_output_group;

    always @(posedge clk) begin
        if (rst) begin
            control_state <= STATE_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            input_features_register <= 0;
            output_groups_register <= 0;
            weight_base_register <= 0;
            parameter_group_base <= 0;
            output_group_index <= 0;
            input_index <= 0;
        end else begin
            done <= 1'b0;
            case (control_state)
                STATE_IDLE: begin
                    if (start) begin
                        input_features_register <= config_input_features;
                        output_groups_register <= config_output_groups;
                        weight_base_register <= config_weight_beat_base;
                        parameter_group_base <= config_parameter_base;
                        output_group_index <= 0;
                        input_index <= 0;
                        busy <= 1'b1;
                        control_state <= STATE_PARAMETERS;
                    end
                end
                STATE_PARAMETERS: begin
                    if (parameter_loaded_count == 5'(OUTPUT_LANES)) begin
                        input_index <= 0;
                        control_state <= STATE_INITIAL_REQUEST;
                    end
                end
                STATE_INITIAL_REQUEST: begin
                    control_state <= STATE_ACCUMULATE_T0;
                end
                STATE_ACCUMULATE_T0: begin
                    control_state <= STATE_ACCUMULATE_T1;
                end
                STATE_ACCUMULATE_T1: begin
                    if (input_index == input_features_register - 1'b1) begin
                        control_state <= STATE_LIF_T0;
                    end else begin
                        input_index <= input_index + 1'b1;
                        control_state <= STATE_ACCUMULATE_T0;
                    end
                end
                STATE_LIF_T0: begin
                    control_state <= STATE_LIF_T1;
                end
                STATE_LIF_T1: begin
                    control_state <= STATE_LIF_WAIT;
                end
                STATE_LIF_WAIT: begin
                    if (output_valid) begin
                        if (last_output_group) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            control_state <= STATE_IDLE;
                        end else begin
                            output_group_index <= output_group_index + 1'b1;
                            parameter_group_base <= parameter_group_base
                                + PARAMETER_ADDRESS_BITS'(OUTPUT_LANES);
                            input_index <= 0;
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
            $error("paired linear accumulator is too narrow");
        end
    end
endmodule

`default_nettype wire
