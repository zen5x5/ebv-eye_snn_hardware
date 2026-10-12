`timescale 1ns/1ps
`default_nettype none

// Bit-exact neuron used by the fixed-point reference:
//   leaked    = floor(3 * previous_membrane / 4)
//   pre_reset = leaked + current
//   spike     = clamp(floor(max(pre_reset, 0) / 2^fraction_bits), 0, 255)
//   membrane  = max(pre_reset - spike * 2^fraction_bits, -2^fraction_bits)
//
// range_error is explicit because a saturated spike can leave a membrane that
// no longer fits STATE_BITS.  The validated deployment vectors never assert it.
module lif_multispike #(
    parameter integer CURRENT_BITS = 24,
    parameter integer STATE_BITS = 16,
    parameter integer SPIKE_BITS = 8,
    parameter integer FRACTION_BITS_WIDTH = 5
) (
    input  wire                                  clk,
    input  wire                                  rst,
    input  wire                                  in_valid,
    input  wire signed [CURRENT_BITS-1:0]         current_integer,
    input  wire signed [STATE_BITS-1:0]           previous_membrane_integer,
    input  wire [FRACTION_BITS_WIDTH-1:0]         fraction_bits,
    output reg                                   out_valid,
    output reg  [SPIKE_BITS-1:0]                 spike_integer,
    output reg  signed [STATE_BITS-1:0]          membrane_integer,
    output reg                                   range_error
);
    localparam integer WORK_BITS = CURRENT_BITS + 2;
    localparam signed [WORK_BITS-1:0] STATE_MINIMUM =
        -(1 <<< (STATE_BITS-1));
    localparam signed [WORK_BITS-1:0] STATE_MAXIMUM =
        (1 <<< (STATE_BITS-1)) - 1;
    localparam [SPIKE_BITS-1:0] SPIKE_MAXIMUM = {SPIKE_BITS{1'b1}};
    localparam [FRACTION_BITS_WIDTH-1:0] MAX_FRACTION_BITS =
        FRACTION_BITS_WIDTH'(STATE_BITS - 1);

    wire signed [WORK_BITS-1:0] previous_extended = {
        {(WORK_BITS-STATE_BITS){previous_membrane_integer[STATE_BITS-1]}},
        previous_membrane_integer
    };
    wire signed [WORK_BITS-1:0] current_extended = {
        {(WORK_BITS-CURRENT_BITS){current_integer[CURRENT_BITS-1]}},
        current_integer
    };
    wire signed [WORK_BITS-1:0] leak_numerator =
        (previous_extended <<< 1) + previous_extended;
    wire signed [WORK_BITS-1:0] leaked_membrane = leak_numerator >>> 2;
    wire signed [WORK_BITS-1:0] pre_reset = leaked_membrane + current_extended;
    wire signed [WORK_BITS-1:0] threshold_integer =
        $signed({{(WORK_BITS-1){1'b0}}, 1'b1}) <<< fraction_bits;
    wire signed [WORK_BITS-1:0] unsaturated_spike =
        (pre_reset > 0) ? (pre_reset >>> fraction_bits) : 0;
    wire spike_overflow = unsaturated_spike > $signed({
        {(WORK_BITS-SPIKE_BITS){1'b0}}, SPIKE_MAXIMUM
    });
    wire [SPIKE_BITS-1:0] clamped_spike = spike_overflow
        ? SPIKE_MAXIMUM
        : unsaturated_spike[SPIKE_BITS-1:0];
    wire signed [WORK_BITS-1:0] reset_amount =
        $signed({{(WORK_BITS-SPIKE_BITS){1'b0}}, clamped_spike})
        <<< fraction_bits;
    wire signed [WORK_BITS-1:0] membrane_before_floor =
        pre_reset - reset_amount;
    wire signed [WORK_BITS-1:0] membrane_after_floor =
        (membrane_before_floor < -threshold_integer)
        ? -threshold_integer
        : membrane_before_floor;
    wire fraction_error = fraction_bits > MAX_FRACTION_BITS;
    wire state_overflow =
        (membrane_after_floor < STATE_MINIMUM)
        || (membrane_after_floor > STATE_MAXIMUM);
    wire signed [STATE_BITS-1:0] clamped_membrane =
        (membrane_after_floor < STATE_MINIMUM)
        ? STATE_MINIMUM[STATE_BITS-1:0]
        : ((membrane_after_floor > STATE_MAXIMUM)
            ? STATE_MAXIMUM[STATE_BITS-1:0]
            : membrane_after_floor[STATE_BITS-1:0]);

    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            spike_integer <= {SPIKE_BITS{1'b0}};
            membrane_integer <= {STATE_BITS{1'b0}};
            range_error <= 1'b0;
        end else begin
            out_valid <= in_valid;
            if (in_valid) begin
                spike_integer <= clamped_spike;
                membrane_integer <= clamped_membrane;
                range_error <= fraction_error || spike_overflow || state_overflow;
            end
        end
    end

    initial begin
        if (CURRENT_BITS < STATE_BITS || SPIKE_BITS <= 0) begin
            $error("invalid LIF width parameters");
        end
    end
endmodule

`default_nettype wire
