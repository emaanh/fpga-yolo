// Convert a 32-bit Q8.24 accumulator value back to a 16-bit Q4.12 word,
// saturating at the Q4.12 range so wrap-around never produces garbage.
//
// Acc layout (signed, 32-bit): [sign | 7 int | 24 frac]
// Take bits [27:12] of the accumulator -> Q4.12.
// If any of the discarded high integer bits differ from the sign bit,
// the value is out of range and we clamp.
`include "include.vh"

module saturate #(
    parameter IN_W   = `ACC_WIDTH,
    parameter OUT_W  = `DATA_WIDTH,
    parameter SHIFT  = `FRAC_BITS               // bits to drop from the bottom
)(
    input  signed [IN_W-1:0]   in,
    output signed [OUT_W-1:0]  out
);
    // The candidate output is in[SHIFT + OUT_W - 1 : SHIFT].
    wire signed [OUT_W-1:0]  candidate = in[SHIFT + OUT_W - 1 : SHIFT];
    // Discarded high bits: in[IN_W-1 : SHIFT + OUT_W - 1] (overlap on sign).
    // If they aren't all sign-extension of `candidate`, we overflow.
    wire signed [IN_W - (SHIFT + OUT_W - 1) - 1:0] hi = in[IN_W-1 : SHIFT + OUT_W - 1];
    wire sign       = in[IN_W-1];
    wire all_sign   = &hi      | ~|hi;          // all-1 (neg) OR all-0 (pos)
    wire overflow   = ~all_sign;

    assign out = overflow ? (sign ? {1'b1, {OUT_W-1{1'b0}}}      // most-negative
                                  : {1'b0, {OUT_W-1{1'b1}}})    // most-positive
                          : candidate;
endmodule
