// 1024-entry combinational sigmoid lookup table.
//
// Input is a Q4.12 signed value clamped (by the producing saturator) to
// the table's effective domain of [-8, +8). The top SIG_LUT_BITS of
// (in + 2^15) - i.e. with the sign bit flipped - form the index.
// Output is Q4.12 unsigned in [0, 1).
`include "include.vh"

module sigmoid_lut #(
    parameter IDX_BITS = `SIG_LUT_BITS,
    parameter DWIDTH   = `DATA_WIDTH
)(
    input  signed [DWIDTH-1:0]   in,
    output        [DWIDTH-1:0]   out
);
    reg [DWIDTH-1:0] lut [0:(1<<IDX_BITS)-1];

    initial begin
        $readmemh(`SIG_LUT_FILE, lut);
    end

    wire [DWIDTH-1:0]    biased = in ^ {1'b1, {DWIDTH-1{1'b0}}};
    wire [IDX_BITS-1:0]  idx    = biased[DWIDTH-1 -: IDX_BITS];

    assign out = lut[idx];
endmodule
