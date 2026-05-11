// Combinational leaky-ReLU on a Q4.12 input.
// Negative inputs are arithmetically shifted right by LEAKY_SHIFT (slope = 1/8).
`include "include.vh"

module leaky_relu #(
    parameter DWIDTH = `DATA_WIDTH,
    parameter SHIFT  = `LEAKY_SHIFT
)(
    input  signed [DWIDTH-1:0] in,
    output signed [DWIDTH-1:0] out
);
    assign out = in[DWIDTH-1] ? (in >>> SHIFT) : in;
endmodule
