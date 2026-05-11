// Feature-map memory. Two of these ping-pong between layers.
// Synchronous write, asynchronous (combinational) read - i.e. distributed
// RAM. This keeps the conv FSM single-cycle per MAC. Swapping to BRAM
// for synthesis just adds a pipeline register on rdata.
`include "include.vh"

module fmap_mem #(
    parameter DEPTH  = `FMAP_DEPTH,
    parameter ADDRW  = `FMAP_ADDRW,
    parameter DWIDTH = `DATA_WIDTH
)(
    input                       clk,
    // write port
    input                       we,
    input      [ADDRW-1:0]      waddr,
    input      [DWIDTH-1:0]     wdata,
    // read port (combinational)
    input      [ADDRW-1:0]      raddr,
    output     [DWIDTH-1:0]     rdata
);
    reg [DWIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
    end

    assign rdata = mem[raddr];
endmodule
