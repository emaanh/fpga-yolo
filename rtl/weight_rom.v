// Async-read ROM initialised from a $readmemh file. One per layer.
`include "include.vh"

module weight_rom #(
    parameter DEPTH    = 256,
    parameter ADDRW    = 8,
    parameter DWIDTH   = `DATA_WIDTH,
    parameter INIT_FILE = ""
)(
    input  [ADDRW-1:0]      addr,
    output [DWIDTH-1:0]     dout
);
    reg [DWIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "")
            $readmemh(INIT_FILE, mem);
    end

    assign dout = mem[addr];
endmodule
