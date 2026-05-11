// Walks an entire feature map and rewrites each element through the
// sigmoid LUT in place.
//
// Used after the detection head to turn raw (tx, ty, tw, th, obj, cls...)
// logits into Q4.12 unsigned values in [0,1) before they get dumped to
// the testbench.
`include "include.vh"

module sigmoid_pass #(
    parameter SIZE   = 4*4*7,                  // total elements
    parameter ADDRW  = `FMAP_ADDRW,
    parameter DWIDTH = `DATA_WIDTH
)(
    input                          clk,
    input                          rst,
    input                          start,
    output reg                     done,

    output reg [ADDRW-1:0]         r_addr,
    input  signed [DWIDTH-1:0]     r_data,

    output reg                     w_en,
    output reg [ADDRW-1:0]         w_addr,
    output reg [DWIDTH-1:0]        w_data
);
    wire [DWIDTH-1:0] sig_out;
    sigmoid_lut sig (.in(r_data), .out(sig_out));

    reg [ADDRW-1:0] cnt;
    reg             running;

    // Read address tracks the counter (combinational on cnt)
    always @(*) r_addr = cnt;

    always @(posedge clk) begin
        if (rst) begin
            running <= 0;
            done    <= 0;
            w_en    <= 0;
            cnt     <= 0;
        end else begin
            w_en <= 0;
            done <= 0;
            if (start && !running) begin
                running <= 1;
                cnt     <= 0;
            end else if (running) begin
                w_en   <= 1;
                w_addr <= cnt;
                w_data <= sig_out;
                if (cnt == SIZE - 1) begin
                    running <= 0;
                    done    <= 1;
                end else
                    cnt <= cnt + 1'b1;
            end
        end
    end
endmodule
