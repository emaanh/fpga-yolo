// 2x2 max-pool, stride 2.  Channel-major iteration over a CHW planar fmap.
// Combinational-read source means we read one position per cycle and use
// 4 cycles to produce one output pixel.
//
// Input  layout: c * H * W,         addr = c*H*W + y*W + x
// Output layout: c * (H/2)*(W/2),   addr = c*(H/2)*(W/2) + oy*(W/2) + ox
`include "include.vh"

module max_pool2x2 #(
    parameter H      = 32,
    parameter W      = 32,
    parameter C      = 8,
    parameter ADDRW  = `FMAP_ADDRW,
    parameter DWIDTH = `DATA_WIDTH
)(
    input                              clk,
    input                              rst,
    input                              start,
    output reg                         done,

    output reg [ADDRW-1:0]             r_addr,
    input      signed [DWIDTH-1:0]     r_data,

    output reg                         w_en,
    output reg [ADDRW-1:0]             w_addr,
    output reg signed [DWIDTH-1:0]     w_data
);
    localparam OH = H/2;
    localparam OW = W/2;

    reg [$clog2(C):0]   c_idx;
    reg [$clog2(OH):0]  oy;
    reg [$clog2(OW):0]  ox;
    reg [1:0]           phase;
    reg                 running;
    reg signed [DWIDTH-1:0] cur_max;

    function signed [DWIDTH-1:0] smax2;
        input signed [DWIDTH-1:0] a;
        input signed [DWIDTH-1:0] b;
        begin smax2 = (a > b) ? a : b; end
    endfunction

    // Read address (combinational on phase + position).
    wire [ADDRW-1:0] base = c_idx * (H*W) + (2*oy)*W + (2*ox);
    always @(*) begin
        case (phase)
            2'd0: r_addr = base;
            2'd1: r_addr = base + 1;
            2'd2: r_addr = base + W;
            2'd3: r_addr = base + W + 1;
        endcase
    end

    always @(posedge clk) begin
        if (rst) begin
            running <= 0;
            done    <= 0;
            w_en    <= 0;
            c_idx   <= 0;
            oy      <= 0;
            ox      <= 0;
            phase   <= 0;
        end else begin
            w_en <= 0;
            done <= 0;
            if (start && !running) begin
                running <= 1;
                c_idx   <= 0;
                oy      <= 0;
                ox      <= 0;
                phase   <= 0;
            end else if (running) begin
                case (phase)
                    2'd0: cur_max <= r_data;
                    2'd1: cur_max <= smax2(cur_max, r_data);
                    2'd2: cur_max <= smax2(cur_max, r_data);
                    2'd3: begin
                        w_en   <= 1;
                        w_addr <= c_idx * (OH*OW) + oy*OW + ox;
                        w_data <= smax2(cur_max, r_data);
                    end
                endcase

                if (phase == 2'd3) begin
                    if (ox == OW-1) begin
                        ox <= 0;
                        if (oy == OH-1) begin
                            oy <= 0;
                            if (c_idx == C-1) begin
                                running <= 0;
                                done    <= 1;
                            end else
                                c_idx <= c_idx + 1;
                        end else
                            oy <= oy + 1;
                    end else
                        ox <= ox + 1;
                    phase <= 0;
                end else begin
                    phase <= phase + 1;
                end
            end
        end
    end
endmodule
