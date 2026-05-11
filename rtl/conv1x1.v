// 1x1 convolution layer (the detection head).
// No spatial neighborhood; for each output channel we just sum over the
// input channels at the same (y,x) position. No activation - the head
// emits raw logits; sigmoid happens in the decoder.
//
// Weight layout: co * CIN, addr = co*CIN + ci
`include "include.vh"

module conv1x1 #(
    parameter H           = 4,
    parameter W           = 4,
    parameter CIN         = 32,
    parameter COUT        = 7,
    parameter WEIGHT_FILE = "",
    parameter BIAS_FILE   = "",
    parameter ADDRW       = `FMAP_ADDRW,
    parameter DWIDTH      = `DATA_WIDTH
)(
    input                                clk,
    input                                rst,
    input                                start,
    output reg                           done,

    output     [ADDRW-1:0]               in_addr,
    input  signed [DWIDTH-1:0]           in_data,

    output reg                           w_en,
    output reg [ADDRW-1:0]               out_addr,
    output reg signed [DWIDTH-1:0]       out_data
);
    localparam W_DEPTH = COUT * CIN;
    localparam W_ADDRW = $clog2(W_DEPTH);
    localparam B_DEPTH = COUT;
    localparam B_ADDRW = (COUT == 1) ? 1 : $clog2(B_DEPTH);

    wire [W_ADDRW-1:0]        w_addr;
    wire signed [DWIDTH-1:0]  w_data;
    wire [B_ADDRW-1:0]        b_addr;
    wire signed [DWIDTH-1:0]  b_data;

    weight_rom #(.DEPTH(W_DEPTH), .ADDRW(W_ADDRW),
                 .DWIDTH(DWIDTH), .INIT_FILE(WEIGHT_FILE))
        wrom (.addr(w_addr), .dout(w_data));
    weight_rom #(.DEPTH(B_DEPTH), .ADDRW(B_ADDRW),
                 .DWIDTH(DWIDTH), .INIT_FILE(BIAS_FILE))
        brom (.addr(b_addr), .dout(b_data));

    localparam CO_W = (COUT == 1) ? 1 : $clog2(COUT);
    localparam OY_W = (H == 1)    ? 1 : $clog2(H);
    localparam OX_W = (W == 1)    ? 1 : $clog2(W);
    localparam CI_W = (CIN == 1)  ? 1 : $clog2(CIN);

    reg [CO_W:0] co_idx;
    reg [OY_W:0] oy;
    reg [OX_W:0] ox;
    reg [CI_W:0] ci;

    assign in_addr = ci * (H*W) + oy * W + ox;
    assign w_addr  = co_idx * CIN + ci;
    assign b_addr  = co_idx[B_ADDRW-1:0];

    wire signed [2*DWIDTH-1:0]    product = $signed(in_data) * $signed(w_data);
    wire signed [`ACC_WIDTH-1:0]  product_ext =
        {{(`ACC_WIDTH-2*DWIDTH){product[2*DWIDTH-1]}}, product};

    reg signed [`ACC_WIDTH-1:0] acc;
    wire signed [`ACC_WIDTH-1:0] bias_aligned =
        $signed({{(`ACC_WIDTH-DWIDTH-`FRAC_BITS){b_data[DWIDTH-1]}}, b_data, {`FRAC_BITS{1'b0}}});

    wire signed [DWIDTH-1:0] sat_out;
    saturate sat (.in(acc), .out(sat_out));

    localparam S_IDLE = 3'd0,
               S_INIT = 3'd1,
               S_MAC  = 3'd2,
               S_FIN  = 3'd3,
               S_DONE = 3'd4;
    reg [2:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            done  <= 1'b0;
            w_en  <= 1'b0;
        end else begin
            w_en <= 1'b0;
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    co_idx <= 0; oy <= 0; ox <= 0; ci <= 0;
                    state  <= S_INIT;
                end
                S_INIT: begin
                    acc   <= bias_aligned;
                    state <= S_MAC;
                end
                S_MAC: begin
                    acc <= acc + product_ext;
                    if (ci == CIN - 1) begin
                        ci    <= 0;
                        state <= S_FIN;
                    end else
                        ci <= ci + 1'b1;
                end
                S_FIN: begin
                    w_en     <= 1'b1;
                    out_addr <= co_idx * (H*W) + oy * W + ox;
                    out_data <= sat_out;
                    if (ox == W - 1) begin
                        ox <= 0;
                        if (oy == H - 1) begin
                            oy <= 0;
                            if (co_idx == COUT - 1) begin
                                state <= S_DONE;
                            end else begin
                                co_idx <= co_idx + 1'b1;
                                state  <= S_INIT;
                            end
                        end else begin
                            oy    <= oy + 1'b1;
                            state <= S_INIT;
                        end
                    end else begin
                        ox    <= ox + 1'b1;
                        state <= S_INIT;
                    end
                end
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
