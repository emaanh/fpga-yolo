// Top-level integrating the three conv blocks, three pools, the 1x1 head
// and an in-place sigmoid pass over the head output. Two feature-map
// memories (A and B) ping-pong between layers:
//
//   load (ext)  -> A
//   conv1 :  A -> B
//   pool1 :  B -> A
//   conv2 :  A -> B
//   pool2 :  B -> A
//   conv3 :  A -> B
//   pool3 :  B -> A
//   head  :  A -> B
//   sigm  :  B -> A          (testbench then reads decoded cells from A)
//
// The testbench writes the frame into A through the ext_* write port
// before pulsing `start`, and reads decoded cells from A once `done`
// rises.
`include "include.vh"

module tiny_yolo_top (
    input                          clk,
    input                          rst,
    input                          start,
    output                         done,
    output reg [3:0]               stage_dbg,   // for $display in tb

    // External access to fmap A
    input                          ext_we,
    input  [`FMAP_ADDRW-1:0]       ext_waddr,
    input  [`DATA_WIDTH-1:0]       ext_wdata,
    input  [`FMAP_ADDRW-1:0]       ext_raddr,
    output [`DATA_WIDTH-1:0]       ext_rdata
);
    // -------- fmap memories --------------------------------------------
    wire                       a_we, b_we;
    wire [`FMAP_ADDRW-1:0]     a_waddr, b_waddr;
    wire [`DATA_WIDTH-1:0]     a_wdata, b_wdata;
    wire [`FMAP_ADDRW-1:0]     a_raddr, b_raddr;
    wire [`DATA_WIDTH-1:0]     a_rdata, b_rdata;

    fmap_mem fmapA (.clk(clk),
        .we(a_we), .waddr(a_waddr), .wdata(a_wdata),
        .raddr(a_raddr), .rdata(a_rdata));
    fmap_mem fmapB (.clk(clk),
        .we(b_we), .waddr(b_waddr), .wdata(b_wdata),
        .raddr(b_raddr), .rdata(b_rdata));

    // -------- compute modules ------------------------------------------
    // conv1: A -> B,  3->8, 32x32
    wire                       c1_start, c1_done, c1_we;
    wire [`FMAP_ADDRW-1:0]     c1_raddr, c1_waddr;
    wire [`DATA_WIDTH-1:0]     c1_wdata;
    conv3x3 #(.H(`IMG_SIZE), .W(`IMG_SIZE), .CIN(`C_IN1), .COUT(`C_OUT1),
              .WEIGHT_FILE(`W_CONV1_FILE), .BIAS_FILE(`B_CONV1_FILE))
        c1 (.clk(clk), .rst(rst), .start(c1_start), .done(c1_done),
            .in_addr(c1_raddr), .in_data(a_rdata),
            .w_en(c1_we), .out_addr(c1_waddr), .out_data(c1_wdata));

    // pool1: B -> A, 32x32x8 -> 16x16x8
    wire                       p1_start, p1_done, p1_we;
    wire [`FMAP_ADDRW-1:0]     p1_raddr, p1_waddr;
    wire [`DATA_WIDTH-1:0]     p1_wdata;
    max_pool2x2 #(.H(`H1), .W(`H1), .C(`C_OUT1))
        p1 (.clk(clk), .rst(rst), .start(p1_start), .done(p1_done),
            .r_addr(p1_raddr), .r_data(b_rdata),
            .w_en(p1_we), .w_addr(p1_waddr), .w_data(p1_wdata));

    // conv2: A -> B,  8->16, 16x16
    wire                       c2_start, c2_done, c2_we;
    wire [`FMAP_ADDRW-1:0]     c2_raddr, c2_waddr;
    wire [`DATA_WIDTH-1:0]     c2_wdata;
    conv3x3 #(.H(`H1P), .W(`H1P), .CIN(`C_IN2), .COUT(`C_OUT2),
              .WEIGHT_FILE(`W_CONV2_FILE), .BIAS_FILE(`B_CONV2_FILE))
        c2 (.clk(clk), .rst(rst), .start(c2_start), .done(c2_done),
            .in_addr(c2_raddr), .in_data(a_rdata),
            .w_en(c2_we), .out_addr(c2_waddr), .out_data(c2_wdata));

    // pool2: B -> A, 16x16x16 -> 8x8x16
    wire                       p2_start, p2_done, p2_we;
    wire [`FMAP_ADDRW-1:0]     p2_raddr, p2_waddr;
    wire [`DATA_WIDTH-1:0]     p2_wdata;
    max_pool2x2 #(.H(`H2), .W(`H2), .C(`C_OUT2))
        p2 (.clk(clk), .rst(rst), .start(p2_start), .done(p2_done),
            .r_addr(p2_raddr), .r_data(b_rdata),
            .w_en(p2_we), .w_addr(p2_waddr), .w_data(p2_wdata));

    // conv3: A -> B,  16->32, 8x8
    wire                       c3_start, c3_done, c3_we;
    wire [`FMAP_ADDRW-1:0]     c3_raddr, c3_waddr;
    wire [`DATA_WIDTH-1:0]     c3_wdata;
    conv3x3 #(.H(`H2P), .W(`H2P), .CIN(`C_IN3), .COUT(`C_OUT3),
              .WEIGHT_FILE(`W_CONV3_FILE), .BIAS_FILE(`B_CONV3_FILE))
        c3 (.clk(clk), .rst(rst), .start(c3_start), .done(c3_done),
            .in_addr(c3_raddr), .in_data(a_rdata),
            .w_en(c3_we), .out_addr(c3_waddr), .out_data(c3_wdata));

    // pool3: B -> A, 8x8x32 -> 4x4x32
    wire                       p3_start, p3_done, p3_we;
    wire [`FMAP_ADDRW-1:0]     p3_raddr, p3_waddr;
    wire [`DATA_WIDTH-1:0]     p3_wdata;
    max_pool2x2 #(.H(`H3), .W(`H3), .C(`C_OUT3))
        p3 (.clk(clk), .rst(rst), .start(p3_start), .done(p3_done),
            .r_addr(p3_raddr), .r_data(b_rdata),
            .w_en(p3_we), .w_addr(p3_waddr), .w_data(p3_wdata));

    // head: A -> B, 32->7, 4x4 (1x1 conv)
    wire                       h_start, h_done, h_we;
    wire [`FMAP_ADDRW-1:0]     h_raddr, h_waddr;
    wire [`DATA_WIDTH-1:0]     h_wdata;
    conv1x1 #(.H(`H3P), .W(`H3P), .CIN(`C_INHEAD), .COUT(`C_OUTHEAD),
              .WEIGHT_FILE(`W_HEAD_FILE), .BIAS_FILE(`B_HEAD_FILE))
        head (.clk(clk), .rst(rst), .start(h_start), .done(h_done),
              .in_addr(h_raddr), .in_data(a_rdata),
              .w_en(h_we), .out_addr(h_waddr), .out_data(h_wdata));

    // sigmoid pass: B -> A, 4*4*7 values
    wire                       s_start, s_done, s_we;
    wire [`FMAP_ADDRW-1:0]     s_raddr, s_waddr;
    wire [`DATA_WIDTH-1:0]     s_wdata;
    sigmoid_pass #(.SIZE(`H3P * `H3P * `C_OUTHEAD))
        sigm (.clk(clk), .rst(rst), .start(s_start), .done(s_done),
              .r_addr(s_raddr), .r_data(b_rdata),
              .w_en(s_we), .w_addr(s_waddr), .w_data(s_wdata));

    // -------- stage FSM ------------------------------------------------
    localparam S_IDLE = 4'd0,
               S_C1   = 4'd1, S_P1 = 4'd2,
               S_C2   = 4'd3, S_P2 = 4'd4,
               S_C3   = 4'd5, S_P3 = 4'd6,
               S_HEAD = 4'd7, S_SIG = 4'd8,
               S_DONE = 4'd9;
    reg [3:0] state;
    reg done_r;
    assign done = done_r;

    // one-cycle start pulses
    reg c1s, p1s, c2s, p2s, c3s, p3s, hs, ss;

    always @(posedge clk) begin
        if (rst) begin
            state  <= S_IDLE;
            done_r <= 0;
            c1s <= 0; p1s <= 0; c2s <= 0; p2s <= 0;
            c3s <= 0; p3s <= 0; hs  <= 0; ss  <= 0;
        end else begin
            c1s <= 0; p1s <= 0; c2s <= 0; p2s <= 0;
            c3s <= 0; p3s <= 0; hs  <= 0; ss  <= 0;
            done_r <= 0;
            case (state)
                S_IDLE: if (start) begin c1s <= 1; state <= S_C1; end
                S_C1:   if (c1_done) begin p1s <= 1; state <= S_P1; end
                S_P1:   if (p1_done) begin c2s <= 1; state <= S_C2; end
                S_C2:   if (c2_done) begin p2s <= 1; state <= S_P2; end
                S_P2:   if (p2_done) begin c3s <= 1; state <= S_C3; end
                S_C3:   if (c3_done) begin p3s <= 1; state <= S_P3; end
                S_P3:   if (p3_done) begin hs  <= 1; state <= S_HEAD; end
                S_HEAD: if (h_done)  begin ss  <= 1; state <= S_SIG; end
                S_SIG:  if (s_done)  begin done_r <= 1; state <= S_DONE; end
                S_DONE: state <= S_IDLE;
            endcase
        end
    end
    always @(*) stage_dbg = state;

    assign c1_start = c1s;
    assign p1_start = p1s;
    assign c2_start = c2s;
    assign p2_start = p2s;
    assign c3_start = c3s;
    assign p3_start = p3s;
    assign h_start  = hs;
    assign s_start  = ss;

    // -------- fmap A muxing --------------------------------------------
    // A is written by: ext (load), pool1, pool2, pool3, sigmoid_pass
    // A is read by:    ext (dump), conv1, conv2, conv3, head
    assign a_we    = ext_we | (state == S_P1 & p1_we)
                            | (state == S_P2 & p2_we)
                            | (state == S_P3 & p3_we)
                            | (state == S_SIG & s_we);
    assign a_waddr = ext_we              ? ext_waddr :
                     (state == S_P1)     ? p1_waddr  :
                     (state == S_P2)     ? p2_waddr  :
                     (state == S_P3)     ? p3_waddr  :
                     (state == S_SIG)    ? s_waddr   : {`FMAP_ADDRW{1'b0}};
    assign a_wdata = ext_we              ? ext_wdata :
                     (state == S_P1)     ? p1_wdata  :
                     (state == S_P2)     ? p2_wdata  :
                     (state == S_P3)     ? p3_wdata  :
                     (state == S_SIG)    ? s_wdata   : {`DATA_WIDTH{1'b0}};
    assign a_raddr = (state == S_C1)     ? c1_raddr  :
                     (state == S_C2)     ? c2_raddr  :
                     (state == S_C3)     ? c3_raddr  :
                     (state == S_HEAD)   ? h_raddr   : ext_raddr;
    assign ext_rdata = a_rdata;

    // -------- fmap B muxing --------------------------------------------
    // B is written by: conv1, conv2, conv3, head
    // B is read by:    pool1, pool2, pool3, sigmoid_pass
    assign b_we    = (state == S_C1   & c1_we)
                   | (state == S_C2   & c2_we)
                   | (state == S_C3   & c3_we)
                   | (state == S_HEAD & h_we);
    assign b_waddr = (state == S_C1)   ? c1_waddr :
                     (state == S_C2)   ? c2_waddr :
                     (state == S_C3)   ? c3_waddr :
                     (state == S_HEAD) ? h_waddr  : {`FMAP_ADDRW{1'b0}};
    assign b_wdata = (state == S_C1)   ? c1_wdata :
                     (state == S_C2)   ? c2_wdata :
                     (state == S_C3)   ? c3_wdata :
                     (state == S_HEAD) ? h_wdata  : {`DATA_WIDTH{1'b0}};
    assign b_raddr = (state == S_P1)   ? p1_raddr :
                     (state == S_P2)   ? p2_raddr :
                     (state == S_P3)   ? p3_raddr :
                     (state == S_SIG)  ? s_raddr  : {`FMAP_ADDRW{1'b0}};
endmodule
