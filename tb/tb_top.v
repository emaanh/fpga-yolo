// Top-level testbench: load a .hex frame into fmap A, run inference,
// then read back the 4x4x7 sigmoided detection-head output and dump it
// to a text file for Python post-processing (NMS + visualization).
//
// Plusarg controls:
//   +FRAME=<path>    .hex file with 3072 lines of Q4.12 input (CHW planar)
//   +OUT=<path>      destination for raw decoded cells (one Q4.12 hex per line)
//   +DUMP_VCD=1      enable VCD dump
//   +MAX_CYCLES=N    safety timeout (default 5,000,000)
`timescale 1ns/1ps
`include "include.vh"

module tb_top;
    localparam IN_SIZE  = `IMG_CHANS * `IMG_SIZE * `IMG_SIZE;   // 3072
    localparam OUT_SIZE = `C_OUTHEAD * `H3P    * `H3P;          // 112

    reg                        clk = 0;
    reg                        rst = 1;
    reg                        start = 0;
    wire                       done;
    wire [3:0]                 stage_dbg;

    reg                        ext_we    = 0;
    reg  [`FMAP_ADDRW-1:0]     ext_waddr = 0;
    reg  [`DATA_WIDTH-1:0]     ext_wdata = 0;
    reg  [`FMAP_ADDRW-1:0]     ext_raddr = 0;
    wire [`DATA_WIDTH-1:0]     ext_rdata;

    tiny_yolo_top dut (
        .clk(clk), .rst(rst), .start(start), .done(done),
        .stage_dbg(stage_dbg),
        .ext_we(ext_we), .ext_waddr(ext_waddr), .ext_wdata(ext_wdata),
        .ext_raddr(ext_raddr), .ext_rdata(ext_rdata)
    );

    always #5 clk = ~clk;                                    // 100 MHz

    // ------ cycle counter ----------------------------------------------
    integer cyc;
    initial cyc = 0;
    always @(posedge clk) if (!rst) cyc <= cyc + 1;

    // ------ stage transition logging -----------------------------------
    reg [3:0] last_stage;
    initial last_stage = 4'hF;
    always @(posedge clk) begin
        if (stage_dbg != last_stage) begin
            $display("[tb] cycle %7d  stage %0d -> %0d", cyc, last_stage, stage_dbg);
            last_stage <= stage_dbg;
        end
    end

    // ------ dump task --------------------------------------------------
    integer fd, i;
    task dump_cells(input [8*256-1:0] path);
        begin
            fd = $fopen(path, "w");
            if (fd == 0) begin
                $display("[tb] ERROR: cannot open %0s", path);
                $finish;
            end
            $fwrite(fd, "# %0d rows, layout c*H*W + gy*W + gx ",  OUT_SIZE);
            $fwrite(fd, "(c=0..6 -> tx,ty,tw,th,obj,cls0,cls1; H=W=%0d)\n", `H3P);
            for (i = 0; i < OUT_SIZE; i = i + 1) begin
                ext_raddr = i[`FMAP_ADDRW-1:0];
                #1;
                $fwrite(fd, "%04x\n", ext_rdata);
            end
            $fclose(fd);
        end
    endtask

    // ------ main stimulus ----------------------------------------------
    reg [`DATA_WIDTH-1:0] frame_buf [0:IN_SIZE-1];
    reg [8*256-1:0] frame_path;
    reg [8*256-1:0] out_path;
    reg [31:0]      dump_vcd;
    reg [31:0]      max_cycles;

    initial begin
        if (!$value$plusargs("FRAME=%s", frame_path))
            frame_path = "data/frames/frame_0000.hex";
        if (!$value$plusargs("OUT=%s", out_path))
            out_path = "data/detections/cells_0000.txt";
        dump_vcd = 0;
        $value$plusargs("DUMP_VCD=%d", dump_vcd);
        max_cycles = 32'd5_000_000;
        $value$plusargs("MAX_CYCLES=%d", max_cycles);

        if (dump_vcd) begin
            $dumpfile("build/tb_top.vcd");
            $dumpvars(0, tb_top);
        end

        $display("[tb] Loading frame: %0s", frame_path);
        $readmemh(frame_path, frame_buf);

        // Reset
        rst = 1;
        repeat (5) @(posedge clk);
        rst = 0;
        @(posedge clk);

        // Stream the frame into fmap A through the external write port.
        $display("[tb] Writing %0d words to fmap A...", IN_SIZE);
        for (i = 0; i < IN_SIZE; i = i + 1) begin
            @(negedge clk);
            ext_we    = 1;
            ext_waddr = i[`FMAP_ADDRW-1:0];
            ext_wdata = frame_buf[i];
        end
        @(negedge clk);
        ext_we = 0;

        // Kick off inference.
        @(posedge clk);
        start = 1;
        @(posedge clk);
        start = 0;
        $display("[tb] Inference started at cycle %0d", cyc);

        // Wait for done (with timeout).
        while (!done && cyc < max_cycles) @(posedge clk);
        if (!done) begin
            $display("[tb] TIMEOUT at cycle %0d (stage=%0d)", cyc, stage_dbg);
            $finish;
        end
        $display("[tb] Inference done at cycle %0d", cyc);

        dump_cells(out_path);
        $display("[tb] Wrote decoded cells to %0s", out_path);
        $finish;
    end
endmodule
