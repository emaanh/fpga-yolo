// =====================================================================
// tiny_yolo: global config / single source of truth for shapes + format.
// Mirror of python/config.py - keep them in sync.
// =====================================================================
`ifndef TINY_YOLO_INCLUDE_VH
`define TINY_YOLO_INCLUDE_VH

// ---------------- fixed-point format -----------------------------------
`define DATA_WIDTH      16        // Q4.12 signed (1 sign + 3 int + 12 frac)
`define FRAC_BITS       12
`define ACC_WIDTH       32        // accumulator for MAC chain (Q8.24)
`define ACC_FRAC_BITS   24

// ---------------- image / grid -----------------------------------------
`define IMG_SIZE        32
`define IMG_CHANS       3
`define GRID            4
`define STRIDE          (`IMG_SIZE / `GRID)

// ---------------- backbone channels ------------------------------------
`define C_IN1           3
`define C_OUT1          8
`define C_IN2           8
`define C_OUT2          16
`define C_IN3           16
`define C_OUT3          32

// ---------------- detection head ---------------------------------------
`define NUM_CLASSES     2
`define PRED_PER_CELL   (5 + `NUM_CLASSES)            // 7
`define C_INHEAD        32
`define C_OUTHEAD       `PRED_PER_CELL

// ---------------- feature-map sizes per stage --------------------------
`define H1   `IMG_SIZE          // post-conv1, pre-pool1   (32)
`define H1P  (`H1/2)            // post-pool1              (16)
`define H2   `H1P               // post-conv2, pre-pool2   (16)
`define H2P  (`H2/2)            // post-pool2              ( 8)
`define H3   `H2P               // post-conv3, pre-pool3   ( 8)
`define H3P  (`H3/2)            // post-pool3              ( 4)
`define H4   `H3P               // post-head               ( 4)

// ---------------- feature-map memory -----------------------------------
// Sized to the largest stage. Post-conv1 = 32*32*8 = 8192 elements.
`define FMAP_DEPTH      8192
`define FMAP_ADDRW      13

// ---------------- activations ------------------------------------------
`define LEAKY_SHIFT     3                    // slope = 1/8

// ---------------- sigmoid LUT ------------------------------------------
`define SIG_LUT_BITS    10                   // 1024 entries
`define SIG_LUT_DEPTH   (1 << `SIG_LUT_BITS)

// ---------------- NMS / decoder thresholds (Q4.12) ---------------------
// 0.40 -> 1638, 0.40 -> 1638
`define OBJ_THRESH_Q    16'd1638             // 0.4 in Q4.12
`define IOU_THRESH_Q    16'd1638             // 0.4 in Q4.12

// ---------------- weight / bias file paths -----------------------------
`define W_CONV1_FILE    "data/weights/w_conv1.mif"
`define B_CONV1_FILE    "data/biases/b_conv1.mif"
`define W_CONV2_FILE    "data/weights/w_conv2.mif"
`define B_CONV2_FILE    "data/biases/b_conv2.mif"
`define W_CONV3_FILE    "data/weights/w_conv3.mif"
`define B_CONV3_FILE    "data/biases/b_conv3.mif"
`define W_HEAD_FILE     "data/weights/w_head.mif"
`define B_HEAD_FILE     "data/biases/b_head.mif"
`define SIG_LUT_FILE    "data/sigmoid/sigmoid_lut.mif"

`endif  // TINY_YOLO_INCLUDE_VH
