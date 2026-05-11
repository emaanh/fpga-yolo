"""Single source of truth for network architecture + fixed-point format.

Mirrored in rtl/include.vh. If you change something here, regenerate weights
and update the RTL header to match.
"""

# Image / grid
IMG_SIZE   = 32          # 32x32 RGB input
IMG_CHANS  = 3
GRID       = 4           # 4x4 detection grid (stride 8)
STRIDE     = IMG_SIZE // GRID

# Backbone channel widths (input chans of each layer's *next* feature map)
CHANS      = [IMG_CHANS, 8, 16, 32]   # in -> b1 -> b2 -> b3

# Classes
CLASS_NAMES = ["circle", "square"]
NUM_CLASSES = len(CLASS_NAMES)

# Detection head: per cell predicts (tx, ty, tw, th, obj, p_cls0, p_cls1...)
PRED_PER_CELL = 5 + NUM_CLASSES

# Fixed-point: Q4.12 signed (1 sign + 3 int + 12 frac)
DATA_WIDTH = 16
FRAC_BITS  = 12
INT_BITS   = DATA_WIDTH - 1 - FRAC_BITS          # = 3
Q_SCALE    = 1 << FRAC_BITS                      # = 4096
Q_MAX_VAL  = (1 << (DATA_WIDTH - 1)) - 1         # 32767  -> ~+7.9998
Q_MIN_VAL  = -(1 << (DATA_WIDTH - 1))            # -32768 -> -8.0

# Accumulator width (used by conv MAC chain)
ACC_WIDTH = 32

# LeakyReLU slope (must be power-of-2 reciprocal so RTL can use a shift)
LEAKY_SHIFT = 3                                  # slope = 1/8 = 0.125
LEAKY_SLOPE = 1.0 / (1 << LEAKY_SHIFT)

# Sigmoid LUT depth (must match rtl/sigmoid_lut.v)
SIG_LUT_BITS = 10                                # 1024 entries

# NMS
NMS_IOU_THRESH = 0.4
OBJ_THRESH     = 0.4
