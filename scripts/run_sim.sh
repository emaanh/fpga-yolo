#!/usr/bin/env bash
# End-to-end demo: generate data + weights, compile RTL, simulate one
# frame, decode + run NMS, and render the prediction as a PNG.
#
# Usage:
#   ./scripts/run_sim.sh                       # frame 0, random weights
#   ./scripts/run_sim.sh -f 12                 # frame 12
#   ./scripts/run_sim.sh -f 12 --trained       # uses data/model.pt + quantize
#   ./scripts/run_sim.sh --regen-data 5000     # regenerate the dataset first
#   ./scripts/run_sim.sh --vcd                 # also dump build/tb_top.vcd
set -euo pipefail

# Run from the project root regardless of caller cwd.
cd "$(dirname "$0")/.."

FRAME_ID=0
TRAINED=0
REGEN_DATA=""
DUMP_VCD=0
MAX_CYCLES=5000000

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--frame)        FRAME_ID="$2"; shift 2 ;;
    --trained)         TRAINED=1; shift ;;
    --regen-data)      REGEN_DATA="$2"; shift 2 ;;
    --vcd)             DUMP_VCD=1; shift ;;
    --max-cycles)      MAX_CYCLES="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done

stem=$(printf "frame_%04d" "$FRAME_ID")
echo "=== tiny_yolo end-to-end :: frame $stem ==="

# ---------- 1. ensure data + weights exist --------------------------------
if [[ -n "$REGEN_DATA" || ! -f "data/frames/${stem}.hex" ]]; then
  N="${REGEN_DATA:-200}"
  echo "[1/5] Generating $N frames..."
  python3 python/gen_dataset.py -n "$N"
fi

if [[ ! -f "data/sigmoid/sigmoid_lut.mif" ]]; then
  echo "[1b ] Generating sigmoid LUT..."
  python3 python/gen_sigmoid_lut.py
fi

if [[ "$TRAINED" == "1" ]]; then
  echo "[2/5] Quantizing trained model..."
  python3 python/quantize.py
elif [[ ! -f "data/weights/w_conv1.mif" ]]; then
  echo "[2/5] No weights yet -- writing random Q4.12 weights..."
  python3 python/random_weights.py
fi

# ---------- 2. compile RTL ------------------------------------------------
mkdir -p build data/detections
echo "[3/5] Compiling RTL with iverilog..."
iverilog -g2012 -Irtl -o build/tiny_yolo.vvp rtl/*.v tb/tb_top.v

# ---------- 3. simulate ---------------------------------------------------
out_cells="data/detections/cells_$(printf %04d "$FRAME_ID").txt"
echo "[4/5] Simulating $stem -> $out_cells ..."
vvp build/tiny_yolo.vvp \
    +FRAME="data/frames/${stem}.hex" \
    +OUT="$out_cells" \
    +DUMP_VCD="$DUMP_VCD" \
    +MAX_CYCLES="$MAX_CYCLES" \
    | tee "build/sim_${stem}.log"

# ---------- 4. decode + visualise -----------------------------------------
echo "[5/5] Decoding + NMS + visualizing..."
python3 python/decode_nms.py --frame "$stem"
python3 python/visualize.py  --frame "$stem"

echo
echo "Done. Outputs:"
echo "  raw cells       : $out_cells"
echo "  detections      : data/detections/${stem}.txt"
echo "  rendered PNG    : data/detections/${stem}_vis.png"
