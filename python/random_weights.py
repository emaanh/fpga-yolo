"""Emit small random Q4.12 weight files so the RTL pipeline can be exercised
end-to-end *before* training (or without PyTorch installed).

Resulting detections will be garbage but the simulator will run, the
data path is exercised, and weight/bias files have the right shapes.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import array_to_hex_mif


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

LAYERS = [
    ("conv1", (C.CHANS[1], C.CHANS[0], 3, 3), (C.CHANS[1],)),
    ("conv2", (C.CHANS[2], C.CHANS[1], 3, 3), (C.CHANS[2],)),
    ("conv3", (C.CHANS[3], C.CHANS[2], 3, 3), (C.CHANS[3],)),
    ("head",  (C.PRED_PER_CELL, C.CHANS[3], 1, 1), (C.PRED_PER_CELL,)),
]


def main():
    w_dir = DATA / "weights"
    b_dir = DATA / "biases"
    w_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for name, wshape, bshape in LAYERS:
        fan_in = int(np.prod(wshape[1:]))
        w = rng.normal(0.0, np.sqrt(2.0 / fan_in), size=wshape).astype(np.float32)
        b = np.zeros(bshape, dtype=np.float32)
        array_to_hex_mif(w, w_dir / f"w_{name}.mif")
        array_to_hex_mif(b, b_dir / f"b_{name}.mif")
        print(f"  {name}: w{wshape}  b{bshape}")
    print("Random Q4.12 weights written.")


if __name__ == "__main__":
    main()
