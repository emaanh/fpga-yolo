"""Generate the 1024-entry sigmoid LUT consumed by rtl/sigmoid_lut.v.

Index i corresponds to a Q4.12 signed input of  x = -8 + i / 64.
LUT[i] stores sigmoid(x) encoded as a Q4.12 unsigned value.
"""
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C


HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "sigmoid" / "sigmoid_lut.mif"

N = 1 << C.SIG_LUT_BITS                          # 1024
RANGE = 16.0                                     # full Q4.12 signed span
STEP = RANGE / N                                 # 1/64
ORIGIN = -8.0


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for i in range(N):
            x = ORIGIN + i * STEP
            s = 1.0 / (1.0 + math.exp(-x))
            q = int(round(s * C.Q_SCALE))
            q = max(0, min(C.Q_MAX_VAL, q))
            f.write(f"{q:04x}\n")
    print(f"Wrote {N} sigmoid entries to {OUT}")


if __name__ == "__main__":
    main()
