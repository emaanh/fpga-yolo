"""Quantize a trained TinyYolo to Q4.12 and emit .mif files the RTL ingests.

Weight memory layout (per layer, one file):
  - 3x3 conv layer "convL":  weights[co, ci, ky, kx] flattened in OIHW order
                             address = co*Cin*9 + ci*9 + ky*3 + kx
  - 1x1 detection head:      weights[co, ci]
                             address = co*Cin + ci
Biases (one file per layer): one Q4.12 word per output channel.

Both files are $readmemh-compatible (4-char hex words, one per line).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from model import TinyYolo
from utils import array_to_hex_mif


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def _export_conv(conv, name, w_dir, b_dir):
    w = conv.weight.detach().cpu().numpy()                   # (Cout, Cin, kH, kW)
    b = conv.bias.detach().cpu().numpy()                     # (Cout,)
    array_to_hex_mif(w, w_dir / f"w_{name}.mif")
    array_to_hex_mif(b, b_dir / f"b_{name}.mif")
    print(f"  {name}: w{tuple(w.shape)} range=[{w.min():.3f},{w.max():.3f}]  "
          f"b{tuple(b.shape)} range=[{b.min():.3f},{b.max():.3f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA / "model.pt"))
    args = ap.parse_args()

    w_dir = DATA / "weights"
    b_dir = DATA / "biases"
    w_dir.mkdir(parents=True, exist_ok=True)
    b_dir.mkdir(parents=True, exist_ok=True)

    model = TinyYolo()
    sd = torch.load(args.model, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    print("Exporting Q4.12 weights:")
    _export_conv(model.b1[0], "conv1", w_dir, b_dir)
    _export_conv(model.b2[0], "conv2", w_dir, b_dir)
    _export_conv(model.b3[0], "conv3", w_dir, b_dir)
    _export_conv(model.head, "head",  w_dir, b_dir)

    # Sanity: report fraction of weights that would saturate.
    all_w = np.concatenate([p.detach().numpy().ravel()
                            for p in model.parameters() if p.dim() > 1])
    sat = np.mean(np.abs(all_w) >= (C.Q_MAX_VAL / C.Q_SCALE))
    print(f"Weights saturating Q{C.INT_BITS+1}.{C.FRAC_BITS}: {sat*100:.2f}%")


if __name__ == "__main__":
    main()
