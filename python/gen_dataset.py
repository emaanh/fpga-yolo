"""Synthesize a dataset of 32x32 RGB frames containing circles + squares.

Each frame contains 1-2 shapes. Ground truth is one row per shape:
    <class_id> <cx> <cy> <w> <h>      (cx/cy/w/h normalized to [0,1])

Outputs:
    data/frames/frame_NNNN.npy          (float32, shape (3,32,32), values 0..1)
    data/frames/frame_NNNN.hex          (Q4.12 hex, planar CHW, $readmemh format)
    data/frames/frame_NNNN.png          (visualizable RGB image)
    data/ground_truth/frame_NNNN.txt
"""
import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import array_to_hex_mif_planar

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def _rasterize_circle(canvas, cx, cy, r, color):
    H, W = canvas.shape[1:]
    yy, xx = np.ogrid[:H, :W]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    for c in range(3):
        canvas[c][mask] = color[c]


def _rasterize_square(canvas, cx, cy, half, color):
    H, W = canvas.shape[1:]
    y0 = max(0, int(cy - half))
    y1 = min(H, int(cy + half))
    x0 = max(0, int(cx - half))
    x1 = min(W, int(cx + half))
    for c in range(3):
        canvas[c, y0:y1, x0:x1] = color[c]


def _random_shape(rng):
    cls = int(rng.integers(0, 2))                   # 0=circle, 1=square
    size = rng.uniform(5, 9)                        # radius / half-side
    margin = size + 1
    cx = rng.uniform(margin, C.IMG_SIZE - margin)
    cy = rng.uniform(margin, C.IMG_SIZE - margin)
    # Random saturated color, biased so the two classes are easily separable
    if cls == 0:
        color = (rng.uniform(0.6, 1.0), rng.uniform(0.0, 0.3), rng.uniform(0.0, 0.3))
    else:
        color = (rng.uniform(0.0, 0.3), rng.uniform(0.6, 1.0), rng.uniform(0.0, 0.3))
    return cls, cx, cy, size, color


def _iou_centers(a, b):
    cls_a, ax, ay, as_, _ = a
    cls_b, bx, by, bs, _ = b
    return abs(ax - bx) < (as_ + bs) and abs(ay - by) < (as_ + bs)


def _make_frame(rng):
    img = np.zeros((3, C.IMG_SIZE, C.IMG_SIZE), dtype=np.float32)
    n_shapes = int(rng.integers(1, 3))
    shapes = []
    attempts = 0
    while len(shapes) < n_shapes and attempts < 20:
        attempts += 1
        s = _random_shape(rng)
        if any(_iou_centers(s, t) for t in shapes):
            continue
        shapes.append(s)

    gt = []
    for cls, cx, cy, size, color in shapes:
        if cls == 0:
            _rasterize_circle(img, cx, cy, size, color)
            w = h = 2 * size
        else:
            _rasterize_square(img, cx, cy, size, color)
            w = h = 2 * size
        gt.append((cls, cx / C.IMG_SIZE, cy / C.IMG_SIZE,
                   w / C.IMG_SIZE, h / C.IMG_SIZE))
    return img, gt


def _save_png(img, path):
    if Image is None:
        return
    rgb = (np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=2000, help="frames to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(DATA))
    args = ap.parse_args()

    out = Path(args.out_dir)
    frames_dir = out / "frames"
    gt_dir = out / "ground_truth"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    for i in range(args.num):
        img, gt = _make_frame(rng)
        stem = f"frame_{i:04d}"
        np.save(frames_dir / f"{stem}.npy", img)
        array_to_hex_mif_planar(img, frames_dir / f"{stem}.hex")
        _save_png(img, frames_dir / f"{stem}.png")
        with open(gt_dir / f"{stem}.txt", "w") as f:
            for cls, cx, cy, w, h in gt:
                f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{args.num}")
    print(f"Wrote {args.num} frames to {frames_dir}")


if __name__ == "__main__":
    main()
