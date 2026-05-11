"""Overlay RTL detections back onto the input frame as a PNG.

The simulator dumps one detections file per inferred frame in data/detections/
with rows: <class_id> <cx> <cy> <w> <h> <conf>     (all in [0,1])
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config as C

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise SystemExit("Pillow required: pip install pillow")


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

CLASS_COLORS = [(255, 64, 64), (64, 255, 64)]


def _load_frame(stem):
    npy = DATA / "frames" / f"{stem}.npy"
    arr = np.load(npy)                                       # (3,32,32)
    rgb = (np.clip(arr.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def _draw(img, dets, scale=8):
    img = img.resize((C.IMG_SIZE * scale, C.IMG_SIZE * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for cls, cx, cy, w, h, conf in dets:
        x0 = (cx - w / 2) * C.IMG_SIZE * scale
        y0 = (cy - h / 2) * C.IMG_SIZE * scale
        x1 = (cx + w / 2) * C.IMG_SIZE * scale
        y1 = (cy + h / 2) * C.IMG_SIZE * scale
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2),
                  f"{C.CLASS_NAMES[cls]} {conf:.2f}", fill=color)
    return img


def _read_dets(path):
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cls, cx, cy, w, h, conf = line.split()
            out.append((int(cls), float(cx), float(cy),
                        float(w), float(h), float(conf)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help='frame stem, e.g. "frame_0000"')
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img = _load_frame(args.frame)
    dets = _read_dets(DATA / "detections" / f"{args.frame}.txt")
    rendered = _draw(img, dets)

    out = Path(args.out) if args.out else DATA / "detections" / f"{args.frame}_vis.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(out)
    print(f"Wrote {out}  ({len(dets)} detections)")


if __name__ == "__main__":
    main()
