"""Decode the testbench's raw cells dump into a detections file.

Input  (data/detections/cells_NNNN.txt):
    112 lines of 4-char Q4.12 hex, layout c*H*W + gy*W + gx with
    c=0..6 -> sigmoid(tx, ty, tw, th, obj, cls0, cls1).

Output (data/detections/frame_NNNN.txt):
    One line per surviving detection:
        <class_id> <cx> <cy> <w> <h> <conf>      (all in [0,1])

NMS is per-class with IoU threshold OBJ_THRESH/IOU_THRESH from config.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from utils import load_hex_mif


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


def _iou(a, b):
    ax0 = a[1] - a[3] / 2; ay0 = a[2] - a[4] / 2
    ax1 = a[1] + a[3] / 2; ay1 = a[2] + a[4] / 2
    bx0 = b[1] - b[3] / 2; by0 = b[2] - b[4] / 2
    bx1 = b[1] + b[3] / 2; by1 = b[2] + b[4] / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms_per_class(dets, iou_th):
    out = []
    by_class = {}
    for d in dets:
        by_class.setdefault(d[0], []).append(d)
    for cls, group in by_class.items():
        group.sort(key=lambda d: d[5], reverse=True)
        kept = []
        for d in group:
            if all(_iou(d, k) <= iou_th for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


def decode_cells(cells_path: Path):
    flat = load_hex_mif(cells_path, C.PRED_PER_CELL * C.GRID * C.GRID)
    chw = flat.reshape(C.PRED_PER_CELL, C.GRID, C.GRID)    # (c, gy, gx)

    dets = []
    for gy in range(C.GRID):
        for gx in range(C.GRID):
            tx, ty, tw, th, obj, *cls = chw[:, gy, gx].tolist()
            if obj < C.OBJ_THRESH:
                continue
            cx = (gx + tx) / C.GRID
            cy = (gy + ty) / C.GRID
            w, h = tw, th
            cls_id = max(range(len(cls)), key=lambda i: cls[i])
            conf = obj * cls[cls_id]
            if conf < C.OBJ_THRESH:
                continue
            dets.append((cls_id, cx, cy, w, h, conf))
    return _nms_per_class(dets, C.NMS_IOU_THRESH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, help='frame stem, e.g. "frame_0000"')
    args = ap.parse_args()

    cells_path = DATA / "detections" / f"cells_{args.frame.split('_')[-1]}.txt"
    out_path = DATA / "detections" / f"{args.frame}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dets = decode_cells(cells_path)
    with open(out_path, "w") as f:
        for cls, cx, cy, w, h, conf in dets:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f}\n")
    print(f"Decoded {len(dets)} detections -> {out_path}")


if __name__ == "__main__":
    main()
