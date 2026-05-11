"""Train TinyYolo on the synthetic dataset.

Loss is YOLOv1-style: coord MSE on tx/ty/tw/th for cells that own a target,
BCE on obj for every cell (with the usual lambda_noobj down-weighting), and
BCE on the class logits for cells that own a target.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import config as C
from model import TinyYolo


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"


class ShapesDataset(Dataset):
    def __init__(self, frames_dir: Path, gt_dir: Path):
        self.frames = sorted(frames_dir.glob("frame_*.npy"))
        self.gt_dir = gt_dir

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = np.load(self.frames[idx])                       # (3,32,32)
        stem = self.frames[idx].stem
        gt = []
        with open(self.gt_dir / f"{stem}.txt") as f:
            for line in f:
                cls, cx, cy, w, h = line.split()
                gt.append((int(cls), float(cx), float(cy), float(w), float(h)))

        target = np.zeros((C.PRED_PER_CELL, C.GRID, C.GRID), dtype=np.float32)
        mask = np.zeros((C.GRID, C.GRID), dtype=np.float32)
        for cls, cx, cy, w, h in gt:
            gx = min(int(cx * C.GRID), C.GRID - 1)
            gy = min(int(cy * C.GRID), C.GRID - 1)
            tx = cx * C.GRID - gx
            ty = cy * C.GRID - gy
            target[0, gy, gx] = tx
            target[1, gy, gx] = ty
            target[2, gy, gx] = w
            target[3, gy, gx] = h
            target[4, gy, gx] = 1.0
            target[5 + cls, gy, gx] = 1.0
            mask[gy, gx] = 1.0
        return torch.from_numpy(img), torch.from_numpy(target), torch.from_numpy(mask)


def yolo_loss(pred, target, mask, lambda_coord=5.0, lambda_noobj=0.5):
    sig = torch.sigmoid(pred)
    m = mask.unsqueeze(1)                                    # (N,1,G,G)

    coord = ((sig[:, 0:4] - target[:, 0:4]) ** 2).sum(dim=1, keepdim=True) * m
    obj = F.binary_cross_entropy_with_logits(
        pred[:, 4:5], target[:, 4:5], reduction="none")
    obj_loss = (obj * m).sum() + lambda_noobj * (obj * (1 - m)).sum()
    cls_loss = (F.binary_cross_entropy_with_logits(
        pred[:, 5:], target[:, 5:], reduction="none") * m).sum()
    coord_loss = coord.sum()

    n = pred.size(0)
    return (lambda_coord * coord_loss + obj_loss + cls_loss) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default=str(DATA / "model.pt"))
    args = ap.parse_args()

    ds = ShapesDataset(DATA / "frames", DATA / "ground_truth")
    if len(ds) == 0:
        raise SystemExit("No training data. Run gen_dataset.py first.")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = TinyYolo()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        total = 0.0
        for img, tgt, mask in dl:
            opt.zero_grad()
            loss = yolo_loss(model(img), tgt, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * img.size(0)
        print(f"epoch {ep+1:3d}/{args.epochs}  loss={total/len(ds):.4f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
