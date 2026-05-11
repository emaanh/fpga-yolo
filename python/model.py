"""Tiny-YOLO model definition. Architecture is locked to match the RTL.

Backbone:
    Conv3x3 (3 -> 8)  + LeakyReLU + MaxPool 2x2  -> 16x16x8
    Conv3x3 (8 -> 16) + LeakyReLU + MaxPool 2x2  ->  8x8x16
    Conv3x3 (16->32)  + LeakyReLU + MaxPool 2x2  ->  4x4x32
Detection head:
    Conv1x1 (32 -> 5 + NUM_CLASSES)              ->  4x4x(5+C)

Per grid cell the head predicts raw logits for (tx, ty, tw, th, obj, cls...).
Decoding applies sigmoid to all of those and treats tx/ty as the offset
within the cell and tw/th as the box dimensions relative to the full image.
"""
import torch
import torch.nn as nn
import config as C


class TinyYolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = self._block(C.CHANS[0], C.CHANS[1])
        self.b2 = self._block(C.CHANS[1], C.CHANS[2])
        self.b3 = self._block(C.CHANS[2], C.CHANS[3])
        self.head = nn.Conv2d(C.CHANS[3], C.PRED_PER_CELL, kernel_size=1, bias=True)

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=C.LEAKY_SLOPE, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return self.head(x)                       # (N, 5+C, 4, 4)


def decode(raw, conf_thresh=C.OBJ_THRESH):
    """Decode raw head output to a list of detections per batch element.

    raw: (N, 5+C, G, G) tensor of logits.
    Returns: list of length N, each a list of (cls, cx, cy, w, h, conf) in [0,1].
    """
    n, _, gh, gw = raw.shape
    raw = raw.permute(0, 2, 3, 1)                 # (N, G, G, 5+C)
    sig = torch.sigmoid(raw)
    out = []
    for b in range(n):
        boxes = []
        for gy in range(gh):
            for gx in range(gw):
                tx, ty, tw, th, obj, *cls = sig[b, gy, gx].tolist()
                if obj < conf_thresh:
                    continue
                cx = (gx + tx) / gw
                cy = (gy + ty) / gh
                cls_id = int(max(range(len(cls)), key=lambda i: cls[i]))
                conf = obj * cls[cls_id]
                boxes.append((cls_id, cx, cy, tw, th, conf))
        out.append(boxes)
    return out
