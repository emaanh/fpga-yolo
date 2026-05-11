"""Fixed-point + .mif/.hex helpers shared across the pipeline."""
import numpy as np
from pathlib import Path
import config as C


def to_q412(x: float) -> int:
    """Float -> signed 16-bit Q4.12, saturating."""
    v = int(round(x * C.Q_SCALE))
    return max(C.Q_MIN_VAL, min(C.Q_MAX_VAL, v))


def q412_to_hex(x: float) -> str:
    """Float -> 4-char hex of Q4.12 two's-complement."""
    v = to_q412(x)
    if v < 0:
        v += 1 << C.DATA_WIDTH
    return f"{v:04x}"


def array_to_hex_mif(arr: np.ndarray, path: Path) -> None:
    """Flatten a float array to a $readmemh-compatible file (one hex word per line)."""
    flat = arr.flatten()
    with open(path, "w") as f:
        for v in flat:
            f.write(q412_to_hex(float(v)) + "\n")


def array_to_hex_mif_planar(arr: np.ndarray, path: Path) -> None:
    """Same as array_to_hex_mif but explicit about CHW planar order for images.

    Input shape: (C, H, W). Stored as c0 row-major, then c1, then c2.
    Address: c*H*W + y*W + x.
    """
    assert arr.ndim == 3
    array_to_hex_mif(arr, path)


def load_hex_mif(path: Path, count: int) -> np.ndarray:
    """Read a .hex/.mif of Q4.12 hex words back to a float vector.

    Blank lines and lines starting with `#` or `//` are skipped.
    """
    out = np.zeros(count, dtype=np.float32)
    i = 0
    with open(path) as f:
        for raw in f:
            line = raw.split("//")[0].split("#")[0].strip()
            if not line:
                continue
            if i >= count:
                break
            v = int(line, 16)
            if v >= 1 << (C.DATA_WIDTH - 1):
                v -= 1 << C.DATA_WIDTH
            out[i] = v / C.Q_SCALE
            i += 1
    return out
