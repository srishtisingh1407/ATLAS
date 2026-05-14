"""One-off helper: print shapes of first pre/post/target TIFF in data/train."""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "data" / "train"


def read_tif(path: Path) -> np.ndarray:
    try:
        import tifffile as tifffile

        return tifffile.imread(str(path))
    except ImportError:
        import cv2

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2 could not read {path}; install tifffile: pip install tifffile")
        return img


def main() -> None:
    pre_dir = ROOT / "pre-event"
    post_dir = ROOT / "post-event"
    tgt_dir = ROOT / "target"
    pre = sorted(pre_dir.glob("*.tif"))[0]
    post = sorted(post_dir.glob("*.tif"))[0]
    m = sorted(tgt_dir.glob("*.tif"))[0]
    a, b, y = read_tif(pre), read_tif(post), read_tif(m)
    print("pre ", pre.name, a.shape, a.dtype)
    print("post", post.name, b.shape, b.dtype)
    print("mask", m.name, y.shape, y.dtype)


if __name__ == "__main__":
    main()
