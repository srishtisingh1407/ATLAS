from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class BinarySegMetrics:
    # Confusion matrix entries for positive class=1
    tp: int
    fp: int
    fn: int
    tn: int

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-9)

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-9)

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        return 2 * p * r / (p + r + 1e-9)

    def iou(self) -> float:
        return self.tp / (self.tp + self.fp + self.fn + 1e-9)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": int(self.tp),
            "fp": int(self.fp),
            "fn": int(self.fn),
            "tn": int(self.tn),
            "precision": float(self.precision()),
            "recall": float(self.recall()),
            "f1": float(self.f1()),
            "iou": float(self.iou()),
        }


def _fast_confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    pred = pred.astype(np.uint8).reshape(-1)
    gt = gt.astype(np.uint8).reshape(-1)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    return tp, fp, fn, tn


def compute_binary_metrics(pred: np.ndarray, gt: np.ndarray) -> BinarySegMetrics:
    tp, fp, fn, tn = _fast_confusion(pred, gt)
    return BinarySegMetrics(tp=tp, fp=fp, fn=fn, tn=tn)


def reduce_metrics(metrics: list[BinarySegMetrics]) -> BinarySegMetrics:
    return BinarySegMetrics(
        tp=sum(m.tp for m in metrics),
        fp=sum(m.fp for m in metrics),
        fn=sum(m.fn for m in metrics),
        tn=sum(m.tn for m in metrics),
    )


def threshold_sweep(
    probs: np.ndarray,
    gts: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Sweep decision thresholds on flat 1-D probability and ground-truth arrays.
    Returns the best threshold (by F1) and the full sweep results.

    Usage:
        probs_flat = np.concatenate([sigmoid(logits).reshape(-1) for each batch])
        gts_flat   = np.concatenate([gt_mask.reshape(-1) for each batch])
        result = threshold_sweep(probs_flat, gts_flat)
        best_thr = result["best_threshold"]
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    probs = probs.astype(np.float32)
    gts = gts.astype(np.uint8)

    sweep: list[dict[str, float]] = []
    for thr in thresholds:
        pred = (probs >= float(thr)).astype(np.uint8)
        tp = int(((pred == 1) & (gts == 1)).sum())
        fp = int(((pred == 1) & (gts == 0)).sum())
        fn = int(((pred == 0) & (gts == 1)).sum())
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        iou = tp / (tp + fp + fn + 1e-9)
        sweep.append({
            "threshold": float(round(float(thr), 3)),
            "f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "iou": float(iou),
        })

    best = max(sweep, key=lambda x: x["f1"])
    return {
        "best_threshold": best["threshold"],
        "best_f1": best["f1"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_iou": best["iou"],
        "sweep": sweep,
    }

