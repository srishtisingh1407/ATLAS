from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ChangeDetectionDataset
from .metrics import compute_binary_metrics
from .tta import tta_predict


# ---------------------------------------------------------------------------
# MC Dropout
# ---------------------------------------------------------------------------

def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Monte-Carlo Dropout inference.

    Call model.enable_mc_dropout() BEFORE this function so dropout layers
    stay stochastic while BatchNorm stays in eval mode.

    Returns:
        mean_prob  (B, 1, H, W)  — averaged sigmoid probability
        std_prob   (B, 1, H, W)  — pixel-wise standard deviation (epistemic uncertainty)
    """
    passes: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(n_passes):
            prob = torch.sigmoid(model(x))
            passes.append(prob)

    stacked = torch.stack(passes, dim=0)          # (n_passes, B, 1, H, W)
    return stacked.mean(dim=0), stacked.std(dim=0)


# ---------------------------------------------------------------------------
# Qualitative visualisation
# ---------------------------------------------------------------------------

def _disp(arr: np.ndarray) -> np.ndarray:
    """Normalise array to [0, 1] for display."""
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-9)).clip(0.0, 1.0).astype(np.float32)


def _save_figure(r: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print(f"matplotlib not available — skipping figure {path.name}")
        return

    img = r["image"]          # (C, H, W)
    pred = r["pred"]          # (H, W) uint8 binary
    gt = r["gt"]              # (H, W) uint8 binary or None
    prob = r["prob"]          # (H, W) float
    unc = r["uncertainty"]    # (H, W) float

    # Split pre / post channels for display
    n_pre = 3 if img.shape[0] >= 4 else 1
    pre_raw = img[:n_pre].transpose(1, 2, 0)
    post_raw = img[n_pre:].mean(axis=0)

    pre_disp = _disp(pre_raw)
    post_disp = _disp(post_raw)

    # Build RGB error map: TP=green, FP=red, FN=blue, TN=black
    H, W = pred.shape
    error_rgb = np.zeros((H, W, 3), dtype=np.float32)
    if gt is not None:
        tp_mask = (pred == 1) & (gt == 1)
        fp_mask = (pred == 1) & (gt == 0)
        fn_mask = (pred == 0) & (gt == 1)
        error_rgb[tp_mask] = [0.0, 0.75, 0.0]
        error_rgb[fp_mask] = [0.9,  0.0, 0.0]
        error_rgb[fn_mask] = [0.0,  0.0, 0.9]

    fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
    titles = ["Pre-event", "Post-event", "Ground Truth", "Prediction",
              "Error  (TP=green, FP=red, FN=blue)", "Uncertainty (MC σ)"]

    axes[0].imshow(pre_disp if pre_disp.ndim == 3 and pre_disp.shape[2] >= 3
                   else pre_disp[:, :, 0], cmap="gray" if n_pre == 1 else None)
    axes[1].imshow(post_disp, cmap="gray")
    axes[2].imshow(gt if gt is not None else np.zeros((H, W), dtype=np.uint8),
                   cmap="gray", vmin=0, vmax=1)
    axes[3].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[4].imshow(error_rgb)
    im = axes[5].imshow(unc, cmap="hot", vmin=0, vmax=max(float(unc.max()), 1e-6))
    plt.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    if gt is not None:
        patches = [
            mpatches.Patch(color=[0.0, 0.75, 0.0], label="TP"),
            mpatches.Patch(color=[0.9,  0.0, 0.0], label="FP"),
            mpatches.Patch(color=[0.0,  0.0, 0.9], label="FN"),
        ]
        axes[4].legend(handles=patches, loc="lower right", fontsize=6)

    iou_str = f"IoU={r['iou']:.3f}" if (gt is not None and not np.isnan(r["iou"])) else "no GT"
    fig.suptitle(f"Sample {r['idx']} | {iou_str}", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_qualitative_examples(
    dataset: ChangeDetectionDataset,
    model: nn.Module,
    device: torch.device,
    threshold: float,
    save_dir: Path,
    n_samples: int = 10,
    n_mc_passes: int = 30,
    use_tta: bool = True,
) -> None:
    """
    Save qualitative visualisation grids for a mix of success and failure cases.

    For each selected sample the PNG shows:
      pre-event | post-event | ground-truth | prediction | error map | uncertainty
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Scan a representative subset (up to 200 samples)
    n_scan = min(len(dataset), 200)
    rng = np.random.default_rng(seed=0)
    indices = rng.choice(len(dataset), n_scan, replace=False).tolist()

    model.eval()
    if n_mc_passes > 0:
        model.enable_mc_dropout()  # type: ignore[attr-defined]

    results: list[dict] = []
    for idx in indices:
        sample = dataset[idx]
        x = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            if use_tta:
                prob_mean = tta_predict(model, x)
            else:
                prob_mean = torch.sigmoid(model(x))

            if n_mc_passes > 0:
                _, mc_std = mc_dropout_predict(model, x, n_mc_passes)
            else:
                mc_std = torch.zeros_like(prob_mean)

        pred = (prob_mean > threshold).squeeze().cpu().numpy().astype(np.uint8)
        prob_np = prob_mean.squeeze().cpu().numpy()
        unc_np = mc_std.squeeze().cpu().numpy()

        gt = None
        iou = float("nan")
        if "mask" in sample:
            gt = (sample["mask"].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            m = compute_binary_metrics(pred, gt)
            iou = m.iou()

        results.append({
            "idx": int(idx),
            "iou": iou,
            "image": sample["image"].numpy(),
            "pred": pred,
            "gt": gt,
            "prob": prob_np,
            "uncertainty": unc_np,
        })

    model.eval()  # restore full eval mode after MC dropout

    # Pick n_samples/2 best and n_samples/2 worst (by IoU)
    valid = [r for r in results if not np.isnan(r["iou"])]
    if valid:
        valid.sort(key=lambda r: r["iou"])
        n_fail = n_samples // 2
        n_succ = n_samples - n_fail
        # Guard: if fewer than needed, take what we have
        selected = valid[:n_fail] + valid[-n_succ:]
    else:
        selected = results[:n_samples]

    for i, r in enumerate(selected):
        tag = "success" if (not np.isnan(r["iou"]) and r["iou"] >= 0.5) else "failure"
        fname = f"{i:03d}_{tag}_iou{r['iou']:.3f}.png"
        _save_figure(r, save_dir / fname)

    print(f"Saved {len(selected)} qualitative examples to {save_dir}")
