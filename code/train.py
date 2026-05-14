from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxeye_cd.config import load_config
from galaxeye_cd.dataset import ChangeDetectionDataset, build_sample_list, discover_split_root
from galaxeye_cd.metrics import compute_binary_metrics, reduce_metrics, threshold_sweep
from galaxeye_cd.model import UNetSmall
from galaxeye_cd.sampler import build_weighted_sampler
from galaxeye_cd.utils import ensure_dir, env_info, resolve_device, save_json, save_yaml, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    return ap.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> dict:
    model.eval()
    metrics = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) > threshold).to(torch.uint8).cpu().numpy()
        gt = (y > 0.5).to(torch.uint8).cpu().numpy()
        for i in range(pred.shape[0]):
            metrics.append(compute_binary_metrics(pred[i, 0], gt[i, 0]))
    return reduce_metrics(metrics).as_dict()


@torch.no_grad()
def collect_val_probs(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Collect all val probabilities and ground-truths as flat 1-D arrays (for threshold sweep)."""
    model.eval()
    all_probs, all_gts = [], []
    for batch in loader:
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float16)
        gts = (batch["mask"].squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        all_probs.append(probs.reshape(-1))
        all_gts.append(gts.reshape(-1))
    return np.concatenate(all_probs).astype(np.float32), np.concatenate(all_gts)


def main() -> None:
    args = parse_args()
    cfg, cfg_raw = load_config(args.config)

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    run_dir = ensure_dir(Path("runs") / cfg.run_name)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    save_yaml(run_dir / "config_resolved.yaml", cfg_raw | {"env": env_info()})

    data_root = cfg.data_root
    train_root = cfg.splits["train"] or discover_split_root(data_root, "train")
    val_root   = cfg.splits["val"]   or discover_split_root(data_root, "val")

    train_layout, train_index = build_sample_list(train_root, cfg.folders)
    val_layout,   val_index   = build_sample_list(val_root,   cfg.folders)
    print(f"Dataset layout: train={train_layout} ({len(train_index)} samples), "
          f"val={val_layout} ({len(val_index)} samples)")

    train_ds = ChangeDetectionDataset(
        train_index, img_size=cfg.img_size, with_mask=True,
        augment=cfg.augment, seed=cfg.seed,
    )
    val_ds = ChangeDetectionDataset(
        val_index, img_size=cfg.img_size, with_mask=True,
        augment=False,
    )
    if cfg.augment:
        print("Augmentation: hflip / vflip / rot90 / brightness / contrast / noise  [train only]")

    # Weighted sampler — oversamples patches that contain change pixels.
    if cfg.change_weight_multiplier > 0:
        sampler = build_weighted_sampler(train_ds, cfg.change_weight_multiplier)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    in_channels = train_ds[0]["image"].shape[0]
    model = UNetSmall(in_channels=in_channels, base=32, dropout_p=cfg.dropout_p).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = torch.tensor([cfg.pos_weight], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.lr_T0, T_mult=cfg.lr_T_mult, eta_min=cfg.lr_eta_min,
        )
        print(f"LR scheduler: CosineAnnealingWarmRestarts  T_0={cfg.lr_T0}  "
              f"T_mult={cfg.lr_T_mult}  eta_min={cfg.lr_eta_min}")
    else:
        scheduler = None

    best_score = -1.0
    best_path  = ckpt_dir / "best.pth"
    history: list[dict] = []

    start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        total_loss = 0.0

        for batch in pbar:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{opt.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step()

        avg_loss   = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, device)
        score = float(val_metrics[cfg.metric_for_best])

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "lr": opt.param_groups[0]["lr"],
            "val_iou": val_metrics["iou"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        })
        save_json(run_dir / "training_history.json", history)
        save_json(run_dir / "metrics_val.json", {
            "epoch": epoch, "train_loss": avg_loss,
            "val": val_metrics, "best_score": best_score,
        })

        if score > best_score:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "in_channels": in_channels,
                "dropout_p": cfg.dropout_p,
                "epoch": epoch,
                "best_score": best_score,
                "config": cfg_raw,
            }, best_path)

        if epoch % cfg.save_every == 0:
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": cfg_raw},
                ckpt_dir / f"epoch_{epoch}.pth",
            )

        print(f"[epoch {epoch:03d}] loss={avg_loss:.4f}  "
              f"val_iou={val_metrics['iou']:.4f}  val_f1={val_metrics['f1']:.4f}  "
              f"val_p={val_metrics['precision']:.4f}  val_r={val_metrics['recall']:.4f}")

    elapsed = time.time() - start
    print(f"\nTraining done. Best {cfg.metric_for_best}={best_score:.4f}. "
          f"Wall-clock={elapsed/60:.1f} min. Saved: {best_path}")

    # ── Post-training threshold sweep ────────────────────────────────────────
    if cfg.threshold_sweep:
        print("\nThreshold sweep on val set (using best checkpoint)...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        probs_flat, gts_flat = collect_val_probs(model, val_loader, device)
        sweep_result = threshold_sweep(probs_flat, gts_flat)
        best_thr = sweep_result["best_threshold"]

        print(f"  Default 0.5  → F1={evaluate(model, val_loader, device, 0.5)['f1']:.4f}")
        print(f"  Best thresh {best_thr:.3f} → F1={sweep_result['best_f1']:.4f}")

        # Persist threshold inside the checkpoint so eval.py can use it automatically
        ckpt["best_threshold"] = best_thr
        ckpt["threshold_sweep"] = sweep_result
        torch.save(ckpt, best_path)
        save_json(run_dir / "threshold_sweep.json", sweep_result)


if __name__ == "__main__":
    main()
