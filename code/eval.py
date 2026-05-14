from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from galaxeye_cd.analysis import save_qualitative_examples
from galaxeye_cd.config import load_config
from galaxeye_cd.dataset import ChangeDetectionDataset, build_sample_list, discover_split_root
from galaxeye_cd.metrics import compute_binary_metrics, reduce_metrics
from galaxeye_cd.model import UNetSmall
from galaxeye_cd.tta import tta_predict
from galaxeye_cd.utils import ensure_dir, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True,  type=str)
    ap.add_argument("--split",   required=True,  choices=["val", "test", "train"])
    ap.add_argument("--weights", required=True,  type=str)
    ap.add_argument("--out",     default=None,   type=str,
                    help="Optional path to write metrics JSON.")
    ap.add_argument("--no-tta",  action="store_true",
                    help="Disable TTA even if config.tta=true.")
    ap.add_argument("--no-mc",   action="store_true",
                    help="Disable MC Dropout even if config.mc_dropout_passes>0.")
    ap.add_argument("--no-vis",  action="store_true",
                    help="Skip saving qualitative visualisations.")
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args  = parse_args()
    cfg, _ = load_config(args.config)
    device = resolve_device(cfg.device)

    # ── Dataset ──────────────────────────────────────────────────────────────
    data_root  = cfg.data_root
    split_root = cfg.splits[args.split] or discover_split_root(data_root, args.split)
    layout, index = build_sample_list(split_root, cfg.folders)
    print(f"eval  split={args.split}  layout={layout}  samples={len(index)}")

    with_mask = bool(index) and index[0].mask is not None
    ds     = ChangeDetectionDataset(index, img_size=cfg.img_size, with_mask=with_mask)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    # ── Model ────────────────────────────────────────────────────────────────
    ckpt        = torch.load(args.weights, map_location="cpu")
    in_channels = int(ckpt.get("in_channels", ds[0]["image"].shape[0]))
    dropout_p   = float(ckpt.get("dropout_p",  cfg.dropout_p))
    model       = UNetSmall(in_channels=in_channels, base=32, dropout_p=dropout_p)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # ── Threshold ────────────────────────────────────────────────────────────
    threshold = float(ckpt.get("best_threshold", 0.5))
    print(f"threshold={threshold:.3f} "
          f"{'(from sweep)' if 'best_threshold' in ckpt else '(default 0.5)'}")

    use_tta = cfg.tta and not args.no_tta
    n_mc    = cfg.mc_dropout_passes if not args.no_mc else 0
    print(f"TTA={'on' if use_tta else 'off'}  MC-dropout-passes={n_mc}")

    # ── Inference loop ───────────────────────────────────────────────────────
    metrics = []
    for batch in tqdm(loader, desc=f"eval:{args.split}"):
        x = batch["image"].to(device)

        if use_tta:
            prob = tta_predict(model, x)                       # (B,1,H,W)
        else:
            prob = torch.sigmoid(model(x))

        pred = (prob > threshold).to(torch.uint8).cpu().numpy()

        if with_mask:
            gt = (batch["mask"] > 0.5).to(torch.uint8).cpu().numpy()
            for i in range(pred.shape[0]):
                metrics.append(compute_binary_metrics(pred[i, 0], gt[i, 0]))

    # ── Metrics output ───────────────────────────────────────────────────────
    run_dir = ensure_dir(Path("runs") / cfg.run_name)
    out: dict = {"split": args.split, "weights": str(Path(args.weights)),
                 "threshold": threshold, "tta": use_tta, "mc_passes": n_mc}

    if with_mask:
        total = reduce_metrics(metrics)
        out["metrics"] = total.as_dict()
        out["confusion_matrix"] = {
            "rows": ["gt_0 (no-change)", "gt_1 (change)"],
            "cols": ["pred_0 (no-change)", "pred_1 (change)"],
            "values": [
                [int(total.tn), int(total.fp)],
                [int(total.fn), int(total.tp)],
            ],
        }
    else:
        out["metrics"] = None
        out["note"] = ("No ground-truth masks found; predictions generated "
                       "but metrics not computed.")

    dest = Path(args.out) if args.out else run_dir / f"metrics_{args.split}.json"
    save_json(dest, out)

    if out["metrics"]:
        m = out["metrics"]
        print(f"\n{'-'*50}")
        print(f"  Split     : {args.split}")
        print(f"  IoU       : {m['iou']:.4f}")
        print(f"  F1        : {m['f1']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  Confusion matrix (gt rows x pred cols):")
        for row in out["confusion_matrix"]["values"]:
            print(f"    {row}")
        print(f"  Saved -> {dest}")
    else:
        print(out.get("note", ""))

    # Qualitative visualisations
    if with_mask and not args.no_vis:
        vis_dir = run_dir / "qualitative" / args.split
        print(f"\nGenerating {cfg.num_vis_samples} qualitative examples -> {vis_dir}")
        save_qualitative_examples(
            dataset=ds,
            model=model,
            device=device,
            threshold=threshold,
            save_dir=vis_dir,
            n_samples=cfg.num_vis_samples,
            n_mc_passes=n_mc,
            use_tta=use_tta,
        )


if __name__ == "__main__":
    main()
