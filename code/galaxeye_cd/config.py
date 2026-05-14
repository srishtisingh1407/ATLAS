from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    run_name: str
    seed: int
    device: str

    data_root: Path
    img_size: int
    batch_size: int
    num_workers: int

    splits: dict[str, Path | None]
    folders: dict[str, str | None]

    epochs: int
    lr: float
    weight_decay: float
    pos_weight: float
    dropout_p: float

    lr_scheduler: str | None   # "cosine" | None
    lr_T0: int
    lr_T_mult: int
    lr_eta_min: float

    # Augmentation (geometric + photometric, training split only)
    augment: bool

    # Smarter training
    change_weight_multiplier: float   # 0 = plain shuffle; >0 = oversample damage patches

    # Post-training threshold sweep on val
    threshold_sweep: bool

    # Inference options (used in eval.py)
    tta: bool
    mc_dropout_passes: int
    num_vis_samples: int

    save_every: int
    metric_for_best: str


def _to_path_or_none(v: Any) -> Path | None:
    if v is None:
        return None
    return Path(v)


def load_config(path: str | Path) -> tuple[Config, dict[str, Any]]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    cfg = Config(
        run_name=str(raw["run_name"]),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cuda")),
        data_root=Path(raw.get("data_root", "data")),
        img_size=int(raw.get("img_size", 256)),
        batch_size=int(raw.get("batch_size", 4)),
        num_workers=int(raw.get("num_workers", 2)),
        splits={
            "train": _to_path_or_none(raw.get("splits", {}).get("train")),
            "val":   _to_path_or_none(raw.get("splits", {}).get("val")),
            "test":  _to_path_or_none(raw.get("splits", {}).get("test")),
        },
        folders=dict(raw.get("folders", {})),
        epochs=int(raw.get("epochs", 20)),
        lr=float(raw.get("lr", 3e-4)),
        weight_decay=float(raw.get("weight_decay", 1e-5)),
        pos_weight=float(raw.get("pos_weight", 5.0)),
        dropout_p=float(raw.get("dropout_p", 0.3)),
        lr_scheduler=raw.get("lr_scheduler", "cosine") or None,
        lr_T0=int(raw.get("lr_T0", 10)),
        lr_T_mult=int(raw.get("lr_T_mult", 2)),
        lr_eta_min=float(raw.get("lr_eta_min", 1e-5)),
        augment=bool(raw.get("augment", True)),
        change_weight_multiplier=float(raw.get("change_weight_multiplier", 5.0)),
        threshold_sweep=bool(raw.get("threshold_sweep", True)),
        tta=bool(raw.get("tta", True)),
        mc_dropout_passes=int(raw.get("mc_dropout_passes", 30)),
        num_vis_samples=int(raw.get("num_vis_samples", 10)),
        save_every=int(raw.get("save_every", 1)),
        metric_for_best=str(raw.get("metric_for_best", "f1")),
    )

    return cfg, raw
