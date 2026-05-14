from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_REMAP = {
    0: 0,  # Background -> No-Change
    1: 0,  # Intact -> No-Change
    2: 1,  # Damaged -> Change
    3: 1,  # Destroyed -> Change
}


def remap_mask_to_binary(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.uint8)
    for k, v in LABEL_REMAP.items():
        out[mask == k] = v
    return out


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = img[:, :, None]
    img = img.astype(np.float32)
    return img


def _read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _resize(img: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if img.ndim == 2:
        return cv2.resize(img, (size, size), interpolation=interp)
    return cv2.resize(img, (size, size), interpolation=interp)


@dataclass(frozen=True)
class SampleRecord:
    """
    One sample: either four EO/SAR paths (quad layout) or two stacked pre/post rasters (prepost layout).
    Exactly one of `quad` or `prepost` must be set.
    """

    mask: Path | None
    quad: tuple[Path, Path, Path, Path] | None = None
    prepost: tuple[Path, Path] | None = None

    def __post_init__(self) -> None:
        q_ok = self.quad is not None
        p_ok = self.prepost is not None
        if q_ok == p_ok:
            raise ValueError("SampleRecord: set exactly one of quad or prepost.")


def _try_find_folder(root: Path, candidates: Iterable[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.exists() and p.is_dir():
            return p
    return None


def discover_split_root(data_root: Path, split: str) -> Path:
    # Common split naming
    candidates = [split, split.lower(), split.upper()]
    p = _try_find_folder(data_root, candidates)
    if p is not None:
        return p

    # Sometimes nested under "DATA" or similar
    for parent in ["DATA", "data", "dataset", "Dataset"]:
        parent_p = data_root / parent
        if not parent_p.exists():
            continue
        p2 = _try_find_folder(parent_p, candidates)
        if p2 is not None:
            return p2

    raise FileNotFoundError(
        f"Could not find split folder '{split}' under data_root={data_root}. "
        f"Set config.splits.{split} explicitly."
    )


def discover_mask_folder(split_root: Path) -> Path | None:
    mask_names = ["mask", "masks", "label", "labels", "gt", "target", "targets", "y"]
    mask_dir = _try_find_folder(split_root, mask_names)
    if mask_dir is not None:
        return mask_dir
    for sub in split_root.iterdir() if split_root.exists() else []:
        if sub.is_dir() and sub.name.lower() in {n.lower() for n in mask_names}:
            return sub
    return None


def discover_modal_folders(split_root: Path) -> dict[str, Path]:
    """
    Returns dict with keys:
      eo_pre, eo_post, sar_pre, sar_post, mask
    Discovery tries common conventions.
    """

    # Time folders
    pre_names = ["pre", "t1", "before", "pre_event", "pre-event"]
    post_names = ["post", "t2", "after", "post_event", "post-event"]

    # Modality folders
    eo_names = ["eo", "EO", "optical", "rgb", "RGB", "opt", "sentinel2", "s2"]
    sar_names = ["sar", "SAR", "s1", "sentinel1", "radar"]

    # Mask/label folders
    mask_names = ["mask", "masks", "label", "labels", "gt", "target", "targets", "y"]

    def find_time_mod(mod_names: list[str], time_names: list[str]) -> Path | None:
        # split_root/<mod>/<time>
        for mod in mod_names:
            mod_dir = split_root / mod
            if not mod_dir.exists():
                continue
            t = _try_find_folder(mod_dir, time_names)
            if t is not None:
                return t

        # split_root/<time>/<mod>
        for tnm in time_names:
            tdir = split_root / tnm
            if not tdir.exists():
                continue
            m = _try_find_folder(tdir, mod_names)
            if m is not None:
                return m

        return None

    eo_pre = find_time_mod(eo_names, pre_names)
    eo_post = find_time_mod(eo_names, post_names)
    sar_pre = find_time_mod(sar_names, pre_names)
    sar_post = find_time_mod(sar_names, post_names)

    mask_dir = discover_mask_folder(split_root)

    if eo_pre is None or eo_post is None or sar_pre is None or sar_post is None:
        raise FileNotFoundError(
            "Could not auto-discover required folders for (eo_pre, eo_post, sar_pre, sar_post). "
            "Please set config.folders.* explicitly."
        )

    return {
        "eo_pre": eo_pre,
        "eo_post": eo_post,
        "sar_pre": sar_pre,
        "sar_post": sar_post,
        "mask": mask_dir,
    }


def _has_prepost_target_layout(split_root: Path) -> bool:
    pre = split_root / "pre-event"
    post = split_root / "post-event"
    tgt = split_root / "target"
    return pre.is_dir() and post.is_dir() and tgt.is_dir()


def build_index_pre_post(pre_dir: Path, post_dir: Path, mask_dir: Path | None) -> list[SampleRecord]:
    """GalaxEye-style: same filename in pre-event, post-event, target; stack pre+post channels."""
    pre_files = _list_images(pre_dir)
    if not pre_files:
        raise FileNotFoundError(f"No images found in {pre_dir}")

    records: list[SampleRecord] = []
    for p in pre_files:
        post_p = post_dir / p.name
        if not post_p.is_file():
            continue
        mask_p: Path | None = None
        if mask_dir is not None:
            cand = mask_dir / p.name
            if not cand.is_file():
                continue
            mask_p = cand
        records.append(SampleRecord(mask=mask_p, prepost=(p, post_p)))

    if not records:
        raise RuntimeError(
            f"No matching pre/post/target triples under {pre_dir}, {post_dir}, {mask_dir}. "
            "Check filenames match across folders."
        )
    return records


def _rel(split_root: Path, rel: str) -> Path:
    p = split_root / str(rel).replace("\\", "/")
    if not p.is_dir():
        raise FileNotFoundError(f"Expected directory {p} (under {split_root})")
    return p


def resolve_modal_folders(split_root: Path, overrides: dict[str, Any] | None) -> dict[str, Path | None]:
    """
    Resolve EO/SAR pre/post and mask directories for one split.

    - If config.folders sets all four modality-time paths (eo_pre, eo_post, sar_pre, sar_post),
      those are used and auto-discovery is skipped for modalities.
    - Otherwise paths come from discover_modal_folders(split_root), then any non-null
      config.folders entries override.
    Relative paths in YAML are under split_root (e.g. train/), not data_root.
    """
    o = overrides or {}
    keys_four = ("eo_pre", "eo_post", "sar_pre", "sar_post")
    if all(o.get(k) not in (None, "") for k in keys_four):
        eo_pre = _rel(split_root, str(o["eo_pre"]))
        eo_post = _rel(split_root, str(o["eo_post"]))
        sar_pre = _rel(split_root, str(o["sar_pre"]))
        sar_post = _rel(split_root, str(o["sar_post"]))
        mask_dir: Path | None = None
        if o.get("mask") not in (None, ""):
            mask_dir = _rel(split_root, str(o["mask"]))
        else:
            mask_dir = discover_mask_folder(split_root)
        return {
            "eo_pre": eo_pre,
            "eo_post": eo_post,
            "sar_pre": sar_pre,
            "sar_post": sar_post,
            "mask": mask_dir,
        }

    found: dict[str, Path | None] = dict(discover_modal_folders(split_root))
    for key in ("eo_pre", "eo_post", "sar_pre", "sar_post", "mask"):
        rel = o.get(key)
        if rel is None or rel == "":
            continue
        found[key] = _rel(split_root, str(rel))

    if found["eo_pre"] is None or found["eo_post"] is None or found["sar_pre"] is None or found["sar_post"] is None:
        raise FileNotFoundError(
            "Could not resolve eo_pre/eo_post/sar_pre/sar_post. "
            "Set all four in config.folders relative to each split, or use a layout discover_modal_folders understands."
        )

    return {
        "eo_pre": found["eo_pre"],
        "eo_post": found["eo_post"],
        "sar_pre": found["sar_pre"],
        "sar_post": found["sar_post"],
        "mask": found.get("mask"),
    }


def build_index_quad(
    eo_pre_dir: Path,
    eo_post_dir: Path,
    sar_pre_dir: Path,
    sar_post_dir: Path,
    mask_dir: Path | None,
) -> list[SampleRecord]:
    """Four-folder EO/SAR pre/post layout; match basenames without extension across folders."""
    eo_pre = _list_images(eo_pre_dir)
    if not eo_pre:
        raise FileNotFoundError(f"No images found in {eo_pre_dir}")

    def map_by_stem(paths: list[Path]) -> dict[str, Path]:
        return {p.stem: p for p in paths}

    eo_post = map_by_stem(_list_images(eo_post_dir))
    sar_pre = map_by_stem(_list_images(sar_pre_dir))
    sar_post = map_by_stem(_list_images(sar_post_dir))
    masks = map_by_stem(_list_images(mask_dir)) if mask_dir is not None else {}

    samples: list[SampleRecord] = []
    for p in eo_pre:
        stem = p.stem
        if stem not in eo_post or stem not in sar_pre or stem not in sar_post:
            continue
        samples.append(
            SampleRecord(
                mask=masks.get(stem),
                quad=(p, eo_post[stem], sar_pre[stem], sar_post[stem]),
            )
        )

    if not samples:
        raise RuntimeError(
            "No matching samples found across modalities/times using filename stems. "
            "If the dataset uses different naming, update build_index_quad accordingly."
        )
    return samples


def build_sample_list(split_root: Path, overrides: dict[str, Any] | None) -> tuple[Literal["quad", "prepost"], list[SampleRecord]]:
    """
    Pick dataset layout and build sample list for one split (train/val/test root folder).

    Priority:
    1) If config.folders sets all four quad paths -> quad layout.
    2) If config.folders sets pre_event and post_event -> prepost (optional target/mask folder).
    3) If split contains pre-event/, post-event/, target/ -> prepost (GalaxEye release layout).
    4) Else quad auto-discovery under split_root.
    """
    o = overrides or {}
    keys_four = ("eo_pre", "eo_post", "sar_pre", "sar_post")

    if all(o.get(k) not in (None, "") for k in keys_four):
        folders = resolve_modal_folders(split_root, o)
        samples = build_index_quad(
            folders["eo_pre"],  # type: ignore[arg-type]
            folders["eo_post"],  # type: ignore[arg-type]
            folders["sar_pre"],  # type: ignore[arg-type]
            folders["sar_post"],  # type: ignore[arg-type]
            folders["mask"],
        )
        return "quad", samples

    pre_ev = o.get("pre_event") or o.get("pre-event")
    post_ev = o.get("post_event") or o.get("post-event")
    tgt = o.get("target") or o.get("mask")

    if pre_ev and post_ev:
        pre_d = _rel(split_root, str(pre_ev))
        post_d = _rel(split_root, str(post_ev))
        mask_d = _rel(split_root, str(tgt)) if tgt not in (None, "") else discover_mask_folder(split_root)
        samples = build_index_pre_post(pre_d, post_d, mask_d)
        return "prepost", samples

    if _has_prepost_target_layout(split_root):
        samples = build_index_pre_post(
            split_root / "pre-event",
            split_root / "post-event",
            split_root / "target",
        )
        return "prepost", samples

    folders = resolve_modal_folders(split_root, o)
    samples = build_index_quad(
        folders["eo_pre"],  # type: ignore[arg-type]
        folders["eo_post"],  # type: ignore[arg-type]
        folders["sar_pre"],  # type: ignore[arg-type]
        folders["sar_post"],  # type: ignore[arg-type]
        folders["mask"],
    )
    return "quad", samples


def _augment(x: np.ndarray, m: np.ndarray | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Geometric + photometric augmentations sourced entirely from the patch itself.

    Geometric ops (applied to BOTH image and mask so they stay aligned):
      - Random horizontal flip  (p=0.5)
      - Random vertical flip    (p=0.5)
      - Random 90° rotation     (p=0.5, k ∈ {1,2,3})

    Photometric ops (applied to image channels only, NOT mask):
      - Random brightness shift  ±10 %  (p=0.5)
      - Random contrast scale    0.85–1.15× (p=0.5)
      - Gaussian noise  σ≤0.02   (p=0.3)  — simulates SAR speckle

    x shape : (C, H, W)   float32 in [0, 1]
    m shape : (H, W)      uint8 binary  or None
    """
    # ── Geometric (same op on x and m) ──────────────────────────────────────
    if rng.random() < 0.5:                          # hflip
        x = x[:, :, ::-1].copy()
        if m is not None:
            m = m[:, ::-1].copy()

    if rng.random() < 0.5:                          # vflip
        x = x[:, ::-1, :].copy()
        if m is not None:
            m = m[::-1, :].copy()

    if rng.random() < 0.5:                          # rot90
        k = int(rng.integers(1, 4))
        x = np.rot90(x, k=k, axes=(1, 2)).copy()
        if m is not None:
            m = np.rot90(m, k=k, axes=(0, 1)).copy()

    # ── Photometric (image only) ─────────────────────────────────────────────
    if rng.random() < 0.5:                          # brightness
        delta = rng.uniform(-0.10, 0.10)
        x = (x + delta).clip(0.0, 1.0)

    if rng.random() < 0.5:                          # contrast
        factor = rng.uniform(0.85, 1.15)
        mean = x.mean(axis=(1, 2), keepdims=True)
        x = ((x - mean) * factor + mean).clip(0.0, 1.0)

    if rng.random() < 0.3:                          # noise (SAR speckle proxy)
        sigma = rng.uniform(0.0, 0.02)
        x = (x + rng.normal(0.0, sigma, x.shape)).clip(0.0, 1.0).astype(np.float32)

    return x.astype(np.float32), m


class ChangeDetectionDataset(Dataset):
    def __init__(
        self,
        samples: list[SampleRecord],
        img_size: int,
        with_mask: bool,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.samples  = samples
        self.img_size = img_size
        self.with_mask = with_mask
        self.augment  = augment
        # Per-worker RNG seeded deterministically so runs are reproducible
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]

        def norm(arr: np.ndarray) -> np.ndarray:
            mx = float(np.max(arr)) if np.max(arr) > 0 else 1.0
            return (arr / mx).astype(np.float32)

        def to_chw(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return np.transpose(arr, (2, 0, 1))

        # ── Load + resize ────────────────────────────────────────────────────
        if s.quad is not None:
            eo_pre_p, eo_post_p, sar_pre_p, sar_post_p = s.quad
            eo_pre  = norm(_resize(_read_image(eo_pre_p),  self.img_size, is_mask=False))
            eo_post = norm(_resize(_read_image(eo_post_p), self.img_size, is_mask=False))
            sar_pre  = norm(_resize(_read_image(sar_pre_p),  self.img_size, is_mask=False))
            sar_post = norm(_resize(_read_image(sar_post_p), self.img_size, is_mask=False))
            x = np.concatenate([to_chw(eo_pre), to_chw(eo_post),
                                 to_chw(sar_pre), to_chw(sar_post)], axis=0)
            ref_name = eo_pre_p.name
        else:
            assert s.prepost is not None
            pre_p, post_p = s.prepost
            pre  = norm(_resize(_read_image(pre_p),  self.img_size, is_mask=False))
            post = norm(_resize(_read_image(post_p), self.img_size, is_mask=False))
            x = np.concatenate([to_chw(pre), to_chw(post)], axis=0)
            ref_name = pre_p.name

        # ── Load mask ────────────────────────────────────────────────────────
        m: np.ndarray | None = None
        if self.with_mask:
            if s.mask is None:
                raise FileNotFoundError(f"Mask missing for sample {ref_name}")
            m = remap_mask_to_binary(
                _resize(_read_mask(s.mask), self.img_size, is_mask=True)
            )

        # ── Augmentation (train split only) ──────────────────────────────────
        if self.augment:
            rng = np.random.default_rng(self._base_seed + idx)
            x, m = _augment(x, m, rng)

        # ── Pack tensors ─────────────────────────────────────────────────────
        out: dict[str, torch.Tensor] = {"image": torch.from_numpy(x).float()}
        if m is not None:
            out["mask"] = torch.from_numpy(m[None, :, :].astype(np.float32))

        return out

