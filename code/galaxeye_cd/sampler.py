from __future__ import annotations

import cv2
import numpy as np
from torch.utils.data import WeightedRandomSampler

from .dataset import ChangeDetectionDataset, remap_mask_to_binary


def build_weighted_sampler(
    dataset: ChangeDetectionDataset,
    multiplier: float = 5.0,
) -> WeightedRandomSampler:
    """
    Weighted sampler that oversamples patches containing change pixels.

    Each sample gets weight = 1 + multiplier * change_pixel_fraction.
    A pure no-change patch gets weight 1; a patch that is 10% change and
    multiplier=5 gets weight 1.5 — so it is 1.5× more likely to be drawn.

    This is not new data — it is smarter batching of the existing training set.
    Replacement=True so every epoch sees roughly len(dataset) steps.
    """
    print("Building weighted sampler (reading masks)...")
    weights: list[float] = []

    for record in dataset.samples:
        frac = 0.0
        if record.mask is not None:
            m = cv2.imread(str(record.mask), cv2.IMREAD_UNCHANGED)
            if m is not None:
                if m.ndim == 3:
                    m = m[:, :, 0]
                binary = remap_mask_to_binary(m.astype(np.uint8))
                frac = float(binary.mean())
        weights.append(1.0 + multiplier * frac)

    print(f"  Sampler: {len(weights)} samples, "
          f"mean weight={np.mean(weights):.3f}, "
          f"max weight={max(weights):.3f}")

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
