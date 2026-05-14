from __future__ import annotations

import torch
from torch import nn

# 8-element dihedral group D4: each entry is (augment_fn, inverse_fn).
# Applying aug to input, running the model, then applying inv to the output
# aligns every prediction back to the original orientation before averaging.
_D4: list[tuple] = [
    # identity
    (lambda x: x,
     lambda x: x),
    # horizontal flip (self-inverse)
    (lambda x: torch.flip(x, [-1]),
     lambda x: torch.flip(x, [-1])),
    # vertical flip (self-inverse)
    (lambda x: torch.flip(x, [-2]),
     lambda x: torch.flip(x, [-2])),
    # 90° CCW  →  inverse is 270° CCW
    (lambda x: torch.rot90(x, 1, [-2, -1]),
     lambda x: torch.rot90(x, 3, [-2, -1])),
    # 180° (self-inverse)
    (lambda x: torch.rot90(x, 2, [-2, -1]),
     lambda x: torch.rot90(x, 2, [-2, -1])),
    # 270° CCW  →  inverse is 90° CCW
    (lambda x: torch.rot90(x, 3, [-2, -1]),
     lambda x: torch.rot90(x, 1, [-2, -1])),
    # hflip then 90° CCW  →  inverse: 270° CCW then hflip
    (lambda x: torch.rot90(torch.flip(x, [-1]), 1, [-2, -1]),
     lambda x: torch.flip(torch.rot90(x, 3, [-2, -1]), [-1])),
    # vflip then 90° CCW  →  inverse: 270° CCW then vflip
    (lambda x: torch.rot90(torch.flip(x, [-2]), 1, [-2, -1]),
     lambda x: torch.flip(torch.rot90(x, 3, [-2, -1]), [-2])),
]


@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    8-fold test-time augmentation using the full dihedral group D4.

    For each of the 8 geometric transforms:
      1. augment the input
      2. run a forward pass
      3. apply the inverse transform to the output probability map
      4. accumulate

    Returns the pixel-wise mean probability map, shape (B, 1, H, W).
    The model should already be in eval() mode before calling this.
    """
    H, W = x.shape[-2], x.shape[-1]
    prob_sum = torch.zeros(x.shape[0], 1, H, W, device=x.device, dtype=torch.float32)

    for aug, inv in _D4:
        logits = model(aug(x))
        prob = torch.sigmoid(logits)
        prob_sum += inv(prob)

    return prob_sum / len(_D4)
