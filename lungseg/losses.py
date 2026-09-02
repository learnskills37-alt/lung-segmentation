"""Segmentation losses used for training the U-Net."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """Weighted sum of pixel-wise BCE and soft Dice; `sample_weight` down-weights
    low-confidence pseudo-labels produced by the classical stage."""

    def __init__(self, bce_weight: float = 0.5, pos_weight: float | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, sample_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        pos_weight = (
            torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        ).mean(dim=(1, 2, 3))

        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dims)
        cardinality = probs.sum(dims) + targets.sum(dims)
        dice = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        per_sample = self.bce_weight * bce + (1.0 - self.bce_weight) * dice
        if sample_weight is not None:
            weight = sample_weight.to(per_sample.device).view(-1)
            return (per_sample * weight).sum() / weight.sum().clamp_min(1e-6)
        return per_sample.mean()
