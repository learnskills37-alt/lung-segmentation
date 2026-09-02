"""Hybrid stage: fuse the classical lung-field mask with the U-Net probability map.

The classical extractor is precise on well-windowed slices but fails on pathological or
low-contrast ones; the U-Net degrades gracefully everywhere. Fusing them and weighting
by the classical confidence score keeps the strengths of both.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .classical import extract_lung_fields
from .postprocess import dilate, refine_lung_mask

FUSION_MODES = ("hybrid", "unet", "classical", "union", "intersection")


@dataclass
class SegmentationResult:
    mask: np.ndarray
    probability: np.ndarray | None
    classical_mask: np.ndarray
    classical_score: float
    mode: str


def fuse(
    unet_probability: np.ndarray | None,
    classical_mask: np.ndarray,
    classical_score: float,
    mode: str = "hybrid",
    threshold: float = 0.5,
    smooth_sigma: float = 2.0,
    guard_dilation: float = 0.06,
) -> np.ndarray:
    if mode not in FUSION_MODES:
        raise ValueError(f"mode must be one of {FUSION_MODES}, got {mode!r}")

    classical = (np.asarray(classical_mask) > 0).astype(np.float32)
    if mode == "classical" or unet_probability is None:
        return refine_lung_mask((classical > 0.5).astype(np.uint8))

    probability = np.asarray(unet_probability, dtype=np.float32)
    if mode == "unet":
        return refine_lung_mask((probability >= threshold).astype(np.uint8))

    unet_mask = (probability >= threshold).astype(np.uint8)
    classical_binary = (classical > 0.5).astype(np.uint8)
    if mode == "union":
        return refine_lung_mask(np.maximum(unet_mask, classical_binary))
    if mode == "intersection":
        return refine_lung_mask(np.minimum(unet_mask, classical_binary))

    # "hybrid": confidence-weighted blend, restricted to a dilated classical guard band
    # so the U-Net can recover missed tissue without leaking into the mediastinum.
    if classical_binary.any():
        blur = cv2.GaussianBlur(classical, (0, 0), smooth_sigma)
        weight = float(np.clip(classical_score, 0.0, 1.0)) * 0.5
        combined = (1.0 - weight) * probability + weight * blur
        guard = dilate(classical_binary, radius=max(3, int(guard_dilation * min(classical.shape))))
        combined = combined * guard
    else:
        combined = probability
    return refine_lung_mask((combined >= threshold).astype(np.uint8))


def segment_slice(
    image: np.ndarray,
    predictor=None,
    mode: str = "hybrid",
    threshold: float = 0.5,
    min_classical_score: float = 0.35,
) -> SegmentationResult:
    """Segment one slice. `predictor` maps an image to a probability map in [0, 1]."""
    classical = extract_lung_fields(image)
    score = classical.score if classical.score >= min_classical_score else 0.0
    mask_in = classical.mask if score > 0 else np.zeros_like(classical.mask)

    probability = predictor(image) if predictor is not None else None
    effective_mode = mode
    if probability is None:
        effective_mode = "classical"
    elif mode == "hybrid" and score == 0.0:
        effective_mode = "unet"  # classical stage is unreliable here, trust the network

    mask = fuse(probability, mask_in, score, mode=effective_mode, threshold=threshold)
    return SegmentationResult(mask, probability, classical.mask, classical.score, effective_mode)
