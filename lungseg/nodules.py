"""Nodule candidate detection inside the extracted lung fields.

A multi-scale Laplacian-of-Gaussian blob detector keyed to bright, compact structures,
followed by shape filtering that rejects the elongated cross-sections of vessels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from skimage.feature import blob_log

from .postprocess import morph_open


@dataclass
class NoduleCandidate:
    y: int
    x: int
    radius_px: float
    mean_intensity: float
    circularity: float
    score: float

    def as_dict(self) -> dict:
        return asdict(self)


def detect_nodule_candidates(
    image: np.ndarray,
    lung_mask: np.ndarray,
    min_radius_px: float = 2.0,
    max_radius_px: float = 16.0,
    intensity_margin: float = 0.12,
    min_circularity: float = 0.55,
    max_candidates: int = 20,
) -> list[NoduleCandidate]:
    lung_mask = (np.asarray(lung_mask) > 0).astype(np.uint8)
    if lung_mask.sum() < 50:
        return []

    interior = morph_open(lung_mask, radius=2)
    parenchyma = image[interior > 0]
    if parenchyma.size == 0:
        return []
    background = float(np.median(parenchyma))

    working = np.where(interior > 0, image, background).astype(np.float32)
    blobs = blob_log(
        working,
        min_sigma=min_radius_px / np.sqrt(2),
        max_sigma=max_radius_px / np.sqrt(2),
        num_sigma=6,
        threshold=0.03,
        overlap=0.4,
    )

    candidates: list[NoduleCandidate] = []
    for y, x, sigma in blobs:
        y, x = int(round(y)), int(round(x))
        radius = float(sigma * np.sqrt(2))
        if interior[y, x] == 0:
            continue

        region = np.zeros_like(interior)
        cv2.circle(region, (x, y), max(1, int(radius)), 1, thickness=cv2.FILLED)
        inside = region.astype(bool) & (interior > 0)
        if inside.sum() < 4:
            continue

        mean_intensity = float(image[inside].mean())
        if mean_intensity - background < intensity_margin:
            continue

        blob_mask = ((image >= (background + mean_intensity) / 2.0) & inside).astype(np.uint8)
        circularity = _circularity(blob_mask)
        if circularity < min_circularity:
            continue

        contrast = float(np.clip((mean_intensity - background) / max(1e-6, 1.0 - background), 0.0, 1.0))
        candidates.append(
            NoduleCandidate(y, x, radius, mean_intensity, circularity, score=contrast * circularity)
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def _circularity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter <= 0 or area <= 0:
        return 0.0
    return float(np.clip(4.0 * np.pi * area / (perimeter**2), 0.0, 1.0))
