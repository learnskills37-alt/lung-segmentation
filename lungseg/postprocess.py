"""Binary-mask morphology helpers shared by the classical, U-Net and hybrid stages."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _disk(radius: int) -> np.ndarray:
    size = max(1, int(radius)) * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def morph_close(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _disk(radius))


def morph_open(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _disk(radius))


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return mask.astype(np.uint8)
    return cv2.dilate(mask.astype(np.uint8), _disk(radius))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    return ndi.binary_fill_holes(np.asarray(mask) > 0).astype(np.uint8)


def keep_largest_components(mask: np.ndarray, count: int = 2, min_area: int = 0) -> np.ndarray:
    mask = np.asarray(mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num)]
    areas.sort(reverse=True)
    keep = [idx for area, idx in areas[:count] if area >= min_area]
    return np.isin(labels, keep).astype(np.uint8)


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = np.asarray(mask).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return np.isin(labels, keep).astype(np.uint8)


def refine_lung_mask(
    mask: np.ndarray,
    max_regions: int = 2,
    close_fraction: float = 0.015,
    min_area_fraction: float = 0.003,
) -> np.ndarray:
    """Standard clean-up applied to any predicted lung mask."""
    mask = np.asarray(mask).astype(np.uint8)
    h, w = mask.shape
    mask = remove_small_objects(mask, int(min_area_fraction * h * w))
    mask = morph_close(mask, radius=max(2, int(close_fraction * min(h, w))))
    mask = fill_holes(mask)
    return keep_largest_components(mask, max_regions)


def mask_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    mask = np.asarray(mask).astype(np.uint8)
    eroded = cv2.erode(mask, _disk(thickness))
    return (mask - eroded).astype(np.uint8)
