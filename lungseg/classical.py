"""Classical computer-vision extraction of the lung fields from a chest CT slice.

Pipeline: denoise -> Otsu split of air/tissue -> body mask -> air regions inside the
body -> geometric filtering of the lung candidates -> morphological repair so that
juxtapleural nodules and vessels stay inside the field.

The result carries a confidence score so that weak masks can be discarded before they
are used as pseudo-labels for the U-Net.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.filters import threshold_otsu

from .postprocess import fill_holes, keep_largest_components, morph_close, morph_open


@dataclass
class LungFieldResult:
    mask: np.ndarray
    body_mask: np.ndarray
    score: float
    stats: dict = field(default_factory=dict)


def _odd(value: int) -> int:
    value = max(3, int(value))
    return value if value % 2 == 1 else value + 1


def body_mask_from(tissue: np.ndarray) -> np.ndarray:
    """Patient body: the largest filled tissue component, minus the scanner table edges."""
    body = keep_largest_components(tissue, 1)
    body = fill_holes(body)
    body = morph_open(body, radius=max(2, int(0.006 * max(tissue.shape))))
    body = keep_largest_components(body, 1)
    return fill_holes(body)


def _clear_border(mask: np.ndarray, border: int = 1) -> np.ndarray:
    """Drop components touching the image frame (outside air, scanner table)."""
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask
    frame = np.zeros_like(mask, dtype=bool)
    frame[:border, :] = frame[-border:, :] = True
    frame[:, :border] = frame[:, -border:] = True
    touching = set(np.unique(labels[frame])) - {0}
    keep = np.isin(labels, list(set(range(1, num)) - touching))
    return keep.astype(np.uint8)


def _score_mask(mask: np.ndarray, components: int) -> tuple[float, dict]:
    h, w = mask.shape
    area_fraction = float(mask.sum()) / float(h * w)
    left = float(mask[:, : w // 2].sum())
    right = float(mask[:, w // 2 :].sum())
    total = left + right
    balance = 1.0 - abs(left - right) / total if total > 0 else 0.0

    # A lung field on an axial CT slice covers roughly 8-55% of the frame.
    if area_fraction <= 0.0:
        area_term = 0.0
    elif area_fraction < 0.08:
        area_term = area_fraction / 0.08
    elif area_fraction > 0.55:
        area_term = max(0.0, 1.0 - (area_fraction - 0.55) / 0.25)
    else:
        area_term = 1.0

    centroid_term = 1.0
    if total > 0:
        ys, xs = np.nonzero(mask)
        cx, cy = xs.mean() / w, ys.mean() / h
        centroid_term = float(np.clip(1.0 - 2.5 * np.hypot(cx - 0.5, cy - 0.5), 0.0, 1.0))

    component_term = 1.0 if components in (1, 2) else 0.6
    score = float(area_term * component_term * (0.45 + 0.35 * balance + 0.20 * centroid_term))
    stats = {
        "area_fraction": area_fraction,
        "balance": balance,
        "components": components,
        "centroid_term": centroid_term,
    }
    return float(np.clip(score, 0.0, 1.0)), stats


def extract_lung_fields(
    image: np.ndarray,
    min_area_fraction: float = 0.004,
    max_regions: int = 2,
    close_fraction: float = 0.02,
    repair_juxtapleural: bool = True,
) -> LungFieldResult:
    """Segment the lung fields of a single grayscale slice scaled to [0, 1]."""
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale slice, got shape {image.shape}")

    h, w = image.shape
    smooth = cv2.GaussianBlur(image.astype(np.float32), (_odd(0.01 * max(h, w)),) * 2, 0)

    try:
        threshold = float(threshold_otsu(smooth))
    except ValueError:  # flat image
        empty = np.zeros((h, w), np.uint8)
        return LungFieldResult(empty, empty, 0.0, {"reason": "flat_image"})

    tissue = (smooth > threshold).astype(np.uint8)
    body = body_mask_from(tissue)
    if body.sum() < 0.10 * h * w:  # tightly cropped scan: the frame is the body
        body = np.ones((h, w), np.uint8)

    air = ((smooth <= threshold).astype(np.uint8) & body).astype(np.uint8)
    air = _clear_border(air, border=max(1, int(0.01 * min(h, w))))
    air = morph_open(air, radius=max(1, int(0.004 * min(h, w))))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(air, connectivity=8)
    min_area = min_area_fraction * h * w
    candidates = [
        (stats[i, cv2.CC_STAT_AREA], i)
        for i in range(1, num)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not candidates:
        empty = np.zeros((h, w), np.uint8)
        return LungFieldResult(empty, body, 0.0, {"reason": "no_air_regions"})

    candidates.sort(reverse=True)
    largest_area = candidates[0][0]
    # A second lung field is comparable in size to the first; the trachea is not.
    selected = [idx for area, idx in candidates[:max_regions] if area >= 0.15 * largest_area]
    mask = np.isin(labels, selected).astype(np.uint8)

    mask = _close_per_component(mask, radius=max(2, int(close_fraction * min(h, w))))
    mask = fill_holes(mask)
    if repair_juxtapleural:
        mask = (include_juxtapleural(mask) & body).astype(np.uint8)
    mask = keep_largest_components(mask, max_regions)

    score, mask_stats = _score_mask(mask, len(selected))
    return LungFieldResult(mask.astype(np.uint8), body, score, mask_stats)


def _close_per_component(mask: np.ndarray, radius: int) -> np.ndarray:
    """Close each lung field on its own so the operation cannot bridge the mediastinum."""
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if num <= 2:
        return morph_close(mask, radius)
    closed = np.zeros_like(mask, dtype=np.uint8)
    for index in range(1, num):
        closed |= morph_close((labels == index).astype(np.uint8), radius)
    return closed


def include_juxtapleural(
    mask: np.ndarray, max_area_fraction: float = 0.06, min_fill: float = 0.35
) -> np.ndarray:
    """Re-attach nodules sitting on the pleural wall.

    A nodule touching the wall carves an indentation that morphological closing cannot
    bridge. Each lung is compared against its convex hull and the small, compact
    indentations are added back; the large mediastinal concavity fails the area test and
    thin slivers along a smooth boundary fail the fill test, so both are left out.
    """
    mask = np.asarray(mask).astype(np.uint8)
    out = mask.copy()
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for index in range(1, num):
        component = (labels == index).astype(np.uint8)
        area = int(stats[index, cv2.CC_STAT_AREA])
        deficiency = (_convex_hull_per_component(component) & (component == 0)).astype(np.uint8)
        d_num, d_labels, d_stats, _ = cv2.connectedComponentsWithStats(deficiency, connectivity=8)
        for d_index in range(1, d_num):
            d_area = int(d_stats[d_index, cv2.CC_STAT_AREA])
            box = d_stats[d_index, cv2.CC_STAT_WIDTH] * d_stats[d_index, cv2.CC_STAT_HEIGHT]
            if d_area > max_area_fraction * area or d_area / max(1, box) < min_fill:
                continue
            out[d_labels == d_index] = 1
    return out


def _convex_hull_per_component(mask: np.ndarray) -> np.ndarray:
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(hull_mask, [cv2.convexHull(contour)], -1, 1, thickness=cv2.FILLED)
    return hull_mask


def extract_lung_region(image: np.ndarray, mask: np.ndarray, background: float = 0.0) -> np.ndarray:
    """Blank out everything outside the lung fields, keeping the original intensities."""
    out = np.full_like(image, background, dtype=np.float32)
    binary = np.asarray(mask) > 0
    out[binary] = image[binary]
    return out


def crop_to_mask(image: np.ndarray, mask: np.ndarray, margin: int = 8) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if ys.size == 0:
        return image, (0, 0, image.shape[0], image.shape[1])
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1], (y0, x0, y1, x1)
