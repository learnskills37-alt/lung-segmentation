"""Numpy segmentation metrics (no torch dependency)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask) > 0


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _binarize(pred), _binarize(target)
    denominator = p.sum() + t.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, t).sum() / denominator)


def iou(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _binarize(pred), _binarize(target)
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, t).sum() / union)


def precision_recall(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    p, t = _binarize(pred), _binarize(target)
    tp = float(np.logical_and(p, t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    fn = float(np.logical_and(~p, t).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0
    return precision, recall


def specificity(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _binarize(pred), _binarize(target)
    tn = float(np.logical_and(~p, ~t).sum())
    fp = float(np.logical_and(p, ~t).sum())
    return tn / (tn + fp) if tn + fp > 0 else 1.0


def hausdorff95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile symmetric surface distance in pixels (inf if one mask is empty)."""
    p, t = _binarize(pred), _binarize(target)
    if not p.any() or not t.any():
        return float("inf")
    dist_to_target = ndi.distance_transform_edt(~t)
    dist_to_pred = ndi.distance_transform_edt(~p)
    forward = dist_to_target[_surface(p)]
    backward = dist_to_pred[_surface(t)]
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndi.binary_erosion(mask, iterations=1, border_value=0)
    return mask & ~eroded


def all_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    precision, recall = precision_recall(pred, target)
    return {
        "dice": dice(pred, target),
        "iou": iou(pred, target),
        "precision": precision,
        "recall": recall,
        "specificity": specificity(pred, target),
        "hd95": hausdorff95(pred, target),
    }


def aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].keys()
    summary = {}
    for key in keys:
        values = np.array([r[key] for r in records], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"{key}_mean"] = float(finite.mean()) if finite.size else float("nan")
        summary[f"{key}_std"] = float(finite.std()) if finite.size else float("nan")
    return summary
