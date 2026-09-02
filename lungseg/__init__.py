"""Lung region extraction from chest CT slices: classical CV + U-Net + hybrid fusion."""

from .classical import LungFieldResult, extract_lung_fields, extract_lung_region
from .hybrid import SegmentationResult, fuse, segment_slice
from .metrics import all_metrics, dice, iou
from .nodules import detect_nodule_candidates

__version__ = "0.1.0"

__all__ = [
    "LungFieldResult",
    "SegmentationResult",
    "all_metrics",
    "detect_nodule_candidates",
    "dice",
    "extract_lung_fields",
    "extract_lung_region",
    "fuse",
    "iou",
    "segment_slice",
]
