"""Image discovery and loading for CT slices stored as images, DICOM or numpy arrays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
ARRAY_EXTS = {".npy"}
DICOM_EXTS = {".dcm", ".dicom", ".ima"}
SUPPORTED_EXTS = IMAGE_EXTS | ARRAY_EXTS | DICOM_EXTS

LUNG_WINDOW = (-600.0, 1500.0)  # (center, width) in Hounsfield units


def find_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    files = [p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(files)


def window_hu(array: np.ndarray, center: float, width: float) -> np.ndarray:
    low = center - width / 2.0
    high = center + width / 2.0
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def _load_dicom(path: Path) -> np.ndarray:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Reading DICOM requires `pip install pydicom`") from exc

    ds = pydicom.dcmread(str(path))
    array = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    hu = array * slope + intercept
    return window_hu(hu, *LUNG_WINDOW)


def load_image(path: str | Path) -> np.ndarray:
    """Load a single slice as float32 grayscale scaled to [0, 1]."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in DICOM_EXTS:
        return _load_dicom(path).astype(np.float32)

    if suffix in ARRAY_EXTS:
        array = np.load(path).astype(np.float32)
        if array.ndim == 3:
            array = array[array.shape[0] // 2] if array.shape[0] < array.shape[-1] else array[..., 0]
        # Raw CT arrays are stored in Hounsfield units; images are already 0-255 or 0-1.
        if array.min() < -100.0:
            return window_hu(array, *LUNG_WINDOW).astype(np.float32)
        return _to_unit_range(array)

    if suffix in IMAGE_EXTS:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise ValueError(f"Could not read image: {path}")
        if array.ndim == 3:
            array = cv2.cvtColor(array[..., :3], cv2.COLOR_BGR2GRAY)
        return _to_unit_range(array.astype(np.float32))

    raise ValueError(f"Unsupported file type: {path}")


def _to_unit_range(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    peak = float(array.max())
    if peak > 255.0:
        return np.clip(array / 65535.0, 0.0, 1.0)
    if peak > 1.0:
        return np.clip(array / 255.0, 0.0, 1.0)
    return np.clip(array, 0.0, 1.0)


def load_mask(path: str | Path) -> np.ndarray:
    array = load_image(path)
    return (array > 0.5).astype(np.uint8)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (np.asarray(mask) > 0).astype(np.uint8) * 255)


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0) * 255.0
        array = array.astype(np.uint8)
    cv2.imwrite(str(path), array)


def resize(array: np.ndarray, size: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(array, (size[1], size[0]), interpolation=interpolation)
    return resized.astype(np.uint8) if is_mask else resized.astype(np.float32)
