"""Synthetic chest-CT phantoms.

Used for smoke tests and for `lungseg demo`, so the whole pipeline can be exercised
end to end on machines that have no access to the Kaggle dataset.
"""

from __future__ import annotations

import numpy as np


def _ellipse(shape: tuple[int, int], cy: float, cx: float, ry: float, rx: float, angle: float = 0.0) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    y = (yy - cy * h) / (ry * h)
    x = (xx - cx * w) / (rx * w)
    cos, sin = np.cos(angle), np.sin(angle)
    return ((x * cos + y * sin) ** 2 + (-x * sin + y * cos) ** 2) <= 1.0


def make_phantom(
    size: int = 256,
    rng: np.random.Generator | None = None,
    n_nodules: int = 2,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
    """Return (slice in [0, 1], lung mask, nodule centres as (y, x, radius))."""
    rng = rng or np.random.default_rng(0)
    shape = (size, size)
    image = np.zeros(shape, np.float32)  # air outside the body

    body = _ellipse(shape, 0.52, 0.5, 0.44, 0.40)
    image[body] = 0.55  # soft tissue

    for cx, sign in ((0.32, -1.0), (0.68, 1.0)):
        offset = rng.uniform(-0.015, 0.015)
        lung = _ellipse(
            shape,
            0.50 + offset,
            cx + offset,
            rng.uniform(0.24, 0.29),
            rng.uniform(0.12, 0.15),
            angle=sign * rng.uniform(0.0, 0.22),
        )
        image[lung] = 0.06

    lung_mask = (image < 0.2) & body

    spine = _ellipse(shape, 0.78, 0.5, 0.07, 0.07)
    image[spine] = 0.95
    for i in range(9):  # ribs
        angle = np.pi * (0.12 + 0.09 * i)
        rib = _ellipse(shape, 0.52, 0.5, 0.43, 0.39) & ~_ellipse(shape, 0.52, 0.5, 0.405, 0.365)
        band = np.zeros(shape, bool)
        yy, xx = np.mgrid[0:size, 0:size]
        band[np.abs((yy - 0.52 * size) * np.cos(angle) - (xx - 0.5 * size) * np.sin(angle)) < 3] = True
        image[rib & band] = 0.92

    vessels = np.zeros(shape, bool)  # hilar vessels inside the fields
    for _ in range(14):
        cy, cx = rng.uniform(0.35, 0.65), rng.uniform(0.28, 0.72)
        vessels |= _ellipse(shape, cy, cx, rng.uniform(0.006, 0.016), rng.uniform(0.006, 0.016))
    image[vessels & lung_mask] = 0.6

    nodules: list[tuple[int, int, int]] = []
    lung_pixels = np.argwhere(lung_mask)
    for _ in range(n_nodules):
        if lung_pixels.size == 0:
            break
        cy, cx = lung_pixels[rng.integers(len(lung_pixels))]
        radius = int(rng.integers(max(3, size // 64), max(5, size // 26)))
        nodule = _ellipse(shape, cy / size, cx / size, radius / size, radius / size)
        image[nodule] = float(rng.uniform(0.55, 0.8))
        nodules.append((int(cy), int(cx), radius))

    image = np.clip(image + rng.normal(0.0, noise, shape).astype(np.float32), 0.0, 1.0)
    return image, lung_mask.astype(np.uint8), nodules


def make_dataset(
    n: int = 24, size: int = 256, seed: int = 0
) -> tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int, int]]]]:
    rng = np.random.default_rng(seed)
    images, masks, nodules = [], [], []
    for _ in range(n):
        image, mask, nods = make_phantom(size=size, rng=rng, n_nodules=int(rng.integers(0, 4)))
        images.append(image)
        masks.append(mask)
        nodules.append(nods)
    return images, masks, nodules
