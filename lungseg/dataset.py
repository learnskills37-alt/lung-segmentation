"""Dataset assembly: pairs slices with masks (curated or classical pseudo-labels)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .classical import extract_lung_fields
from .io_utils import find_images, load_image, load_mask, resize

MASK_DIR_NAMES = ("mask", "masks", "label", "labels", "segmentation", "segmentations", "gt")


@dataclass
class Sample:
    image_path: Path
    mask_path: Path | None = None
    weight: float = 1.0


def find_mask_for(image_path: Path, mask_root: Path | None) -> Path | None:
    """Locate a curated mask: same stem in `mask_root`, or in a sibling masks/ folder."""
    candidates: list[Path] = []
    if mask_root is not None:
        candidates.append(mask_root / image_path.name)
        candidates.extend(mask_root.glob(f"{image_path.stem}.*"))
    for name in MASK_DIR_NAMES:
        sibling = image_path.parent.parent / name
        if sibling.is_dir():
            candidates.extend(sibling.glob(f"{image_path.stem}.*"))
    for candidate in candidates:
        if candidate.is_file() and candidate.resolve() != image_path.resolve():
            return candidate
    return None


def collect_samples(
    image_root: str | Path,
    mask_root: str | Path | None = None,
    exclude_dirs: tuple[str, ...] = MASK_DIR_NAMES,
) -> list[Sample]:
    mask_root = Path(mask_root) if mask_root else None
    samples = []
    for path in find_images(image_root):
        parts = {part.lower() for part in path.parts}
        if parts & set(exclude_dirs):
            continue
        samples.append(Sample(image_path=path, mask_path=find_mask_for(path, mask_root)))
    return samples


def generate_pseudo_masks(
    samples: list[Sample],
    out_dir: str | Path,
    min_score: float = 0.5,
    verbose: bool = True,
) -> list[Sample]:
    """Run the classical extractor over every slice and keep the confident masks."""
    from .io_utils import save_mask

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept: list[Sample] = []
    scores: dict[str, float] = {}

    for index, sample in enumerate(samples):
        image = load_image(sample.image_path)
        result = extract_lung_fields(image)
        scores[sample.image_path.name] = round(result.score, 4)
        if result.score < min_score or result.mask.sum() == 0:
            continue
        mask_path = out_dir / f"{sample.image_path.stem}.png"
        save_mask(mask_path, result.mask)
        kept.append(Sample(sample.image_path, mask_path, weight=result.score))
        if verbose and (index + 1) % 200 == 0:
            print(f"  pseudo-labelled {index + 1}/{len(samples)} slices")

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    return kept


def split_samples(
    samples: list[Sample], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Sample], list[Sample]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    n_val = max(1, int(round(val_fraction * len(samples)))) if len(samples) > 1 else 0
    val_idx = set(order[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def augment(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    if rng.random() < 0.5:
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()

    angle = float(rng.uniform(-12.0, 12.0))
    scale = float(rng.uniform(0.9, 1.1))
    tx, ty = (float(rng.uniform(-0.05, 0.05) * s) for s in (w, h))
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    image = image * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.05, 0.05))
    if rng.random() < 0.3:
        image = image + rng.normal(0.0, 0.02, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32), mask.astype(np.uint8)


class LungSegmentationDataset(Dataset):
    """Yields (image[1,H,W], mask[1,H,W], weight) tensors."""

    def __init__(
        self,
        samples: list[Sample],
        image_size: int = 256,
        train: bool = False,
        seed: int = 0,
        fallback_to_classical: bool = True,
    ):
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.fallback_to_classical = fallback_to_classical
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = load_image(sample.image_path)
        if sample.mask_path is not None:
            mask = load_mask(sample.mask_path)
        elif self.fallback_to_classical:
            mask = extract_lung_fields(image).mask
        else:
            mask = np.zeros_like(image, dtype=np.uint8)

        size = (self.image_size, self.image_size)
        image = resize(image, size)
        mask = resize(mask, size, is_mask=True)
        if self.train:
            image, mask = augment(image, mask, self.rng)

        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            torch.tensor(sample.weight, dtype=torch.float32),
        )


class ArrayDataset(Dataset):
    """In-memory dataset, used by the phantom demo and the tests."""

    def __init__(
        self,
        images: list[np.ndarray],
        masks: list[np.ndarray],
        image_size: int = 256,
        train: bool = False,
        seed: int = 0,
    ):
        self.images = images
        self.masks = masks
        self.image_size = image_size
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        size = (self.image_size, self.image_size)
        image = resize(self.images[index], size)
        mask = resize(self.masks[index], size, is_mask=True)
        if self.train:
            image, mask = augment(image, mask, self.rng)
        return (
            torch.from_numpy(image[None]).float(),
            torch.from_numpy(mask[None]).float(),
            torch.tensor(1.0, dtype=torch.float32),
        )
