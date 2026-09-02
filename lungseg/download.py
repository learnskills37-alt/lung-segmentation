"""Kaggle dataset retrieval and inspection."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

DATASET_ID = "ucimachinelearning/lung-nodule-dataset"


def download_dataset(dataset_id: str = DATASET_ID, force: bool = False) -> Path:
    """Download via kagglehub and return the local cache path.

    Requires Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)
    and outbound access to kaggle.com.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub first: pip install kagglehub") from exc

    path = kagglehub.dataset_download(dataset_id, force_download=force)
    print(f"Path to dataset files: {path}")
    return Path(path)


def summarize_dataset(root: str | Path, max_examples: int = 5) -> dict:
    """Report the layout of a downloaded dataset: extensions, counts, sample paths."""
    from .io_utils import SUPPORTED_EXTS, find_images, load_image

    root = Path(root)
    all_files = [p for p in root.rglob("*") if p.is_file()]
    extensions = Counter(p.suffix.lower() for p in all_files)
    images = find_images(root)

    shapes: Counter = Counter()
    for path in images[:20]:
        try:
            shapes[load_image(path).shape] += 1
        except Exception:  # noqa: BLE001 - a corrupt sample must not abort the survey
            shapes["unreadable"] += 1

    summary = {
        "root": str(root),
        "n_files": len(all_files),
        "n_images": len(images),
        "extensions": dict(extensions.most_common()),
        "supported_extensions": sorted(SUPPORTED_EXTS),
        "subdirectories": sorted({str(p.relative_to(root).parent) for p in all_files})[:40],
        "sample_images": [str(p) for p in images[:max_examples]],
        "sampled_shapes": {str(k): v for k, v in shapes.items()},
    }
    return summary
