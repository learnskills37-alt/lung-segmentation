"""Compare the classical, U-Net and hybrid stages against reference masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .classical import extract_lung_fields
from .hybrid import fuse
from .io_utils import load_image, load_mask
from .metrics import aggregate, all_metrics
from .postprocess import refine_lung_mask
from .visualize import comparison_panel, save_panel


def evaluate_paths(
    pairs: list[tuple[Path, Path]],
    predictor=None,
    output_dir: str | Path | None = None,
    threshold: float = 0.5,
    n_panels: int = 8,
    verbose: bool = True,
) -> dict:
    per_method: dict[str, list[dict]] = {"classical": [], "unet": [], "hybrid": []}

    for index, (image_path, mask_path) in enumerate(pairs):
        image = load_image(image_path)
        reference = load_mask(mask_path)
        if reference.shape != image.shape:
            from .io_utils import resize

            reference = resize(reference, image.shape, is_mask=True)

        classical = extract_lung_fields(image)
        masks = {"classical": refine_lung_mask(classical.mask)}

        if predictor is not None:
            probability = predictor(image)
            masks["unet"] = fuse(probability, classical.mask, classical.score, "unet", threshold)
            masks["hybrid"] = fuse(probability, classical.mask, classical.score, "hybrid", threshold)

        for name, mask in masks.items():
            per_method[name].append(all_metrics(mask, reference))

        if output_dir and index < n_panels:
            panel = comparison_panel(image, masks, reference=reference)
            save_panel(Path(output_dir) / "panels" / f"{image_path.stem}.png", panel)

        if verbose and (index + 1) % 50 == 0:
            print(f"  evaluated {index + 1}/{len(pairs)} slices")

    summary = {name: aggregate(records) for name, records in per_method.items() if records}
    summary["n_slices"] = len(pairs)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
    return summary


def format_summary(summary: dict) -> str:
    header = f"{'method':<12}{'Dice':>10}{'IoU':>10}{'Prec':>10}{'Recall':>10}{'HD95':>10}"
    lines = [header, "-" * len(header)]
    for method in ("classical", "unet", "hybrid"):
        stats = summary.get(method)
        if not stats:
            continue
        lines.append(
            f"{method:<12}"
            f"{stats['dice_mean']:>10.4f}{stats['iou_mean']:>10.4f}"
            f"{stats['precision_mean']:>10.4f}{stats['recall_mean']:>10.4f}"
            f"{stats['hd95_mean']:>10.2f}"
        )
    lines.append(f"\nslices: {summary.get('n_slices', 0)}")
    return "\n".join(lines)


def evaluate_arrays(
    images: list[np.ndarray],
    references: list[np.ndarray],
    predictor=None,
    threshold: float = 0.5,
) -> dict:
    per_method: dict[str, list[dict]] = {"classical": [], "unet": [], "hybrid": []}
    for image, reference in zip(images, references):
        classical = extract_lung_fields(image)
        masks = {"classical": refine_lung_mask(classical.mask)}
        if predictor is not None:
            probability = predictor(image)
            masks["unet"] = fuse(probability, classical.mask, classical.score, "unet", threshold)
            masks["hybrid"] = fuse(probability, classical.mask, classical.score, "hybrid", threshold)
        for name, mask in masks.items():
            per_method[name].append(all_metrics(mask, reference))

    summary = {name: aggregate(records) for name, records in per_method.items() if records}
    summary["n_slices"] = len(images)
    return summary
