"""Inference: U-Net probability maps, hybrid fusion and lung-region extraction."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from .classical import crop_to_mask, extract_lung_region
from .hybrid import segment_slice
from .io_utils import load_image, save_image, save_mask
from .nodules import detect_nodule_candidates
from .train import resolve_device
from .unet import UNet
from .visualize import contact_sheet, draw_candidates, overlay_mask, result_figure, save_panel


class UNetPredictor:
    """Wraps a trained checkpoint: image in [0, 1] -> probability map at input resolution."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto", image_size: int | None = None):
        self.device = resolve_device(device)
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        config = checkpoint.get("config", {})
        self.image_size = image_size or config.get("image_size", 256)
        self.model = UNet(
            base_channels=config.get("base_channels", 32),
            depth=config.get("depth", 4),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        resized = cv2.resize(image.astype(np.float32), (self.image_size, self.image_size), cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized[None, None]).float().to(self.device)
        probability = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()
        return cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)


def load_predictor(checkpoint_path: str | Path | None, device: str = "auto") -> UNetPredictor | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return UNetPredictor(path, device=device)


def predict_paths(
    image_paths: list[Path],
    output_dir: str | Path,
    predictor: UNetPredictor | None = None,
    mode: str = "hybrid",
    threshold: float = 0.5,
    detect_nodules: bool = False,
    save_overlays: bool = True,
    crop_to_lungs: bool = False,
    max_sheet_slices: int = 12,
    verbose: bool = True,
) -> list[dict]:
    output_dir = Path(output_dir)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "lung_regions").mkdir(parents=True, exist_ok=True)
    if save_overlays:
        (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
        (output_dir / "results").mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    nodule_rows: list[dict] = []
    figures: list[np.ndarray] = []

    for index, path in enumerate(image_paths):
        image = load_image(path)
        result = segment_slice(image, predictor=predictor, mode=mode, threshold=threshold)
        stem = path.stem

        save_mask(output_dir / "masks" / f"{stem}.png", result.mask)
        region = extract_lung_region(image, result.mask)
        if crop_to_lungs and result.mask.any():
            region = crop_to_mask(region, result.mask)[0]
        save_image(output_dir / "lung_regions" / f"{stem}.png", region)

        candidates = []
        if detect_nodules:
            candidates = detect_nodule_candidates(image, result.mask)
            nodule_rows.extend({"image": path.name, **c.as_dict()} for c in candidates)

        if save_overlays:
            panel = overlay_mask(image, result.mask)
            if candidates:
                panel = draw_candidates(panel, candidates)
            save_panel(output_dir / "overlays" / f"{stem}.png", panel)

            figure = result_figure(image, result.mask, candidates, label=stem)
            save_panel(output_dir / "results" / f"{stem}.png", figure)
            if len(figures) < max_sheet_slices:
                figures.append(figure)

        records.append(
            {
                "image": str(path),
                "mask": str(output_dir / "masks" / f"{stem}.png"),
                "mode": result.mode,
                "classical_score": round(result.classical_score, 4),
                "lung_area_fraction": round(float(result.mask.mean()), 4),
                "n_nodule_candidates": len(candidates),
            }
        )
        if verbose and (index + 1) % 50 == 0:
            print(f"  segmented {index + 1}/{len(image_paths)} slices")

    if figures:
        save_panel(output_dir / "contact_sheet.png", contact_sheet(figures))
    _write_csv(output_dir / "predictions.csv", records)
    if detect_nodules:
        _write_csv(output_dir / "nodule_candidates.csv", nodule_rows)
    return records


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
