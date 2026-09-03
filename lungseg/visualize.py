"""Overlay and figure rendering for qualitative review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .postprocess import mask_boundary


def to_bgr(image: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0) * 255.0
    return cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 200, 255),
    alpha: float = 0.35,
    outline: bool = True,
) -> np.ndarray:
    canvas = to_bgr(image)
    binary = np.asarray(mask) > 0
    if binary.any():
        tint = np.zeros_like(canvas)
        tint[binary] = color
        canvas = cv2.addWeighted(canvas, 1.0, tint, alpha, 0.0)
        if outline:
            canvas[mask_boundary(binary.astype(np.uint8), 2) > 0] = color
    return canvas


def draw_candidates(canvas: np.ndarray, candidates, color: tuple[int, int, int] = (60, 60, 255)) -> np.ndarray:
    canvas = canvas.copy()
    for candidate in candidates:
        cv2.circle(canvas, (candidate.x, candidate.y), max(4, int(candidate.radius_px * 1.6)), color, 2)
    return canvas


def result_figure(
    image: np.ndarray,
    mask: np.ndarray,
    candidates=None,
    label: str | None = None,
) -> np.ndarray:
    """The deliverable strip for one slice: input, detected field, extracted lung region."""
    from .classical import extract_lung_region

    outlined = overlay_mask(image, mask)
    if candidates:
        outlined = draw_candidates(outlined, candidates)
    extracted = to_bgr(extract_lung_region(image, mask))

    panels = [to_bgr(image), outlined, extracted]
    titles = ["input slice", "lung field", "extracted region"]
    for panel, title in zip(panels, titles):
        cv2.putText(panel, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    figure = np.hstack(panels)
    if label:
        cv2.putText(figure, label, (6, figure.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return figure


def contact_sheet(figures: list[np.ndarray], columns: int = 2) -> np.ndarray:
    """Stack per-slice result strips into one reviewable sheet."""
    if not figures:
        return np.zeros((1, 1, 3), np.uint8)
    height = min(f.shape[0] for f in figures)
    width = min(f.shape[1] for f in figures)
    tiles = [cv2.resize(f, (width, height)) for f in figures]
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def comparison_panel(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Horizontal strip: original slice, then one overlay per mask."""
    panels = [to_bgr(image)]
    labels = ["input"]
    for name, mask in masks.items():
        panel = overlay_mask(image, mask)
        if reference is not None:
            panel[mask_boundary((np.asarray(reference) > 0).astype(np.uint8), 2) > 0] = (0, 255, 0)
        panels.append(panel)
        labels.append(name)

    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.hstack(panels)


def save_panel(path: str | Path, panel: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), panel)


def plot_history(history: dict, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()
    ax_dice.plot(epochs, history["val_dice"], color="tab:green")
    ax_dice.set(xlabel="epoch", ylabel="Dice", title="Validation Dice")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
