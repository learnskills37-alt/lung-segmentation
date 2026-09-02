"""U-Net training loop with confidence-weighted pseudo-labels."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .losses import BCEDiceLoss
from .unet import UNet, count_parameters


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    image_size: int = 256
    base_channels: int = 32
    depth: int = 4
    dropout: float = 0.1
    bce_weight: float = 0.5
    num_workers: int = 2
    amp: bool = True
    patience: int = 10
    seed: int = 0


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses, dices = [], []
    for images, masks, weights in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        losses.append(float(criterion(logits, masks, weights)))
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        intersection = (predictions * masks).sum(dim=(1, 2, 3))
        cardinality = predictions.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        dices.extend(((2 * intersection + 1e-6) / (cardinality + 1e-6)).cpu().tolist())
    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(dices)) if dices else 0.0


def train_unet(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: TrainConfig,
    output_dir: str | Path,
    device: str = "auto",
    verbose: bool = True,
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=len(train_dataset) > config.batch_size,
        pin_memory=resolved_device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers
    )

    model = UNet(
        base_channels=config.base_channels, depth=config.depth, dropout=config.dropout
    ).to(resolved_device)
    criterion = BCEDiceLoss(bce_weight=config.bce_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    use_amp = config.amp and resolved_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if verbose:
        print(f"device={resolved_device} params={count_parameters(model):,} "
              f"train={len(train_dataset)} val={len(val_dataset)}")

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice, best_epoch = -1.0, -1
    checkpoint_path = output_dir / "unet_best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        started = time.time()
        for images, masks, weights in train_loader:
            images, masks = images.to(resolved_device), masks.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), masks, weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        scheduler.step()

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss, val_dice = evaluate_epoch(model, val_loader, criterion, resolved_device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "val_dice": val_dice,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        if verbose:
            print(
                f"epoch {epoch:3d}/{config.epochs}  train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  ({time.time() - started:.1f}s)"
                + ("  *" if epoch == best_epoch else "")
            )

        if epoch - best_epoch >= config.patience:
            if verbose:
                print(f"early stop: no improvement for {config.patience} epochs")
            break

    summary = {
        "history": history,
        "best_val_dice": best_dice,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "config": asdict(config),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
