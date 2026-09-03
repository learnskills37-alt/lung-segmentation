"""Command line interface: `python -m lungseg <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lungseg", description="Lung region extraction from CT slices")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="Download the Kaggle lung nodule dataset")
    p_download.add_argument("--dataset-id", default="ucimachinelearning/lung-nodule-dataset")
    p_download.add_argument("--force", action="store_true")

    p_inspect = sub.add_parser("inspect", help="Summarize the layout of a dataset directory")
    p_inspect.add_argument("data_dir")

    p_masks = sub.add_parser("masks", help="Generate classical lung masks (pseudo-labels)")
    p_masks.add_argument("data_dir")
    p_masks.add_argument("--out", default="outputs/pseudo_masks")
    p_masks.add_argument("--min-score", type=float, default=0.5)
    p_masks.add_argument("--limit", type=int, default=None)

    p_train = sub.add_parser("train", help="Train the U-Net")
    p_train.add_argument("data_dir")
    p_train.add_argument("--mask-dir", default=None, help="Curated masks; defaults to pseudo-labels")
    p_train.add_argument("--out", default="outputs/run")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--image-size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--base-channels", type=int, default=32)
    p_train.add_argument("--depth", type=int, default=4)
    p_train.add_argument("--val-fraction", type=float, default=0.2)
    p_train.add_argument("--min-score", type=float, default=0.5)
    p_train.add_argument("--num-workers", type=int, default=2)
    p_train.add_argument("--limit", type=int, default=None)
    _add_common(p_train)

    p_predict = sub.add_parser("predict", help="Segment slices and extract the lung regions")
    p_predict.add_argument("data_dir")
    p_predict.add_argument("--checkpoint", default=None, help="Omit to run the classical stage alone")
    p_predict.add_argument("--out", default="outputs/predictions")
    p_predict.add_argument("--mode", default="hybrid", choices=["hybrid", "unet", "classical", "union", "intersection"])
    p_predict.add_argument("--threshold", type=float, default=0.5)
    p_predict.add_argument("--nodules", action="store_true", help="Also detect nodule candidates")
    p_predict.add_argument(
        "--crop", action="store_true", help="Trim each extracted region to the lung bounding box"
    )
    p_predict.add_argument("--no-overlays", action="store_true")
    p_predict.add_argument("--limit", type=int, default=None)
    _add_common(p_predict)

    p_eval = sub.add_parser("evaluate", help="Score the stages against reference masks")
    p_eval.add_argument("data_dir")
    p_eval.add_argument("--mask-dir", required=True)
    p_eval.add_argument("--checkpoint", default=None)
    p_eval.add_argument("--out", default="outputs/evaluation")
    p_eval.add_argument("--threshold", type=float, default=0.5)
    p_eval.add_argument("--limit", type=int, default=None)
    _add_common(p_eval)

    p_demo = sub.add_parser("demo", help="Run the full pipeline on synthetic CT phantoms")
    p_demo.add_argument("--out", default="outputs/demo")
    p_demo.add_argument("--n-train", type=int, default=48)
    p_demo.add_argument("--n-val", type=int, default=12)
    p_demo.add_argument("--epochs", type=int, default=8)
    p_demo.add_argument("--image-size", type=int, default=128)
    p_demo.add_argument("--batch-size", type=int, default=4)
    p_demo.add_argument("--base-channels", type=int, default=16)
    p_demo.add_argument("--depth", type=int, default=3)
    p_demo.add_argument("--num-workers", type=int, default=0)
    _add_common(p_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return globals()[f"_cmd_{args.command}"](args)


def _cmd_download(args) -> int:
    from .download import download_dataset, summarize_dataset

    path = download_dataset(args.dataset_id, force=args.force)
    print(json.dumps(summarize_dataset(path), indent=2))
    return 0


def _cmd_inspect(args) -> int:
    from .download import summarize_dataset

    print(json.dumps(summarize_dataset(args.data_dir), indent=2))
    return 0


def _cmd_masks(args) -> int:
    from .dataset import collect_samples, generate_pseudo_masks

    samples = collect_samples(args.data_dir)[: args.limit]
    if not samples:
        print(f"No supported images found under {args.data_dir}")
        return 1
    kept = generate_pseudo_masks(samples, args.out, min_score=args.min_score)
    print(f"{len(kept)}/{len(samples)} slices produced a confident lung mask (score >= {args.min_score})")
    print(f"masks written to {args.out}")
    return 0


def _cmd_train(args) -> int:
    from .dataset import LungSegmentationDataset, collect_samples, generate_pseudo_masks, split_samples
    from .train import TrainConfig, train_unet
    from .visualize import plot_history

    out_dir = Path(args.out)
    samples = collect_samples(args.data_dir, args.mask_dir)[: args.limit]
    if not samples:
        print(f"No supported images found under {args.data_dir}")
        return 1

    labelled = [s for s in samples if s.mask_path is not None]
    if labelled:
        print(f"Using {len(labelled)} curated masks")
    else:
        print("No curated masks found - generating classical pseudo-labels")
        labelled = generate_pseudo_masks(samples, out_dir / "pseudo_masks", min_score=args.min_score)
        print(f"{len(labelled)}/{len(samples)} slices kept as training targets")
    if len(labelled) < 4:
        print("Not enough usable training slices; lower --min-score or check the data")
        return 1

    train_samples, val_samples = split_samples(labelled, args.val_fraction)
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        base_channels=args.base_channels,
        depth=args.depth,
        num_workers=args.num_workers,
    )
    summary = train_unet(
        LungSegmentationDataset(train_samples, args.image_size, train=True),
        LungSegmentationDataset(val_samples, args.image_size, train=False),
        config,
        out_dir,
        device=args.device,
    )
    plot_history(summary["history"], out_dir / "training_curves.png")
    print(f"best val Dice {summary['best_val_dice']:.4f} at epoch {summary['best_epoch']}")
    print(f"checkpoint: {summary['checkpoint']}")
    return 0


def _cmd_predict(args) -> int:
    from .io_utils import find_images
    from .predict import load_predictor, predict_paths

    paths = find_images(args.data_dir)[: args.limit]
    if not paths:
        print(f"No supported images found under {args.data_dir}")
        return 1
    predictor = load_predictor(args.checkpoint, device=args.device)
    records = predict_paths(
        paths,
        args.out,
        predictor=predictor,
        mode=args.mode,
        threshold=args.threshold,
        detect_nodules=args.nodules,
        save_overlays=not args.no_overlays,
        crop_to_lungs=args.crop,
    )
    print(f"segmented {len(records)} slices -> {args.out}")
    return 0


def _cmd_evaluate(args) -> int:
    from .dataset import find_mask_for
    from .evaluate import evaluate_paths, format_summary
    from .io_utils import find_images
    from .predict import load_predictor

    mask_root = Path(args.mask_dir)
    pairs = []
    for path in find_images(args.data_dir)[: args.limit]:
        mask_path = find_mask_for(path, mask_root)
        if mask_path is not None:
            pairs.append((path, mask_path))
    if not pairs:
        print(f"No image/mask pairs found between {args.data_dir} and {args.mask_dir}")
        return 1

    predictor = load_predictor(args.checkpoint, device=args.device)
    summary = evaluate_paths(pairs, predictor=predictor, output_dir=args.out, threshold=args.threshold)
    print(format_summary(summary))
    print(f"\nwritten to {args.out}/evaluation.json")
    return 0


def _cmd_demo(args) -> int:
    from .dataset import ArrayDataset
    from .evaluate import evaluate_arrays, format_summary
    from .phantom import make_dataset
    from .predict import UNetPredictor
    from .train import TrainConfig, train_unet
    from .visualize import comparison_panel, plot_history, save_panel

    out_dir = Path(args.out)
    train_images, train_masks, _ = make_dataset(args.n_train, size=args.image_size, seed=0)
    val_images, val_masks, _ = make_dataset(args.n_val, size=args.image_size, seed=1)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        base_channels=args.base_channels,
        depth=args.depth,
        num_workers=args.num_workers,
        patience=args.epochs,
    )
    summary = train_unet(
        ArrayDataset(train_images, train_masks, args.image_size, train=True),
        ArrayDataset(val_images, val_masks, args.image_size, train=False),
        config,
        out_dir,
        device=args.device,
    )
    plot_history(summary["history"], out_dir / "training_curves.png")

    predictor = UNetPredictor(summary["checkpoint"], device=args.device)
    scores = evaluate_arrays(val_images, val_masks, predictor=predictor)
    print("\n" + format_summary(scores))

    from .classical import extract_lung_fields
    from .hybrid import fuse

    for index in range(min(4, len(val_images))):
        image = val_images[index]
        classical = extract_lung_fields(image)
        probability = predictor(image)
        panel = comparison_panel(
            image,
            {
                "classical": classical.mask,
                "unet": fuse(probability, classical.mask, classical.score, "unet"),
                "hybrid": fuse(probability, classical.mask, classical.score, "hybrid"),
            },
            reference=val_masks[index],
        )
        save_panel(out_dir / "panels" / f"phantom_{index:02d}.png", panel)

    (out_dir / "demo_metrics.json").write_text(json.dumps(scores, indent=2))
    print(f"\nartifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
