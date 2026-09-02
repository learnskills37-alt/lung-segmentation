#!/usr/bin/env python3
"""End-to-end run on the Kaggle lung nodule dataset.

    python scripts/run_kaggle_pipeline.py --epochs 40

Downloads the dataset, generates classical pseudo-labels, trains the U-Net on them and
segments every slice with the hybrid stage. Needs Kaggle credentials
(~/.kaggle/kaggle.json, or KAGGLE_USERNAME / KAGGLE_KEY).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lungseg.cli import main as cli_main  # noqa: E402
from lungseg.download import download_dataset, summarize_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="ucimachinelearning/lung-nodule-dataset")
    parser.add_argument("--out", default="outputs/kaggle")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N slices")
    parser.add_argument("--skip-download", default=None, help="Path to an already downloaded copy")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)

    data_dir = Path(args.skip_download) if args.skip_download else download_dataset(args.dataset_id)
    summary = summarize_dataset(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{summary['n_images']} readable slices under {data_dir}")
    if summary["n_images"] == 0:
        print("No supported image files were found - inspect dataset_summary.json for the layout")
        return 1

    limit = ["--limit", str(args.limit)] if args.limit else []
    steps = [
        ["train", str(data_dir), "--out", str(out / "run"),
         "--epochs", str(args.epochs), "--image-size", str(args.image_size),
         "--batch-size", str(args.batch_size), "--device", args.device, *limit],
        ["predict", str(data_dir), "--checkpoint", str(out / "run" / "unet_best.pt"),
         "--out", str(out / "predictions"), "--mode", "hybrid", "--nodules",
         "--device", args.device, *limit],
    ]
    for step in steps:
        print(f"\n=== lungseg {step[0]} ===")
        code = cli_main(step)
        if code != 0:
            return code

    print(f"\nDone. Masks, extracted lung regions and overlays are in {out / 'predictions'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
