from pathlib import Path

import numpy as np
import pytest
import torch

from lungseg.dataset import ArrayDataset, collect_samples, generate_pseudo_masks, split_samples
from lungseg.hybrid import fuse, segment_slice
from lungseg.io_utils import find_images, load_image, load_mask, save_image, save_mask
from lungseg.metrics import all_metrics, dice, iou
from lungseg.nodules import detect_nodule_candidates
from lungseg.phantom import make_dataset, make_phantom
from lungseg.predict import UNetPredictor, predict_paths
from lungseg.train import TrainConfig, train_unet
from lungseg.unet import UNet


@pytest.fixture(scope="module")
def phantom_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("phantoms")
    images, masks, _ = make_dataset(10, size=96, seed=4)
    for index, (image, mask) in enumerate(zip(images, masks)):
        save_image(root / "images" / f"slice_{index:02d}.png", image)
        save_mask(root / "masks" / f"slice_{index:02d}.png", mask)
    return root


def test_unet_preserves_spatial_shape():
    model = UNet(base_channels=8, depth=3)
    output = model(torch.zeros(2, 1, 64, 64))
    assert output.shape == (2, 1, 64, 64)


def test_unet_handles_non_power_of_two_input():
    model = UNet(base_channels=8, depth=2)
    output = model(torch.zeros(1, 1, 70, 54))
    assert output.shape == (1, 1, 70, 54)


def test_io_roundtrip(tmp_path):
    image, mask, _ = make_phantom(64, np.random.default_rng(0), n_nodules=1)
    save_image(tmp_path / "a.png", image)
    save_mask(tmp_path / "a_mask.png", mask)
    assert np.allclose(load_image(tmp_path / "a.png"), image, atol=1 / 255)
    assert np.array_equal(load_mask(tmp_path / "a_mask.png"), mask)


def test_collect_samples_pairs_images_with_sibling_masks(phantom_dir):
    samples = collect_samples(phantom_dir / "images")
    assert len(samples) == 10
    assert all(s.mask_path is not None for s in samples)
    assert len(find_images(phantom_dir / "images")) == 10


def test_pseudo_masks_are_confident_and_accurate(phantom_dir, tmp_path):
    samples = collect_samples(phantom_dir / "images")
    kept = generate_pseudo_masks(samples, tmp_path / "pseudo", min_score=0.5, verbose=False)
    assert len(kept) >= 8
    for sample in kept[:3]:
        reference = load_mask(phantom_dir / "masks" / f"{sample.image_path.stem}.png")
        assert dice(load_mask(sample.mask_path), reference) > 0.85
        assert 0.0 < sample.weight <= 1.0


def test_split_is_disjoint_and_covers_everything(phantom_dir):
    samples = collect_samples(phantom_dir / "images")
    train, val = split_samples(samples, val_fraction=0.3, seed=0)
    assert len(train) + len(val) == len(samples)
    assert not {s.image_path for s in train} & {s.image_path for s in val}


@pytest.mark.parametrize("mode", ["hybrid", "unet", "classical", "union", "intersection"])
def test_fusion_modes_return_a_binary_mask(mode):
    image, reference, _ = make_phantom(96, np.random.default_rng(6), n_nodules=1)
    probability = reference.astype(np.float32) * 0.9
    mask = fuse(probability, reference, 0.9, mode=mode)
    assert mask.shape == reference.shape
    assert set(np.unique(mask)) <= {0, 1}
    assert dice(mask, reference) > 0.8


def test_fusion_rejects_unknown_mode():
    with pytest.raises(ValueError):
        fuse(np.zeros((8, 8), np.float32), np.zeros((8, 8), np.uint8), 1.0, mode="nope")


def test_segment_slice_without_predictor_falls_back_to_classical():
    image, reference, _ = make_phantom(128, np.random.default_rng(7), n_nodules=1)
    result = segment_slice(image, predictor=None)
    assert result.mode == "classical"
    assert result.probability is None
    assert dice(result.mask, reference) > 0.85


def test_hybrid_uses_the_network_when_the_classical_stage_is_unreliable():
    image = np.random.default_rng(0).random((64, 64)).astype(np.float32) * 0.1
    result = segment_slice(image, predictor=lambda img: np.ones_like(img), mode="hybrid")
    assert result.mode == "unet"


def test_nodule_detector_finds_synthetic_nodules():
    from lungseg.classical import extract_lung_fields

    rng = np.random.default_rng(11)
    hits = total = 0
    for _ in range(4):
        image, _, nodules = make_phantom(256, rng, n_nodules=3)
        candidates = detect_nodule_candidates(image, extract_lung_fields(image).mask)
        for y, x, radius in nodules:
            total += 1
            hits += any(np.hypot(c.y - y, c.x - x) <= radius + 4 for c in candidates)
    assert hits / total > 0.7


def test_nodule_detector_returns_nothing_for_an_empty_mask():
    image, _, _ = make_phantom(64, np.random.default_rng(0), n_nodules=1)
    assert detect_nodule_candidates(image, np.zeros_like(image, np.uint8)) == []


def test_result_figure_shows_input_field_and_extracted_region():
    from lungseg.visualize import contact_sheet, result_figure

    image, reference, _ = make_phantom(96, np.random.default_rng(2), n_nodules=1)
    figure = result_figure(image, reference, label="slice")
    assert figure.shape == (96, 96 * 3, 3), "three side-by-side panels"

    # The extracted-region panel keeps the lungs and blanks everything else.
    # Row 0-23 carries the caption, so compare below it.
    extracted = figure[24:, 96 * 2 :, 0]
    outside = reference[24:] == 0
    assert extracted[outside].max() <= 40, "background outside the lungs is blanked"
    assert extracted[reference[24:] > 0].mean() > 0

    sheet = contact_sheet([figure, figure, figure], columns=2)
    assert sheet.shape == (96 * 2, 96 * 3 * 2, 3), "3 figures pad out to a 2x2 sheet"


def test_metrics_on_identical_and_disjoint_masks():
    mask = np.zeros((16, 16), np.uint8)
    mask[4:12, 4:12] = 1
    assert dice(mask, mask) == 1.0
    assert iou(mask, mask) == 1.0
    assert dice(mask, 1 - mask) == 0.0
    scores = all_metrics(mask, mask)
    assert scores["precision"] == scores["recall"] == 1.0
    assert scores["hd95"] == 0.0


def test_train_predict_roundtrip(tmp_path, phantom_dir):
    images, masks, _ = make_dataset(8, size=64, seed=9)
    config = TrainConfig(
        epochs=2, batch_size=2, image_size=64, base_channels=8, depth=2, num_workers=0, amp=False
    )
    summary = train_unet(
        ArrayDataset(images[:6], masks[:6], 64, train=True),
        ArrayDataset(images[6:], masks[6:], 64, train=False),
        config,
        tmp_path / "run",
        device="cpu",
        verbose=False,
    )
    assert len(summary["history"]["val_dice"]) == 2

    predictor = UNetPredictor(summary["checkpoint"], device="cpu")
    probability = predictor(images[0])
    assert probability.shape == images[0].shape
    assert 0.0 <= probability.min() <= probability.max() <= 1.0

    paths = sorted((phantom_dir / "images").glob("*.png"))[:3]
    records = predict_paths(
        paths, tmp_path / "pred", predictor=predictor, detect_nodules=True, verbose=False
    )
    assert len(records) == 3
    assert (tmp_path / "pred" / "predictions.csv").is_file()
    for record in records:
        assert (tmp_path / "pred" / "masks" / f"{Path(record['image']).stem}.png").is_file()
