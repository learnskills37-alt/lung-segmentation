import cv2
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from lungseg.classical import crop_to_mask, extract_lung_fields, extract_lung_region
from lungseg.metrics import dice
from lungseg.phantom import make_phantom


def test_extracts_both_lung_fields_from_phantom():
    rng = np.random.default_rng(0)
    scores = []
    for _ in range(5):
        image, reference, _ = make_phantom(256, rng, n_nodules=2)
        result = extract_lung_fields(image)
        assert result.score > 0.7
        scores.append(dice(result.mask, reference))
    assert np.mean(scores) > 0.9


def test_hull_repair_keeps_juxtapleural_nodules_inside_the_field():
    """Nodules on the pleural wall must not be carved out of the lung mask."""
    rng = np.random.default_rng(0)
    with_repair = without_repair = total = 0
    for _ in range(10):
        image, _, nodules = make_phantom(256, rng, n_nodules=3)
        repaired = extract_lung_fields(image).mask
        plain = extract_lung_fields(image, repair_juxtapleural=False).mask
        for y, x, _ in nodules:
            total += 1
            with_repair += int(repaired[y, x])
            without_repair += int(plain[y, x])
    assert with_repair > without_repair
    assert with_repair / total > 0.85


def test_closing_does_not_bridge_the_two_lungs():
    image, _, _ = make_phantom(256, np.random.default_rng(3), n_nodules=0)
    result = extract_lung_fields(image)
    labels = cv2.connectedComponents(result.mask, connectivity=8)[0] - 1
    assert labels == 2, "the mediastinum must keep the fields separate"


def soft_tissue_window(image: np.ndarray) -> np.ndarray:
    """Re-map a phantom the way a soft-tissue window would: parenchyma lands at mid-grey
    instead of near-black, while the air outside the patient stays dark."""
    body = ndi.binary_fill_holes(image > 0.03)
    return np.where(body, 0.45 + 0.55 * image, image).astype(np.float32)


def test_survives_a_display_window_that_puts_parenchyma_at_mid_grey():
    """An image dataset carries whatever window its exporter chose.

    A single global Otsu is dominated by the dark air background and splits background
    from body, leaving parenchyma on the tissue side of the threshold and the lungs
    unfound; the search over intra-body thresholds is what keeps this case working.
    """
    image, reference, _ = make_phantom(256, np.random.default_rng(0), n_nodules=2)
    windowed = soft_tissue_window(image)

    assert windowed[reference > 0].mean() > 0.4, "parenchyma should be mid-grey here"
    global_split = (windowed <= threshold_otsu(windowed)).astype(np.uint8)
    assert dice(global_split, reference) < 0.2, "a single global threshold must fail this case"

    result = extract_lung_fields(windowed)
    assert dice(result.mask, reference) > 0.85
    assert result.score > 0.5


@pytest.mark.parametrize(
    "image",
    [np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32), np.full((32, 32), 0.5, np.float32)],
)
def test_degenerate_inputs_score_zero(image):
    result = extract_lung_fields(image)
    assert result.score == 0.0
    assert result.mask.shape == image.shape


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        extract_lung_fields(np.zeros((8, 8, 3), np.float32))


def test_extract_lung_region_blanks_the_background():
    image, reference, _ = make_phantom(128, np.random.default_rng(1), n_nodules=1)
    region = extract_lung_region(image, reference)
    assert np.array_equal(region[reference > 0], image[reference > 0])
    assert region[reference == 0].max() == 0.0


def test_crop_to_mask_returns_bounds_inside_the_image():
    image, reference, _ = make_phantom(128, np.random.default_rng(2), n_nodules=0)
    cropped, (y0, x0, y1, x1) = crop_to_mask(image, reference, margin=4)
    assert cropped.shape == (y1 - y0, x1 - x0)
    assert 0 <= y0 < y1 <= image.shape[0]
    assert 0 <= x0 < x1 <= image.shape[1]
