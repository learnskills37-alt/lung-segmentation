# Lung Region Extraction from Chest CT

Extracts the lung fields from chest CT slices using three interchangeable stages —
a classical computer-vision segmenter, a U-Net, and a hybrid that fuses them — plus an
optional nodule-candidate detector that runs inside the extracted region.

Target dataset: [`ucimachinelearning/lung-nodule-dataset`](https://www.kaggle.com/datasets/ucimachinelearning/lung-nodule-dataset) on Kaggle.

## Why a hybrid

The classical stage is precise on well-windowed slices and needs no labels, but it fails
on low-contrast or pathological lungs. The U-Net degrades gracefully everywhere but needs
training targets. Combining them gives labels for free and a segmenter that survives the
cases the thresholding cannot handle:

1. The classical stage segments every slice and scores its own output.
2. Slices scoring above a threshold become **pseudo-labels**; the score also weights each
   sample in the loss, so uncertain masks pull the network less.
3. At inference, the U-Net probability map is blended with the classical mask (weighted by
   that same confidence) inside a dilated guard band, so the network can recover tissue
   the threshold missed without leaking into the mediastinum. Where the classical score is
   near zero, the fusion falls back to the network alone.

On the synthetic validation set this ordering holds — the hybrid beats both parts:

| method    |   Dice |    IoU | Precision | Recall | HD95 |
|-----------|-------:|-------:|----------:|-------:|-----:|
| classical | 0.9910 | 0.9822 |    0.9830 | 0.9992 | 1.00 |
| U-Net     | 0.9969 | 0.9939 |    0.9942 | 0.9996 | 0.92 |
| hybrid    | 0.9977 | 0.9954 |    0.9957 | 0.9997 | 0.47 |

(`lungseg demo`, 60 train / 15 validation phantoms, 20 epochs on CPU. Reproduce with the
demo command below.)

**Phantoms flatter the classical stage badly, so do not read those numbers as real
performance.** Checked against 8 real annotated axial chest CT slices, the classical stage
scored Dice 0.29 where it scores 0.99 on phantoms — five of the eight returned an empty
mask. Fixing that (see *Display windowing* below) brought it to **0.865** mean Dice on the
same slices, at 0.979 precision / 0.794 recall. Phantoms are useful as a fast, dependency-
free regression harness; they are not a substitute for validating on your own data, which
is what `lungseg evaluate` is for.

## Install

```bash
pip install -r requirements.txt      # add pydicom if your copy ships .dcm slices
pip install -e .                     # optional: installs the `lungseg` command
```

## Quick start

Run the whole pipeline on synthetic CT phantoms — no dataset or GPU needed, ~1 minute on
a laptop CPU. This is the fastest way to check the install:

```bash
python -m lungseg demo --out outputs/demo
```

It writes a checkpoint, training curves, per-method metrics and side-by-side comparison
panels (green outline = ground truth, orange = prediction).

## Running on the Kaggle dataset

```bash
python scripts/run_kaggle_pipeline.py --epochs 40
```

That script is the four commands below chained together. Run them individually for more
control:

```bash
# 1. Download (needs Kaggle credentials); it prints the cache path, then survey the layout
python -m lungseg download
python -m lungseg inspect <printed_path>

# 2. Classical masks only - no training involved
python -m lungseg masks <data_dir> --out outputs/pseudo_masks --min-score 0.5

# 3. Train the U-Net (uses curated masks if present, else pseudo-labels)
python -m lungseg train <data_dir> --out outputs/run --epochs 40 --image-size 256

# 4. Segment and extract the lung regions
python -m lungseg predict <data_dir> \
    --checkpoint outputs/run/unet_best.pt \
    --out outputs/predictions --mode hybrid --nodules
```

If the dataset ships reference masks, score every stage against them:

```bash
python -m lungseg evaluate <data_dir> --mask-dir <masks_dir> --checkpoint outputs/run/unet_best.pt
```

The original kagglehub snippet works too — point any command at the path it prints:

```python
import kagglehub
path = kagglehub.dataset_download("ucimachinelearning/lung-nodule-dataset")
print("Path to dataset files:", path)
```

### Outputs

`predict` writes, per slice:

| path | contents |
|---|---|
| **`lung_regions/`** | **the extracted lung region: the original slice with everything outside the lungs blanked** |
| `results/` | three-panel figure per slice — input, detected field, extracted region |
| `contact_sheet.png` | the first 12 result figures on one reviewable sheet |
| `masks/` | binary lung mask (PNG) |
| `overlays/` | mask overlay, with nodule candidates circled if `--nodules` |
| `predictions.csv` | fusion mode used, classical confidence, lung area fraction |
| `nodule_candidates.csv` | position, radius, intensity, circularity, score |

## Method

### 1. Classical stage (`lungseg/classical.py`)

Denoise → body mask (largest filled tissue component) → air regions inside the body, with
frame-touching components dropped → keep the up-to-two largest regions that are comparable
in size, which excludes the trachea → repair.

**Display windowing.** The threshold is not a fixed rule, because an image dataset carries
whatever intensity mapping its exporter chose — the same anatomy lands near black under a
lung window and at mid-grey under a soft-tissue window or per-image autoscaling. A single
global Otsu is dominated by the large air background: it splits background from body, which
leaves parenchyma on the *tissue* side of the threshold and finds no lungs at all. This is
what produced the 0.29 Dice above. Instead the mask is built at each of three meaningful
thresholds — the global Otsu, plus Otsu and 3-class multi-Otsu computed inside the body
mask — and the best-scoring result wins. (Arbitrary quantile candidates were tried here
too; they never won except by noise, and cost phantom accuracy, so they were dropped.)

Three repair steps matter more than they look:

- **Per-component closing.** Closing the mask as a whole bridges the mediastinum on slices
  where the fields sit close together, which then breaks the hull repair below. Each field
  is closed independently instead.
- **Juxtapleural nodule recovery.** A nodule on the pleural wall carves an indentation that
  closing cannot bridge, so the mask cuts the nodule out — exactly the structure you want to
  keep on a nodule dataset. Each field is compared against its convex hull and the small,
  compact indentations are added back; the large mediastinal concavity fails the area test
  and thin slivers fail the fill test. On the phantoms this lifts nodule retention from
  50/75 to 70/75 for a 0.002 Dice cost.
- **Boundary contrast in the score.** Geometry alone cannot tell a correct lung field from
  one that has spilled into the chest wall — both look plausibly sized and placed, and on
  real CT the score rated a mask with 0.35 precision at 0.97 confidence. The score now also
  measures how sharply intensity steps across the mask boundary. It compares thin bands
  either side of the edge rather than whole-region means, because a regional mean is
  lowered by bright nodules inside the field and would reward carving them out.

Each mask gets a confidence score from its boundary contrast, area fraction, left/right
balance, component count and centroid position. That score both selects the threshold and
gates everything downstream, so a slice the stage cannot handle is flagged rather than
silently wrong.

### 2. U-Net (`lungseg/unet.py`)

Standard encoder/decoder with skip connections; batch-norm double-conv blocks, configurable
depth and width, bilinear upsampling by default (`--base-channels 32 --depth 4` ≈ 7.8M
parameters). Inputs are padded internally, so non-power-of-two sizes work. Trained with a
BCE + soft-Dice loss where each sample is weighted by its pseudo-label confidence, AdamW,
cosine LR decay, gradient clipping, AMP on CUDA and early stopping on validation Dice.

Expect validation Dice to sit at 0.0 for the first two or three epochs: background
dominates the frame, so the network collapses to all-background before the Dice term pulls
it out. Re-weighting the loss does not shorten this, so give it at least ~10 epochs before
concluding a run has failed.

### 3. Hybrid fusion (`lungseg/hybrid.py`)

`--mode` selects the behaviour: `hybrid` (default), `unet`, `classical`, `union`,
`intersection`. All modes finish with the same clean-up — small-object removal, closing,
hole filling, keep the two largest components.

### 4. Nodule candidates (`lungseg/nodules.py`)

Multi-scale Laplacian-of-Gaussian blob detection restricted to the lung interior, filtered
on contrast against the parenchyma median and on circularity to reject vessel
cross-sections. This is a **high-sensitivity candidate generator, not a classifier** — it
recalls ~86% of the synthetic nodules while also flagging vessels. Treat the output as a
shortlist for a downstream classifier or a human reader.

## Data layout

`collect_samples` walks the dataset recursively and reads `.png/.jpg/.bmp/.tif`, `.npy`
and `.dcm` (with `pydicom`). DICOM and Hounsfield-unit arrays are windowed to the lung
window (centre −600 HU, width 1500); everything else is scaled to `[0, 1]`.

Masks are paired by filename stem, either from `--mask-dir` or from a sibling directory
named `masks/`, `labels/`, `gt/` and similar. Directories with those names are skipped when
collecting images, so masks never get mistaken for inputs. When no masks exist, training
falls back to pseudo-labels automatically.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```

29 tests covering the classical stage (degenerate inputs, alternate display windowing,
mediastinum separation, juxtapleural recovery), fusion modes, metrics, I/O round-trips,
result figures, the U-Net's shape handling and a full train → predict round-trip. They run
on synthetic phantoms, so no dataset download is required.

## Layout

```
lungseg/
  classical.py    thresholding / morphology lung field extraction + confidence score
  hybrid.py       fusion of the classical mask and the U-Net probability map
  unet.py         U-Net
  losses.py       BCE + soft Dice with per-sample weighting; Tversky
  dataset.py      sample collection, pseudo-label generation, augmentation, Datasets
  train.py        training loop, checkpointing, early stopping
  predict.py      checkpoint loading, batch inference, region extraction
  evaluate.py     per-method metric comparison
  nodules.py      LoG nodule candidate detection
  metrics.py      Dice, IoU, precision/recall, specificity, HD95
  postprocess.py  shared binary-mask morphology
  phantom.py      synthetic chest CT generator (demo + tests)
  visualize.py    overlays, comparison panels, training curves
  io_utils.py     image/DICOM/npy loading, HU windowing
  cli.py          command line interface
scripts/run_kaggle_pipeline.py
tests/
```

## Notes and limitations

- Everything is 2D and slice-wise. There is no 3D consistency between adjacent slices.
- The classical stage assumes an axial CT with air-filled lungs darker than surrounding
  tissue. It is not tuned for chest radiographs; on those, train the U-Net against curated
  masks and predict with `--mode unet`.
- Pseudo-labels inherit the classical stage's biases. If the dataset ships reference masks,
  pass `--mask-dir` and use them — the confidence weighting then has nothing to correct.
- Research and educational code. Not a medical device, and not for clinical use.
