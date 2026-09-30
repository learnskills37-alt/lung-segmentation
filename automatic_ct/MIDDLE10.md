# Middle 10 CT images per LIDC-IDRI folder

The catalog contains **1,010 patients and 1,018 CT series**, all with at least ten
instances. One series per patient gives **10,100 images**; all series give
**10,180 images**. These are catalog candidates; geometry is checked after download.
Eight patients have two CT series, so patient count and series-folder count differ.

The default batch selects the first 10 patients in ascending ID order, using the
lexicographically first eligible series per patient. This is a convenience sample,
not a random or clinically validated cohort. Completion is recorded explicitly in
`batch_results.json`; an inventory entry alone does not mean a case was processed.

For N canonical axial slices, select indices `range((N-10)//2, (N-10)//2+10)`.
Indices are zero-based and ordered by physical DICOM geometry, not filenames.
For 133 slices, the selected indices are 61–70 (one-based slices 62–71).
The lower-index window resolves the tie for odd-length volumes.

Both models predict on the **complete CT volume**. Only PNG exports are restricted
to the central ten slices, preserving full-volume context. The NIfTI masks remain
full-volume predictions. Ten central slices do not cover the entire lungs.

## Reproduce

Install the Python environment in [README.md](README.md), then run:

```bash
python automatic_ct/middle10.py --output outputs/middle10 --inventory-only
python automatic_ct/middle10.py --output outputs/middle10 --limit 10
```

Change `--limit` to expand the batch; `0` means all patient candidates. All CT
series total approximately 128.4 GB of DICOM downloads, before NIfTI data and
predictions. Inputs are retained, so larger runs need sufficient disk space.
Models run sequentially to bound memory. Downloads use public IDC AWS objects
with object-size and single-part ETag/MD5 checks, without cloud credentials.

The downloader verifies patient/series IDs, unique SOP instances, uniform slice
spacing, orientation, and physical slice order. It independently checks all ten
selected NIfTI slices against original DICOM HU pixels and physical origins.
Resume uses hashes to check CT and mask identity. Export validation checks every
mask, original display image, lung-region image, and crop against the CT and mask.

Outputs for each patient/model contain `ct_png`, `masks_png`, `lung_regions`,
`lung_regions_cropped`, the full `lung_mask.nii.gz`, metadata, and previews.
Display images use the fixed [-1000, 400] HU window. Outside-mask pixels are black.
The two model folders contain the same ten source CT images, not twenty unique inputs.

No independent whole-lung reference masks are provided here, so no Dice/IoU
accuracy is claimed. The models are pretrained R231 U-Net and TotalSegmentator
nnU-Net (3 mm fast model), not newly trained models or the latest architectures.
Source attribution and CC BY 3.0 data licensing are in [LIDC_IDRI.md](LIDC_IDRI.md).
