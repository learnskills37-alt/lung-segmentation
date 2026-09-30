"""Verify exported masks/regions/crops against the CT and the predicted 3D mask."""
import argparse
import csv
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', required=True, type=Path)
    p.add_argument('--results', required=True, type=Path)
    args = p.parse_args()
    ct_image = nib.load(args.ct)
    mask_image = nib.load(args.results / 'lung_mask.nii.gz')
    if ct_image.shape != mask_image.shape or not np.allclose(ct_image.affine, mask_image.affine, atol=1e-4, rtol=0):
        raise ValueError('Voxel grids differ')
    if not set(np.unique(np.asarray(mask_image.dataobj))).issubset({0, 1}):
        raise ValueError('Mask is not binary')
    ct = nib.as_closest_canonical(ct_image).get_fdata(dtype=np.float32)
    masks = nib.as_closest_canonical(mask_image).get_fdata(dtype=np.float32) > 0
    rows = list(csv.DictReader((args.results / 'slice_index.csv').open()))
    meta = json.loads((args.results / "metrics.json").read_text())
    indices = meta.get("exported_slice_indices", list(range(ct.shape[2])))
    if len(rows) != len(indices) or indices != sorted(set(indices)):
        raise ValueError('Missing slice index rows')
    crops = 0
    for row, z in zip(rows, indices):
        if not 0 <= z < ct.shape[2]:
            raise ValueError("Invalid slice index")
        name = f'slice_{z:03d}.png'
        binary = masks[:, :, z].T[::-1, ::-1]
        hu = ct[:, :, z].T[::-1, ::-1]
        window = np.rint(np.clip((hu + 1000) / 1400, 0, 1) * 255).astype(np.uint8)
        expected = np.where(binary, window, 0).astype(np.uint8)
        if not np.array_equal(np.asarray(Image.open(args.results / 'masks_png' / name)), binary.astype(np.uint8)*255):
            raise ValueError(f'Incorrect mask: {name}')
        if not np.array_equal(np.asarray(Image.open(args.results / 'lung_regions' / name)), expected):
            raise ValueError(f'Incorrect extracted image: {name}')
        crop_path = args.results / 'lung_regions_cropped' / name
        if "exported_slice_indices" in meta:
            if not np.array_equal(np.asarray(Image.open(args.results / "ct_png" / name)), window):
                raise ValueError("Original CT display pixels differ")
        if int(row['slice']) != z or int(row['lung_pixels']) != int(binary.sum()):
            raise ValueError(f'Incorrect slice index: {name}')
        if binary.any():
            yy, xx = np.where(binary)
            bbox = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
            recorded = [int(row[k]) for k in ['x0', 'y0', 'x1_exclusive', 'y1_exclusive']]
            if recorded != bbox or row['has_crop'] != 'True':
                raise ValueError(f'Incorrect crop coordinates: {name}')
            x0, y0, x1, y1 = bbox
            if not np.array_equal(np.asarray(Image.open(crop_path)), expected[y0:y1, x0:x1]):
                raise ValueError(f'Incorrect crop pixels: {name}')
            crops += 1
        elif crop_path.exists() or row['has_crop'] != 'False':
            raise ValueError(f'Unexpected crop for empty slice: {name}')
    for folder, expected_count in [('masks_png', len(indices)), ('lung_regions', len(indices)), ('lung_regions_cropped', crops)]:
        if len(list((args.results / folder).glob('*.png'))) != expected_count:
            raise ValueError(f'Unexpected file count in {folder}; use an empty output directory')
    report = {'status': 'passed', 'mask_slices_checked': len(indices),
              'lung_region_slices_checked': len(indices), 'cropped_views_checked': crops,
              'checks': ['binary masks', 'NIfTI grid and affine', 'every PNG pixel', 'crop coordinates', 'file counts'],
              'scope': 'Export integrity only; not anatomical accuracy validation.'}
    (args.results / 'export_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
