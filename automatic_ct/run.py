"""Pretrained deep learning: CT -> binary lung mask -> extracted lung images."""
import argparse
import csv
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def installed_versions():
    result = {}
    for name in ['torch', 'lungmask', 'TotalSegmentator', 'nnunetv2', 'nibabel']:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def export_regions(ct_path, mask_path, destination):
    import nibabel as nib
    import numpy as np
    from PIL import Image
    original = nib.load(ct_path)
    mask_image = nib.load(mask_path)
    if original.shape != mask_image.shape or not np.allclose(original.affine, mask_image.affine, atol=1e-4, rtol=0):
        raise ValueError('CT and mask voxel grids differ')
    ct = nib.as_closest_canonical(original).get_fdata(dtype=np.float32)
    mask = nib.as_closest_canonical(mask_image).get_fdata(dtype=np.float32) > 0
    directories = [destination / name for name in ['masks_png', 'lung_regions', 'lung_regions_cropped']]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in range(ct.shape[2]):
        # Radiological axial display, anterior up; no intensity changes inside mask
        # other than the declared fixed CT display window (-1000 to 400 HU).
        hu = ct[:, :, z].T[::-1, ::-1]
        binary = mask[:, :, z].T[::-1, ::-1]
        display = np.rint(np.clip((hu + 1000) / 1400, 0, 1) * 255).astype(np.uint8)
        extracted = np.where(binary, display, 0).astype(np.uint8)
        name = f'slice_{z:03d}.png'
        Image.fromarray(binary.astype(np.uint8) * 255).save(directories[0] / name)
        Image.fromarray(extracted).save(directories[1] / name)
        record = {'slice': z, 'lung_pixels': int(binary.sum()), 'has_crop': False,
                  'x0': '', 'y0': '', 'x1_exclusive': '', 'y1_exclusive': ''}
        if binary.any():
            yy, xx = np.where(binary)
            x0, x1, y0, y1 = int(xx.min()), int(xx.max()+1), int(yy.min()), int(yy.max()+1)
            Image.fromarray(extracted[y0:y1, x0:x1]).save(directories[2] / name)
            record.update(has_crop=True, x0=x0, y0=y0, x1_exclusive=x1, y1_exclusive=y1)
        rows.append(record)
    with (destination / 'slice_index.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return len(rows), sum(r['has_crop'] for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ct', type=Path, required=True)
    parser.add_argument('--case', help='Case identifier for the report; defaults to the input filename')
    parser.add_argument('--model', choices=['unet', 'nnunet'], required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('model_cache'))
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--single-process', action='store_true', help='Sequential nnU-Net I/O for restricted CPU environments')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache = args.cache.resolve(); args.cache.mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(args.cache / 'torch')
    os.environ['TOTALSEG_HOME_DIR'] = str(args.cache / 'totalseg')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    import numpy as np
    import nibabel as nib
    import torch
    torch.set_num_threads(args.threads)
    ct = nib.load(args.ct)
    if len(ct.shape) != 3:
        raise ValueError('Input must be a 3D HU CT NIfTI volume')
    output = args.output / 'lung_mask.nii.gz'
    start = time.perf_counter()
    if args.model == 'unet':
        import SimpleITK as sitk
        from lungmask import LMInferer
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
        model = LMInferer(modelname='R231', force_cpu=True, batch_size=2)
        scan = sitk.ReadImage(str(args.ct))
        labels = model.apply(scan)
        binary = sitk.GetImageFromArray((labels > 0).astype(np.uint8))
        binary.CopyInformation(scan)
        sitk.WriteImage(binary, str(output))
        model_name = 'lungmask R231: 2D U-Net'
    else:
        if args.single_process:
            from sequential_nnunet import enable_sequential_io
            enable_sequential_io()
        from totalsegmentator.config import setup_totalseg, set_config_key
        from totalsegmentator.python_api import totalsegmentator
        from totalsegmentator.map_to_binary import class_map
        setup_totalseg(); set_config_key('send_usage_stats', False)
        set_config_key('statistics_disclaimer_shown', True)
        lobes = ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
                 'lung_middle_lobe_right', 'lung_lower_lobe_right']
        labels = totalsegmentator(args.ct, output=None, task='total', fast=True, ml=True,
                                  roi_subset=lobes, device='cpu', nr_thr_resamp=1,
                                  nr_thr_saving=1, statistics=False, preview=False)
        ids = [k for k, v in class_map['total'].items() if v in lobes]
        data = np.isin(np.asarray(labels.dataobj), ids).astype(np.uint8)
        header = ct.header.copy(); header.set_data_dtype(np.uint8)
        binary = nib.Nifti1Image(data, labels.affine, header)
        binary.header.set_slope_inter(1, 0)
        nib.save(binary, output)
        model_name = 'TotalSegmentator: 3D nnU-Net, 3 mm fast model'
    inference_seconds = time.perf_counter() - start
    saved = nib.load(output)
    if saved.shape != ct.shape or not np.allclose(saved.affine, ct.affine, atol=1e-4, rtol=0):
        raise ValueError('Output mask does not match the CT grid')
    pred = np.asarray(saved.dataobj) > 0
    info = {'case': args.case or args.ct.name.split('.nii')[0], 'model': model_name,
            'input_sha256': digest(args.ct), 'mask_sha256': digest(output),
            'shape_xyz': list(ct.shape), 'device': 'cpu', 'threads': args.threads,
            'inference_wall_seconds_including_model_loading': inference_seconds,
            'single_process_io': args.single_process,
            'nnunet_aggregation': 'FP32 CPU' if args.single_process else 'upstream default',
            'versions': installed_versions()}
    if args.reference:
        reference = nib.load(args.reference)
        if reference.shape != ct.shape or not np.allclose(reference.affine, ct.affine, atol=1e-4, rtol=0):
            raise ValueError('Reference grid mismatch')
        ref = np.asarray(reference.dataobj)
        if not set(np.unique(ref).tolist()).issubset({0, 1, 2}):
            raise ValueError('Use the lung-only reference labels')
        truth = ref > 0
        a, b, intersection = int(pred.sum()), int(truth.sum()), int((pred & truth).sum())
        info['metrics'] = {'dice': 2*intersection/(a+b) if a+b else 1.,
                           'iou': intersection/(a+b-intersection) if a+b-intersection else 1.,
                           'precision': intersection/a if a else 0.,
                           'recall': intersection/b if b else 0.}
        info['reference_sha256'] = digest(args.reference)
    count, cropped = export_regions(args.ct, output, args.output)
    info['exported_full_frame_slices'] = count
    info['exported_cropped_slices'] = cropped
    (args.output / 'metrics.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
