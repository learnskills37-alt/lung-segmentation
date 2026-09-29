"""Download one complete LIDC-IDRI CT series from NCI IDC and convert to HU NIfTI."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patient', default='LIDC-IDRI-0001')
    p.add_argument('--series', help='Exact SeriesInstanceUID; required if multiple CT series exist')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not re.fullmatch(r'LIDC-IDRI-\d{4}', args.patient):
        p.error('Use an LIDC-IDRI patient identifier, for example LIDC-IDRI-0001')
    if args.series and not re.fullmatch(r'[0-9.]+', args.series):
        p.error('Invalid DICOM series UID')
    from idc_index import IDCClient
    import SimpleITK as sitk
    import numpy as np
    client = IDCClient.client()
    rows = client.sql_query("SELECT * FROM index WHERE collection_id='lidc_idri' "
                            f"AND PatientID='{args.patient}' AND Modality='CT'")
    if args.series:
        rows = rows[rows.SeriesInstanceUID == args.series]
    if len(rows) != 1:
        raise ValueError('Select exactly one CT series using --series. Candidates: '
                         + rows[['SeriesInstanceUID', 'instanceCount']].to_json(orient='records'))
    source = json.loads(rows.to_json(orient='records'))[0]
    uid = source['SeriesInstanceUID']
    args.output.mkdir(parents=True, exist_ok=True)
    dicom_dir = args.output / 'dicom'
    dicom_dir.mkdir(exist_ok=True)
    client.download_dicom_series(uid, str(dicom_dir), dirTemplate=None,
                                  show_progress_bar=False, use_s5cmd_sync=True)
    files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), uid))
    if len(files) != int(source['instanceCount']) or len(files) < 2:
        raise ValueError(f'Incomplete CT series: {len(files)} files; expected {source["instanceCount"]}')
    # GDCM sorts by physical slice position. Validate geometry before assembling.
    positions, orientations, spacings, identities = [], [], [], []
    for filename in files:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filename); reader.ReadImageInformation()
        get = lambda tag: reader.GetMetaData(tag).strip()
        if get('0020|000e') != uid or get('0010|0020') != args.patient or get('0008|0060') != 'CT':
            raise ValueError('DICOM identity or modality mismatch')
        positions.append([float(x) for x in get('0020|0032').split('\\')])
        orientations.append([float(x) for x in get('0020|0037').split('\\')])
        spacings.append([float(x) for x in get('0028|0030').split('\\')])
        identities.append(get('0008|0018'))
    if len(set(identities)) != len(files):
        raise ValueError('Duplicate SOP instances')
    if not np.allclose(orientations, orientations[0], atol=1e-5) or not np.allclose(spacings, spacings[0]):
        raise ValueError('Inconsistent in-plane geometry')
    normal = np.cross(orientations[0][:3], orientations[0][3:])
    offsets = np.asarray(positions) @ normal
    steps = np.diff(offsets)
    if not (np.all(steps > 0) or np.all(steps < 0)) or not np.allclose(steps, np.median(steps), atol=0.01):
        raise ValueError('Duplicate or nonuniform slice positions')
    reader = sitk.ImageSeriesReader(); reader.SetFileNames(files)
    reader.SetOutputPixelType(sitk.sitkFloat32)
    image = reader.Execute()  # GDCM applies DICOM rescale slope/intercept to HU.
    ct_path = args.output / 'ct.nii.gz'
    sitk.WriteImage(image, str(ct_path))
    source.update(dataset='LIDC-IDRI', idc_index_version=version('idc-index'),
                  dicom_instances=len(files), shape_xyz=list(image.GetSize()),
                  spacing_mm=list(image.GetSpacing()), units='HU',
                  ct_sha256=hashlib.sha256(ct_path.read_bytes()).hexdigest(),
                  reference_mask_available=False,
                  reference_note='LIDC nodule annotations are not whole-lung ground truth.')
    (args.output / 'source.json').write_text(json.dumps(source, indent=2)+'\n')
    print(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
