"""Catalog LIDC-IDRI and run both full-volume models, exporting ten central slices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from run import digest, middle_indices


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read()
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt + 1)


def download_ct(row, output):
    """Use public IDC AWS objects, verify object sizes/MD5, then DICOM geometry."""
    import numpy as np
    import pydicom
    import SimpleITK as sitk
    import nibabel as nib
    output.mkdir(parents=True, exist_ok=True)
    ct = output / 'ct.nii.gz'
    if (output/'source.json').exists() and ct.exists():
        old = json.loads((output/'source.json').read_text())
        if old['SeriesInstanceUID'] == row['SeriesInstanceUID'] and old['ct_sha256'] == digest(ct):
            return ct
        raise ValueError('Existing CT provenance mismatch')
    bucket = row['aws_bucket']; prefix = row['crdc_series_uuid'] + '/'
    if not bucket.startswith('idc-'):
        raise ValueError('Expected an IDC public bucket')
    endpoint = 'https://' + bucket + '.s3.amazonaws.com/'
    ns = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    objects, token = [], None
    while True:
        query = {'list-type': '2', 'prefix': prefix}
        if token: query['continuation-token'] = token
        root = ET.fromstring(get(endpoint+'?'+urllib.parse.urlencode(query)))
        for item in root.findall('s:Contents', ns):
            key = item.findtext('s:Key', namespaces=ns)
            if key.endswith('.dcm'):
                objects.append((key, int(item.findtext('s:Size', namespaces=ns)), item.findtext('s:ETag', namespaces=ns).strip('"')))
        if root.findtext('s:IsTruncated', namespaces=ns) != 'true': break
        token = root.findtext('s:NextContinuationToken', namespaces=ns)
    if len(objects) != row['instanceCount']:
        raise ValueError('Object listing and catalog instance counts differ')
    dicom = output/'dicom';dicom.mkdir(exist_ok=True)
    def fetch(item):
        key, size, md5 = item
        path = dicom/Path(key).name
        data = path.read_bytes() if path.exists() else get(endpoint+urllib.parse.quote(key))
        if len(data) != size or ('-' not in md5 and hashlib.md5(data).hexdigest() != md5):
            raise ValueError('Object integrity failure: '+key)
        if not path.exists():
            partial = path.with_suffix('.part');partial.write_bytes(data);partial.replace(path)
        return path
    with ThreadPoolExecutor(max_workers=24) as pool:
        files = list(pool.map(fetch, objects))
    headers = [pydicom.dcmread(f, stop_before_pixels=True) for f in files]
    if any(str(d.SeriesInstanceUID) != row['SeriesInstanceUID'] or str(d.PatientID) != row['PatientID'] or d.Modality != 'CT' for d in headers):
        raise ValueError('DICOM identity mismatch')
    if len({str(d.SOPInstanceUID) for d in headers}) != len(headers):
        raise ValueError('Duplicate DICOM instances')
    orientations = np.asarray([d.ImageOrientationPatient for d in headers], dtype=float)
    positions = np.asarray([d.ImagePositionPatient for d in headers], dtype=float)
    spacing = np.asarray([d.PixelSpacing for d in headers], dtype=float)
    if not np.allclose(orientations, orientations[0], atol=1e-5) or not np.allclose(spacing, spacing[0]):
        raise ValueError('Inconsistent geometry')
    normal = np.cross(orientations[0,:3],orientations[0,3:])
    offset = positions @ normal; order = np.argsort(offset); steps = np.diff(offset[order])
    if not np.all(steps > 0) or not np.allclose(steps, np.median(steps), atol=.01):
        raise ValueError('Duplicate or nonuniform slice positions')
    files = [files[int(i)] for i in order]
    reader = sitk.ImageSeriesReader();reader.SetFileNames([str(f) for f in files])
    reader.SetOutputPixelType(sitk.sitkFloat32)
    image = reader.Execute();sitk.WriteImage(image,str(ct))
    # Independently verify middle slices against the original DICOM pixels and origins.
    native = nib.load(ct); canonical = nib.as_closest_canonical(native)
    transform = np.linalg.inv(native.affine) @ canonical.affine
    if nib.aff2axcodes(native.affine)[2] not in ['S','I']:
        raise ValueError('Expected axial CT')
    array = native.get_fdata(dtype=np.float32); checked=[]
    for z in middle_indices(canonical.shape[2]):
        native_z = int(round((transform @ [0,0,z,1])[2]))
        ds = pydicom.dcmread(files[native_z])
        hu = ds.pixel_array.astype(np.float32)*float(ds.RescaleSlope)+float(ds.RescaleIntercept)
        point = (native.affine @ [0,0,native_z,1])[:3]*[-1,-1,1]
        if not np.allclose(point,ds.ImagePositionPatient,atol=.001,rtol=0) or not np.array_equal(hu,array[:,:,native_z].T):
            raise ValueError('DICOM/NIfTI origin or HU mismatch')
        checked.append({'slice':z,'SOPInstanceUID':str(ds.SOPInstanceUID),'origin_LPS_mm':[float(v) for v in ds.ImagePositionPatient]})
    source = dict(row, ct_sha256=digest(ct), shape_xyz=list(native.shape), units='HU',
                  spacing_mm=list(image.GetSpacing()), source_validation='passed', checked_middle_slices=checked,
                  data_license='CC BY 3.0', source_DOI='10.7937/K9/TCIA.2015.LO9QL9SX')
    (output/'source.json').write_text(json.dumps(source,indent=2)+'\n')
    return ct


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=10,help='One series per patient; 0 means all patients')
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--cache',type=Path,default=Path('model_cache'))
    p.add_argument('--threads',type=int,default=6)
    args=p.parse_args()
    if args.limit<0 or args.threads<1:p.error('Invalid limit or thread count')
    args.output.mkdir(parents=True,exist_ok=True)
    from idc_index import IDCClient
    c=IDCClient.client()
    frame=c.sql_query("SELECT PatientID, SeriesInstanceUID, StudyInstanceUID, instanceCount, series_size_MB, aws_bucket, crdc_series_uuid FROM index WHERE collection_id='lidc_idri' AND Modality='CT' ORDER BY PatientID, SeriesInstanceUID")
    rows=json.loads(frame.to_json(orient='records'))
    chosen={}
    for row in rows:
        if row['instanceCount']>=10:
            row['middle_indices']=middle_indices(row['instanceCount'])
            chosen.setdefault(row['PatientID'],row)
    inventory={'queried_utc':datetime.now(timezone.utc).isoformat(),'idc_index':version('idc-index'),
               'patient_folders':len({r['PatientID'] for r in rows}), 'ct_series_folders':len(rows),
               'eligible_patient_folders':len(chosen),'eligible_ct_series':sum(r['instanceCount']>=10 for r in rows),
               'images_one_series_per_patient':10*len(chosen),'images_all_series':10*sum(r['instanceCount']>=10 for r in rows),
               'full_ct_download_GB':sum(r['series_size_MB'] for r in rows)/1000,
               'qualification':'Catalog CT series with >=10 instances; geometry verified after download.',
               'series_selection':'Lexicographically first eligible SeriesInstanceUID per patient.',
               'middle_selection':'start=floor((N-10)/2); 10 consecutive canonical axial slices.',
               'all_ct_series':rows}
    (args.output/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(json.dumps({k:v for k,v in inventory.items() if k!='all_ct_series'},indent=2),flush=True)
    if args.inventory_only:return
    selected=list(chosen.values())[:args.limit or None]
    (args.output/'selected_cohort.json').write_text(json.dumps(selected,indent=2)+'\n')
    scripts=Path(__file__).resolve().parent;results=[]
    # Full-volume neural inference is deliberately sequential to bound memory.
    for row in selected:
        patient=row['PatientID'];root=args.output/patient;root.mkdir(exist_ok=True)
        try:
            print('Download/verify '+patient,flush=True)
            ct=download_ct(row,root/'data');models={}
            for model in ['unet','nnunet']:
                out=root/model;out.mkdir(exist_ok=True)
                def call(script,options,log):
                    with (root/log).open('w') as f:
                        subprocess.run([sys.executable,str(scripts/script),*map(str,options)],stdout=f,stderr=subprocess.STDOUT,check=True)
                if not (out/'metrics.json').exists():
                    print('Predict '+patient+' '+model,flush=True)
                    options=['--ct',ct,'--case',patient,'--model',model,'--output',out,'--cache',args.cache.resolve(),'--threads',args.threads,'--middle-slices',10]
                    if model=='nnunet':options.append('--single-process')
                    call('run.py',options,model+'.log')
                meta=json.loads((out/'metrics.json').read_text())
                if meta['input_sha256']!=digest(ct) or meta['mask_sha256']!=digest(out/'lung_mask.nii.gz') or meta['exported_slice_indices']!=row['middle_indices']:
                    raise ValueError('Prediction provenance/selection mismatch')
                call('verify_exports.py',['--ct',ct,'--results',out],model+'_validation.log')
                call('preview.py',['--ct',ct,'--results',out,'--output',out],model+'_preview.log')
                models[model]={'masks':meta['exported_full_frame_slices'],'lung_regions':meta['exported_full_frame_slices'],'crops':meta['exported_cropped_slices'],'export_integrity':'passed'}
            result={'patient':patient,'status':'completed','series':row['SeriesInstanceUID'],'selected_slices':row['middle_indices'],'models':models}
            print('Completed '+patient,flush=True)
        except Exception as exc:
            result={'patient':patient,'status':'failed','error':str(exc)}
            print('Failed '+patient+': '+str(exc),flush=True)
        results.append(result)
        (args.output/'batch_results.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status']!='completed' for r in results):raise SystemExit('One or more cases failed; inspect logs')


if __name__=='__main__':main()
