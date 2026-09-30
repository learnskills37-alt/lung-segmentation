"""Download one actual CT/reference pair from the original dataset archive."""
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import argparse


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0: raise ValueError('Invalid seek')
        return self.position
    def read(self, n=-1):
        end = self.size if n < 0 else min(self.size, self.position+n)
        if end <= self.position: return b''
        request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.position}-{end-1}'})
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status != 206:
                raise RuntimeError('Byte-range requests unavailable; download the archive manually')
            result = response.read()
        if len(result) != end-self.position: raise IOError('Incomplete archive response')
        self.position = end
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('example_data'))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/3757476', timeout=60) as response:
        record = json.load(response)
    for archive, filename in [('COVID-19-CT-Seg_20cases.zip','ct.nii.gz'),('Lung_Mask.zip','reference.nii.gz')]:
        item = next(f for f in record['files'] if f['key'] == archive)
        with zipfile.ZipFile(RemoteZip(item['links']['self'],item['size'])) as z:
            members = [n for n in z.namelist() if Path(n).name == 'coronacases_002.nii.gz']
            if len(members) != 1: raise ValueError('Expected exactly one matching archive entry')
            (args.output/filename).write_bytes(z.read(members[0]))  # CRC checked by ZipFile
        print('Saved',args.output/filename,flush=True)
