"""Preview actual CT, predicted binary masks, and extracted lung-region images."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ct', required=True, type=Path)
    p.add_argument('--results', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    volume = nib.as_closest_canonical(nib.load(args.ct)).get_fdata(dtype=np.float32)
    metrics = json.loads((args.results / 'metrics.json').read_text())
    indices = [int(round((volume.shape[2]-1)*f)) for f in [0.25,0.5,0.75]]
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, selected in [('lung_region_preview.png', [indices[1]]),
                               ('three_slice_preview.png', indices)]:
        fig, axes = plt.subplots(len(selected), 4, figsize=(14, 3.7*len(selected)), squeeze=False)
        for r,z in enumerate(selected):
            name=f'slice_{z:03d}.png'
            ct=volume[:,:,z].T[::-1,::-1]
            ct=np.rint(np.clip((ct+1000)/1400,0,1)*255).astype(np.uint8)
            for c,(title,image) in enumerate([
                    ('Original CT',ct),
                    ('Predicted lung mask',np.asarray(Image.open(args.results/'masks_png'/name))),
                    ('Extracted lung region',np.asarray(Image.open(args.results/'lung_regions'/name))),
                    ('Cropped lung region',np.asarray(Image.open(args.results/'lung_regions_cropped'/name)))]):
                ax=axes[r,c];ax.imshow(image,cmap='gray',vmin=0,vmax=255)
                ax.set_title(title,fontsize=11,fontweight='bold');ax.axis('off')
            axes[r,0].text(0.03,0.04,f'Axial slice {z}',transform=axes[r,0].transAxes,color='white')
        fig.suptitle('CoronaCases 002 | '+metrics['model'],fontsize=14)
        fig.tight_layout();fig.savefig(args.output/filename,dpi=140,bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__ == '__main__':main()
