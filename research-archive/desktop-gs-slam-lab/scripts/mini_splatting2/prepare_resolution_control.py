"""Matched lossless low-resolution input for a later resolution ablation."""
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def main():
    cv2.setNumThreads(3)
    stage=ROOT/'source-resolution-rectified-002';source=stage/'dataset'
    out=stage/'matched-low-resolution-control';out.mkdir(exist_ok=False)
    image_dir=out/'images';image_dir.mkdir();sparse=out/'sparse/0';sparse.mkdir(parents=True)
    manifest=json.loads((stage/'manifest.json').read_text());rows=[]
    for row in manifest['rows']:
        path=source/'images'/row['name'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['rectified_sha256']
        image=cv2.imread(str(path));assert image.shape[:2]==(2694,1514)
        small=cv2.resize(image,(757,1347),interpolation=cv2.INTER_AREA)
        target=image_dir/row['name'];assert cv2.imwrite(str(target),small,[cv2.IMWRITE_PNG_COMPRESSION,3])
        rows.append(dict(name=row['name'],split=row['split'],full_resolution_sha256=row['rectified_sha256'],
            low_resolution_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    sys.path.insert(0,str(LAB/'repos/hierarchical-3d-gaussians/preprocess'));import read_write_model as rw
    cams,ims,points=rw.read_model(str(source/'sparse/0'),ext='.bin')
    cams={key:cam._replace(width=cam.width//2,height=cam.height//2,params=cam.params/2) for key,cam in cams.items()}
    ims={key:im._replace(xys=im.xys/2) for key,im in ims.items()}
    rw.write_model(cams,ims,points,str(sparse),ext='.bin')
    old=LAB/'outputs/mini-splatting2-computer00-001/hover-sfm-001/dataset/sparse/0'
    assert (sparse/'cameras.bin').read_bytes()==(old/'cameras.bin').read_bytes()
    assert (sparse/'points3D.bin').read_bytes()==(old/'points3D.bin').read_bytes()
    proposal=dict(status='paired_resolution_inputs_prepared_no_training',created=time.time(),
        full_resolution_dataset=str(source),matched_low_resolution_dataset=str(out),
        full_resolution=[1514,2694],low_resolution=[757,1347],rows=rows,
        training_images=51,development_images=8,cameras_and_structure_preserved=True,
        primary_comparison='Same decoded, rectified, lossless source images; factor2area downsampling only',
        secondary_comparison='Legacy resized JPEG pipeline remains a separate reference',
        used_for_training=False,ready_for_gpu_training=False,
        next=['Run representative native full-resolution forward/backward memory probe only after pending GPU audits',
            'If memory permits, freeze identical native settings and iteration budget for the matched low/full pair',
            'Evaluate both on the same source-resolution real images and the same downsampled references',
            'Report model count, memory, runtime and head fidelity; do not mix different GT resolutions in a claimed gain'])
    (stage/'resolution-ablation-inputs.json').write_text(json.dumps(proposal,indent=2));print(json.dumps({k:v for k,v in proposal.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
