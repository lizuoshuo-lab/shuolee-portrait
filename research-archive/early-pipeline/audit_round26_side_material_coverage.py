"""Measure the intervention roster's alpha weight in signed R22 side crops."""

import argparse
import json
import sys

import numpy as np
from PIL import Image
import torch

from build_round26_native_face_material import LABELS, OUT, composite_view
from trace_round22_face_pixels import DELIVERY, PROJECT, REVIEW_SHA, checked, load_subject, need, pin, preprocess, read, sha


VIEWS=('close_yaw_m90_pitch_p00','close_yaw_p90_pitch_p00','close_yaw_p00_pitch_p30','close_yaw_p00_pitch_m30')


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    source=OUT/f'{subject}-material-evidence.json'
    proof=read(source)
    archive=pin(OUT/f'{subject}-material-evidence.npz',proof['material_evidence_npz_sha256'])
    with np.load(archive,allow_pickle=False) as data:
        labels,owners,stable,spill=(data[key] for key in ('labels','owner_faces','stable_ids','silhouette_spill'))
    context=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    need(np.array_equal(owners,context['owner']) and np.array_equal(stable,context['stable']),
         'A side attribution cannot identify the frozen material rows.')
    results={}
    for key in VIEWS:
        row=context['grid']['records'][key]
        panel=row['panel']
        x0,y0,x1,y1=panel['crop_xyxy']; w,h=x1-x0,y1-y0
        with Image.open(checked(panel)) as image:
            pixels=np.asarray(image.convert('RGB'),dtype=np.int16)
        need(pixels.shape==(h,w*4,3),'Unexpected signed head crop layout.')
        width,height=row['camera']['wh']
        truth=np.zeros((height,width,3),np.int16)
        truth[y0:y1,x0:x1]=pixels[:,:w]
        pre=preprocess(context['arrays'],row['camera'])
        mass,parity=composite_view(pre,np.zeros((h,w),np.uint8),panel['crop_xyxy'],truth,
                                  np.arange(len(stable)),(panel['crop_xyxy'],pixels[:,2*w:3*w,0]))
        weights=mass[:,0]
        total=float(weights.sum())
        groups={label:float(weights[labels==i].sum()) for i,label in enumerate(LABELS)}
        groups['unknown_reliable_silhouette_spill']=float(weights[spill].sum())
        results[key]=dict(camera=row['camera'],panel=panel,parity=parity,alpha_mass=total,
            material_alpha_mass=groups,alpha_share={k:v/total for k,v in groups.items()},
            highest_alpha_rows=[dict(stable_id=int(stable[r]),owner_face=int(owners[r]),label=LABELS[labels[r]],
                repeated_native_spill=bool(spill[r]),alpha_mass=float(weights[r])) for r in np.argsort(-weights)[:30]])
    target=OUT/f'{subject}-side-material-coverage.json'
    with target.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='shuolee-r26-signed-novel-head-material-coverage/1',subject=subject,
            source_material_receipt_sha256=sha(source),source_material_npz_sha256=sha(archive),views=results,
            scope='After freezing the native-only labels, audit their actual opacity weight in all signed pixels of four original R22 synthetic head crops. Novel views contribute no classification evidence.'),
            f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(dict(file=str(target),sha256=sha(target),coverage={k:v['alpha_share'] for k,v in results.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
