"""Measure frozen R27 eligibility on the previously marked R22 streak rays."""

import argparse
import json
import sys

import numpy as np
import torch

from build_round27_material_visibility import OUT
from trace_round22_face_pixels import (DELIVERY, PICKS, PROJECT, REVIEW_SHA, VIEW, checked, load_subject,
                                       need, parity_panel, pin, preprocess, read, sha)


ROLES=('head_scope','observed','geometry_eligible','silhouette_spill','unknown_unobserved_elongated',
       'possible_protected_material')


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    path=OUT/f'{subject}-material-visibility.json'
    receipt=read(path)
    evidence=pin(path.with_suffix('.npz'),receipt['material_visibility_npz_sha256'])
    context=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    with np.load(evidence,allow_pickle=False) as data:
        need(np.array_equal(context['stable'],data['stable_ids']) and np.array_equal(context['owner'],data['owner_faces']),
             'Material evidence does not describe the exact R22 rows.')
        groups={key:data[key] for key in ROLES}
        groups['observed_unknown']=data['observed']&(data['labels']==0)
        groups['observed_unknown_without_protected_material']=groups['observed_unknown']&~groups['possible_protected_material']
        groups['unobserved_unknown']=data['head_scope']&~data['observed']&(data['labels']==0)
    views={}
    for suffix,(description,pairs) in PICKS[subject].items():
        key=VIEW+suffix; view=context['grid']['records'][key]; panel=view['panel']
        pre=preprocess(context['arrays'],view['camera'])
        parity,trace=parity_panel(pre,checked(panel),panel['crop_xyxy'],pairs)
        need(parity['passed'],'Signed R22 rays did not reproduce.')
        sample=[]
        for points in pairs:
            pair={}
            for label, xy in zip(('artifact','control'),points,strict=True):
                item=trace[xy]; rows=item['rows']; weights=item['weights']
                pair[label]=dict(xy=xy,total_alpha=item['opacity'],role_alpha={key:float(weights[mask[rows]].sum())
                                for key,mask in groups.items()})
            sample.append(pair)
        mean={role:{key:float(np.mean([row[role]['role_alpha'][key] for row in sample])) for key in groups}
              for role in ('artifact','control')}
        views[key]=dict(description=description,panel=panel,parity=parity,samples=sample,mean_alpha_by_role=mean)
    target=OUT/f'{subject}-prior-ray-visibility.json'
    with target.open('x',encoding='utf-8') as file:
        json.dump(dict(schema='shuolee-r27-prior-marked-ray-visibility/1',subject=subject,
            material_visibility_receipt_sha256=sha(path),material_visibility_npz_sha256=sha(evidence),
            source_sha256=sha(__file__),views=views,
            scope='The fixed R22 synthetic rays quantify existing compositing weights after the native-only evidence was frozen. '
                  'Overlapping role weights are not additive. These pixels were never used to classify the rows.'),
            file,indent=2,ensure_ascii=False,allow_nan=False)
        file.write('\n')
    print(dict(path=str(target),sha256=sha(target),coverage={k:v['mean_alpha_by_role'] for k,v in views.items()}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
