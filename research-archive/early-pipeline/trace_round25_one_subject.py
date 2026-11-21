"""Run matched R24/R25 CPU attribution from one signed R25 subject review."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from compare_round25_garment_attribution import R24_ATTRIBUTION, board, groups
from trace_round22_face_pixels import checked, geometry, need, pin, read, sha
from trace_round25_garment_pixels import (OUT, PICKS, PROJECT, R24_REVIEW, attribute, row_identity,
                                        samples, world_rows)
from trace_round25_native_garment_material import evidence


def summarize(old,new):
    keys=('region','artifact_xy_px','control_xy_px')
    need(all(old[key]==new[key] for key in keys),'The R25 ray differs from R24.')
    record={key:old[key] for key in keys}
    record['parity']=[old['parity'],new['parity']]
    if not all(row['parity']['passed'] for row in (old,new)):
        return record
    values=('artifact_alpha','control_alpha','artifact_rgb','control_rgb','delta_luma','background_luma')
    record['r24']={key:old[key] for key in values}|dict(layer_contributions=groups(old))
    record['r25']={key:new[key] for key in values}|dict(layer_contributions=groups(new))
    record['r25_minus_r24']=dict(artifact_alpha=new['artifact_alpha']-old['artifact_alpha'],
        control_alpha=new['control_alpha']-old['control_alpha'],
        abs_luma_contrast=abs(new['delta_luma'])-abs(old['delta_luma']))
    return record


def run(subject, review_path, review_sha):
    torch.set_num_threads(2)
    sys.path.insert(0,PROJECT+'25/src')
    r24_dir=OUT/'smpl-delivery-round24'
    r25_dir=OUT/'smpl-delivery-round25/input-audit'
    r25_dir.mkdir(parents=True,exist_ok=True)
    old_receipt=read(pin(r24_dir/'review-receipt.json',R24_REVIEW))
    old_result=read(checked(old_receipt['subjects'][subject]['review']))
    old=read(pin(r24_dir/'input-audit/side-waist-hem-attribution.json',R24_ATTRIBUTION))['subjects'][subject]['views']
    result=read(pin(review_path,review_sha))
    need(result['subject']==subject and result['static_camera_count']==50 and result['native_poses']==28,
         'R25 QA is incomplete or belongs to a different subject.')
    plan=read(checked(result['plan']))
    arm='r25'
    arm_plan=plan['static_grid']['arms'][arm]
    context=world_rows(arm_plan,checked(plan['graphs'][arm]['manifest']))
    row_identity(context,plan,arm)
    grid=read(checked(result['static'][arm]['receipt']))
    need(grid['ply_sha256']==arm_plan['ply_sha256'],'R25 QA uses a different PLY.')
    new,comparison={},{}
    arrays=geometry(context['portable'])
    for key,probes in PICKS[subject].items():
        row=grid['records'][key]
        panel=row['panel']
        before=old_result['static']['r24']['records'][key]
        need(row['fixed_world_camera']==before['fixed_world_camera'] and panel['crop_xyxy']==before['panel']['crop_xyxy'],
             'The R25 world camera or crop changed from R24.')
        pre,traces=samples(row['camera'],panel,arrays,probes)
        pairs=[attribute(context,pre,traces,panel['crop_xyxy'],probe) for probe in probes]
        new[key]=dict(camera=row['camera'],panel=panel,pairs=pairs)
        comparison[key]=[summarize(a,b) for a,b in zip(old[key]['pairs'],pairs,strict=True)]
    candidates=np.array(sorted({row['row'] for item in new.values() for p in item['pairs'] if p['parity']['passed']
                                for row in p['contributors']}),dtype=np.int64)
    by_row,native=evidence(dict(subject=subject,review=result,plan=plan,arm=arm),context['scene'],context['drive'],candidates)
    native_by_stable={str(int(context['stable'][row])):items for row,items in by_row.items()}
    image=board(r25_dir,subject,old,new)
    report=r25_dir/f'{subject}-r24-r25-side-waist-hem-attribution.json'
    with report.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='shuolee-r24-r25-signed-subject-attribution/1',subject=subject,
            r25_review_sha256=review_sha,r24_attribution_sha256=R24_ATTRIBUTION,
            r25_inputs=context['proof'],r25_trace=new,paired=comparison,r25_native_material=native,
            r25_native_material_by_stable_id=native_by_stable,marked_board=image,
            method='Exact world camera and same pixel pairs. Positive material overlap requires the source photo, matte, actual raster alpha and saved native pixel parity. Source views do not establish novel-view ground truth.',
            optimizer_updates=0,gpu_used=False,quality_accepted=False,script_sha256=sha(__file__)),
            f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(dict(subject=subject,file=str(report),sha256=sha(report),marked_board=image),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=tuple(PICKS))
    parser.add_argument('review',type=Path)
    parser.add_argument('review_sha256')
    args=parser.parse_args()
    run(args.subject,args.review,args.review_sha256)
