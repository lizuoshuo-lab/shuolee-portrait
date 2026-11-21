"""Audit whether unknown elongated head splats bridge native facial regions."""

import argparse
from pathlib import Path
import json
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from build_round26_native_face_material import (MATERIAL_REVIEW, OUT, alpha_panel, array_sha, composite_view,
                                                material_grid, protected_faces)
from trace_round22_face_pixels import (DELIVERY, PROJECT, REVIEW_SHA, VIEW, PICKS, checked, load_subject,
                                       need, numbered, parity_panel, pin, posed_model, preprocess,
                                       quaternion_matrix, read, sha)


POLICY=dict(min_tangent_axis_ratio=3., min_major_axis_tangent_projection=.9,
            max_native_zone_gap_px=24., min_zone_alpha_px=.75, min_smaller_zone_share=.10,
            min_distinct_native_bins_for_same_pair=2)


def crossing_grid(labels):
    # Keep both anatomical interior regions; each skin neighbor gets its own bin.
    grid=np.zeros_like(labels)
    for label, adjacent in ((2,3),(5,4)):
        anatomy=labels==label
        if not anatomy.any():
            continue
        grid[anatomy]=label
        distance=cv2.distanceTransform((~anatomy).astype(np.uint8),cv2.DIST_L2,5)
        skin=(labels==1)&(distance<=POLICY['max_native_zone_gap_px'])
        # A skin pixel close to both zones is ambiguous; count neither neighbor.
        collision=skin&(grid!=0)
        grid[skin]=adjacent
        grid[collision]=0
    return grid


def matched_rows(context, material, head):
    q=numbered(context['local'],'rot',4)
    q=q/np.linalg.norm(q,axis=1,keepdims=True)
    axes=quaternion_matrix(q)
    log_scale=numbered(context['local'],'scale',3)
    scale=np.exp(log_scale)
    tangent=axes[:,[0,2],:]*scale[:,None,:]
    tangent_variance=np.linalg.eigvalsh(tangent@tangent.transpose(0,2,1))
    axis_ratio=np.sqrt(tangent_variance[:,1]/tangent_variance[:,0])
    major=scale.argmax(1)
    along_normal=axes[np.arange(len(major)),1,major]
    tangent_projection=np.sqrt(np.maximum(0.,1.-along_normal**2))
    geometry=(axis_ratio>=POLICY['min_tangent_axis_ratio'])&(
        tangent_projection>=POLICY['min_major_axis_tangent_projection'])
    rows=np.flatnonzero(head[context['owner']]&(material==0)&geometry)
    return rows,axis_ratio,tangent_projection,major


def ray_coverage(subject, context, eligible, paired_bins):
    report={}
    for suffix,(description,pairs) in PICKS[subject].items():
        row=context['grid']['records'][VIEW+suffix]
        panel=row['panel']
        pre=preprocess(context['arrays'],row['camera'])
        parity,traced=parity_panel(pre,checked(panel),panel['crop_xyxy'],pairs)
        need(parity['passed'],'Historical ray replay differs.')
        records=[]
        for pair in pairs:
            matched={}
            for name,xy in zip(('artifact','control'),pair):
                item=traced[xy]
                rows=item['rows']; weights=item['weights']
                top=np.argsort(-weights)[:8]
                matched[name]=dict(xy=xy, alpha_weight=float(weights[eligible[rows]].sum()),
                    top_rows=[dict(stable_id=int(context['stable'][rows[i]]),weight=float(weights[i]),
                        accepted=bool(eligible[rows[i]]),same_pair_native_bins=paired_bins[rows[i]].tolist()) for i in top])
            records.append(matched)
        report[VIEW+suffix]=dict(description=description,parity=parity,pairs=records)
    return report


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    receipt_path=OUT/f'{subject}-material-evidence.json'
    receipt=read(receipt_path)
    material_path=pin(receipt_path.with_suffix('.npz'),receipt['material_evidence_npz_sha256'])
    with np.load(material_path,allow_pickle=False) as data:
        stable,owner,material=(data[key] for key in ('stable_ids','owner_faces','labels'))
    context=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    need(np.array_equal(owner,context['owner']) and np.array_equal(stable,context['stable']),
         'Native crossing observations cannot identify the material rows.')
    model,slots=posed_model(context)
    arm=context['arm']; run=Path(arm['run']); result=read(run/'result.json')
    replay_path=pin(run/result['bound_replay'],result['bound_replay_sha256'])
    replay=read(replay_path)
    records={item['frame']:item for item in replay['records']}
    rows=read(pin(run/'data/training-inputs.json',receipt['training_inputs_sha256']))['records']
    with np.load(pin(run/'fit/fitted.npz',receipt['fit_sha256']),allow_pickle=False) as archive:
        obs,proj,faces=(archive[k] for k in ('observed_native_keypoints','joints_projected_native','faces'))
    asset=pin(context['plan']['asset'],read(replay_path.parent.parent/'bound-replay.json')['asset_sha256'])
    candidate,ratios,tangent_projection,major=matched_rows(context,material,protected_faces(asset,faces))
    row_lookup=np.full(len(stable),-1,np.int64); row_lookup[candidate]=np.arange(len(candidate))
    opts=read(run/'gaussian-run/config.json')['unified_main']['constraints']
    attachment=read(run/'gaussian-run/raw-step-018000/attachment.json')
    old={item['frame']:item for item in attachment['support']['views']}
    annotations=read(pin(DELIVERY/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW))['subjects'][subject]['native_annotation']
    native=context['result']['native']['r22']
    roles={row['frame']:row['role'] for row in context['plan']['poses']['r22']['pairs']}
    _,_,_,opacity,rgb=context['arrays']
    bins=receipt['coverage']['source_view_bins']
    bin_votes=np.zeros((len(bins),len(candidate),2),bool)
    per_view=[]
    need([row['frame'] for row in rows]==receipt['classification_frames'],'Native crossing roster differs.')
    for row,prior in zip(rows,receipt['views']):
        frame=row['frame']; slot=slots[frame]; saved=records[frame]
        box=prior['head_crop_xyxy']
        labels,_=material_grid(row,obs[slot],proj[slot],box,opts,old[frame],annotations.get(frame,{}))
        need(array_sha(labels)==prior['label_mask_sha256'],'Frozen source material pixels changed.')
        grid=crossing_grid(labels)
        yy,xx=np.where(grid!=0)
        if not len(xx) or not (((grid==3).any() and (grid==2).any()) or ((grid==4).any() and (grid==5).any())):
            continue
        a,b,c,d=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
        tight=(box[0]+a,box[1]+b,box[0]+c,box[1]+d)
        model.select_mesh_by_timestep(slot)
        arrays=(model.get_xyz.numpy(),model.get_scaling.numpy(),model.get_rotation.numpy(),opacity,rgb)
        pre=preprocess(arrays,saved['camera'])
        with Image.open(pin(replay_path.parent/saved['image'],saved['image_sha256'])) as image:
            truth=np.asarray(image.convert('RGB'),dtype=np.int16)
        mass,parity=composite_view(pre,grid[b:d,a:c],tight,truth,row_lookup,alpha_panel(native,roles.get(frame)))
        first=mass[:,[2,5]]; second=mass[:,[3,4]]
        smaller=np.minimum(first,second)
        vote=(smaller>=POLICY['min_zone_alpha_px'])&(smaller>=POLICY['min_smaller_zone_share']*(first+second))
        source_bin=prior['source_bin']; bin_votes[bins.index(source_bin)]|=vote
        per_view.append(dict(frame=frame,source_bin=source_bin,crop_xyxy=tight,parity=parity,
                             eligible_overlaps=vote.sum(0).tolist()))
    counts=bin_votes.sum(0)
    accepted=(counts>=POLICY['min_distinct_native_bins_for_same_pair']).any(1)
    full=np.zeros(len(stable),bool); full[candidate]=accepted
    full_counts=np.zeros((len(stable),2),int); full_counts[candidate]=counts
    payload=dict(schema='shuolee-r26-unknown-internal-crossing-diagnostic/1',subject=subject,
        material_receipt_sha256=sha(receipt_path),material_npz_sha256=sha(material_path),
        source_sha256=sha(__file__),policy=POLICY,pair_columns=['eye_near_skin','oral_lip_near_skin'],
        geometry_eligible_unknown_rows=len(candidate),native_confirmed_rows=int(accepted.sum()),
        native_confirmed_by_pair=(counts>=2).sum(0).tolist(),source_views=per_view,
        rows=[dict(stable_id=int(stable[row]),owner_face=int(owner[row]),major_scale_axis=int(major[row]),
            tangent_ratio=float(ratios[row]),major_axis_tangent_projection=float(tangent_projection[row]),
            eligible_source_bins=[np.asarray(bins)[bin_votes[:,i,k]].tolist() for k in range(2)])
            for i in np.flatnonzero(accepted) for row in (candidate[i],)],
        prior_rays=ray_coverage(subject,context,full,full_counts),
        scope='Positive compositing in adjacent trusted native skin and eye/lip interiors supports geometric footprint overlap only. Unknown material identity remains unknown. No heldout source frame used, no training or baseline changes.')
    path=OUT/f'{subject}-unknown-internal-crossing.json'
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    print(dict(subject=subject,path=str(path),sha256=sha(path),candidates=len(candidate),accepted=int(accepted.sum()),
               native_views=len(per_view)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
