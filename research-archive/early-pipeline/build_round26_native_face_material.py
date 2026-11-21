"""Freeze cross-view material evidence for the actual R22 SH0 head rows."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch

from trace_round22_face_pixels import (DELIVERY, F, PICKS, PROJECT, REVIEW_SHA, VIEW, checked, code,
                                       load_subject, material_mask, need, parity_panel, pin, posed_model,
                                       preprocess, read, sha)


OUT = DELIVERY.with_name('smpl-delivery-round26')/'input-audit'
MATERIAL_REVIEW = 'b09b58120cfabab797d149b1c73b08753db6419e85a2c0c0ebf0e50e80e5422a'
LABELS = ['unknown', 'trusted_skin', 'eye', 'hair', 'eyewear', 'oral_lip']
BACKGROUND = len(LABELS)
POLICY = dict(min_native_bins=2, min_pixel_equivalent_alpha=1., min_alpha_fraction_in_bin=.55,
              min_consensus_fraction=.75, max_background_fraction=.25,
              min_landmark_confidence=.5, max_landmark_error_px=6., max_landmark_error_over_span=.08,
              min_eye_points=5, min_lip_points=8, min_roi_px=12,
              spill_min_bins=2, spill_min_pixel_equivalent_alpha=2., spill_min_alpha_fraction_in_bin=.10,
              spill_max_near_silhouette_px=32.,
              stock_cpu_validation_max_code=1)


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def protected_faces(asset, faces):
    with np.load(asset, allow_pickle=False) as archive:
        need(np.array_equal(faces,archive['f']), 'The frozen mesh differs from its SMPL-X asset.')
        dominant = archive['weights'].argmax(-1)
        landmark_faces=np.r_[archive['lmk_faces_idx'].reshape(-1),archive['dynamic_lmk_faces_idx'].reshape(-1)]
    protected=np.isin(dominant,[12,15,22,23,24])
    protected[faces[landmark_faces].ravel()]=True
    return protected[faces].all(1)


def head_crop(vertices, faces, head, camera):
    ids=np.unique(faces[head])
    E,K=np.asarray(camera['w2c']),np.asarray(camera['K'])
    xyz=vertices[ids]@E[:3,:3].T+E[:3,3]
    need(np.all(xyz[:,2]>.2), 'The source head crosses the renderer near plane.')
    q=xyz@K.T
    xy=q[:,:2]/q[:,2:]-0.5
    padding=max(16.,.35*np.ptp(xy,axis=0).max())
    left,top=np.maximum(np.floor(xy.min(0)-padding),[0,0]).astype(int)
    right,bottom=np.minimum(np.ceil(xy.max(0)+padding)+1,camera['wh']).astype(int)
    need(left<right and top<bottom, 'Empty native head region.')
    return int(left),int(top),int(right),int(bottom)


def local_hull(xy, box):
    x0,y0,x1,y1=box
    local=np.rint(xy-[x0,y0]).astype(np.int32)
    output=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
    hull=cv2.convexHull(local)
    if len(hull)>=3:
        cv2.fillConvexPoly(output,hull,1)
    return output.astype(bool)


def rois_from_landmarks(obs, proj, box):
    from shuolee_portrait.face_supplement import REGIONS

    cfg=POLICY
    reliable=np.isfinite(obs).all(1)&np.isfinite(proj).all(1)&(obs[:,2]>=cfg['min_landmark_confidence'])
    head=obs[67:118,:2][reliable[67:118]]
    span=np.ptp(head,axis=0).max() if len(head)>=8 else 0.
    shape=(box[3]-box[1],box[2]-box[0])
    eye=np.zeros(shape,dtype=bool)
    lip=eye.copy()
    if span<24:
        return eye,lip
    error=np.linalg.norm(obs[:,:2]-proj,axis=1)
    usable=reliable&(error<=min(cfg['max_landmark_error_px'],cfg['max_landmark_error_over_span']*span))
    for name in ('left_eye','right_eye','mouth'):
        ids=REGIONS[name]
        if name=='mouth':
            ids=ids[:12]
        ids=ids[usable[ids]]
        if len(ids)<(cfg['min_lip_points'] if name=='mouth' else cfg['min_eye_points']):
            continue
        admitted=binary_erosion(local_hull(obs[ids,:2],box)&local_hull(proj[ids],box),iterations=1)
        if admitted.sum()<cfg['min_roi_px']:
            continue
        if name=='mouth':
            lip|=admitted
        else:
            eye|=admitted
    return eye,lip


def material_grid(row,obs,proj,box,opts,old_view,annotations):
    from shuolee_portrait.coverage_refinement import _reliable_regions
    from shuolee_portrait.surface_attachment import _skin_pixels
    from shuolee_portrait.training import _native_foreground_target

    pin(row['image'],row['source_sha256'])
    pin(row['mask'],row['mask_sha256'])
    rgb,mask,foreground_box=_native_foreground_target(row)
    inside,outside=_reliable_regions(mask,opts)
    view=SimpleNamespace(rgb=rgb,inside=inside,box=foreground_box)
    skin,proof=_skin_pixels(view,row,obs,proj)
    need(proof.get('native_material_mask_sha256')==old_view.get('native_material_mask_sha256'),
         'The reconstructed CPU skin source mask differs from the actual R22 training mask.')
    x0,y0,x1,y1=box
    shape=(y1-y0,x1-x0)
    region=np.s_[y0:y1,x0:x1]
    skin_mask=np.zeros(shape,dtype=bool)
    if skin is not None:
        (left,top,right,bottom),values=skin
        need(x0<=left<right<=x1 and y0<=top<bottom<=y1,'Trusted skin lies outside the head audit region.')
        skin_mask[top-y0:bottom-y0,left-x0:right-x0]=values.numpy()
    eye,lip=rois_from_landmarks(obs,proj,box)
    known={name:material_mask(row['wh'],meta['polygons'])[region] for name,meta in annotations.get('masks',{}).items()
           if meta['validated']}
    for name,meta in annotations.get('masks',{}).items():
        if meta['validated']:
            need(array_sha(material_mask(row['wh'],meta['polygons']))==meta['full_roi_sha256'],
                 'The signed original RGB material polygon differs.')
    if annotations:
        need(annotations['source_sha256']==row['source_sha256'],'A material annotation came from another frame.')
    clear=inside.numpy()[region]
    masks=[skin_mask,eye,known.get('hair',np.zeros(shape,bool)),known.get('glasses_rim',np.zeros(shape,bool)),lip]
    stack=np.stack(masks)&clear
    count=stack.sum(0)
    labels=np.zeros(shape,np.uint8)
    for i,area in enumerate(stack,1):
        labels[area&(count==1)]=i
    nearest=cv2.distanceTransform((mask[0].numpy()<=opts['reliable_foreground']).astype(np.uint8),cv2.DIST_L2,5)
    labels[outside.numpy()[region]&(nearest[region]<=POLICY['spill_max_near_silhouette_px'])]=BACKGROUND
    return labels,dict(material_pixels=np.bincount(labels.ravel(),minlength=BACKGROUND+1).tolist(),
                       skin_mask_sha256=proof.get('native_material_mask_sha256'), conflicting_pixels=int((count>1).sum()),
                       signed_material_sources=sorted(known))


def native_weights(pre, tile, xy):
    rows=np.flatnonzero(pre['valid'] & ((pre['low']<=tile)&(pre['high']>tile)).all(1))
    rows=rows[np.argsort(pre['depth'][rows],kind='stable')]
    d=pre['xy'][rows][None,:,:]-xy[:,None,:].astype(np.float32)
    x,y=d[:,:,0],d[:,:,1]
    a,b,c=pre['conic'][rows].T
    power=F(-.5)*(a*x*x+c*y*y)-b*x*y
    alpha=np.minimum(F(.99),pre['opacity'][rows]*np.exp(np.minimum(power,F(0.))))
    alpha[(power>0)|(alpha<F(1./255.))]=0.
    T=np.cumprod(1-alpha,axis=1,dtype=np.float32)
    prior=np.column_stack((np.ones(len(xy),np.float32),T[:,:-1])) if len(rows) else T
    return rows,np.where(T>=F(.0001),prior*alpha,F(0.))


def composite_view(pre,labels,box,truth,head_index,alpha_truth=None):
    x0,y0,x1,y1=box
    result=np.zeros((int(head_index.max())+1,BACKGROUND+1),np.float64)
    used=np.zeros((BACKGROUND+1,),np.int64)
    stats=dict(pixels=0,exact_pixels=0,excluded_pixels=0,max_code_error=0,
               alpha_checked_pixels=0,max_alpha_code_error=0)
    for tx in range(x0//16,(x1+15)//16):
        for ty in range(y0//16,(y1+15)//16):
            xx,yy=np.meshgrid(np.arange(max(x0,tx*16),min(x1,(tx+1)*16)),np.arange(max(y0,ty*16),min(y1,(ty+1)*16)))
            x,y=xx.ravel(),yy.ravel()
            cls=labels[y-y0,x-x0]
            rows,weights=native_weights(pre,(tx,ty),np.column_stack((x,y)))
            recovered=code(weights@pre['rgb'][rows])
            error=abs(recovered-truth[y,x]).max(1)
            if alpha_truth is not None:
                bounds,ref=alpha_truth
                a,b,c,d=bounds
                within=(x>=a)&(x<c)&(y>=b)&(y<d)
                if within.any():
                    expected=ref[y[within]-b,x[within]-a]
                    alpha_error=abs(code(weights[within].sum(1))-expected)
                    stats['max_alpha_code_error']=max(stats['max_alpha_code_error'],int(alpha_error.max()))
                    stats['alpha_checked_pixels']+=int(within.sum())
                    error[within]=np.maximum(error[within],alpha_error)
            passed=error<=POLICY['stock_cpu_validation_max_code']
            stats['pixels']+=len(error)
            stats['exact_pixels']+=int((error==0).sum())
            stats['excluded_pixels']+=int((~passed).sum())
            stats['max_code_error']=max(stats['max_code_error'],int(error.max()))
            ids=head_index[rows]
            selection=ids>=0
            if not selection.any():
                continue
            for label in np.unique(cls[passed]):
                keep=passed&(cls==label)
                result[ids[selection],label]+=weights[keep][:,selection].sum(0,dtype=np.float64)
                used[label]+=int(keep.sum())
    stats['class_pixels_with_head_tile']=used.tolist()
    return result,stats


def alpha_panel(native,role):
    if role is None:
        return None
    panel=native['panels']['dark_'+role]
    path=pin(Path(native['directory'])/panel['file'],panel['sha256'])
    x0,y0,x1,y1=panel['crop_xyxy']
    w=x1-x0
    with Image.open(path) as image:
        data=np.asarray(image.convert('RGB'),dtype=np.int16)
    need(data.shape==(y1-y0,w*4,3),'Unexpected signed native black/white/alpha layout.')
    return (x0,y0,x1,y1),data[:,3*w:,0]


def classify(mass,head_rows,stable,skin):
    cfg=POLICY
    pooled=mass.sum(0)
    total=pooled.sum(1)
    known=pooled[:,1:BACKGROUND]
    usable=(mass[:,:,1:BACKGROUND]>=cfg['min_pixel_equivalent_alpha']) & (
        mass[:,:,1:BACKGROUND]>=cfg['min_alpha_fraction_in_bin']*mass.sum(2,keepdims=True))
    bins=usable.sum(0)
    best=known.argmax(1)
    index=np.arange(len(best))
    enough=(bins[index,best]>=cfg['min_native_bins'])&(known[index,best]>=cfg['min_consensus_fraction']*known.sum(1))&(
        pooled[:,BACKGROUND]<=cfg['max_background_fraction']*total)
    actual_skin=np.array([len({r['bin'] for r in skin.get(int(stable[row]),())}) for row in head_rows])
    enough&=(best!=0)|(actual_skin>=cfg['min_native_bins'])
    labels=np.where(enough,best+1,0).astype(np.uint8)
    spill_votes=((mass[:,:,BACKGROUND]>=cfg['spill_min_pixel_equivalent_alpha']) & (
        mass[:,:,BACKGROUND]>=cfg['spill_min_alpha_fraction_in_bin']*mass.sum(2))).sum(0)
    spill=(labels==0)&(spill_votes>=cfg['spill_min_bins'])
    return labels,pooled,bins,actual_skin,spill,spill_votes


def bad_rays(context,labels,head_index,spill,subject):
    report={}
    for suffix,(description,pairs) in PICKS[subject].items():
        row=context['grid']['records'][VIEW+suffix]
        panel=row['panel']
        pre=preprocess(context['arrays'],row['camera'])
        proof,traced=parity_panel(pre,checked(panel),panel['crop_xyxy'],pairs)
        need(proof['passed'],'The historical novel ray no longer matches native QA.')
        checks=[]
        for a,b in pairs:
            groups={}
            for key,xy in (('artifact',a),('control',b)):
                item=traced[xy]
                masses=defaultdict(float)
                top=[]; spill_alpha=0.
                for index in np.argsort(-item['weights'],kind='stable'):
                    row=int(item['rows'][index]); local=head_index[row]
                    code_=int(labels[row]); label=LABELS[code_] if local>=0 else 'outside_head_scope'
                    weight=float(item['weights'][index]); masses[label]+=weight
                    if spill[row]:
                        spill_alpha+=weight
                    if len(top)<12:
                        top.append(dict(stable_id=int(context['stable'][row]),owner_face=int(context['owner'][row]),
                                        label=label,alpha_weight=weight,silhouette_spill=bool(spill[row])))
                groups[key]=dict(crop_local_xy=xy,alpha=item['opacity'],weights=dict(masses),
                                 silhouette_spill_alpha=spill_alpha,top_rows=top)
            checks.append(groups)
        report[VIEW+suffix]=dict(description=description,parity=proof,pairs=checks)
    return report


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    review=read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA))
    historical=read(pin(DELIVERY/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW))
    context=load_subject(subject,review)
    model,slots=posed_model(context)
    arm=context['arm']; run=Path(arm['run']); result=read(run/'result.json')
    replay_path=pin(run/result['bound_replay'],result['bound_replay_sha256'])
    replay=read(replay_path)
    rows_path=pin(run/'data/training-inputs.json',replay['training_inputs_sha256'])
    rows=read(rows_path)['records']
    records={item['frame']:item for item in replay['records']}
    fit_path=pin(run/'fit/fitted.npz',context['plan']['poses']['r22']['new']['fitted_sha256'])
    with np.load(fit_path,allow_pickle=False) as archive:
        fit={key:archive[key] for key in ('observed_native_keypoints','joints_projected_native','faces','frame_ids')}
    face_ids=fit['faces']
    manifest=read(replay_path.parent.parent/'bound-replay.json')
    asset=pin(context['plan']['asset'],manifest['asset_sha256'])
    head=protected_faces(asset,face_ids)
    head_rows=np.flatnonzero(head[context['owner']])
    head_index=np.full(len(context['stable']),-1,dtype=np.int64)
    head_index[head_rows]=np.arange(len(head_rows))
    config=read(run/'gaussian-run/config.json')
    opts=config['unified_main']['constraints']
    attachment=read(run/'gaussian-run/raw-step-018000/attachment.json')
    old_views={item['frame']:item for item in attachment['support']['views']}
    bins=sorted(set(context['source_bins'].values()))
    masses=np.zeros((len(bins),len(head_rows),BACKGROUND+1),np.float64)
    annotations=historical['subjects'][subject]['native_annotation']
    native=context['result']['native']['r22']
    roles={row['frame']:row['role'] for row in context['plan']['poses']['r22']['pairs']}
    _,_,_,opacity,rgb=context['arrays']
    reports=[]
    ids=[row['frame'] for row in rows]
    need(len(ids)==24 and ids==fit['frame_ids'].tolist()==list(slots), 'The frozen selected observation roster differs.')
    for row in rows:
        frame=row['frame']; slot=slots[frame]; saved=records[frame]; camera=saved['camera']
        need(all(camera[k]==row[k] for k in ('K','w2c','wh')) and saved['source_sha256']==row['source_sha256'] and
             saved['mask_sha256']==row['mask_sha256'], 'Native source camera or content changed.')
        model.select_mesh_by_timestep(slot)
        box=head_crop(model.vertices_sequence[slot].numpy(),face_ids,head,camera)
        labels,roi=material_grid(row,fit['observed_native_keypoints'][slot],fit['joints_projected_native'][slot],
                                 box,opts,old_views[frame],annotations.get(frame,{}))
        arrays=(model.get_xyz.numpy(),model.get_scaling.numpy(),model.get_rotation.numpy(),opacity,rgb)
        pre=preprocess(arrays,camera)
        with Image.open(pin(replay_path.parent/saved['image'],saved['image_sha256'])) as image:
            true=np.asarray(image.convert('RGB'),dtype=np.int16)
        weight,proof=composite_view(pre,labels,box,true,head_index,alpha_panel(native,roles.get(frame)))
        src_bin=context['source_bins'][frame]
        masses[bins.index(src_bin)]+=weight
        report=dict(frame=frame,source_bin=src_bin,head_crop_xyxy=box,source_sha256=row['source_sha256'],
                    source_mask_sha256=row['mask_sha256'],native_render_sha256=saved['image_sha256'],
                    label_mask_sha256=array_sha(labels),roi=roi,parity=proof)
        reports.append(report)
        print(json.dumps(dict(subject=subject,frame=frame,parity=proof,pixels=roi['material_pixels'])),flush=True)
    labels,pooled,votes,skin_bins,spill,spill_votes=classify(masses,head_rows,context['stable'],context['skin'])
    all_labels=np.zeros(len(context['stable']),np.uint8)
    all_labels[head_rows]=labels
    all_spill=np.zeros(len(context['stable']),np.bool_)
    all_spill[head_rows]=spill
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'{subject}-material-evidence.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,stable_ids=context['stable'],labels=all_labels,
                            owner_faces=context['owner'],silhouette_spill=all_spill)
    supplemental=[row['frame'] for row in context['plan']['poses']['r22']['supplement']['frames']]
    holdouts=context['plan']['poses']['r22']['new']['protected_holdouts']
    need(set(ids).isdisjoint(supplemental+holdouts),'Training evidence includes an evaluation frame.')
    metrics=dict(head_rows=len(head_rows),label_counts=dict(zip(LABELS,np.bincount(labels,minlength=len(LABELS)).tolist())),
        mass_by_material=pooled.sum(0).tolist(),source_view_bins=bins,rows_with_two_signed_native_skin_bins=int((skin_bins>=2).sum()),
        rows_with_two_bin_class_votes=dict(zip(LABELS[1:],(votes>=2).sum(0).tolist())),
        unknown_repeated_native_silhouette_spill_rows=int(spill.sum()),
        silhouette_spill_supported_bins_histogram=np.bincount(spill_votes[spill]).tolist())
    receipt=dict(schema='shuolee-r26-cross-view-head-material/1',subject=subject,
        baseline_raw_ply_sha256=context['hashes']['world'], material_evidence_npz_sha256=sha(path),
        fit_sha256=sha(fit_path),training_inputs_sha256=sha(rows_path),baseline_bound_ply_sha256=context['hashes']['bound_local'],
        material_evidence_file=str(path),selected_frames=ids,classification_frames=ids,admitted_auxiliary_frames=supplemental,
        auxiliary_classification_frames=[],retained_holdouts=holdouts,heldout_source_images_read=0,
        label_codes=dict(enumerate(LABELS)),policy=POLICY,coverage=metrics,views=reports,
        previous_novel_ray_coverage=bad_rays(context,all_labels,head_index,all_spill,subject),
        provenance=dict(review_sha256=REVIEW_SHA, prior_material_review_sha256=MATERIAL_REVIEW,source_sha256=sha(__file__)),
        scope='Native compositing on the existing selected images, with conservative material support. Landmark-aligned eye/lip regions are anatomical observations. Hair/rim support comes from prior signed original RGB regions. Unknown rows retain no claim of material identity.',
        optimizer_updates=0,gpu_used=False,quality_accepted=False)
    target=path.with_suffix('.json')
    with target.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')
    print(dict(subject=subject,evidence=str(target),sha256=sha(target),coverage=metrics),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
