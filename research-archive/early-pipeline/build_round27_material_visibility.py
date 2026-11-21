"""Freeze R27 face visibility and material support from signed original views."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
import torch

from build_round26_native_face_material import (BACKGROUND, LABELS, MATERIAL_REVIEW, alpha_panel, array_sha,
    bad_rays, head_crop, local_hull, native_weights, protected_faces, rois_from_landmarks)
from trace_round22_face_pixels import (DELIVERY, PROJECT, REVIEW_SHA, checked, code, load_subject,
    material_mask, need, numbered, pin, posed_model, preprocess, quaternion_matrix, read, sha)


OUT=DELIVERY.with_name('smpl-delivery-round27')/'input-audit'
INTERIOR=BACKGROUND+1
POLICY=dict(code_parity_max=1, foreground_alpha_min=252, background_alpha_max=3,
    max_near_silhouette_px=32., min_landmark_confidence=.5, max_landmark_flip_px=5.,
    max_fitted_error_px=6., min_face_span_px=24., min_eye_mouth_clearance_px=3,
    min_skin_seed_pixels=10, min_view_alpha_px=.75, min_material_alpha_px=.85,
    min_material_fraction=.45, min_known_consensus_fraction=.78, min_material_bins=2,
    protect_material_alpha_px=1., protect_material_fraction=.20,
    spill_alpha_px=2., spill_fraction=.10, spill_bins=2,
    long_tangent_ratio=3., major_axis_min_tangent_projection=.90)


def decode(path, box=None, mode='RGB'):
    with Image.open(path) as image:
        image=image.convert(mode)
        return np.asarray(image if box is None else image.crop(box)).copy()


def safe_source_regions(obs, proj, detector, box, rgb, clear):
    from shuolee_portrait.face_supplement import REGIONS

    c=POLICY
    eye,lip=rois_from_landmarks(obs,proj,box)
    observed_face=obs[67:118,:2]
    reliable=np.isfinite(obs).all(1)&(obs[:,2]>=c['min_landmark_confidence'])
    inner=observed_face[reliable[67:118]]
    skin=np.zeros(clear.shape,bool)
    if len(inner)<8 or (span:=np.ptp(inner,axis=0).max())<c['min_face_span_px']:
        return skin,skin,eye,lip,dict(reason='insufficient_original_facial_measurements')
    paired=reliable&np.isfinite(proj).all(1)&(
        np.linalg.norm(obs[:,:2]-proj,axis=1)<=min(c['max_fitted_error_px'],.08*span))
    core=np.arange(67,118)[paired[67:118]]
    if len(core)<8:
        return skin,skin,eye,lip,dict(reason='insufficient_original_and_fitted_agreement')
    contour_xy,confidence,flip=detector
    contour_ok=np.isfinite(contour_xy).all(1)&(confidence>=c['min_landmark_confidence'])&(
        flip<=min(c['max_landmark_flip_px'],.07*span))
    # Keep a paired interior anchor. The observed jaw can extend the visible
    # face polygon when native detector confidence and flip consistency agree.
    interior=local_hull(obs[core,:2],box)&local_hull(proj[core],box)
    if contour_ok.sum()>=7 and contour_ok[:8].any() and contour_ok[9:].any():
        jaw=contour_xy[contour_ok]
        points=np.concatenate((obs[core,:2],jaw))
        hull=local_hull(points,box)
        radius=max(4,round(.25*span))
        distance=cv2.distanceTransform((~interior).astype(np.uint8),cv2.DIST_L2,5)
        interior=hull&(distance<=radius)
    interior=binary_erosion(interior,iterations=max(1,round(.025*span)))&clear
    mouth=REGIONS['mouth'][paired[REGIONS['mouth']]]
    nose=REGIONS['nose'][paired[REGIONS['nose']]]
    eyes=np.r_[REGIONS['right_eye'],REGIONS['left_eye']]
    eyes=eyes[reliable[eyes]]
    detail=dict(interior_pixels=int(interior.sum()),contour_points=int(contour_ok.sum()),
                paired_nose_points=len(nose),paired_mouth_points=len(mouth))
    if len(nose)<4 or len(mouth)<6 or len(eyes)<4:
        return interior,skin,eye,lip,dict(detail,reason='skin_region_landmarks_incomplete')
    # A subject-specific color estimate is measured again in each native RGB
    # frame near the paired nose. It is conditioned on local geometry and
    # never used to label the full head or the eye/mouth neighborhoods.
    nose_xy=(obs[nose,:2]+proj[nose])*.5
    anchor=nose_xy[np.argsort(nose)[-min(5,len(nose)):]].mean(0)
    axis=obs[mouth,:2].mean(0)-obs[eyes,:2].mean(0)
    axis_length=np.linalg.norm(axis)
    if axis_length<.2*span:
        return interior,skin,eye,lip,dict(detail,reason='degenerate_eye_mouth_direction')
    axis/=axis_length
    x0,y0,x1,y1=box
    yy,xx=np.mgrid[y0:y1,x0:x1]
    down=xx*axis[0]+yy*axis[1]
    below_eyes=down>=(obs[eyes,:2]@axis).max()+max(3,round(.08*span))
    lip_hull=local_hull(obs[mouth,:2],box)|local_hull(proj[mouth],box)
    excluded=binary_dilation(eye|lip_hull,iterations=max(c['min_eye_mouth_clearance_px'],round(.05*span)))
    inside=interior&below_eyes&~excluded
    color=rgb.astype(np.float32)/255.
    luma=color@np.asarray([.2126,.7152,.0722],np.float32)
    chroma=color[..., [0,2]]/color.sum(-1,keepdims=True).clip(1e-5)
    seed=inside&((xx-anchor[0])**2+(yy-anchor[1])**2<=max(3.,.095*span)**2)&(luma>=.22)&(luma<=.92)
    if seed.sum()<c['min_skin_seed_pixels']:
        return interior,skin,eye,lip,dict(detail,reason='insufficient_visible_nose_skin_color')
    center=np.median(chroma[seed],axis=0)
    delta=np.linalg.norm(chroma-center,axis=-1)
    median_luma=float(np.median(luma[seed]))
    mad=float(np.median(delta[seed]))
    if mad>.032:
        return interior,skin,eye,lip,dict(detail,reason='mixed_native_nose_seed',seed_chroma_mad=mad)
    skin=binary_erosion(inside&(delta<=np.clip(3.5*mad,.022,.063))&
                        (luma>=max(.18,.61*median_luma))&(luma<=min(.96,1.35*median_luma)),iterations=1)
    return interior,skin,eye,lip,dict(detail,seed_pixels=int(seed.sum()),seed_chroma_mad=mad,skin_pixels=int(skin.sum()))


def source_grid(row, obs, proj, detector, box, annotations, allowed=None):
    rgb=decode(pin(row['image'],row['source_sha256']),box)
    matte=decode(pin(row['mask'],row['mask_sha256']),box,'L')
    clear=binary_erosion(matte>=POLICY['foreground_alpha_min'],iterations=1)
    if allowed is not None:
        need(allowed.shape==clear.shape and np.all(matte[allowed]>=252),'The auxiliary face-only mask changed source support.')
        clear&=allowed
    x0,y0,x1,y1=box
    for l,t,r,b in row.get('other_person_boxes_xyxy',()):
        left,top=max(x0,int(l)),max(y0,int(t)); right,bottom=min(x1,int(r)),min(y1,int(b))
        if left<right and top<bottom:
            clear[top-y0:bottom-y0,left-x0:right-x0]=False
    interior,skin,eye,lip,summary=safe_source_regions(obs,proj,detector,box,rgb,clear)
    masks={}
    if annotations:
        need(annotations['source_sha256']==row['source_sha256'],'Signed material polygon source differs.')
    for name,item in annotations.get('masks',{}).items():
        if item['validated']:
            full=material_mask(row['wh'],item['polygons'])
            need(array_sha(full)==item['full_roi_sha256'],'Signed material polygon differs.')
            masks[name]=full[y0:y1,x0:x1]
    blank=np.zeros(clear.shape,bool)
    hair=masks.get('hair',blank)
    eyewear=masks.get('glasses_rim',blank)
    # A specific material boundary may veto skin, but conflicting eye/rim
    # pixels remain unlabelled unless their masks actually agree on a class.
    skin&=~binary_dilation(hair|eyewear,iterations=2)
    stack=np.stack((skin,eye,hair,eyewear,lip))&clear
    grid=np.zeros(clear.shape,np.uint8)
    unique=stack.sum(0)==1
    for index, region in enumerate(stack,1):
        grid[region&unique]=index
    if allowed is None:
        near=cv2.distanceTransform((~clear).astype(np.uint8),cv2.DIST_L2,5)
        grid[(matte<=POLICY['background_alpha_max'])&(near<=POLICY['max_near_silhouette_px'])]=BACKGROUND
    else:
        # The signed auxiliary region already binds observed landmarks,
        # foreground reliability, mesh visibility and original alpha.
        interior=clear.copy()
    interior|=(grid>0)&(grid<BACKGROUND)
    scope=np.ones(clear.shape,bool) if allowed is None else allowed
    return grid,interior,scope,dict(summary,material_pixels=np.bincount(grid[scope],minlength=BACKGROUND+1).tolist(),
                                   actual_interior_pixels=int(interior.sum()),annotation_materials=sorted(masks))


def compositor(pre, grid, interior, scope, box, truth, lookup, alpha=None):
    x0,y0,x1,y1=box
    out=np.zeros((int(lookup.max())+1,INTERIOR+1),np.float64)
    proof=dict(pixels=0,exact_pixels=0,excluded_pixels=0,max_code_error=0,
               alpha_checked_pixels=0,max_alpha_code_error=0)
    for tx in range(x0//16,(x1+15)//16):
        for ty in range(y0//16,(y1+15)//16):
            xx,yy=np.meshgrid(np.arange(max(x0,tx*16),min(x1,(tx+1)*16)),np.arange(max(y0,ty*16),min(y1,(ty+1)*16)))
            x,y=xx.ravel(),yy.ravel()
            inside=scope[y-y0,x-x0]
            x,y=x[inside],y[inside]
            if not len(x):
                continue
            rows,weights=native_weights(pre,(tx,ty),np.column_stack((x,y)))
            error=np.abs(code(weights@pre['rgb'][rows])-truth[y-y0,x-x0]).max(1)
            if alpha is not None:
                (left,top,right,bottom),image=alpha
                valid=(x>=left)&(x<right)&(y>=top)&(y<bottom)
                if valid.any():
                    delta=np.abs(code(weights[valid].sum(1))-image[y[valid]-top,x[valid]-left])
                    error[valid]=np.maximum(error[valid],delta)
                    proof['alpha_checked_pixels']+=int(valid.sum())
                    proof['max_alpha_code_error']=max(proof['max_alpha_code_error'],int(delta.max()))
            ok=error<=POLICY['code_parity_max']
            proof['pixels']+=len(error)
            proof['exact_pixels']+=int((error==0).sum())
            proof['excluded_pixels']+=int((~ok).sum())
            proof['max_code_error']=max(proof['max_code_error'],int(error.max()))
            ids=lookup[rows]; selected=ids>=0
            if not selected.any():
                continue
            w=weights[:,selected]
            cls=grid[y-y0,x-x0]
            for label in np.unique(cls[ok]):
                out[ids[selected],label]+=w[ok&(cls==label)].sum(0,dtype=np.float64)
            out[ids[selected],INTERIOR]+=w[ok&interior[y-y0,x-x0]].sum(0,dtype=np.float64)
    return out,proof


def classify(view_mass, source_bins):
    c=POLICY
    interior=view_mass[:,:,INTERIOR]
    visible=interior>=c['min_view_alpha_px']
    bins=np.asarray(source_bins)
    groups=[bins==value for value in sorted(set(source_bins))]
    visible_bins=np.stack([visible[g].any(0) for g in groups]).sum(0)
    total=view_mass[:,:,:INTERIOR].sum(2,keepdims=True)
    score=view_mass[:,:,1:BACKGROUND]
    votes=(score>=c['min_material_alpha_px'])&(score>=c['min_material_fraction']*total)
    vote_bins=np.stack([votes[g].any(0) for g in groups]).sum(0)
    protect=(score[:,:,1:]>=c['protect_material_alpha_px'])&(score[:,:,1:]>=c['protect_material_fraction']*total)
    protected=protect.any(axis=(0,2))
    pooled=score.sum(0)
    winner=np.argmax(pooled,axis=1)
    ids=np.arange(len(winner))
    accepted=(vote_bins[ids,winner]>=c['min_material_bins'])&(
        pooled[ids,winner]>=c['min_known_consensus_fraction']*pooled.sum(1))
    accepted&=(winner!=0)|~protected
    labels=np.where(accepted,winner+1,0).astype(np.uint8)
    spill_mass=view_mass[:,:,BACKGROUND]
    spill_votes=(spill_mass>=c['spill_alpha_px'])&(spill_mass>=c['spill_fraction']*total[:,:,0])
    spill_bins=np.stack([spill_votes[g].any(0) for g in groups]).sum(0)
    return dict(labels=labels,observed=visible.any(0),visible_views=visible.sum(0).astype(np.uint8),
                visible_bins=visible_bins.astype(np.uint8),
                material_bins=np.where(accepted,vote_bins[ids,winner],0).astype(np.uint8),
                evidence_interior_mass=interior.sum(0).astype(np.float32),geometry_eligible=labels==1,
                silhouette_spill=(labels==0)&~protected&(spill_bins>=c['spill_bins']),
                possible_protected_material=protected)


def elongated(local):
    q=numbered(local,'rot',4).astype(np.float64)
    q/=np.linalg.norm(q,axis=1,keepdims=True)
    axes=quaternion_matrix(q)
    scales=np.exp(numbered(local,'scale',3).astype(np.float64))
    tangents=axes[:,[0,2],:]*scales[:,None,:]
    sv=np.linalg.svd(tangents,compute_uv=False)
    major_normal=axes[np.arange(len(q)),1,scales.argmax(1)]
    return (sv[:,0]>=POLICY['long_tangent_ratio']*sv[:,1]) & (
        1.-major_normal**2>=POLICY['major_axis_min_tangent_projection']**2)


def write_report(subject, context, head, rows, reports, masses, auxiliary, holdouts, provenance):
    output=classify(masses,[view['source_bin'] for view in reports])
    initial=classify(masses[:24],[view['source_bin'] for view in reports[:24]])
    all_rows={name:np.zeros(len(context['stable']),dtype=values.dtype) for name,values in output.items()}
    for name,values in output.items():
        all_rows[name][head]=values
    head_scope=np.zeros(len(context['stable']),bool); head_scope[head]=True
    unknown=head_scope&(all_rows['labels']==0)&~all_rows['observed']&~all_rows['possible_protected_material']
    all_rows['unknown_unobserved_elongated']=unknown&elongated(context['local'])
    all_rows['head_scope']=head_scope
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'{subject}-material-visibility.npz'
    with path.open('xb') as file:
        np.savez_compressed(file,stable_ids=context['stable'],owner_faces=context['owner'],**all_rows)
    def coverage(data):
        return dict(label_counts=dict(zip(LABELS,np.bincount(data['labels'],minlength=len(LABELS)).tolist())),
            **{name:int(data[name].sum()) for name in ('observed','geometry_eligible','silhouette_spill','possible_protected_material')})
    index=np.full(len(head_scope),-1,np.int64); index[head]=np.arange(len(head))
    old_ray=bad_rays(context,all_rows['labels'],index,all_rows['silhouette_spill'],subject)
    for view in old_ray.values():
        for pair in view['pairs']:
            for role in ('artifact','control'):
                for contributor in pair[role]['top_rows']:
                    local=int(np.searchsorted(context['stable'],contributor['stable_id']))
                    contributor.update(observed=bool(all_rows['observed'][local]),
                        geometry_eligible=bool(all_rows['geometry_eligible'][local]),
                        unknown_unobserved_elongated=bool(all_rows['unknown_unobserved_elongated'][local]))
    proof=dict(schema='shuolee-r27-native-face-material-visibility/1',subject=subject,
        baseline_raw_ply_sha256=context['hashes']['world'],baseline_bound_ply_sha256=context['hashes']['bound_local'],
        material_visibility_npz_sha256=sha(path),fit_sha256=provenance['fit_sha256'],
        training_inputs_sha256=provenance['training_inputs_sha256'],
        classification_frames=[view['frame'] for view in reports],selected_frames=[row['frame'] for row in rows],
        admitted_auxiliary_frames=auxiliary,retained_holdouts=holdouts,heldout_source_images_read=0,
        label_codes=dict(enumerate(LABELS)),policy=POLICY,coverage=dict(head_rows=len(head),main_24=coverage(initial),
            main_24_plus_aux_4=coverage(output),unknown_unobserved_elongated=int(all_rows['unknown_unobserved_elongated'].sum()),
            distinct_source_bins=sorted({view['source_bin'] for view in reports}),
            added_observed_from_aux=int((output['observed']&~initial['observed']).sum()),
            added_skin_from_aux=int((output['geometry_eligible']&~initial['geometry_eligible']).sum())),
        views=reports,previous_novel_ray_coverage=old_ray,provenance=provenance,
        scope='Original foreground and original landmark support, verified through actual compositing. '
              'The former R22 and R26 per-row material labels are not a prerequisite. Auxiliary contributes only signed face-only pixels. '
              'Unobserved elongated geometry is a diagnosis about current source coverage, without material identity evidence.')
    target=path.with_suffix('.json')
    with target.open('x',encoding='utf-8') as file:
        json.dump(proof,file,ensure_ascii=False,indent=2,allow_nan=False)
        file.write('\n')
    print(dict(subject=subject,file=str(target),sha256=sha(target),npz_sha256=sha(path),coverage=proof['coverage']),flush=True)


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    context=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    model,slots=posed_model(context)
    pose=context['plan']['poses']['r22']
    root=Path(context['arm']['run']); result=read(root/'result.json')
    replay_path=pin(root/result['bound_replay'],result['bound_replay_sha256'])
    replay=read(replay_path)
    replay_rows={row['frame']:row for row in replay['records']}
    main_path=pin(root/'data/training-inputs.json',pose['new']['training_inputs_sha256'])
    main=read(main_path)['records']
    fit_path=pin(root/'fit/fitted.npz',pose['new']['fitted_sha256'])
    with np.load(fit_path,allow_pickle=False) as fit:
        main_obs,main_proj,faces=(fit[name] for name in ('observed_native_keypoints','joints_projected_native','faces'))
    proof=pose['proof']; admission=read(checked(proof['supplement_admission']))
    auxiliary=pose['supplement']['frames']
    aux_path=checked(proof['supplement_replay']); aux=read(aux_path)
    aux_ids=[row['frame'] for row in auxiliary]
    holdouts=pose['new']['protected_holdouts']
    need(len(main)==24 and len(aux_ids)==4 and len(holdouts)==4 and len(set(list(slots)+aux_ids+holdouts))==32
         and aux_ids==admission['chosen_frame_ids']==[row['frame'] for row in aux['records']]
         and aux['shared_bound_ply_sha256']==context['hashes']['bound_local'],'Wrong R27 source roles or terminal bound model.')
    aux_rows={row['frame']:row for row in read(pin(root/'data/inputs.json',admission['input_sha256']['native_inputs']))['records']}
    with np.load(checked(proof['supplement_poses']),allow_pickle=False) as fit:
        aux_vertices=fit['vertices_world']
        need(fit['frame_ids'].tolist()==aux_ids and np.array_equal(fit['faces'],faces)
             and fit['pose_indices'].tolist()==list(range(24,28)), 'The signed auxiliary poses differ.')
    model.vertices_sequence=torch.cat((model.vertices_sequence,torch.from_numpy(aux_vertices)))
    candidates=pin(root/'fit/joint-candidates.npz',admission['input_sha256']['joint_candidates'])
    with np.load(candidates,allow_pickle=False) as fit:
        names=fit['frame_ids'].tolist(); source_slots=[names.index(frame) for frame in aux_ids]
        aux_obs,aux_proj=(fit[name][source_slots] for name in ('observed_native_keypoints','joints_projected_native'))
        for key in ('K_integer_centers','image_sha256','mask_sha256'):
            expected=[row['admission']['native_camera'][key] if key=='K_integer_centers' else
                      row['source_sha256' if key=='image_sha256' else key] for row in auxiliary]
            need(np.array_equal(fit[key][source_slots],expected),f'Auxiliary {key} differs from signed cameras.')
    observe=pin(root/'data/observations.npz',admission['input_sha256']['observations'])
    with np.load(observe,allow_pickle=False) as data:
        names=data['frame_ids'].tolist()
        joints=[data['joint_names'].tolist().index(f'face-{index}') for index in range(17)]
        detections={name:tuple(data[key][names.index(name),joints] for key in ('keypoints','scores','flip_disagreement'))
                    for name in list(slots)+aux_ids}
    material=read(pin(DELIVERY/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW))['subjects'][subject]['native_annotation']
    manifest=read(replay_path.parent.parent/'bound-replay.json')
    faces_scope=protected_faces(pin(context['plan']['asset'],manifest['asset_sha256']),faces)
    head=np.flatnonzero(faces_scope[context['owner']])
    lookup=np.full(len(context['stable']),-1,np.int64); lookup[head]=np.arange(len(head))
    native=context['result']['native']['r22']
    roles={row['frame']:row['role'] for row in pose['pairs']}
    _,_,_,opacity,rgb=context['arrays']
    reports,masses=[],[]
    for slot,name in enumerate(list(slots)+aux_ids):
        model.select_mesh_by_timestep(slot)
        valid=None
        if slot<24:
            row=main[slot]; replay_row=replay_rows[name]; cam=replay_row['camera']
            need(name==row['frame'] and all(cam[key]==row[key] for key in ('K','w2c','wh'))
                 and replay_row['source_sha256']==row['source_sha256'] and replay_row['mask_sha256']==row['mask_sha256'],
                 'Main source/native renderer mismatch.')
            box=head_crop(model.vertices_sequence[slot].numpy(),faces,faces_scope,cam)
            true_path=pin(replay_path.parent/replay_row['image'],replay_row['image_sha256'])
            truth=decode(true_path,box).astype(np.int16)
            alpha=alpha_panel(native,roles.get(name))
            obs,proj,bin_,role=main_obs[slot],main_proj[slot],context['source_bins'][name],'selected_fit'
        else:
            a=slot-24; saved=auxiliary[a]; row=aux_rows[name]; cam=saved['camera']; box=saved['box']
            need(saved['pose_index']==slot and saved['role']=='face_only' and cam['K']==row['K'] and cam['wh']==row['wh']
                 and saved['source_sha256']==row['source_sha256'] and saved['mask_sha256']==row['mask_sha256'],
                 'Auxiliary source camera or content differs.')
            valid=decode(pin(saved['valid_path'],saved['valid_mask_sha256']),mode='L')>0
            need(int(valid.sum())==saved['valid_pixels'],'Auxiliary source supervision support changed.')
            files=saved['files']
            true_path=pin(aux_path.parent/files['gaussian']['file'],files['gaussian']['sha256'])
            truth=decode(true_path).astype(np.int16)
            alpha=box,decode(pin(aux_path.parent/files['alpha']['file'],files['alpha']['sha256']),mode='L')
            obs,proj,bin_,role=aux_obs[a],aux_proj[a],saved['admission']['source_bin'],'auxiliary_face'
        grid,interior,scope,stats=source_grid(row,obs,proj,detections[name],box,material.get(name,{}),valid)
        arrays=(model.get_xyz.numpy(),model.get_scaling.numpy(),model.get_rotation.numpy(),opacity,rgb)
        mass,parity=compositor(preprocess(arrays,cam),grid,interior,scope,box,truth,lookup,alpha)
        need(parity['pixels']>0 and parity['excluded_pixels']/parity['pixels']<.01,
             'Native compositor lost R22 renderer parity.')
        report=dict(frame=name,role=role,source_bin=int(bin_),source_sha256=row['source_sha256'],
            mask_sha256=row['mask_sha256'],native_render_sha256=sha(true_path),box=list(box),
            label_mask_sha256=array_sha(grid),interior_sha256=array_sha(interior),pixel_scope_sha256=array_sha(scope),
            parity=parity,source_evidence=stats)
        masses.append(mass); reports.append(report)
        print(dict(subject=subject,frame=name,role=role,pixels=stats['actual_interior_pixels'],
                   skin_pixels=stats.get('skin_pixels',0),parity=parity['excluded_pixels']),flush=True)
    write_report(subject,context,head,main,reports,np.stack(masses),aux_ids,holdouts,
        dict(fit_sha256=sha(fit_path),training_inputs_sha256=sha(main_path),observations_sha256=sha(observe),
             fitted_auxiliary_landmarks_sha256=sha(candidates),auxiliary_admission_sha256=proof['supplement_admission']['sha256'],
             auxiliary_replay_sha256=sha(aux_path),auxiliary_pose_sha256=proof['supplement_poses']['sha256'],
             baseline_review_index_sha256=REVIEW_SHA,source_sha256=sha(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
