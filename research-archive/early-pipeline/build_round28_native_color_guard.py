"""Estimate guarded SH0 source RGB intervals from directly contributing native pixels."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from build_round26_native_face_material import MATERIAL_REVIEW, alpha_panel, array_sha, native_weights
from build_round27_material_visibility import OUT as R27, decode, source_grid
from trace_round22_face_pixels import (DELIVERY, PROJECT, REVIEW_SHA, checked, code, load_subject, need, pin,
                                       posed_model, preprocess, read, sha)


OUT=DELIVERY.with_name('smpl-delivery-round28')/'input-audit'
POLICY=dict(min_pixel_composite_alpha=.055,min_pixel_contribution_fraction=.22,min_pixels_in_bin=4,
    min_equivalent_alpha_px_in_bin=.75,max_bin_rgb_std=.10,min_angular_bins=2,
    max_across_bin_rgb_quantile_span=.19,source_rgb_margin=.04,code_parity_max=1)
ARRAYS=('source_rgb_low','source_rgb_high','source_rgb_center','source_color_bins','source_color_guard','source_color_mass')


def source_inputs(context, subject, main, pose, expected):
    root=Path(context['arm']['run'])
    model,slots=posed_model(context)
    result=read(root/'result.json')
    replay_path=pin(root/result['bound_replay'],result['bound_replay_sha256'])
    replay={row['frame']:row for row in read(replay_path)['records']}
    proof=pose['proof']; admission=read(checked(proof['supplement_admission']))
    aux=pose['supplement']['frames']
    auxiliary=[row['frame'] for row in aux]
    auxiliary_path=pin(root/'data/inputs.json',admission['input_sha256']['native_inputs'])
    aux_rows={row['frame']:row for row in read(auxiliary_path)['records']}
    aux_replay=checked(proof['supplement_replay'])
    fit_path=pin(root/'fit/fitted.npz',expected['fit_sha256'])
    with np.load(fit_path,allow_pickle=False) as fit:
        main_obs,main_proj,faces=(fit[name] for name in ('observed_native_keypoints','joints_projected_native','faces'))
    with np.load(checked(proof['supplement_poses']),allow_pickle=False) as fit:
        need(np.array_equal(fit['faces'],faces) and fit['frame_ids'].tolist()==auxiliary and
             fit['pose_indices'].tolist()==list(range(24,28)), 'The signed auxiliary poses changed.')
        model.vertices_sequence=torch.cat((model.vertices_sequence,torch.from_numpy(fit['vertices_world'])))
    candidates=pin(root/'fit/joint-candidates.npz',admission['input_sha256']['joint_candidates'])
    with np.load(candidates,allow_pickle=False) as fit:
        names=fit['frame_ids'].tolist(); indices=[names.index(frame) for frame in auxiliary]
        aux_obs,aux_proj=(fit[name][indices] for name in ('observed_native_keypoints','joints_projected_native'))
    observe=pin(root/'data/observations.npz',admission['input_sha256']['observations'])
    with np.load(observe,allow_pickle=False) as data:
        names=data['frame_ids'].tolist()
        joints=[data['joint_names'].tolist().index(f'face-{index}') for index in range(17)]
        detections={name:tuple(data[key][names.index(name),joints] for key in ('keypoints','scores','flip_disagreement'))
                    for name in list(slots)+auxiliary}
    annotations=read(pin(DELIVERY/'input-audit/face-pixel-attribution.json',MATERIAL_REVIEW))['subjects'][subject]['native_annotation']
    native=context['result']['native']['r22']
    roles={row['frame']:row['role'] for row in pose['pairs']}
    need(expected['classification_frames']==list(slots)+auxiliary and expected['admitted_auxiliary_frames']==auxiliary
         and expected['selected_frames']==[row['frame'] for row in main] and
         not set(expected['classification_frames'])&set(expected['retained_holdouts']), 'The R28 source roster differs from R27.')
    _,_,_,opacity,rgb=context['arrays']
    for slot,item in enumerate(expected['views']):
        name=item['frame']; box=item['box']; valid=None
        model.select_mesh_by_timestep(slot)
        if slot<24:
            row=main[slot]; saved=replay[name]; cam=saved['camera']
            rendered=pin(replay_path.parent/saved['image'],item['native_render_sha256'])
            truth=decode(rendered,box).astype(np.int16)
            alpha=alpha_panel(native,roles.get(name))
            obs,proj=main_obs[slot],main_proj[slot]
        else:
            i=slot-24; saved=aux[i]; row=aux_rows[name]; cam=saved['camera']
            need(saved['box']==box and saved['pose_index']==slot and saved['role']=='face_only','Auxiliary source control changed.')
            valid=decode(pin(saved['valid_path'],saved['valid_mask_sha256']),mode='L')>0
            rendered=pin(aux_replay.parent/saved['files']['gaussian']['file'],item['native_render_sha256'])
            truth=decode(rendered).astype(np.int16)
            alpha=box,decode(pin(aux_replay.parent/saved['files']['alpha']['file'],saved['files']['alpha']['sha256']),mode='L')
            obs,proj=aux_obs[i],aux_proj[i]
        need(row['frame']==name and row['source_sha256']==item['source_sha256']==saved['source_sha256'] and
             row['mask_sha256']==item['mask_sha256']==saved['mask_sha256'] and cam['K']==row['K'] and cam['wh']==row['wh'],
             'Original RGB/alpha/camera identity changed.')
        label,interior,scope,_=source_grid(row,obs,proj,detections[name],box,annotations.get(name,{}),valid)
        need(array_sha(label)==item['label_mask_sha256'] and array_sha(interior)==item['interior_sha256'] and
             array_sha(scope)==item['pixel_scope_sha256'], 'R27 original native material masks failed byte-for-byte replay.')
        source=decode(pin(row['image'],row['source_sha256']),box).astype(np.float32)/255.
        parameters=(model.get_xyz.numpy(),model.get_scaling.numpy(),model.get_rotation.numpy(),opacity,rgb)
        yield item,preprocess(parameters,cam),label,interior&scope,source,truth,alpha


def measure(pre, label, interior, source, truth, alpha, box, lookup, labels):
    x0,y0,x1,y1=box
    count=len(labels)
    alpha_mass=np.zeros(count,np.float64)
    rgb_sum=np.zeros((count,3),np.float64)
    rgb_square=np.zeros_like(rgb_sum)
    used=np.zeros(count,np.int64)
    proof=dict(interior_pixels=0,parity_excluded=0,max_code_error=0,alpha_checked=0)
    yy,xx=np.nonzero(interior)
    if not len(xx):
        return alpha_mass,rgb_sum,rgb_square,used,proof
    left,top,right,bottom=int(xx.min())+x0,int(yy.min())+y0,int(xx.max())+x0+1,int(yy.max())+y0+1
    for tx in range(left//16,(right+15)//16):
        for ty in range(top//16,(bottom+15)//16):
            u,v=np.meshgrid(np.arange(max(left,tx*16),min(right,(tx+1)*16)),np.arange(max(top,ty*16),min(bottom,(ty+1)*16)))
            x,y=u.ravel(),v.ravel()
            keep=interior[y-y0,x-x0]
            x,y=x[keep],y[keep]
            if not len(x):
                continue
            rows,weights=native_weights(pre,(tx,ty),np.column_stack((x,y)))
            rendered=code(weights@pre['rgb'][rows])
            error=np.abs(rendered-truth[y-y0,x-x0]).max(1)
            full_mass=weights.sum(1)
            if alpha is not None:
                (a,b,c,d),image=alpha
                check=(x>=a)&(x<c)&(y>=b)&(y<d)
                if check.any():
                    error[check]=np.maximum(error[check],np.abs(code(full_mass[check])-image[y[check]-b,x[check]-a]))
                    proof['alpha_checked']+=int(check.sum())
            ok=error<=POLICY['code_parity_max']
            proof['interior_pixels']+=len(error)
            proof['parity_excluded']+=int((~ok).sum())
            proof['max_code_error']=max(proof['max_code_error'],int(error.max()))
            ids=lookup[rows]; eligible=ids>=0
            if not eligible.any():
                continue
            ids=ids[eligible]
            w=weights[:,eligible]
            pixels=label[y-y0,x-x0,None]
            classes=labels[ids][None,:]
            same_material=(classes==pixels)|((classes==0)&(pixels<=1))
            support=ok[:,None]&same_material&(w>=POLICY['min_pixel_composite_alpha'])&(
                w>=POLICY['min_pixel_contribution_fraction']*full_mass[:,None])
            observed=np.where(support,w,0.).astype(np.float64)
            color=source[y-y0,x-x0].astype(np.float64)
            alpha_mass[ids]+=observed.sum(0)
            used[ids]+=support.sum(0)
            rgb_sum[ids]+=observed.T@color
            rgb_square[ids]+=observed.T@(color*color)
    return alpha_mass,rgb_sum,rgb_square,used,proof


def intervals(mass,total,squared,count):
    denom=np.maximum(mass[:,:,None],1e-10)
    colors=total/denom
    deviation=np.sqrt(np.maximum(0.,squared/denom-colors**2))
    accepted=(mass>=POLICY['min_equivalent_alpha_px_in_bin'])&(count>=POLICY['min_pixels_in_bin'])&(
        deviation.max(2)<=POLICY['max_bin_rgb_std'])
    bins=accepted.sum(0).astype(np.uint8)
    guarded=np.zeros(len(bins),bool)
    low=np.zeros((len(bins),3),np.float32)
    high=np.ones_like(low); center=np.zeros_like(low)
    for index in np.flatnonzero(bins>=POLICY['min_angular_bins']):
        use=accepted[:,index]
        observed=colors[use,index]
        a,b=np.quantile(observed,[.1,.9],axis=0)
        if (b-a).max()>POLICY['max_across_bin_rgb_quantile_span']:
            continue
        tolerance=np.maximum(POLICY['source_rgb_margin'],np.median(deviation[use,index],axis=0))
        guarded[index]=True
        low[index]=np.maximum(0.,a-tolerance)
        high[index]=np.minimum(1.,b+tolerance)
        center[index]=np.median(observed,axis=0)
    return low,high,center,bins,guarded,np.where(accepted,mass,0.).sum(0).astype(np.float32)


def run(subject):
    sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'src/avatar_core')]
    torch.set_num_threads(2)
    material_path=R27/f'{subject}-material-visibility.json'; material=read(material_path)
    archive=pin(material_path.with_suffix('.npz'),material['material_visibility_npz_sha256'])
    base=load_subject(subject,read(pin(DELIVERY/'review-receipt.json',REVIEW_SHA)))
    pose=base['plan']['poses']['r22']
    main=read(pin(Path(base['arm']['run'])/'data/training-inputs.json',material['training_inputs_sha256']))['records']
    with np.load(archive,allow_pickle=False) as evidence:
        stable,owner,labels,observed,protect=(evidence[key] for key in ('stable_ids','owner_faces','labels','observed','possible_protected_material'))
    need(np.array_equal(stable,base['stable']) and np.array_equal(owner,base['owner']) and material['heldout_source_images_read']==0,
         'R28 color guard lost frozen source identity.')
    eligible=np.flatnonzero(observed&((labels!=0)|~protect))
    lookup=np.full(len(stable),-1,np.int64); lookup[eligible]=np.arange(len(eligible))
    angular=material['coverage']['distinct_source_bins']
    shape=(len(angular),len(eligible))
    mass=np.zeros(shape,np.float64); counts=np.zeros(shape,np.int64)
    color=np.zeros((*shape,3),np.float64); square=np.zeros_like(color)
    views=[]
    for item,project,grid,interior,source,truth,alpha in source_inputs(base,subject,main,pose,material):
        a,b,c,d,proof=measure(project,grid,interior,source,truth,alpha,item['box'],lookup,labels[eligible])
        i=angular.index(item['source_bin'])
        mass[i]+=a; color[i]+=b; square[i]+=c; counts[i]+=d
        if proof['interior_pixels']:
            need(proof['parity_excluded']/proof['interior_pixels']<.01,'Native RGB/alpha parity failed on an R28 source ROI.')
        views.append(dict(frame=item['frame'],role=item['role'],source_bin=item['source_bin'],
                          source_sha256=item['source_sha256'],mask_sha256=item['mask_sha256'],
                          native_render_sha256=item['native_render_sha256'],label_mask_sha256=item['label_mask_sha256'],
                          interior_sha256=item['interior_sha256'],eligible_source_rows=int((a>0).sum()),parity=proof))
    values=intervals(mass,color,square,counts)
    proof={name:np.zeros((len(stable),*value.shape[1:]),dtype=value.dtype) for name,value in zip(ARRAYS,values,strict=True)}
    proof['source_rgb_high'].fill(1.)
    for name,value in zip(ARRAYS,values,strict=True):
        proof[name][eligible]=value
    guard=proof['source_color_guard']
    need(guard.any() and np.isfinite(proof['source_rgb_low']).all() and np.isfinite(proof['source_rgb_high']).all() and
         np.all(proof['source_rgb_low']<=proof['source_rgb_high']) and np.all(observed[guard]), 'R28 color evidence is unusable.')
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/f'{subject}-native-color-guard.npz'
    with path.open('xb') as file:
        np.savez_compressed(file,stable_ids=stable,owner_faces=owner,**proof)
    counts=dict(eligible_rows=len(eligible),guarded_rows=int(guard.sum()),observed_rows=int(observed.sum()),
                guarded_by_label=np.bincount(labels[guard],minlength=6).tolist(),
                guarded_bin_histogram=np.bincount(proof['source_color_bins'][guard]).tolist())
    receipt=dict(schema='shuolee-r28-native-sh0-color-guard/1',subject=subject,source_color_guard_npz_sha256=sha(path),
        baseline_raw_ply_sha256=base['hashes']['world'],baseline_bound_ply_sha256=base['hashes']['bound_local'],
        material_visibility_npz_sha256=sha(archive),material_visibility_receipt_sha256=sha(material_path),
        source_frames=material['classification_frames'],selected_frames=material['selected_frames'],
        admitted_auxiliary_frames=material['admitted_auxiliary_frames'],retained_holdouts=material['retained_holdouts'],
        heldout_source_images_read=0,policy=POLICY,coverage=counts,views=views,source_sha256=sha(__file__),
        rgb_convention='Native PNG RGB code values divided by 255. These are the values represented by SH0 before raster composition.',
        scope='Weighted native RGB from parity-checked foreground pixels where the frozen Gaussian contributes materially. '
              'The interval is an appearance guard, with unknown geometry and unsupported color retaining their source uncertainty.')
    target=path.with_suffix('.json')
    with target.open('x',encoding='utf-8') as file:
        json.dump(receipt,file,indent=2,ensure_ascii=False,allow_nan=False)
        file.write('\n')
    print(dict(subject=subject,file=str(target),sha256=sha(target),npz_sha256=sha(path),coverage=counts),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject',choices=('016','006'))
    run(parser.parse_args().subject)
