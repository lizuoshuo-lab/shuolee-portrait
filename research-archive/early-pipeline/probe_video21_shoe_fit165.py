"""Matched SMPLify-X final stages with optional relative shoe depth."""
from datetime import datetime, timezone
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

import fit_video21_seated149 as seated
import official_smplifyx_entry123 as entry
from prepare_video21_shoe_binding163 import CAMERA, FOOT_BONES, bind_one
from track_video21_shoe_overlap164 import OUT as MATERIAL
from update_leafwoman129_state import atomic, read, sha


BASE=seated.base
OUT=BASE.OUT/'shoe-depth-refine165'
BRANCHES=('control','depth')
# These tracks were checked on the frozen material gallery before fitting.
REVIEWED={
    'frame_000029':['left_01','left_02','right_00','right_01'],
    'frame_000053':['left_00','left_01','left_02','right_00','right_01'],
}


def prepare_pairs(plan, source, source_rows):
    camera={r['frame']:r for r in read(CAMERA)}
    camera_indices={r['source_index']:camera[r['frame']]
                    for r in read(BASE.OUT/'dense-camera-tracks158/result.json')['camera_frame_receipts']}
    seeds={r['frame']:r for r in source['frames']}
    K=np.asarray(plan['K_integer_centers'])
    with np.load(BASE.original.ASSET,allow_pickle=False) as z:
        weights=z['weights']
    result={}
    for frame,index in zip(plan['source_frames'],plan['native_indices'],strict=True):
        rows=[r for r in source_rows if index in r.get('fit_native_indices',[])
              and r['material'] in REVIEWED[r['reference']]]
        if not rows:
            continue
        assert frame not in plan['retained_holdouts']
        seed=seeds[frame]
        assert sha(seed['output'])==seed['sha256']
        with np.load(seed['output'],allow_pickle=False) as z:
            verts=z['vertices_camera'][0].astype(float); faces=z['faces']
            assert np.array_equal(K,z['K_integer_centers'])
        tags=np.full(len(faces),'other',dtype='<U5')
        for side,indices in FOOT_BONES.items():
            foot=weights[:,indices].sum(1)>.5
            tags[foot[faces].all(1)]=side
        C=np.asarray(camera[frame]['C_world'])
        R=np.asarray(camera[frame]['R_w2c'])
        anchors=[]
        for side in ('left','right'):
            for r in sorted(rows,key=lambda a:a['material']):
                if not r['material'].startswith(side+'_'):
                    continue
                assert r['state']=='local_prediction' and r['held_max_px']<1.
                uv=r['observed_uv'][str(index)]
                bind=bind_one(verts,faces,tags,K,uv,side)
                if not bind['bound'] or bind['pixel_distance']!=0.:
                    continue
                X=np.asarray(r['xyz_arbitrary_scale'])
                z=float((R@(X-C))[2])
                rays=[X-np.asarray(camera_indices[i]['C_world']) for i in r['triangulated_from']]
                rays=[q/np.linalg.norm(q) for q in rays]
                angle=float(np.arccos(np.clip(rays[0]@rays[1],-1.,1.)))
                if z<=0 or angle<np.deg2rad(.5):
                    continue
                anchors.append(dict(side=side,window=r['reference'],material=r['material'],uv=uv,
                                    z_scene_gauge=z,angle_rad=angle,**bind))
                break
        record=dict(seed=seed,native_index=index,anchors=anchors,admitted=len(anchors)==2)
        if record['admitted']:
            a,b=anchors
            record.update(target_log_ratio=float(np.log(a['z_scene_gauge']/b['z_scene_gauge'])),
                initial_log_ratio=float(np.log(a['seed_z_model']/b['seed_z_model'])),
                equivalent_px_per_log_ratio=float(min(K[0,0],K[1,1])/
                    np.sqrt(sum(1./np.sin(r['angle_rad'])**2 for r in anchors))))
        result[frame]=record
        print('FOOT_DEPTH_READY',frame,[r['side'] for r in anchors],
              record.get('target_log_ratio'),record.get('initial_log_ratio'),flush=True)
    return result


def fit_shard(folder, index):
    plan=read(folder/'plan.json')
    use_depth=folder.name=='depth'

    def fit(**kwargs):
        import torch
        import smplify_seeded_refine124 as refine

        frame=kwargs['result_path'].parent.name
        pair=plan['pairs'][frame]
        assert pair['admitted']
        row=pair['seed']; assert sha(row['output'])==row['sha256']
        with np.load(row['output'],allow_pickle=False) as z:
            seed=dict(z)
        assert np.array_equal(seed['observed_native_keypoints'],kwargs['keypoints'])
        assert np.array_equal(seed['K_integer_centers'],kwargs['K_integer_centers'])
        calls=0
        pixel_scale=pair['equivalent_px_per_log_ratio']
        target=pair['target_log_ratio']
        ids=torch.as_tensor([a['vertices'] for a in pair['anchors']],dtype=torch.long)
        bary=torch.as_tensor([a['barycentric'] for a in pair['anchors']],dtype=torch.float32)

        def anchored_loader(root,anchor,receipt):
            utils,prior,fitting,optimizer=refine.load_with_anchor(root,anchor,receipt)
            factory=fitting.create_loss

            def make(loss_type='smplify',**options):
                loss=factory(loss_type=loss_type,**options)
                if loss_type!='smplify' or not use_depth:
                    return loss
                forward=loss.forward

                def with_depth(obj,*args,**kw):
                    nonlocal calls
                    original=forward(obj,*args,**kw)
                    cam=kw['camera']
                    z=(obj.vertices[:,ids,2]*bary).sum(2)+cam.translation[:,2,None]
                    logz=z.clamp_min(.05).log()
                    mismatch=pixel_scale*(logz[:,0]-logz[:,1]-target)
                    calls+=1
                    return original+loss.robustifier(mismatch).sum()*loss.data_weight.square()
                loss.forward=with_depth
                return loss
            fitting.create_loss=make
            return utils,prior,fitting,optimizer

        scope=dict(refine.__dict__,load_with_anchor=anchored_loader)
        solve=types.FunctionType(refine.fit.__code__,scope)
        inputs=dict(kwargs,vposer2=seated.SeatedDecoder(kwargs['vposer2'],plan['seated_prior']))
        replay,rep=solve(saved_seed=seed,replay_only=True,**inputs)
        difference=float(np.max(np.abs(replay['joints_projected_native']-seed['joints_projected_native'])))
        assert difference<.001 and calls==0 and rep['anchor_calls']==0
        arrays,receipt=solve(saved_seed=seed,replay_only=False,**inputs)
        assert (calls>0)==use_depth and receipt['original_stage_indices']==[3,4]
        z=(arrays['vertices_camera'][0,ids.numpy(),2]*bary.numpy()).sum(1)
        assert min(z)>0
        achieved=float(np.log(z[0]/z[1]))
        receipt.update(depth_term_enabled=use_depth,depth_calls=calls,seed_replay_max_px=difference,
            depth_log_ratio=achieved,depth_abs_log_error=abs(achieved-target),
            depth_error_equivalent_px=abs(achieved-target)*pixel_scale)
        return arrays,receipt

    adapter=types.SimpleNamespace(**dict(entry.__dict__,fit_one=fit))
    worker=types.SimpleNamespace(**dict(BASE.original.__dict__,entry=adapter))
    scope=dict(BASE.__dict__,original=worker,__file__=__file__)
    types.FunctionType(BASE.fit_shard.__code__,scope)(folder,index)


def main():
    assert os.name=='nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    material=read(MATERIAL/'result.json')
    assert material['plan_sha256']==sha(MATERIAL/'plan.json')
    assert material['review_image_sha256']==sha(MATERIAL/'source-material.jpg')
    assert read(MATERIAL/'plan.json')['camera_sha256']==sha(CAMERA)
    source,plan=read(seated.OUT/'result.json'),read(seated.OUT/'plan.json')
    assert source['plan_sha256']==sha(seated.OUT/'plan.json') and source['fitting_complete']
    pairs=prepare_pairs(plan,source,material['rows'])
    frames=[f for f in plan['source_frames'] if f in pairs and pairs[f]['admitted']]
    assert {'frame_000029','frame_000053'}<=set(frames)
    helper=Path(__file__).with_name('smplify_seeded_refine124.py')
    paths=[Path(__file__),Path(seated.__file__),Path(entry.__file__),helper,BASE.ARCHIVE,BASE.CONTACTS,
           seated.OUT/'result.json',seated.OUT/'plan.json',MATERIAL/'plan.json',MATERIAL/'result.json',
           MATERIAL/'source-material.jpg',CAMERA]
    registered=dict(plan,updated=datetime.now(timezone.utc).isoformat(),runner_sha256=sha(__file__),
        pairs=pairs,frames=frames,branches=list(BRANCHES),source_hashes={str(p):sha(p) for p in paths},
        material_actual_visual_review=True,reviewed_labels=REVIEWED,
        hypothesis='A scale-independent left-to-right surface-depth ratio constrains limb depth beyond native detector points.',
        single_change='One robust pixel-equivalent log-depth ratio residual in the existing final stages; use the original stage data weight.',
        weight='Min focal length divided by the root-sum of squared inverse-sine source parallax. No sweep.',
        association='First reviewed eligible corner of each foot with an exact visible seed-foot hit. Use source triangulation FIT views only in this diagnostic.',
        protections='Same seed, full K, shared shape, all native observations, original stages/optimizer/rotation anchor and free body placement; four original holdouts excluded.',
        quality_accepted=False,gaussian_started=False)
    atomic(OUT/'plan.json',registered)
    atomic(OUT/'launch.json',dict(pid=os.getpid(),command=[sys.executable,*sys.argv],frames=frames))
    scope=dict(BASE.__dict__,fit_shard=fit_shard)
    stage=types.FunctionType(BASE.stage.__code__,scope)
    indices=[plan['source_frames'].index(f) for f in frames]
    started=time.monotonic()
    try:
        for branch in BRANCHES:
            stage(OUT.name+'/'+branch,registered,indices,dict(vector=source['fixed_shape']))
        assert all(sha(path)==value for path,value in registered['source_hashes'].items())
        atomic(OUT/'result.json',dict(completed=True,frames=frames,branches=list(BRANCHES),
            plan_sha256=sha(OUT/'plan.json'),seconds=time.monotonic()-started,
            branch_result_sha256={b:sha(OUT/b/'result.json') for b in BRANCHES},gaussian_started=False))
    except BaseException as exc:
        atomic(OUT/'failure.json',dict(error=repr(exc)))
        raise


if __name__=='__main__':
    main()
