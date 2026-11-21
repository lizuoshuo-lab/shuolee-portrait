"""Read-only pose-source counterfactual on the immutable SH0 18k snapshot.

Reuse the original mesh binding and native Gaussian renderer. No optimizer,
Gaussian parameter update, new fitting, frame selection, or model promotion.
"""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image
from run_official_gaussians123 import PROJECT, ASSETS, read, save, sha, now
from train_demo_body_upstream123 import DEMO, UP, module
from train_demo_body_root_sh123 import process

OLD = DEMO/'body-gaussian-smplify-eight-dc18000'
SNAP = OLD/'gaussian-run/step-018000'
FIT = DEMO/'fullbody-smplifyx/face-priority-guarded-sequence'
OUT = OLD/'pose-source-replay'
HERE = Path(__file__).resolve().parent


def checked():
    plan = read(OUT/'plan.json')
    for path, digest in plan['files'].items():
        assert sha(path) == digest, path
    return plan


def prepare():
    assert not OUT.exists(), 'Preserve the previous read-only result.'
    OUT.mkdir()
    paths = [Path(__file__).resolve(), OLD/'fit/fitted.npz', OLD/'fit/surface-cache.npz',
             OLD/'selection.json', OLD/'data/training-inputs.json', SNAP/'bound-local.ply',
             SNAP/'binding.npz', SNAP/'result.json', SNAP/'current-meshes.npz',
             FIT/'fit/fitted.npz', FIT/'closeout.json', FIT/'selection-audit.json',
             HERE/'frozen_skeleton_surface123.py', ASSETS/'smplx/SMPLX_NEUTRAL.npz',
             UP/'scene/gaussian_model.py', PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',
             PROJECT/'src/shuolee_portrait/training.py', PROJECT/'src/avatar_core/gaussian_renderer/__init__.py']
    plan = dict(updated=now(), state='registered_before_replay',
        hypothesis='The direct Gaussian route still consumes the earlier independent SMPLify-X fit. Test the already reviewed whole-sequence fit as a pose-only rendering counterfactual before any new training.',
        changed_variable='Source of fitted SMPL-X pose/expression and its derived per-frame mesh; full fitted candidate, not handpicked joints or frames.',
        unchanged=['all59 real images/masks', 'K/w2c', 'shared shape', 'reference33',
                   'same immutable SH0 local Gaussian PLY', 'same clothing delta', 'same triangle binding',
                   'same native renderer', 'same eight training frame identities'],
        optimization_updates=0, gaussian_training_started=False,
        frames=59, gpu_timeout_seconds=180, CPU_timeout_seconds=180,
        stop=['input/hash/identity mismatch', 'reference mesh changes', 'forward parity >2e-6m',
              'non-finite', 'native old-snapshot mean RGB difference >2e-5',
              'new relative mesh fold or protected-face displacement from clothing'],
        warning='SH0 cannot change colour with direction. Better pose replay is neither a new trained model nor proof of clothing geometry, independent facial truth, or15min end-to-end completion.',
        quality_accepted=False, model_promoted=False,
        files={str(p):sha(p) for p in paths})
    save(OUT/'plan.json', plan)
    with np.load(OLD/'fit/fitted.npz', allow_pickle=False) as z: old = dict(z)
    with np.load(FIT/'fit/fitted.npz', allow_pickle=False) as z: new = dict(z)
    ids = list(old['frame_ids'].astype(str)); new_ids = list(new['frame_ids'].astype(str))
    assert len(ids)==len(set(ids))==len(new_ids)==len(set(new_ids))==59
    assert set(ids)==set(new_ids) and ids[0]=='frame_000033'
    order = [new_ids.index(f) for f in ids]
    new = {k:(v[order] if v.ndim and v.shape[0]==59 else v) for k,v in new.items()}
    unchanged = ['frame_ids','reference_frame_id','shared_betas','betas','source_frames',
                 'image_sha256','mask_sha256','K_integer_centers','w2c','source_w2c',
                 'observed_native_keypoints','faces','world_scale','camera_translation',
                 'camera_length_contract_json','source_camera_length_contract_json','root_translation_semantics']
    for key in unchanged: assert np.array_equal(old[key],new[key]), key
    pose_keys = ['global_orient','body_pose_axis_angle','expression','jaw_pose',
                 'left_hand_pose','right_hand_pose','leye_pose','reye_pose']
    changed = np.zeros(59,dtype=bool)
    for key in pose_keys: changed |= np.any(old[key]!=new[key],axis=1)
    assert not changed[0]
    assert np.array_equal(old['vertices_world'][0],new['vertices_world'][0])
    closeout=read(FIT/'closeout.json')
    assert closeout['regional_protections_passed'] and closeout['controlled_change_passed']
    assert not closeout['quality_accepted']
    assert int(changed.sum())==closeout['selected_new']==12
    rows=read(OLD/'data/training-inputs.json')['records']
    assert [r['frame'] for r in rows]==ids
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['source_sha256']==str(old['image_sha256'][i])
        assert sha(r['mask'])==r['mask_sha256']==str(old['mask_sha256'][i])
        assert np.array_equal(r['w2c'],np.eye(4))
    new['static_offset']=np.zeros((1,10475,3),np.float32)
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    torch.set_num_threads(4); assert not torch.cuda.is_initialized()
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z: binding=dict(z)
    with np.load(OLD/'fit/surface-cache.npz',allow_pickle=False) as z: cache=dict(z)
    delta=torch.as_tensor(binding['static_offset'],dtype=torch.float32)
    clothing=torch.as_tensor(cache['clothing_vertices']); protected=torch.as_tensor(cache['protected_face_vertices'])
    assert not torch.any(delta[~clothing]) and not torch.any(delta[protected])
    started=time.monotonic()
    bridge=FrozenSkeletonSurface(new,new['static_offset'][0],ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    replay_error=float((bridge.base_vertices-torch.as_tensor(new['vertices_world'])).abs().max())
    assert replay_error<2e-6, replay_error
    current_old=torch.as_tensor(cache['base_vertices'])+(torch.as_tensor(cache['linear'])@delta[None,:,:,None]).squeeze(-1)
    current_new=bridge.base_vertices+(bridge.linear@delta[None,:,:,None]).squeeze(-1)
    assert torch.equal(current_new[0],current_old[0])
    stable_error=float((current_new[~changed]-current_old[~changed]).abs().max())
    assert stable_error<2e-6,stable_error
    oracle=[]
    with torch.no_grad():
        for f in ('frame_000033','frame_000018','frame_000058'):
            i=ids.index(f);v,j=bridge.independent_original_forward(delta,i)
            e=float((current_new[i]-v).abs().max());je=float((bridge.joints(i)-j).abs().max())
            assert e<2e-6 and je<2e-6
            oracle.append(dict(frame=f,vertex_error_m=e,joint_error_m=je))
    assert not torch.any(current_new[:,protected]-bridge.base_vertices[:,protected])
    faces=torch.as_tensor(old['faces'],dtype=torch.long)
    tri0=bridge.base_vertices[:,faces];tri1=current_new[:,faces]
    n0=torch.cross(tri0[:,:,1]-tri0[:,:,0],tri0[:,:,2]-tri0[:,:,0],dim=-1)
    n1=torch.cross(tri1[:,:,1]-tri1[:,:,0],tri1[:,:,2]-tri1[:,:,0],dim=-1)
    valid=n0.norm(dim=-1)>1e-12
    folds=int(((n0*n1).sum(-1)[valid]<=0).sum())
    assert folds==0,folds
    np.savez_compressed(OUT/'pose-replay.npz', frame_ids=old['frame_ids'],faces=old['faces'],
        old_vertices=current_old.numpy(),new_vertices=current_new.numpy(),changed=changed)
    selected=read(OLD/'selection.json')['selected_frames']
    audit=dict(updated=now(),state='prepared_read_only',frame_alignment='explicit frame-ID permutation; candidate is chronological, current training archive starts with33',
        changed_frames=[f for f,c in zip(ids,changed) if c],unchanged_frames=[f for f,c in zip(ids,changed) if not c],
        selected_training_frames=selected,changed_training_frames=[f for f,c in zip(ids,changed) if c and f in selected],
        immutable_fields_equal=unchanged,reference_identical=True,clothing_delta_unchanged=True,
        previous_face_and_body_protections=True,source_geometry_quality_accepted=False,
        replay_max_abs_m=replay_error,unchanged_pose_mesh_max_abs_m=stable_error,forward_oracles=oracle,
        new_relative_normal_reversals=folds,protected_face_clothing_motion_m=0,
        seconds=time.monotonic()-started,optimization_updates=0,quality_accepted=False)
    save(OUT/'input-audit.json',audit)
    plan['files'][str(OUT/'pose-replay.npz')]=sha(OUT/'pose-replay.npz')
    plan['state']='prepared';save(OUT/'plan.json',plan)
    print('POSE_SOURCE_READ_ONLY_PREPARED',audit['changed_frames'],flush=True)


def replay(pid):
    checked();assert process(pid)[1]=='T' and not (OUT/'result.json').exists()
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    rt.configure(ASSETS)
    from shuolee_portrait.training import camera,image
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    official=module(UP/'scene/gaussian_model.py','pose_replay_original_gaussian123')
    with np.load(OUT/'pose-replay.npz',allow_pickle=False) as z: data=dict(z)
    with np.load(SNAP/'binding.npz',allow_pickle=False) as z:binding=dict(z)
    rows=read(OLD/'data/training-inputs.json')['records']
    class Body(official.GaussianModel):
        select_mesh_by_timestep=SMPLXGaussianModel.select_mesh_by_timestep
        get_covariance=SMPLXGaussianModel.get_covariance
    model=Body(0);model.static_offset=None
    model.faces=torch.as_tensor(data['faces'],device='cuda',dtype=torch.long)
    model.num_timesteps=59;model.load_ply(str(SNAP/'bound-local.ply'))
    assert np.array_equal(model.binding.cpu().numpy(),binding['triangle_ids'])
    tensors=[model._xyz,model._features_dc,model._features_rest,model._opacity,model._scaling,model._rotation,model.binding]
    before=[t.detach().clone() for t in tensors]
    oldverts=torch.as_tensor(data['old_vertices'],device='cuda');newverts=torch.as_tensor(data['new_vertices'],device='cuda')
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    bg=torch.zeros(3,device='cuda');records=[];started=time.monotonic()
    with torch.no_grad():
        for i,row in enumerate(rows):
            cam,_=camera(row,i,pose_index=i);rgb=[]
            for label,verts in (('old',oldverts),('new',newverts)):
                model.vertices_sequence=verts;model.select_mesh_by_timestep(i)
                val=render(cam,model,pipe,bg)['render'];assert torch.isfinite(val).all()
                pixels=image(val);pixels.save(OUT/f"{row['frame']}_{label}.png");rgb.append(val.detach().clone())
            png=np.asarray(image(rgb[0])).astype(np.int16)
            previous=np.asarray(Image.open(SNAP/(row['frame']+'_gaussian.png')).convert('RGB')).astype(np.int16)
            parity=float(np.abs(png-previous).mean()/255)
            assert parity<2e-5,(row['frame'],parity)
            diff=float((rgb[0]-rgb[1]).abs().mean())
            if not data['changed'][i]:assert diff<2e-6,(row['frame'],diff)
            records.append(dict(frame=row['frame'],changed=bool(data['changed'][i]),
                old_snapshot_mean_abs=parity,old_new_render_mean_abs=diff))
            save(OUT/'progress.json',dict(updated=now(),state='read_only_rendering',pid=os.getpid(),frames=i+1,optimization_updates=0))
    assert all(torch.equal(a,b) for a,b in zip(tensors,before))
    checked();assert process(pid)[1]=='T'
    save(OUT/'result.json',dict(updated=now(),state='rendered_pending_review',frames=59,records=records,
        points=len(model._xyz),active_sh_degree=0,gaussian_parameters_bitwise_unchanged=True,
        optimizer_created=False,optimizer_updates=0,seconds=time.monotonic()-started,
        quality_accepted=False,model_promoted=False,actual_visual_review=False))
    print('POSE_SOURCE_COUNTERFACTUAL_RENDERED_NO_OPTIMIZER',flush=True)


def run(pid):
    plan=checked();assert not (OUT/'result.json').exists()
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert pid==331 and identity=='9229' and 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    others=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (pid,os.getpid()):continue
        try:
            c,s,_=process(int(p.name))
            executable=(p/'exe').resolve().name
            if executable.startswith('python') and any(k in c for k in (' train','replay_smplify','demo_static_upsample','compare_static')) and s!='T':others.append((p.name,c))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not others,others
    receipt=dict(started=now(),pid=pid,identity=identity,command=cmd,state='suspending',read_only=True)
    save(OUT/'gpu-handoff.json',receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        subprocess.run([sys.executable,'-B',__file__,'replay','--pid',str(pid)],check=True,timeout=plan['gpu_timeout_seconds'])
    finally:
        if process(pid)[2]==identity:os.kill(pid,signal.SIGCONT)
        receipt.update(finished=now(),state='resumed',trainer_state=process(pid)[1]);save(OUT/'gpu-handoff.json',receipt)
        print('BASELINE_RESUMED',pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','replay']);p.add_argument('--pid',type=int);a=p.parse_args()
    try:
        if a.mode=='prepare':prepare()
        else:globals()[a.mode](a.pid)
    except BaseException as e:
        if OUT.exists():save(OUT/(a.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
        raise
