"""Zero-update interface preflight for the original VHAP direct-body option.

Also test the minimal proposed Gaussian initial-offset loader in memory,
against all59 existing nonzero-offset meshes. Do not edit production sources.
"""
from pathlib import Path
import dataclasses
import difflib
import fcntl
import inspect
import json
import os
import textwrap
import time
import traceback
import types

import numpy as np
import torch
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
from run_official_gaussians123 import read,save,sha,now,PROJECT,ASSETS

ROOT=base.mask.joint.ROOT
OUT=ROOT/'smplify-vhap-direct-body-preflight'
PLAN=ROOT/'smplify-vhap-direct-body-plan.json'
CONSUMER=PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py'


def alternative_forward(tracker):
    original=tracker.forward_smplx.__func__
    source=textwrap.dedent(inspect.getsource(original))
    old="body_pose=self.vposer.decode(self.pose_embedding[timesteps])['pose_body'].reshape(len(timesteps),-1)"
    new="body_pose=(self.vposer.decode(self.pose_embedding[timesteps])['pose_body'].reshape(len(timesteps),-1) if self.cfg.model.use_vposer else self.body_pose[timesteps])"
    assert source.count(old)==1
    updated=source.replace(old,new,1)
    namespace={**original.__globals__,**inspect.getclosurevars(original).nonlocals}
    exec(compile(updated,'<proposed-original-direct-body-forward>','exec'),namespace)
    (OUT/'proposed-forward.py').write_text(updated)
    (OUT/'forward.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile='existing-adapter-forward',tofile='proposed-adapter-forward')))
    return types.MethodType(namespace['forward_smplx'],tracker)


def alternative_consumer(cls):
    original=cls.enable_shared_static_offset
    source=textwrap.dedent(inspect.getsource(original))
    old='self.static_offset = nn.Parameter(torch.zeros_like(self.offset_body.v_template))'
    new='''initial_offset = saved.get('static_offset')
    if initial_offset is None:
        initial_offset = torch.zeros_like(self.offset_body.v_template)
    else:
        initial_offset = torch.as_tensor(initial_offset, dtype=torch.float32, device='cuda')
        if initial_offset.shape == (1, *self.offset_body.v_template.shape):
            initial_offset = initial_offset[0]
        if initial_offset.shape != self.offset_body.v_template.shape or not torch.isfinite(initial_offset).all():
            raise ValueError('Invalid fitted canonical static_offset shape or values.')
    self.static_offset = nn.Parameter(initial_offset.detach().clone())'''
    assert source.count(old)==1
    updated=source.replace(old,new,1)
    namespace=dict(original.__globals__)
    exec(compile(updated,'<proposed-vhap-offset-consumer>','exec'),namespace)
    (OUT/'proposed-consumer-method.py').write_text(updated)
    (OUT/'consumer.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),updated.splitlines(True),fromfile='existing-enable_shared_static_offset',tofile='proposed-enable_shared_static_offset')))
    return namespace['enable_shared_static_offset']


def fixture(saved,poses,path):
    ids=list(saved['frame_ids'])
    arrays=dict(frame_ids=np.asarray(ids),faces=saved['faces'],shared_betas=saved['betas'][None],
        vertices_world=saved['vertices_camera'],camera_translation=saved['transl'],
        body_pose_axis_angle=np.asarray(poses,np.float32),w2c=saved['w2c'],static_offset=saved['static_offset'])
    for k in ('global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose'):
        arrays[k]=saved[k]
    np.savez_compressed(path,**arrays)
    return arrays


def main():
    plan=read(PLAN)
    assert plan['preflight_authorized'] and not plan['execution_authorized']
    assert plan['source_sha256']==sha(__file__) and not OUT.exists()
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic();torch.set_num_threads(4)
    torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    tracker,rows=base.make_tracker(OUT)
    receipts=bridge.initialize(tracker,rows)
    assert [r['frame'] for r in rows]==plan['roster']
    original_params={k:bridge.digest_tensor(getattr(tracker,k)) for k in bridge.PARAMS}
    poses=[]
    for r in rows:
        with np.load(bridge.SOURCE/'frames'/r['frame']/'winner-camera-space.npz',allow_pickle=False) as z:
            poses.append(z['body_pose_axis_angle'][0].copy())
    pose=np.asarray(poses,np.float32)
    with torch.no_grad():
        latent_decoded=torch.cat([tracker.vposer.decode(tracker.pose_embedding[i:i+1])['pose_body'].reshape(1,63) for i in range(59)])
    latent_parity=float(np.max(np.abs(latent_decoded.cpu().numpy()-pose)))
    assert latent_parity<5e-5
    tracker.body_pose=torch.tensor(pose,device='cuda',requires_grad=True)
    tracker.pose_embedding.requires_grad_(False)
    tracker.cfg.model.use_vposer=False
    tracker.forward_smplx=alternative_forward(tracker)
    groups=tracker.get_train_parameters('rgb_global_tracking')
    assert any(p is tracker.body_pose for p in groups['body'])
    assert all(p is not tracker.pose_embedding for v in groups.values() for p in v)
    direct_params=tuple('body_pose' if k=='pose_embedding' else k for k in bridge.PARAMS)
    active={id(p) for g in groups.values() for p in g}
    for k in direct_params:getattr(tracker,k).requires_grad_(id(getattr(tracker,k)) in active)
    # After this point no VPoser call is permitted, including in the energy.
    def forbidden(*args,**kwargs):raise AssertionError('Direct-body branch unexpectedly decoded VPoser.')
    tracker.vposer.decode=forbidden
    records=[];initial_vertices=[]
    for i,row in enumerate(rows):
        assert time.monotonic()-start<plan['max_seconds']
        tracker.clear_cache();sample=tracker.dataset.one(i)
        with np.load(bridge.SOURCE/'frames'/row['frame']/'winner-camera-space.npz',allow_pickle=False) as z:
            expected_vertices=z['vertices_camera'].copy();expected_uv=z['joints_projected_native'].copy()
        with torch.no_grad():
            vertices,_,joints,_,_=tracker.forward_smplx(sample['timestep_index'])
            K=torch.tensor(row['K_integer_centers'],device='cuda',dtype=torch.float32)
            q=joints@K.T;uv=q[...,:2]/q[...,2:]
            vertex_error=float(np.abs(vertices.cpu().numpy()-expected_vertices).max())
            pixel_error=float(np.abs(uv.cpu().numpy()-expected_uv).max())
            assert vertex_error<2e-6 and pixel_error<.002
            initial_vertices.append(vertices[0].cpu().numpy())
            original_body=tracker.body_pose[i].clone()
            tracker.body_pose[i,57]+=.001
            _,_,perturbed,_,_=tracker.forward_smplx(sample['timestep_index'])
            p=perturbed@K.T;p=p[...,:2]/p[...,2:]
            elbow_motion=float((p[0,6]-uv[0,6]).abs().max())
            hand_motion=float((p[0,26:46]-uv[0,26:46]).norm(dim=-1).max())
            tracker.body_pose[i].copy_(original_body)
            assert elbow_motion<.0001 and hand_motion>0
        records.append(dict(frame=row['frame'],vertices_max_abs_m=vertex_error,projection_max_abs_px=pixel_error,
                            wrist_1mrad_same_side_elbow_motion_px=elbow_motion,wrist_1mrad_hand_max_motion_px=hand_motion))
    assert np.array_equal(tracker.body_pose.detach().cpu().numpy(),pose)
    sample=tracker.dataset.one(17);tracker.clear_cache()
    loss,logs,*_=tracker.compute_energy(sample,stage='rgb_global_tracking')
    body_gradient,hand_gradient=torch.autograd.grad(loss,(tracker.body_pose,tracker.left_hand_pose),retain_graph=True)
    assert torch.isfinite(body_gradient).all() and float(body_gradient.abs().max())>0
    assert torch.isfinite(hand_gradient).all() and float(hand_gradient.abs().max())>0
    energy=dict(losses={k:float(v.detach()) for k,v in logs.items()},body_gradient_max=float(body_gradient.abs().max()),
                wrist_gradient_max=float(body_gradient[17,57:63].abs().max()),hand_gradient_max=float(hand_gradient.abs().max()),
                original_direct_pose_prior_weight=tracker.cfg.w.reg_body,original_latent_prior_weight=tracker.cfg.w.reg_vposer,
                original_temporal_body_weight=tracker.cfg.w.smooth_body,optimizer_updates=0)
    # The research shared-offset replay path is exercised, without any Gaussian
    # optimization or generated PLY. Frozen-vertex deployment is a separate path.
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    checkpoint=ROOT/'smplify-vhap-normalized-laplacian/epoch-30/tracked-smplx-params.npz'
    with np.load(checkpoint,allow_pickle=False) as z:saved=dict(z)
    decoded=read(ROOT/'smplify-vhap-sequential-stage/limb-transfer-audit/result.json')['runs']['normalized_control']['records']
    assert list(saved['frame_ids'])==[r['frame'] for r in decoded]==plan['roster']
    fitted_fixture=OUT/'fitted-nonzero-offset-fixture.npz'
    fixture(saved,[r['body_axis_angle'] for r in decoded],fitted_fixture)
    model=SMPLXGaussianModel(3);model.load_fresh_meshes(fitted_fixture)
    failure=None
    try:model.enable_shared_static_offset(fitted_fixture,ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    except RuntimeError as error:failure=str(error)
    assert failure and 'Zero shared offset changed official geometry' in failure
    proposed=alternative_consumer(SMPLXGaussianModel)
    model.enable_shared_static_offset=types.MethodType(proposed,model)
    replay=model.enable_shared_static_offset(fitted_fixture,ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    fitted_offset=torch.tensor(saved['static_offset'][0],device='cuda')
    assert torch.equal(model.static_offset,fitted_offset)
    offset_before=bridge.digest_tensor(model.static_offset)
    replay_errors=[]
    with torch.no_grad():
        for i in range(59):
            replay_errors.append(float((model.shared_offset_vertices(i)-model.vertices_sequence[i]).abs().max()))
    assert max(replay_errors)<2e-6 and bridge.digest_tensor(model.static_offset)==offset_before
    initial_saved={k:getattr(tracker,k).detach().cpu().numpy() for k in
        ('betas','transl','global_orient','expression','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','static_offset')}
    initial_saved.update(frame_ids=np.asarray(plan['roster']),faces=tracker.smplx.faces,
                         vertices_camera=np.asarray(initial_vertices),w2c=np.asarray([r['w2c'] for r in rows]))
    initial_fixture=OUT/'initial-zero-offset-fixture.npz'
    arrays=fixture(initial_saved,pose,OUT/'initial-with-zero-offset-fixture.npz')
    arrays.pop('static_offset')
    np.savez_compressed(initial_fixture,**arrays)
    legacy=SMPLXGaussianModel(3);legacy.load_fresh_meshes(initial_fixture)
    legacy.enable_shared_static_offset=types.MethodType(proposed,legacy)
    backward=legacy.enable_shared_static_offset(initial_fixture,ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    assert float(legacy.static_offset.abs().max())==0 and backward['checked_frames']==59
    assert all(bridge.digest_tensor(getattr(tracker,k))==h for k,h in original_params.items())
    assert np.array_equal(tracker.body_pose.detach().cpu().numpy(),pose)
    for p,h in plan['input_hashes'].items():assert sha(p)==h,p
    result=dict(updated=now(),state='passed_zero_update_preflight',frames=59,optimizer_updates=0,original_source_unchanged=True,
        seconds=time.monotonic()-start,source_sha256=sha(__file__),latent_to_saved_axis_angle_max_abs_rad=latent_parity,
        records=records,energy=energy,hand_pca_dimensions=12,original_optimizer_created=False,
        original_forward_change='Only select the original direct body_pose branch instead of unconditional VPoser decoding.',
        gaussian_nonzero_offset=dict(original_loader_failure=failure,proposed_loader_replay=replay,
            replay_max_abs_m=max(replay_errors),fitted_offset_preserved=True,fixture_sha256=sha(fitted_fixture),
            old_missing_offset_fixture_backward_compatible=True,old_missing_offset_replay=backward,
            contract='Fitted canonical static_offset initializes the one trainable absolute shared offset before LBS, exactly once.'),
        production_source_changed=False,quality_accepted=False,new_training_started=False,
        limits=['This is interface equivalence and local wrist independence, not a trained quality result.',
            'Disabling VPoser also switches to original direct rotation prior and axis-angle temporal metric, with quality risks.',
            'The nonzero-offset check replays an existing unaccepted mesh; it does not accept that mesh or export a new model.'])
    save(OUT/'result.json',result);save(OUT/'config.json',json.loads(json.dumps(dataclasses.asdict(tracker.cfg),default=str)))
    save(OUT/'progress.json',dict(updated=now(),state='passed_zero_update_preflight',pid=None,frames=59,optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','energy')}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
