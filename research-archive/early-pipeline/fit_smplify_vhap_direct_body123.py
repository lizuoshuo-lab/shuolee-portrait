"""Prepared, approval-gated original VHAP direct63-body option trial.

Keep official optimizer, stages and handPCA12. Only adapt the representation
selection/serialization. Never execute without this exact trial's approval.
"""
import dataclasses
import fcntl
import json
import os
import subprocess
import time
import traceback
import types
import numpy as np
import torch
from torch.utils.data import DataLoader
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
import preflight_vhap_direct_body123 as interface
from run_official_gaussians123 import read,save,sha,now,ASSETS
from vhap_demo_adapter123 import REPO,COMMIT

ROOT=base.mask.joint.ROOT
OUT=ROOT/'smplify-vhap-direct-body'
PLAN=ROOT/'smplify-vhap-direct-body-trial-plan.json'


def export_fitted_mesh():
    path=OUT/'epoch-30/tracked-smplx-params.npz'
    with np.load(path,allow_pickle=False) as z:saved=dict(z)
    assert saved['body_pose'].shape==(59,63) and saved['left_hand_pose'].shape==(59,12)
    interface.fixture(saved,saved['body_pose'],OUT/'gaussian-fit-export.npz')
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    model=SMPLXGaussianModel(3);model.load_fresh_meshes(OUT/'gaussian-fit-export.npz')
    replay=model.enable_shared_static_offset(OUT/'gaussian-fit-export.npz',ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    assert np.array_equal(model.static_offset.detach().cpu().numpy(),saved['static_offset'][0])
    save(OUT/'gaussian-handoff-check.json',dict(updated=now(),state='passed_parameter_replay',frames=59,
        fit_sha256=sha(OUT/'gaussian-fit-export.npz'),receipt=replay,optimizer_updates=0,
        coordinate_contract='Existing OpenCV camera-relative gauge, identity w2c; not a world/+Y preview PLY.',
        quality_accepted=False,gaussian_training_started=False))


def main():
    plan=read(PLAN)
    assert plan['execution_authorized'] and plan.get('user_approval'), 'Pending explicit user decision on the original direct-body option.'
    assert plan['source_sha256']==sha(__file__) and not OUT.exists()
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True)==''
    for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();bridge.OUT=OUT;interface.OUT=OUT;start=time.monotonic()
    torch.set_num_threads(4);torch.manual_seed(6060);torch.backends.cuda.matmul.allow_tf32=False
    save(OUT/'progress.json',dict(updated=now(),state='checking_direct_body_interface',pid=os.getpid(),steps=0,quality_accepted=False))
    try:
        tracker,rows=base.make_tracker(OUT)
        tracker.cfg.render.background_train='white'
        receipts=bridge.initialize(tracker,rows)  # Original latent import first.
        poses=[]
        for row in rows:
            with np.load(bridge.SOURCE/'frames'/row['frame']/'winner-camera-space.npz',allow_pickle=False) as z:
                poses.append(z['body_pose_axis_angle'][0].copy())
        tracker.body_pose=torch.tensor(np.asarray(poses),device='cuda',requires_grad=True)
        tracker.pose_embedding.requires_grad_(False)
        tracker.cfg.model.use_vposer=False
        tracker.forward_smplx=interface.alternative_forward(tracker)
        bridge.PARAMS=tuple('body_pose' if k=='pose_embedding' else k for k in bridge.PARAMS)
        bridge.PER_FRAME=tuple('body_pose' if k=='pose_embedding' else k for k in bridge.PER_FRAME)
        params=tracker.get_train_parameters('rgb_global_tracking')
        trainable={id(p) for group in params.values() for p in group}
        for key in bridge.PARAMS:getattr(tracker,key).requires_grad_(id(getattr(tracker,key)) in trainable)
        assert id(tracker.body_pose) in trainable and id(tracker.pose_embedding) not in trainable
        assert tracker.left_hand_pose.shape==(59,12) and tracker.right_hand_pose.shape==(59,12)
        tracker.vposer.eval();tracker.vposer.requires_grad_(False)
        save(OUT/'bridge-check.json',dict(updated=now(),state='passed_same_smplify_initialization',frames=receipts,
            body_representation='axis_angle63',hand_representation='PCA12',optimizer_updates=0,quality_accepted=False))
        save(OUT/'config.json',json.loads(json.dumps(dataclasses.asdict(tracker.cfg),default=str)))
        bridge.checkpoint(tracker,rows,0,{})
        def evaluate(self,make_visualization=True,epoch=0,evaluate=True):
            bridge.checkpoint(self,rows,epoch,{})
            if time.monotonic()-start>plan['max_seconds']:raise RuntimeError('Direct-body trial budget reached.')
        def log_scalars(self,logs,timestep,**kw):
            elapsed=time.monotonic()-start
            if elapsed>plan['max_seconds']:raise RuntimeError('Direct-body trial budget reached.')
            entry=dict(updated=now(),state='direct_body_fitting',pid=os.getpid(),step=self.global_step+1,
                frame=rows[int(timestep)]['frame'],stage=kw.get('stage'),seconds=elapsed,frames_total=59,
                losses={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in logs.items()},quality_accepted=False)
            save(OUT/'progress.json',entry);print(json.dumps(entry),flush=True)
        tracker.evaluate=types.MethodType(evaluate,tracker);tracker.log_scalars=types.MethodType(log_scalars,tracker)
        anchor=next(i for i,r in enumerate(rows) if r['frame']=='frame_000033')
        tracker.optimize_stage('rgb_init_offset',sample=tracker.dataset.one(anchor),lr_scale=1.)
        assert tracker.global_step==500
        np.savez_compressed(OUT/'offset-initialization-params.npz',**{k:getattr(tracker,k).detach().cpu().numpy() for k in bridge.PARAMS})
        loader=DataLoader(tracker.dataset,batch_size=1,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(6060))
        tracker.optimize_stage('rgb_global_tracking',dataloader=loader,lr_scale=.1)
        assert tracker.global_step==2270
        export_fitted_mesh()
        for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
        result=dict(updated=now(),state='complete_pending_visual_review',frames=59,steps=2270,
            offset_initialization_updates=500,sequential_refinement_updates=0,global_refinement_updates=1770,
            elapsed_seconds=time.monotonic()-start,body_representation='original_direct_axis_angle63',hand_pca_dimensions=12,
            original_optimizer_and_scheduler=True,original_energy_terms_preserved=True,
            added_loss='Existing photo30*sum(abs(pred_alpha-target_alpha))/sum(target_alpha)',
            option_effect='Original direct rotation prior and63D axis-angle temporal metric replace latent-space counterparts.',
            camera_K_fixed=True,quality_accepted=False,gaussian_training_started=False)
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(**result,pid=None));print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,quality_accepted=False));raise


if __name__=='__main__':main()
