"""SMPLify-X saved full-sequence initialization -> upstream VHAP surface stage.

Only an input bridge and explicit upstream parameter-group configuration.
No network body seed, custom optimizer, per-frame selection, or changed loss.
"""
from pathlib import Path
import dataclasses
import fcntl
import json
import os
import subprocess
import time
import traceback
import types

import numpy as np
from vhap_demo_adapter123 import ROOT, REPO, COMMIT, DATA, make_tracker
from continue_vhap_demo123 import PARAMS, PER_FRAME
from run_official_gaussians123 import read, save, sha, now

SOURCE = ROOT/'fullbody-smplifyx/full-sequence'
OUT = ROOT/'smplify-vhap-surface'
PLAN = ROOT/'smplify-vhap-surface-plan.json'
STAGE = 'rgb_global_tracking'


def digest_tensor(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def initialize(tracker, rows):
    """Join by frame identity; the fit source starts at 33, VHAP is chronological."""
    import torch
    fit_rows = {r['frame']: r for r in read(SOURCE/'data/training-inputs.json')['records']}
    receipts = []
    shared_beta = None
    assert len(rows) == len(fit_rows) == 59
    for slot, row in enumerate(rows):
        path = SOURCE/'frames'/row['frame']/'winner-camera-space.npz'
        with np.load(path, allow_pickle=False) as archive:
            saved = dict(archive)
        assert str(saved['frame_id']) == row['frame']
        assert np.array_equal(saved['faces'], tracker.smplx.faces)
        assert np.array_equal(saved['K_integer_centers'], row['K_integer_centers'])
        assert np.array_equal(saved['w2c'], np.eye(4))
        assert np.array_equal(row['w2c'], np.eye(4))
        assert str(saved['image_sha256']) == row['source_sha256'] == sha(row['image'])
        assert str(saved['mask_sha256']) == row['mask_sha256'] == sha(row['mask'])
        assert fit_rows[row['frame']]['source_sha256'] == row['source_sha256']
        if shared_beta is None:
            shared_beta = saved['betas'][0].copy()
        assert np.array_equal(saved['betas'][0], shared_beta)
        with torch.no_grad():
            tracker.betas.copy_(torch.tensor(shared_beta, device='cuda'))
            for key in PER_FRAME:
                source_key = 'camera_translation' if key == 'transl' else key
                getattr(tracker, key)[slot].copy_(torch.tensor(saved[source_key][0], device='cuda'))
            v, _, joints, _, _ = tracker.forward_smplx(torch.tensor([slot]))
            K = torch.tensor(saved['K_integer_centers'], device='cuda', dtype=v.dtype)
            q = joints @ K.T
            uv = q[..., :2] / q[..., 2:]
            vertex_error = float((v-torch.tensor(saved['vertices_camera'], device='cuda')).abs().max())
            joint_error = float((joints-torch.tensor(saved['joints_camera'], device='cuda')).abs().max())
            pixel_error = float((uv-torch.tensor(saved['joints_projected_native'], device='cuda')).abs().max())
            sample = tracker.dataset.one(slot)
            ndc = tracker.render.world_to_ndc(joints, sample['extrinsic'].cuda(), sample['intrinsic'].cuda(), tracker.image_size, flip_y=True)
            gpu_uv = (ndc[..., :2]+1)*torch.tensor(row['wh'], device='cuda')/2-.5
            renderer_error = float((gpu_uv-uv).abs().max())
        assert vertex_error < 2e-6 and joint_error < 2e-6
        assert pixel_error < .002 and renderer_error < .002
        receipts.append(dict(frame=row['frame'],source_sha256=sha(path),
            vertex_error_m=vertex_error,joint_error_m=joint_error,
            saved_projection_error_px=pixel_error,renderer_projection_error_px=renderer_error))
    assert float(tracker.static_offset.abs().max()) == 0
    return receipts


def checkpoint(tracker, rows, epoch, frozen):
    import torch
    label = 'initial' if epoch == 0 else f'epoch-{epoch:02d}'
    folder = OUT/label
    folder.mkdir(exist_ok=False)
    for key, digest in frozen.items():
        assert digest_tensor(getattr(tracker,key)) == digest, 'Frozen parameter changed: '+key
    arrays = {key:getattr(tracker,key).detach().cpu().numpy() for key in PARAMS}
    arrays.update(frame_ids=np.array([r['frame'] for r in rows]),faces=tracker.smplx.faces,
        K_integer_centers=np.array([r['K_integer_centers'] for r in rows]),w2c=np.array([r['w2c'] for r in rows]))
    records=[]; vertices=[]; projections=[]
    for slot,row in enumerate(rows):
        sample=tracker.dataset.one(slot)
        tracker.clear_cache()
        with torch.no_grad():
            v,_,j,_,_=tracker.forward_smplx(sample['timestep_index'])
            K=torch.tensor(row['K_integer_centers'],device='cuda',dtype=v.dtype)
            q=j@K.T; uv=(q[...,:2]/q[...,2:])[0].cpu().numpy()
            rast=tracker.rasterize_smplx(sample,v,tracker.smplx.faces_tensor,train_mode=False)
            visible=(rast['rast_out'][0,...,-1].flip(0)>0).cpu().numpy()
            mask=sample['alpha_map'][0,0].numpy()>.5
            target=np.zeros((118,3),np.float32)
            dst=np.r_[tracker.dataset.COCO,np.arange(19,25),np.arange(25,67),np.arange(67,118)]
            src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
            target[dst,:2]=tracker.dataset.obs['keypoints'][slot,src]
            target[dst,2]=tracker.input_confidence[slot,src]
            err=np.linalg.norm(uv-target[:,:2],axis=-1)
            groups={}
            for name,ids in dict(body=[i for i in tracker.dataset.COCO if i not in (9,12)],
                face=range(67,118),left_hand=range(25,46),right_hand=range(46,67),feet=range(19,25)).items():
                ids=np.array(list(ids));e=err[ids[target[ids,2]>0]]
                groups[name]=dict(count=len(e),median=float(np.median(e)),max=float(e.max()))
            records.append(dict(frame=row['frame'],groups=groups,elbow_px=float(err[6]),
                mask_iou=float((visible&mask).sum()/max(1,(visible|mask).sum())),
                target_uncovered=float((mask&~visible).sum()/max(1,mask.sum())),
                mesh_outside=float((visible&~mask).sum()/max(1,visible.sum()))))
            vertices.append(v[0].cpu().numpy());projections.append(uv)
    arrays['vertices_camera']=np.array(vertices);arrays['joints_projected_native']=np.array(projections)
    np.savez_compressed(folder/'tracked-smplx-params.npz',**arrays)
    offsets=np.linalg.norm(arrays['static_offset'][0],axis=-1)*1000
    result=dict(updated=now(),state='saved_pending_visual_review',epoch=epoch,steps=tracker.global_step,
        frames=59,records=records,offset_mm=dict(median=float(np.median(offsets)),p90=float(np.quantile(offsets,.9)),max=float(offsets.max())),
        frozen_parameters_unchanged=True,fit_detector_residuals_not_independent_truth=True,quality_accepted=False)
    save(folder/'result.json',result)
    save(OUT/'latest-checkpoint.json',dict(updated=now(),folder=str(folder),epoch=epoch,steps=tracker.global_step))
    print(json.dumps(dict(checkpoint=label,step=tracker.global_step,offset_mm=result['offset_mm'],
        mask_iou_median=float(np.median([r['mask_iou'] for r in records])))),flush=True)
    return result


def main():
    import torch
    from torch.utils.data import DataLoader
    plan=read(PLAN)
    assert plan['execution_authorized'] and plan['source_sha256']==sha(__file__)
    assert not OUT.exists(), 'Preserve prior run; do not launch twice.'
    assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==COMMIT
    assert subprocess.check_output(['git','-C',str(REPO),'-c','core.autocrlf=true','-c','core.filemode=false','status','--porcelain'],text=True)==''
    for path,digest in plan['input_hashes'].items():
        assert sha(path)==digest,path
    for path,digest in plan['source_hashes'].items():
        assert sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir();start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32=False
    save(OUT/'progress.json',dict(updated=now(),state='checking_smplify_vhap_bridge',pid=os.getpid(),steps=0,quality_accepted=False))
    try:
        tracker,rows=make_tracker(OUT)
        tracker.cfg.render.background_train='white'
        # Existing VHAP parameter-group option, not a new sequence solver.
        tracker.cfg.pipeline.rgb_global_tracking.optimizable_params=('texture','lights','static_offset')
        tracker.cfg.pipeline.rgb_global_tracking.num_epochs=30
        receipts=initialize(tracker,rows)
        save(OUT/'bridge-check.json',dict(updated=now(),state='passed',frames=receipts,
            exact_same_smplx_topology=True,translation_applied_once=True,join_by_frame_id=True,
            zero_initial_shared_offset=True,optimizer_updates=0,quality_accepted=False))
        save(OUT/'config.json',json.loads(json.dumps(dataclasses.asdict(tracker.cfg),default=str)))
        frozen={key:digest_tensor(getattr(tracker,key)) for key in ('betas',)+PER_FRAME}
        params=tracker.get_train_parameters(STAGE)
        trainable={id(p) for group in params.values() for p in group}
        for key in PARAMS:
            getattr(tracker,key).requires_grad_(id(getattr(tracker,key)) in trainable)
        tracker.vposer.eval();tracker.vposer.requires_grad_(False)
        assert id(tracker.static_offset) in trainable
        assert all(id(getattr(tracker,key)) not in trainable for key in frozen)
        # Check a finite nonzero surface gradient using the frozen designated reference,
        # before the unchanged optimizer performs any update.
        anchor=next(i for i,r in enumerate(rows) if r['frame']=='frame_000033')
        loss,logs,*_=tracker.compute_energy(tracker.dataset.one(anchor),stage=STAGE)
        gradient,=torch.autograd.grad(loss,tracker.static_offset)
        assert torch.isfinite(gradient).all() and float(gradient.abs().max())>0
        save(OUT/'gradient-check.json',dict(updated=now(),offset_gradient_max=float(gradient.abs().max()),
            initial_terms={k:float(v.detach()) for k,v in logs.items()},optimizer_updates=0))
        del loss,logs,gradient
        checkpoint(tracker,rows,0,frozen)
        def evaluate(self,make_visualization=True,epoch=0,evaluate=True):
            checkpoint(self,rows,epoch,frozen)
            if time.monotonic()-start>1800:
                raise RuntimeError('Bounded 30-minute surface trial exhausted.')
        def log_scalars(self,logs,timestep,**kw):
            elapsed=time.monotonic()-start
            if elapsed>1800:raise RuntimeError('Bounded 30-minute surface trial exhausted.')
            state=dict(updated=now(),state='vhap_surface_fitting',pid=os.getpid(),step=self.global_step+1,
                frame=rows[int(timestep)]['frame'],seconds=elapsed,frames_total=59,
                smplifyx_fitting_used=True,quality_accepted=False,
                losses={k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in logs.items()})
            save(OUT/'progress.json',state);print(json.dumps(state),flush=True)
        tracker.evaluate=types.MethodType(evaluate,tracker)
        tracker.log_scalars=types.MethodType(log_scalars,tracker)
        loader=DataLoader(tracker.dataset,batch_size=1,shuffle=True,num_workers=0,
            generator=torch.Generator().manual_seed(6060))
        tracker.optimize_stage(STAGE,dataloader=loader,lr_scale=.1)
        assert tracker.global_step==1770
        for path,digest in plan['source_hashes'].items():assert sha(path)==digest,path
        for path,digest in plan['input_hashes'].items():assert sha(path)==digest,path
        result=dict(updated=now(),state='complete_pending_visual_review',steps=tracker.global_step,frames=59,
            elapsed_seconds=time.monotonic()-start,smplifyx_fitting_used=True,
            initialization_quality_accepted=False,quality_accepted=False,gaussian_training_started=False,
            original_optimizer_loss_scheduler=True,frozen_parameters_unchanged=True,
            scope='Surface trial; inherited SMPLify-X pose defects are not corrected by claiming completion.')
        save(OUT/'result.json',result);save(OUT/'progress.json',dict(**result,pid=None))
        print(json.dumps(result),flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,quality_accepted=False))
        raise


if __name__=='__main__':main()
