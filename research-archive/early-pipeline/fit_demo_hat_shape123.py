"""Geometry-only control requested by user: mask, landmarks and mesh priors."""
import fcntl,time,traceback,types
import numpy as np
import torch
from torch.utils.data import DataLoader
import fit_demo_flame_hatmask123 as h
from run_official_gaussians123 import read,save,sha,now

BASE=h.OUT
OUT=BASE/'geometry-only-control'
SOURCE=h.SOURCE

def save_geometry(t,rows,label):
    h.g.f.export(t,rows,label)
    params={n:getattr(t,n).detach().cpu().numpy() for n in ['shape',*h.g.f.PER_FRAME,'scene_scale','static_offset']}
    params.update(frame_ids=np.array([r['frame'] for r in rows]),K_integer_centers=np.array([r['K_integer_centers'] for r in rows]))
    np.savez_compressed(OUT/label/'tracked-flame-geometry.npz',**params)

def main():
    assert not OUT.exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(BASE/'plan.json');plan.update(updated=now(),source=str(SOURCE),source_sha256=sha(SOURCE),
        script_sha256=sha(__file__),parent_script_sha256=sha(h.__file__),
        authorization='User: 只需要学形状，不需要学纹理颜色吧。',
        single_change='Compared with the1003-update mask stage, same initialization, crop, optimizer, masks and priors; set RGB energy to zero. Mask weight remains30.',
        appearance_optimization=False,appearance_objective=False,num_epochs=17,max_updates=1003,max_train_seconds=90,
        output='Geometry checkpoint and all59 original/mesh/mask comparisons. No texture learning or appearance deliverable.',quality_accepted=False)
    OUT.mkdir();save(OUT/'plan.json',plan);h.OUT=OUT;start=time.monotonic()
    try:
        t,rows,hair=h.setup();original=t.compute_photometric_energy
        def alpha_only(self,*args,**kwargs):
            photo,render=original(*args,**kwargs)
            return photo*0,render
        t.compute_photometric_energy=types.MethodType(alpha_only,t)
        s=t.dataset.one(32);loss,logs,*_=t.compute_energy(s,stage=h.STAGE)
        grad=torch.autograd.grad(loss,t.static_offset)[0]
        assert float(logs['photo'])==0 and float(logs['silhouette'])>0 and torch.isfinite(grad).all() and float(grad.abs().max())>0
        save(OUT/'preflight.json',dict(updated=now(),optimizer_updates=0,photo_energy=0,mask_energy=float(logs['silhouette']),gradient_max=float(grad.abs().max())))
        frozen={n:getattr(t,n).detach().clone() for n in ['shape',*h.g.f.PER_FRAME,'scene_scale','tex_extra','lights']}
        old=t.static_offset.detach().clone();nonhair=torch.ones(old.shape[1],device='cuda',dtype=torch.bool);nonhair[hair]=False
        save_geometry(t,rows,'before');t.started=time.monotonic();optimize=t.optimize_iter
        def bounded(*args,**kwargs):
            if time.monotonic()-t.started>90:raise RuntimeError('Geometry-only90s budget')
            return optimize(*args,**kwargs)
        t.optimize_iter=bounded
        t.optimize_stage(h.STAGE,dataloader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0),lr_scale=.1)
        seconds=time.monotonic()-t.started
        assert t.global_step==1003 and all(torch.equal(getattr(t,n),v) for n,v in frozen.items())
        assert torch.equal(t.static_offset[:,nonhair],old[:,nonhair])
        save_geometry(t,rows,'epoch-30');h.mask_review(t,rows)
        save(OUT/'result.json',dict(updated=now(),state='completed_pending_geometry_review',frames=59,steps=1003,
            train_seconds=seconds,elapsed_seconds=time.monotonic()-start,photo_energy_zero=True,
            shape_pose_appearance_camera_parameters_unchanged=True,nonhair_offsets_unchanged=True,quality_accepted=False,Gaussian_started=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_pending_geometry_review',pid=None,step=1003))
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
