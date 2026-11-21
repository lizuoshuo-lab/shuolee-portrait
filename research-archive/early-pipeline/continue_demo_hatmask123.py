"""One bounded convergence check of the same mask objective; no weight search."""
import fcntl,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
import fit_demo_flame_hatmask123 as h
from run_official_gaussians123 import read,save,sha,now

BASE=h.OUT
OUT=BASE/'convergence-control'
SOURCE=BASE/'epoch-30/tracked-flame-with-surface.npz'

def main():
    assert not OUT.exists(), 'Single convergence control only.'
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=read(BASE/'review/geometry-mask-audit.json')
    assert audit['median_iou_after']>audit['median_iou_before'] and audit['reference_triangle_normals_over90deg']==0
    plan=read(BASE/'plan.json');plan.update(updated=now(),source=str(SOURCE),source_sha256=sha(SOURCE),
        script_sha256=sha(__file__),parent_script_sha256=sha(h.__file__),
        stage='One same-objective convergence control after1003 updates; fresh original stage Adam from saved mesh, not exact optimizer-state resume.',
        rationale='Mask IoU0.753->0.837; no reference triangle normal reversals; face parameters unchanged; brim still under-covered.',
        num_epochs=50,max_updates=2950,max_train_seconds=90,
        stop='One additional fixed-budget joint stage. Preserve face and camera, reject silhouette regressions, folding or visible remaining brim gap. Do not scan weights.')
    OUT.mkdir();save(OUT/'plan.json',plan)
    h.OUT=OUT;h.SOURCE=SOURCE;start=time.monotonic()
    try:
        t,rows,hair=h.setup();t.cfg.pipeline[h.STAGE].num_epochs=50
        frozen={n:getattr(t,n).detach().clone() for n in ['shape',*h.g.f.PER_FRAME,'scene_scale','tex_extra','lights']}
        offset=t.static_offset.detach().clone();nonhair=torch.ones(offset.shape[1],dtype=torch.bool,device='cuda');nonhair[hair]=False
        h.g.export(t,rows,'before');t.started=time.monotonic();original=t.optimize_iter
        def bounded(*args,**kwargs):
            if time.monotonic()-t.started>90:raise RuntimeError('Same-objective90s limit')
            return original(*args,**kwargs)
        t.optimize_iter=bounded
        t.optimize_stage(h.STAGE,dataloader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0),lr_scale=.1)
        seconds=time.monotonic()-t.started
        assert t.global_step==2950 and all(torch.equal(getattr(t,n),x) for n,x in frozen.items())
        assert torch.equal(t.static_offset[:,nonhair],offset[:,nonhair])
        h.g.export(t,rows,'epoch-30');h.mask_review(t,rows)
        save(OUT/'result.json',dict(updated=now(),state='completed_pending_visual_review',frames=59,steps=2950,
            cumulative_mask_updates=3953,train_seconds=seconds,elapsed_seconds=time.monotonic()-start,
            face_and_camera_parameters_unchanged=True,nonhair_offsets_unchanged=True,quality_accepted=False,Gaussian_started=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_pending_visual_review',pid=None,step=2950))
    except BaseException as e:
        save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
