"""Run existing VHAP photometric stages after reviewed FLAME landmark fitting."""
import argparse,dataclasses,fcntl,json,os,time,types,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import fit_demo_flame_face123 as f
from run_official_gaussians123 import read,save,sha,now

BASE=f.OUT/'pose-seeded-control';OUT=f.OUT/'photometric-face'
OCCLUDED=set(range(25,32))

def setup():
    os.chdir(f.WORK);torch.set_num_threads(4);torch.manual_seed(6060)
    f.OUT=OUT;t,rows=f.make_tracker()
    from vhap.model.flame import FlameTexPainted
    t.cfg.exp.photometric=True;t.cfg.model.tex_extra=True;t.cfg.model.tex_resolution=2048
    t.cfg.model.use_static_offset=True;t.cfg.data.use_alpha_map=True
    t.flame_tex_painted=FlameTexPainted(tex_size=2048).cuda()
    t.tex_extra=torch.zeros(3,2048,2048,device='cuda',requires_grad=True)
    t.static_offset=torch.zeros(1,len(t.flame.v_template),3,device='cuda',requires_grad=True)
    with np.load(BASE/'epoch-30/face-parameters-and-points.npz',allow_pickle=False) as z:
        with torch.no_grad():
            for name in ['shape',*f.PER_FRAME,'scene_scale']:getattr(t,name).copy_(torch.tensor(z[name],device='cuda'))
    for i,row in enumerate(rows):
        xy=t.dataset.crops[i];side=t.dataset.side
        alpha=torch.from_numpy(np.array(Image.open(row['mask']).convert('L').crop((*xy,*(xy+side))),copy=True)).float()[None]/255
        sample=t.dataset.samples[i];sample['rgb']=sample['rgb']*alpha+(1-alpha)
        sample['alpha']=alpha;sample['alpha_map']=alpha
    original_photo=t.compute_photometric_energy
    def observed_photo(self,sample,*args,**kwargs):
        energy,render=original_photo(sample,*args,**kwargs)
        # Predeclared raw-image occlusion, not rejection by residual. Keep all
        # 59 frames in the original landmark/temporal objective.
        if int(sample['timestep_index'][0]) in OCCLUDED:energy=energy*0
        return energy,render
    t.compute_photometric_energy=types.MethodType(observed_photo,t)
    t.started=time.monotonic()
    original_iter=t.optimize_iter
    def bounded(sample,optimizer,stage):
        if time.monotonic()-t.started>420:raise RuntimeError('Original RGB stage420s budget reached')
        return original_iter(sample,optimizer,stage)
    t.optimize_iter=bounded
    t.evaluate=lambda **kwargs: None
    return t,rows

def export(t,rows,label):
    f.export(t,rows,label)
    folder=OUT/label
    params={k:getattr(t,k).detach().cpu().numpy() for k in ['shape',*f.PER_FRAME,'scene_scale','tex_extra','lights','static_offset']}
    params.update(frame_ids=np.array([r['frame'] for r in rows]),K_integer_centers=np.array([r['K_integer_centers'] for r in rows]))
    np.savez_compressed(folder/'tracked-flame-with-surface.npz',**params)
    t.cfg.render.background_eval='white';errors=[];offset=t.static_offset.detach().norm(dim=-1).cpu().numpy()
    for i,row in enumerate(rows):
        with torch.no_grad():
            t.clear_cache();s=t.dataset.one(i);_,_,v,faces,lm,albedo,render=t.compute_energy(s,stage=None)
        rgba=render['rgba'][0].detach().cpu().numpy();rgb=np.uint8(np.clip(rgba[:3].transpose(1,2,0),0,1)*255)
        xy=t.dataset.crops[i];side=t.dataset.side;raw=Image.open(row['image']).convert('RGB').crop((*xy,*(xy+side)))
        image=Image.new('RGB',(2*side,side),'white');image.paste(raw);image.paste(Image.fromarray(rgb),(side,0))
        image.save(folder/(row['frame']+'_原图_学习纹理模型独立渲染.png'))
        # Fixed raw-observation convex hull, identical for before/after; no
        # reconstruction silhouette selection and no claim of heldout RGB.
        import cv2
        pts=t.dataset.obs['keypoints'][i,23+17:23+68]-xy
        mask=np.zeros((side,side),np.uint8);cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pts).astype(np.int32)),1)
        target=np.array(raw)/255.;err=np.abs(rgb/255.-target).mean(2)
        errors.append(dict(frame=row['frame'],face_hull_L1=float(err[mask>0].mean()),
            face_hull_coverage=float((rgba[3][mask>0]>.5).mean()),occluded=i in OCCLUDED))
    np.savez_compressed(folder/'shared-surface-offset.npz',offset=params['static_offset'])
    save(folder/'appearance-review-data.json',dict(updated=now(),rows=errors,template_units_not_measured_metres=True,
        shared_offset_norm=dict(median=float(np.median(offset)),p90=float(np.quantile(offset,.9)),max=float(offset.max())),
        median_face_hull_L1=float(np.median([r['face_hull_L1'] for r in errors if not r['occluded']])),
        model_only_render_on_white=True,raw_image_not_used_as_display_background=True,hat_brim_reconstruction_accepted=False))

def main(mode):
    if mode=='register':
        assert not OUT.exists();OUT.mkdir()
        save(OUT/'plan.json',dict(updated=now(),state='registered',authorization='User accepted third head pose and requested learning facial and hat surface features; use existing original RGB/texture/shared-offset stages.',
            source=str(BASE),source_sha256=sha(BASE/'epoch-30/face-parameters-and-points.npz'),script_sha256=sha(__file__),
            stages=['rgb_init_texture:500 at reviewed33','rgb_init_all:500 at reviewed33','rgb_init_offset:500 at reviewed33','rgb_global_tracking:30epochs on59frames'],
            optimizer='Unchanged original VHAP Adam, parameters, learning rates, losses, regularizers and texture2048 settings; global lr_scale0.1',
            masks='Existing SAM2 alpha, white compositing in native crops. Conditional K unchanged. No resizing or second undistortion.',
            occlusion='Predeclared slots25..31: photo energy zero; preserve original landmark/temporal terms and59 frame identities.',
            scope='Existing head topology, shared texture and canonical vertex offsets. No network-regressed template, no new sequence solver, no Gaussian training.',
            limitation='Target background is the upstream setting: no gradient for absent brim/hat region. This cannot establish cap silhouette recovery; downstream clothed/hat surface Gaussians remain required.',
            risk='RGB can confuse lighting/texture/geometry; offsets may damage profile or absorb mask errors. Retain baseline and actual all-frame mesh and model-only RGB review.',
            budget=dict(max_updates=3270,max_seconds=420),quality_accepted=False));return
    plan=read(OUT/'plan.json');assert plan['script_sha256']==sha(__file__)
    assert plan['source_sha256']==sha(BASE/'epoch-30/face-parameters-and-points.npz')
    marker=OUT/(mode+'-started.json');assert not marker.exists()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(marker,dict(updated=now(),pid=os.getpid()));started=time.monotonic()
    try:
        t,rows=setup();t.get_train_parameters('rgb_init_offset');s=t.dataset.one(32)
        loss,logs,*_=t.compute_energy(s,stage='rgb_init_offset')
        grads=torch.autograd.grad(loss,[t.tex_extra,t.static_offset,t.shape,t.rotation],allow_unused=False)
        check={n:dict(finite=bool(torch.isfinite(g).all()),max_abs=float(g.abs().max())) for n,g in zip(['texture','surface_offset','shape','rotation'],grads)}
        assert all(g['finite'] and g['max_abs']>0 for g in check.values())
        with torch.no_grad():
            t.clear_cache();_,occlusion_logs,*_=t.compute_energy(t.dataset.one(28),stage='rgb_init_offset')
        assert float(occlusion_logs['photo'])==0
        save(OUT/'preflight.json',dict(updated=now(),state='passed',optimizer_updates=0,gradients=check,
            uv_mask_shape=list(t.flame_uvmask.get_uvmask_by_region(['sclerae']).shape),occluded_photo_energy=0))
        save(OUT/'effective-config.json',json.loads(json.dumps(dataclasses.asdict(t.cfg),default=str)))
        if mode=='preflight':return
        export(t,rows,'before')
        t.started=time.monotonic()
        for stage in ['rgb_init_texture','rgb_init_all','rgb_init_offset']:t.optimize_stage(stage,sample=t.dataset.one(32))
        t.optimize_stage('rgb_global_tracking',dataloader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0),lr_scale=.1)
        assert t.global_step==3270
        train_seconds=time.monotonic()-t.started
        export(t,rows,'epoch-30')
        save(OUT/'result.json',dict(updated=now(),state='completed_pending_visual_review',frames=59,steps=3270,
            train_seconds=train_seconds,elapsed_seconds=time.monotonic()-started,quality_accepted=False,Gaussian_started=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_pending_visual_review',pid=None,step=3270))
    except BaseException as e:
        save(OUT/(mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['register','preflight','fit']);args=p.parse_args();main(args.mode)
