"""Original VHAP face pose initialization control, without a new solver."""
import argparse,copy,fcntl,os,time
import numpy as np
import torch
from PIL import Image
import cv2
import fit_demo_flame_face123 as f
from run_official_gaussians123 import read,save,sha,now

BASE=f.OUT;OUT=BASE/'pose-seeded-control'

def run(fit=False):
    if not fit:
        assert not OUT.exists();OUT.mkdir()
        save(OUT/'plan.json',dict(updated=now(),state='prepared_not_authorized_to_train',
            hypothesis='Head pose seed rather than geometry/K changes: identical final shared face, per-frame expression and original losses; initialize only root rigid pose from the fixed EPNP feasibility result.',
            initialization='Full 59-frame parameter transfer checked geometrically; select equivalent Rodrigues branch closest to existing root angle.',
            budget=dict(updates=1770,max_seconds=180,GPU_tasks=1),
            original_optimizer='Original VHAP lmk_global_tracking,30 epochs,lr_scale0.1; no new loss or sequence solver',
            risk='Occluded/inaccurate 2D landmarks can seed a wrong head pose; shared shape/global stage may regress near-front frames. The frozen baseline and held labels are retained.',
            script_sha256=sha(__file__),checkpoint_sha256=sha(BASE/'epoch-30/face-parameters-and-points.npz'),
            pose_seed_sha256=sha(BASE/'rigid-pose-diagnostic/rigid-head-correction.npz'),
            training_authorized=False,quality_accepted=False))
    else:
        plan=read(OUT/'plan.json');assert plan['training_authorized'], 'Specific pose-control approval is pending.'
        assert plan['script_sha256']==sha(__file__) and not (OUT/'fit-started.json').exists()
        assert sha(BASE/'epoch-30/face-parameters-and-points.npz')==plan['checkpoint_sha256']
        assert sha(BASE/'rigid-pose-diagnostic/rigid-head-correction.npz')==plan['pose_seed_sha256']
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.chdir(f.WORK);torch.set_num_threads(4);torch.manual_seed(6060)
    f.OUT=OUT;t,rows=f.make_tracker();started=time.monotonic()
    with np.load(BASE/'epoch-30/face-parameters-and-points.npz',allow_pickle=False) as z:data=dict(z)
    with np.load(BASE/'rigid-pose-diagnostic/rigid-head-correction.npz',allow_pickle=False) as z:seed=dict(z)
    maximum=0.;original_crop_images=[]
    from review_full_sequence123 import mesh_renderer
    render_mesh,_=mesh_renderer()
    with torch.no_grad():
        for name in ['shape',*f.PER_FRAME,'scene_scale']:getattr(t,name).copy_(torch.tensor(data[name],device='cuda'))
        for i,row in enumerate(rows):
            rc=seed['rotation'][i];tc=seed['translation'][i]
            desired=data['landmarks3d_camera'][i]@rc.T+tc
            old=data['rotation'][i];r=cv2.Rodrigues(rc@cv2.Rodrigues(old)[0])[0].ravel()
            axis=r/np.linalg.norm(r);choices=[r,r+2*np.pi*axis,r-2*np.pi*axis];r=min(choices,key=lambda x:np.linalg.norm(x-old))
            t.rotation[i]=torch.tensor(r,device='cuda');t.translation[i].zero_()
            _,_,zero,_=t.forward_flame(torch.tensor([i]));delta=desired-zero[0].cpu().numpy()
            assert np.max(np.ptp(delta,axis=0))<1e-5
            t.translation[i]=torch.tensor(delta.mean(0),device='cuda')
            v,_,lm,_=t.forward_flame(torch.tensor([i]));err=float(abs(lm[0].cpu().numpy()-desired).max());maximum=max(maximum,err)
            assert err<1e-5 and lm[...,2].min()>.1
            if not fit and i in [0,12,23,28,32,35,46,58]:
                raw=Image.open(row['image']).convert('RGB');xy=t.dataset.crops[i];side=t.dataset.side;box=(*xy,*(xy+side))
                previous=Image.open(BASE/'epoch-30'/(row['frame']+'_原图_FLAME投影.png')).convert('RGB')
                over=render_mesh(raw,np.array([0,0,*raw.size]),dict(vertices_camera=v.cpu().numpy(),faces=t.flame.faces.cpu().numpy(),K_integer_centers=np.array(row['K_integer_centers'])))
                joined=Image.new('RGB',(3*side,side),'white');joined.paste(previous);joined.paste(over.crop(box),(2*side,0))
                joined.save(OUT/(row['frame']+'_原图_原拟合_仅姿态种子.png'))
    if not fit:
        save(OUT/'preflight.json',dict(updated=now(),state='passed_transfer_only',frames=59,
            transformed_landmark_max_abs_units=maximum,optimizer_updates=0,source_geometry_unchanged=True,quality_accepted=False))
        print(read(OUT/'preflight.json'));return
    from torch.utils.data import DataLoader
    save(OUT/'fit-started.json',dict(updated=now(),pid=os.getpid(),source='approved pose-seed control'))
    original_iter=t.optimize_iter
    def bounded(sample,optimizer,stage):
        if time.monotonic()-started>180:raise RuntimeError('Pose-seed control 180s budget exceeded')
        return original_iter(sample,optimizer,stage)
    t.optimize_iter=bounded
    t.optimize_stage('lmk_global_tracking',dataloader=DataLoader(t.dataset,batch_size=1,shuffle=True,num_workers=0),lr_scale=.1)
    assert t.global_step==1770
    save(OUT/'result.json',dict(updated=now(),state='complete_pending_review',frames=59,steps=1770,
        elapsed_seconds=time.monotonic()-started,quality_accepted=False,network_template_used=False))
    save(OUT/'progress.json',dict(updated=now(),state='complete_pending_review',pid=None,step=1770))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','fit']);args=p.parse_args();run(args.mode=='fit')
