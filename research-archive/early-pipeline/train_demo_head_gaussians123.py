"""Thin cached-FLAME interface to the existing GaussianAvatars mesh-bound core.

No new fitter, camera solver, skinning or source-repository edits. The original
GaussianModel, triangle frames, optimizer and existing local-unit split repair
are reused. This run is explicitly head-only and exploratory.
"""
import argparse,fcntl,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from run_official_gaussians123 import read,save,sha,now

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/demo-vhap-initialization')
GEO=ROOT/'flame-face-prestage/hat-mask-control/geometry-only-control'
OUT=ROOT/'head-gaussian-first-cycle'
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
REFERENCE=32

def prepare():
    import torch
    import fit_demo_flame_hatmask123 as h
    assert not OUT.exists(), 'Do not overwrite or repeat this run.'
    OUT.mkdir();start=time.monotonic()
    h.OUT=GEO;t,rows,hair=h.setup()
    source=GEO/'epoch-30/tracked-flame-geometry.npz'
    with np.load(source,allow_pickle=False) as z,torch.no_grad():
        for n in ['shape',*h.g.f.PER_FRAME,'scene_scale','static_offset']:
            getattr(t,n).copy_(torch.as_tensor(z[n],device='cuda'))
    vertices=[];canonical=[];landmarks=[];records=[]
    imagefolder=OUT/'inputs';imagefolder.mkdir()
    points=t.dataset.obs['keypoints'][:,23:91]
    plan=read(GEO/'plan.json')
    for i,(row,r) in enumerate(zip(rows,plan['records'])):
        with torch.no_grad():v,c,l,_=t.forward_flame(torch.tensor([i],device='cuda'))
        vertices.append(v[0].cpu().numpy());canonical.append(c[0].cpu().numpy());landmarks.append(l[0].cpu().numpy())
        xy=np.array(r['crop_integer_xy']);side=plan['crop_side'];box=(*xy,*(xy+side))
        imagefolder_i=imagefolder/row['frame'];imagefolder_i.mkdir()
        for key,name in [('image','image.png'),('mask','mask.png')]:
            Image.open(row[key]).convert('RGB' if key=='image' else 'L').crop(box).save(imagefolder_i/name)
        # Head-only observation region: raw jaw landmark height, not fitted mesh
        # silhouette. Predeclared device-occluded frames contribute cap alpha only.
        lower=int(np.ceil(points[i,:17,1].max()))
        occluded=25<=i<=31
        if occluded:lower=r['upper_head_band_ymax']
        roi=np.broadcast_to((np.arange(side)+xy[1])[:,None]<=lower,(side,side)).astype('uint8')*255
        Image.fromarray(roi).save(imagefolder_i/'roi.png')
        K=np.array(row['K']);K[:2,2]-=xy
        records.append(dict(frame=row['frame'],source_frame=row['source_frame_index'],image=str(imagefolder_i/'image.png'),
            mask=str(imagefolder_i/'mask.png'),roi=str(imagefolder_i/'roi.png'),K=K.tolist(),wh=[side,side],
            w2c=np.eye(4).tolist(),crop_integer_xy=xy.tolist(),photo_valid=not occluded,
            source_image_sha256=sha(row['image']),source_mask_sha256=sha(row['mask']),
            image_sha256=sha(imagefolder_i/'image.png'),mask_sha256=sha(imagefolder_i/'mask.png')))
    np.savez_compressed(OUT/'fitted-head-meshes.npz',vertices_world=np.array(vertices),vertices_canonical=np.array(canonical),
        landmarks3d=np.array(landmarks),faces=t.flame.faces.cpu().numpy(),frame_ids=np.array([r['frame'] for r in rows]))
    assert len(rows)==59 and np.isfinite(vertices).all()
    # Projection replay includes the actual integer crop / explicit half pixel.
    errs=[]
    for i,row in enumerate(records):
        l=landmarks[i];q=l@np.array(row['K']).T;crop=q[:,:2]/q[:,2:]-.5
        full=l@np.array(rows[i]['K_integer_centers']).T
        errs.append(float(np.abs(crop-(full[:,:2]/full[:,2:]-row['crop_integer_xy'])).max()))
    assert max(errs)<1e-4
    save(OUT/'inputs.json',dict(records=records,source_geometry=str(source),source_geometry_sha256=sha(source),mesh_sha256=sha(OUT/'fitted-head-meshes.npz'),max_projection_replay_error_px=max(errs)))
    save(OUT/'plan.json',dict(updated=now(),authorization='User explicitly selected current face plus hat, and directly starting Gaussian training without another mesh round.',
        scope='Current fitted FLAME head and hat; not whole-body SMPL-X.',frames=59,reference_slot=REFERENCE,
        source_geometry=str(source),source_geometry_sha256=sha(source),initialization='One original Gaussian per fitted triangle, fresh optimizer and SH.',
        fitted_mesh_pose_and_offsets='Frozen input to this Gaussian round; no further VHAP fitting or texture optimization.',
        supervision='Original first SAM2 mask; native integer crop, no resize. Pixels below raw jaw height excluded. Slots25..31 use upper-head alpha only because of predeclared device occlusion.',
        appearance='No VHAP texture imported. Gaussian SH learns foreground appearance during Gaussian reconstruction.',
        algorithm='Original GaussianAvatars GaussianModel and FlameGaussianModel.update_mesh_properties, with existing fullK camera and local-coordinate split/export adapters.',
        schedule_steps=18000,first_review_step=1000,point_cap=200000,densify_from=300,densify_interval=60,
        max_train_seconds=900,known_limitations=['Hat brim missing in initial mesh','Mask may include held objects','Conditional K, no independently calibrated scale','Unobserved back is not validated'],
        fit_quality_accepted=False,training_admission='Explicit exploratory user authorization; not a geometric quality pass.',
        script_sha256=sha(__file__),prepared_seconds=time.monotonic()-start))
    save(OUT/'progress.json',dict(updated=now(),state='prepared',training_started=False,quality_accepted=False))
    print('HEAD_GAUSSIAN_PREPARED',flush=True)

def train():
    import torch
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera,foreground_loss,image
    from scene.gaussian_model import GaussianModel
    from scene.flame_gaussian_model import FlameGaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel
    from gaussian_renderer import render
    from arguments import OptimizationParams
    plan=read(OUT/'plan.json');inputs=read(OUT/'inputs.json');rows=inputs['records']
    assert sha(__file__)==plan['script_sha256'] and sha(OUT/'fitted-head-meshes.npz')==inputs['mesh_sha256']
    run=OUT/'gaussian-run';run.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(6060);torch.cuda.manual_seed_all(6060);np.random.seed(6060)
    torch.backends.cuda.matmul.allow_tf32=False
    with np.load(OUT/'fitted-head-meshes.npz',allow_pickle=False) as z:data=dict(z)
    class CachedFlame(GaussianModel):
        def __init__(self):
            super().__init__(3)
            self.vertices_sequence=torch.tensor(data['vertices_world'],device='cuda')
            self.canonical=torch.tensor(data['vertices_canonical'],device='cuda')
            self.faces=torch.tensor(data['faces'],device='cuda',dtype=torch.long)
            self.flame_model=SimpleNamespace(faces=self.faces)
            self.binding=torch.arange(len(self.faces),device='cuda')
            self.binding_counter=torch.ones(len(self.faces),device='cuda',dtype=torch.int32)
            self.num_timesteps=len(rows);self.select_mesh_by_timestep(REFERENCE)
        def select_mesh_by_timestep(self,timestep):
            self.timestep=int(timestep)
            FlameGaussianModel.update_mesh_properties(self,self.vertices_sequence[self.timestep:self.timestep+1],self.canonical[self.timestep:self.timestep+1])
        densify_and_split=SMPLXGaussianModel.densify_and_split
        get_covariance=SMPLXGaussianModel.get_covariance
        export_world_ply=SMPLXGaussianModel.export_world_ply
    model=CachedFlame();model.create_from_pcd(None,1.)
    parser=argparse.ArgumentParser();opt=OptimizationParams(parser).extract(parser.parse_args([]))
    opt.iterations=opt.position_lr_max_steps=opt.densify_until_iter=18000
    opt.densify_from_iter=300;opt.densification_interval=60;opt.opacity_reset_interval=1800
    model.training_setup(opt);assert len(model.optimizer.state)==0
    cams=[];rgbs=[];alphas=[];rois=[];camrecords=[]
    for i,r in enumerate(rows):
        assert sha(r['image'])==r['image_sha256'] and sha(r['mask'])==r['mask_sha256']
        cam,record=camera(r,i,pose_index=i);cams.append(cam);camrecords.append(record)
        rgbs.append(torch.tensor(np.array(Image.open(r['image'])),device='cuda').permute(2,0,1).float()/255)
        alphas.append(torch.tensor(np.array(Image.open(r['mask'])),device='cuda')[None].float()/255)
        rois.append(torch.tensor(np.array(Image.open(r['roi'])),device='cuda')[None].float()/255)
    save(run/'training-cameras.json',dict(records=camrecords,note='Each camera requires its named bound mesh pose; not59 static-viewer cameras.'))
    save(run/'effective-options.json',dict(vars(opt)))
    save(run/'provenance.json',dict(updated=now(),script_sha256=sha(__file__),input_mesh_sha256=inputs['mesh_sha256'],
        source_files={str(p):sha(p) for p in [PROJECT/'src/avatar_core/scene/gaussian_model.py',PROJECT/'src/avatar_core/scene/flame_gaussian_model.py',PROJECT/'src/avatar_core/scene/smplx_gaussian_model.py',PROJECT/'src/shuolee_portrait/training.py']},
        initial_points=len(model._xyz),optimizer_initial_state_entries=0,from_scratch=True,head_only=True,quality_accepted=False))
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False);bg=torch.zeros(3,device='cuda')
    def losses(i):
        model.select_mesh_by_timestep(i);p=render(cams[i],model,pipe,bg)
        valid_alpha=alphas[i]*rois[i]
        if rows[i]['photo_valid']:photo,l1,ss=foreground_loss(p['render'],rgbs[i],valid_alpha)
        else:photo=l1=ss=p['render'].sum()*0
        alpha=render(cams[i],model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
        silhouette=((alpha-alphas[i]).abs()*rois[i]).sum()/rois[i].sum().clamp_min(1)
        visible=p['visibility_filter'];assert visible.any()
        reg=torch.relu(model._xyz[visible].norm(dim=-1)-opt.threshold_xyz).mean()*opt.lambda_xyz
        reg+=torch.relu(model._scaling[visible].exp()-opt.threshold_scale).norm(dim=-1).mean()*opt.lambda_scale
        return photo+.1*silhouette+reg,p,dict(l1=l1,ssim=ss,alpha=silhouette,regularizer=reg)
    loss,p,detail=losses(REFERENCE);loss.backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for g in model.optimizer.param_groups for v in g['params'])
    original=rgbs[REFERENCE].clone();probe=alphas[REFERENCE]*rois[REFERENCE]
    with torch.no_grad():
        test=original.clone();test[:,probe[0]<=.99]=.37
        difference=float(abs(foreground_loss(p['render'],original,probe)[0]-foreground_loss(p['render'],test,probe)[0]))
    assert difference<1e-7
    model.optimizer.zero_grad(set_to_none=True)
    save(run/'preflight.json',dict(updated=now(),finite_backward=True,optimizer_updates=0,outside_RGB_effect=difference,
        camera_projection_max_error_px=max(x['projection_error_px'] for x in camrecords),initial_points=len(model._xyz)))
    extent=float(np.linalg.norm(np.ptp(data['vertices_world'][REFERENCE],axis=0)))
    def snapshot(step):
        folder=run/f'step-{step:06d}';folder.mkdir()
        with torch.no_grad():
            model.export_world_ply(folder/'world.ply',REFERENCE);model.save_ply(str(folder/'bound-local.ply'))
            np.savez_compressed(folder/'binding.npz',triangle_ids=model.binding.cpu().numpy(),frame_ids=data['frame_ids'])
            torch.save(dict(step=step,capture=model.capture(),binding=model.binding,binding_counter=model.binding_counter),folder/'checkpoint.pt')
            plain=GaussianModel(3);plain.load_ply(str(folder/'world.ply'));plain.active_sh_degree=model.active_sh_degree
            model.select_mesh_by_timestep(REFERENCE)
            delta=(render(cams[REFERENCE],model,pipe,bg)['render']-render(cams[REFERENCE],plain,pipe,bg)['render']).abs()
            assert float(delta.max())<.005 and float(delta.mean())<2e-6
            check=dict(mean_abs=float(delta.mean()),max_abs=float(delta.max()),passed=True)
            for i,r in enumerate(rows):
                model.select_mesh_by_timestep(i);rgb=render(cams[i],model,pipe,bg)['render']
                alpha=render(cams[i],model,pipe,bg,override_color=torch.ones_like(model.get_xyz))['render'][:1]
                image(rgb).save(folder/(r['frame']+'_gaussian.png'))
                image(alpha.repeat(3,1,1)).save(folder/(r['frame']+'_alpha.png'))
            save(folder/'result.json',dict(updated=now(),step=step,points=len(model._xyz),active_sh_degree=model.active_sh_degree,
                reference_frame=rows[REFERENCE]['frame'],reference_slot=REFERENCE,world_ply_sha256=sha(folder/'world.ply'),native_ply_parity=check,
                pose_binding_retained=True,mesh_source_sha256=inputs['mesh_sha256'],quality_accepted=False))
        return folder
    rng=np.random.default_rng(6060);stack=[];counts=np.zeros(59,dtype=int);start=time.monotonic();wait_seconds=0
    with (run/'loss.jsonl').open('x') as logfile:
        for step in range(1,18001):
            if time.monotonic()-start-wait_seconds>900:raise RuntimeError('Registered900s Gaussian compute budget reached')
            model.update_learning_rate(step)
            if step%1000==0:model.oneupSHdegree()
            if not stack:stack=rng.permutation(59).tolist()
            i=stack.pop();counts[i]+=1
            loss,p,detail=losses(i);assert torch.isfinite(loss)
            loss.backward()
            with torch.no_grad():
                visible=p['visibility_filter'];model.max_radii2D[visible]=torch.maximum(model.max_radii2D[visible],p['radii'][visible])
                model.add_densification_stats(p['viewspace_points'],visible)
                model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
                if 300<step<18000 and step%60==0:model.densify_and_prune(opt.densify_grad_threshold,.005,extent,None,max_points=200000)
                if step<18000 and step%opt.opacity_reset_interval==0:model.reset_opacity()
                assert len(model._xyz)<=200000 and model.binding_counter.min()>=1
            if step==1 or step%100==0:
                status=dict(updated=now(),state='training',pid=os.getpid(),step=step,points=len(model._xyz),loss=float(loss),
                    seconds=time.monotonic()-start-wait_seconds,views_seen=int((counts>0).sum()),quality_accepted=False,**{k:float(v) for k,v in detail.items()})
                save(OUT/'progress.json',status);logfile.write(json.dumps(status)+'\n');logfile.flush();print(status,flush=True)
            if step in [1000,6000,12000,18000]:
                folder=snapshot(step)
                if step==1000:
                    decision=OUT/'review-1000-decision.json'
                    assert not decision.exists()
                    save(OUT/'review-1000-request.json',dict(updated=now(),step=step,snapshot=str(folder),world_ply_sha256=sha(folder/'world.ply')))
                    save(OUT/'progress.json',dict(updated=now(),state='review_pending',pid=os.getpid(),step=step,optimizer_waiting=True,quality_accepted=False))
                    waiting=time.monotonic()
                    while not decision.exists():time.sleep(2)
                    d=read(decision);assert d['world_ply_sha256']==sha(folder/'world.ply') and d['decision'] in ['continue','stop']
                    wait_seconds+=time.monotonic()-waiting
                    if d['decision']=='stop':break
        save(run/'result.json',dict(updated=now(),state='completed_quality_review_pending' if step==18000 else 'stopped_after_review',
            steps=step,points=len(model._xyz),per_frame_updates=counts.tolist(),seconds=time.monotonic()-start-wait_seconds,
            final_snapshot=f'step-{step:06d}',quality_accepted=False))
        save(OUT/'progress.json',dict(updated=now(),state='completed_quality_review_pending',pid=None,step=step,points=len(model._xyz),quality_accepted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','train']);args=p.parse_args()
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:prepare() if args.mode=='prepare' else train()
    except BaseException as e:
        save(OUT/(args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False));raise
