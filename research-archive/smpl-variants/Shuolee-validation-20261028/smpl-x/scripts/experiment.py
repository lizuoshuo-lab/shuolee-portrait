"""Portable launch adapter for the preserved, static 114 comparison code.

This wrapper changes file locations and execution gates, not the loss or camera model.
It never imports the old subject-fitting pipeline or loads a learned person checkpoint.
"""
import argparse, hashlib, importlib.util, json, os, shutil, sys, types
from contextlib import contextmanager
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
CORE=PROJECT/'src/avatar_core'
ARMS=('gaussianavatar','smplx')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()
def resolve(p):
    p=Path(p);return p.resolve() if p.is_absolute() else (PROJECT/p).resolve()

def audit(cfg):
    import numpy as np
    from PIL import Image
    dataset=resolve(cfg['dataset']);items=read(dataset/'training-inputs.json')['records']
    assert len(items)==cfg['frames']==175
    assert len({r['frame'] for r in items})==175
    assert sorted(r['source_frame_index'] for r in items)==list(range(1,176))
    mask_count=0
    for r in items:
        assert r['camera_kind']=='fresh_full_scene_sfm' and not r['camera_registration']['interpolation_used']
        E=np.array(r['w2c']);K=np.array(r['K']);R=E[:3,:3]
        assert E.shape==(4,4) and np.isfinite(E).all() and np.isfinite(K).all()
        assert np.max(abs(R@R.T-np.eye(3)))<1e-5 and np.linalg.det(R)>.999
        assert K[0,0]>0 and K[1,1]>0 and np.allclose(K[2],[0,0,1])
        for field,digest in [('image','source_sha256'),('mask','mask_sha256')]:
            path=dataset/r[field];assert sha(path)==r[digest],str(path)
            with Image.open(path) as image:assert image.size==tuple(r['wh'])
        mask_count+=1
    manifest=read(PROJECT/'configs/assets-manifest.json')['files']
    for r in manifest:
        p=PROJECT/r['path'];assert p.exists(),f'Missing WPS asset: {p}'
        assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],f'Asset mismatch: {p}'
    return dict(frames=175,training_masks=mask_count,fit_masks=sum(r['mask'] is not None for r in read(dataset/'inputs.json')['records']),camera_matrices_valid=True,input_bytes_verified=True,physical_calibration_accuracy_accepted=False,old_subject_fit_loaded=False)

def prepare(cfg,run):
    report=audit(cfg)
    if run.exists():raise RuntimeError('Fresh run only: output directory already exists. Choose a new run name.')
    dataset=resolve(cfg['dataset']);run.mkdir(parents=True)
    for arm in ARMS:
        root=run/arm
        for directory in ['data','inference','fit','gaussian-run']:(root/directory).mkdir(parents=True)
        for name in ['inputs.json','training-inputs.json']:
            doc=read(dataset/name)
            for r in doc['records']:
                for key in ['image','mask']:
                    if r[key] is not None:r[key]=str((dataset/r[key]).resolve())
            save(root/'data'/name,doc)
        shutil.copy2(dataset/'observations.npz',root/'data/observations.npz')
        for name in ['prediction-postprocessed.npz','time-camera-inputs.npz','vitpose-keypoints.pt']:
            shutil.copy2(dataset/'inference'/name,root/'inference'/name)
    runtime=run/'runtime'
    targets={'hand-observations.npz':'current/smplx-hand-depth-fresh-011/hand-observations.npz','inspection/floor-reference.json':'current/smplx-observation-support-014/floor-reference.json','inspection/surface-support.npz':'current/smplx-observation-support-014/surface-support.npz','inspection/route-comparison.json':'current/smplx-whole-body-prior-fresh-006/gaussianavatar-comparison/route-comparison.json'}
    for source,dest in targets.items():
        d=runtime/dest;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dataset/source,d)
    save(runtime/'work-status.json',dict(local_training_running=False,quality_accepted=False))
    save(run/'input-audit.json',report);save(run/'config.json',cfg)
    print(json.dumps({'state':'prepared_fresh','run':str(run),**report},ensure_ascii=False))

def configure(cfg,run,arm):
    # Optional local wheel directories are not written into the portable config.
    extra=[p for p in os.environ.get('SHUOLEE_EXTRA_PYTHONPATH','').split(os.pathsep) if p]
    sys.path[:0]=[str(CORE),*extra]
    import shuolee_common as c
    c.LAB=PROJECT/'third_party';c.BASE=run/'runtime';c.ROOT=run/arm
    c.PRIVATE=PROJECT/'assets';c.MODEL=resolve(cfg['smplx_model'])
    def runtime():
        import torch
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('A working NVIDIA CUDA runtime is required.')
        return torch
    c.runtime=runtime
    shim=types.ModuleType('fresh_sequence_pipeline');shim.c=c;shim.REPO=resolve(cfg['gvhmr_maps']);shim.RUN=run/arm
    sys.modules['fresh_sequence_pipeline']=shim
    import shuolee_vposer as vp
    vp.MODEL=resolve(cfg['vposer']);vp.CODE=PROJECT/'third_party/human_body_prior'
    return c

@contextmanager
def gpu_lock():
    lock=PROJECT/'.gpu-task.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError(f'Another task lock exists: {lock}. Check its PID before removing a stale lock.')
    try:
        os.write(fd,json.dumps({'pid':os.getpid(),'command':sys.argv}).encode());os.close(fd)
        yield
    finally:lock.unlink(missing_ok=True)

def fit_review(a,root):
    import numpy as np
    from PIL import Image
    import nvdiffrast.torch as dr
    torch=a.torch;fit=a.load(root/'fit/fitted.npz');rows=read(root/'data/training-inputs.json')['records']
    ctx=dr.RasterizeCudaContext();faces=torch.tensor(fit['faces'],device='cuda',dtype=torch.int32);v=a.G(fit['vertices_world'][0])
    out=root/'fit/review';out.mkdir(exist_ok=True);items=[]
    for i in a.SELECT:
        r=rows[i];H=768;W=round(r['wh'][0]*H/r['wh'][1]);K=a.G(np.diag([W/r['wh'][0],H/r['wh'][1],1])@np.array(r['K']));E=a.G(r['w2c'])
        cam=v@E[:3,:3].T+E[:3,3];z=cam[:,2]
        clip=torch.stack([2*K[0,0]/W*cam[:,0]+(2*K[0,2]/W-1)*z,-2*K[1,1]/H*cam[:,1]+(1-2*K[1,2]/H)*z,1.00002*z-.0200002,z],-1)
        rast,_=dr.rasterize(ctx,clip[None],faces,(H,W));mask=(rast[0,:,:,3]>0).flip(0).cpu().numpy()
        rgb=np.array(Image.open(r['image']).convert('RGB').resize((W,H),Image.Resampling.LANCZOS))
        overlay=rgb.astype(float);overlay[mask]=overlay[mask]*.6+np.array([15,190,210])*.4
        Image.fromarray(rgb).save(out/f'{r["frame"]}_input.png');Image.fromarray(overlay.astype('uint8')).save(out/f'{r["frame"]}_mesh_overlay.png')
        items.append(r['frame'])
    save(root/'fit/visual-review.json',dict(state='manual_review_pending',images=items,usable_as_coarse_static_initialization=False,geometry_accepted=False,source_fit_sha256=sha(root/'fit/fitted.npz')))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['audit','prepare','doctor','fit','review-fit','preflight-gaussians','train','export'])
    p.add_argument('--config',default='configs/hover175.json');p.add_argument('--run');p.add_argument('--arm',choices=ARMS,default='gaussianavatar')
    args=p.parse_args();cfg=read(resolve(args.config))
    assert (cfg['frames'],cfg['initial_points'],cfg['iterations'],cfg['fit_steps'],cfg['seed'],cfg['training_height'])==(175,200000,18000,1200,6060,1536),'This snapshot only supports its frozen protocol.'
    if args.stage=='audit':print(json.dumps(audit(cfg),ensure_ascii=False));return
    if not args.run:p.error('--run is required for this stage')
    run=resolve(args.run)
    if args.stage=='prepare':prepare(cfg,run);return
    if not (run/'config.json').exists():raise RuntimeError('Run prepare first; never point at an old reconstruction.')
    assert read(run/'config.json')==cfg,'Run configuration changed after preparation.'
    root=run/args.arm;c=configure(cfg,run,args.arm)
    if args.stage=='doctor':
        torch=c.runtime()
        import diff_gaussian_rasterization,simple_knn._C,nvdiffrast.torch,pytorch3d,smplx,roma
        import avatar_pair_114 as a
        print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'imports':'passed','training_started':False}));return
    with gpu_lock():
        if args.stage=='fit':
            import fit_avatar_pair_static_114 as fit
            fit.main(root,cfg['fit_steps']);return
        import avatar_pair_114 as a
        a.PAIR=run;a.PUBLIC=run/'reviews'
        if args.stage=='review-fit':fit_review(a,root);return
        if args.stage in ['train','preflight-gaussians']:
            if args.stage=='train':
                review=read(root/'fit/visual-review.json')
                assert review['usable_as_coarse_static_initialization'] is True
                assert review['source_fit_sha256']==sha(root/'fit/fitted.npz'),'Fit changed after its review.'
            a.train(root,args.arm,args.stage=='preflight-gaussians');return
        result=read(root/'gaussian-run/result.json');assert result['steps']==18000 and result['from_scratch']
        native=read(root/'gaussian-run/native-vs-ply.json');assert native['passed'] and native['sha256']==sha(root/'gaussian-run/point_cloud.ply')
        model=a.GaussianModel(3);model.load_ply(str(root/'gaussian-run/point_cloud.ply'));model.representation='shared_static_world'
        model.select_mesh_by_timestep=types.MethodType(lambda self,t:setattr(self,'timestep',int(t)),model)
        model.export_world_ply=types.MethodType(lambda self,path,t:self.save_ply(str(path)),model)
        from compatible_gaussian_export import export_and_verify
        export_and_verify(model,root,18000,99,existing_only=False)
        print('Export parity passed; actual viewer UI and ghosting quality review remain required.')

if __name__=='__main__':main()
