"""Registered R3: one 200k update over two full-size residual cameras."""
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, os, signal, time, traceback

H = Path(__file__).resolve().parent
R = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
O = R / 'native-residual-capacity'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def mount(p):
    p=p.replace('\\','/')
    return Path('/mnt/'+p[0].lower()+p[2:]) if len(p)>2 and p[1]==':' else Path(p)

def main():
    if (O/'progress.json').exists() or (O/'result.json').exists(): raise RuntimeError('R3 already attempted; no duplicate')
    plan=read(O/'plan.json');reg=read(O/'execution-registration.json')
    if sha(O/'plan.json')!=reg['plan_sha256'] or any(sha(mount(p))!=d for p,d in reg['sources'].items()):
        raise RuntimeError('Registered source changed')
    if os.environ.get('SHUOLEE_R3_LOCK')!='held-by-flock': raise RuntimeError('Shared GPU lock required')
    started=time.monotonic();checks=[];stage='preflight'
    def progress(s):
        nonlocal stage
        stage=s;save(O/'progress.json',dict(state='running',stage=s,pid=os.getpid(),updated=now(),elapsed=time.monotonic()-started));print(s,flush=True)
    def check(name,v,**detail):
        checks.append(dict(name=name,passed=bool(v),**detail));save(O/'checks.json',checks)
        if not v:raise AssertionError(name)
    def timeout(*args):raise TimeoutError('R3 registered wall budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['limits']['wall_seconds'])
    progress(stage)
    try:
        import numpy as np
        import torch
        from PIL import Image
        import native_linear_archive122 as archive
        from native_portrait_renderer122_v3 import render_cameras
        from native_residual_camera122 import ResidualRenderContract
        from native_scene_package122_v5 import load_scene,render_recorded,runtime_record
        from native_training122_v4 import FreshNativeGaussians,create_view,accumulate_update,export_native_checkpoint
        from native_gaussian_ply122 import read_standard_ply
        check('Pinned CUDA runtime',torch.__version__=='2.7.1+cu128' and runtime_record()['version']=='1.5.3')
        fixture=R/'native-training-integration/subpixel-culling-diagnosis/frozen-large-arrays.npz'
        with np.load(fixture,allow_pickle=False) as z:initial={k:z[k] for k in z.files}
        check('Frozen 200k complete SH3 initialization',initial['means'].shape==(200000,3) and initial['sh'].shape==(200000,16,3))
        contract=ResidualRenderContract('synthetic_known_camera');cameras=plan['cameras']
        def render(parameters,c):
            p={n:torch.as_tensor(v,dtype=torch.float32,device='cuda') for n,v in parameters.items()}
            with torch.no_grad():rgb,alpha,meta=render_cameras(p,[c],contract=contract,sh_degree=3)
            torch.cuda.synchronize()
            if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():raise ValueError('Nonfinite render')
            return rgb[0].cpu().numpy(),alpha[0].cpu().numpy()
        torch.cuda.reset_peak_memory_stats();target={k:v.copy() for k,v in initial.items()}
        target['means'][:,0]+=.002;target['sh'][:,0]+=.025
        views=[]
        for c in cameras:
            progress(c['id']+':known_target')
            rgb,alpha=render(target,c);h,w=alpha.shape[:2]
            check(c['id']+':full native nonempty target',rgb.shape==(c['height'],c['width'],3) and float(alpha.max())>0)
            valid=np.ones_like(alpha,dtype=bool);valid[h//2-3:h//2+3,w//2-3:w//2+3]=False
            views.append(create_view(c,np.clip(rgb,0,1),alpha,(alpha>.05)&valid,valid,device='cuda'))
        del target,rgb,alpha
        model=FreshNativeGaussians(initial,device='cuda');opt=model.optimizer(plan['optimizer'])
        check('Fresh empty optimizer',len(opt.state)==0)
        before=model.means.detach().clone();progress('two_full_native_views_one_update');tick=time.monotonic()
        result=accumulate_update(model,views,contract,opt);torch.cuda.synchronize();update_seconds=time.monotonic()-tick
        gradients={n:dict(finite=bool(torch.isfinite(p.grad).all()),nonzero=int(torch.count_nonzero(p.grad))) for n,p in model.named_parameters()}
        for n,v in gradients.items():check('Gradient '+n,v['finite'] and v['nonzero']>0)
        moved=int((model.means.detach()!=before).any(-1).sum())
        check('Actual free center update',moved>0,moved=moved)
        check('Full frozen K E D unchanged',[v.camera for v in views]==cameras)
        parameters=model.numpy_linear();archive.write(O/'preexport-linear.npz',parameters)
        progress('preexport_full_float_references')
        for c in cameras:
            rgb,alpha=render(parameters,c)
            np.savez_compressed(O/(c['id']+'-preexport-float.npz'),rgb=rgb,alpha=alpha)
            Image.fromarray(np.uint8(np.clip(rgb,0,1)*255+.5)).save(O/(c['id']+'-rgb.png'))
            Image.fromarray(np.uint8(np.clip(alpha[...,0],0,1)*255+.5)).save(O/(c['id']+'-alpha.png'))
        progress('exact_package_export')
        path=export_native_checkpoint(O/'native-package',model,views,contract,scope='synthetic-control-only')
        peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
        del model,opt,views,before,rgb,alpha
        gc.collect();torch.cuda.empty_cache()
        loaded=load_scene(path)
        check('All 200k native arrays exact',all(np.array_equal(parameters[k],loaded.parameters[k]) for k in archive.FIELDS))
        check('All K E D records exact',list(loaded.cameras.values())==cameras)
        preview=read_standard_ply(path.parent/'point_cloud.ply')
        check('Standard preview complete SH3 finite',preview['sh'].shape==(200000,16,3) and all(np.isfinite(preview[n]).all() for n in archive.FIELDS))
        def covariance(p):
            q=p['quats'].astype(float);q/=np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T
            rot=np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(-1,3,3)
            return (rot*p['scales'].astype(float)[:,None,:]**2)@rot.transpose(0,2,1)
        covmax=float(np.max(np.abs(covariance(parameters)-covariance(preview))))
        check('Preview covariance and SH codec',covmax<1e-8 and np.array_equal(parameters['sh'],preview['sh']),covariance_max_abs=covmax)
        comparisons=[]
        for c in cameras:
            progress(c['id']+':independent_package_replay')
            with np.load(O/(c['id']+'-preexport-float.npz')) as z:ref={k:z[k] for k in z.files}
            rgb,alpha,receipt=render_recorded(loaded,c['id'])
            error=dict(rgb=float(np.max(np.abs(rgb-ref['rgb']))),alpha=float(np.max(np.abs(alpha-ref['alpha']))))
            check(c['id']+':native response replay',max(error.values())<=plan['limits']['native_float_max'],**error)
            comparisons.append(dict(camera_id=c['id'],width=c['width'],height=c['height'],max_abs=error,receipt=receipt))
        check('All registered sources preserved',all(sha(mount(p))==d for p,d in reg['sources'].items()))
        save(O/'result.json',dict(state='completed_pending_review',updated=now(),elapsed=time.monotonic()-started,
            checks=checks,points=200000,optimizer_steps=1,view_exposures=2,moved_centers=moved,update_seconds=update_seconds,
            gradients=gradients,per_view=result['per_view'],training_peak=peak,comparisons=comparisons,
            covariance_max_abs=covmax,native_package=str(path),device=torch.cuda.get_device_name(),
            synthetic_only=True,physical_quality_accepted=False,person_training=False,
            preview_image_equivalence=False,adaptive_density_tested=False,full_18k_training_tested=False))
        save(O/'progress.json',dict(state='completed',updated=now(),pid=None))
        print(json.dumps(dict(state='completed',checks=len(checks),elapsed=time.monotonic()-started,update_seconds=update_seconds)),flush=True)
    except BaseException as e:
        save(O/'result.json',dict(state='failed',stage=stage,error=repr(e),traceback=traceback.format_exc(),checks=checks,elapsed=time.monotonic()-started,
             person_training=False,physical_quality_accepted=False,no_point_or_resolution_fallback=True))
        save(O/'progress.json',dict(state='failed',updated=now(),pid=None,stage=stage,error=repr(e)))
        raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
