"""Once-only T7: two native-size synthetic 200k one-step and PLY controls.

No image/camera estimation, human initialization, threshold change or retry.
The wrapper holds the shared GPU flock; this file records every terminal result.
"""
from pathlib import Path
from datetime import datetime, timezone
import gc, hashlib, json, os, signal, sys, time, traceback

H = Path(__file__).resolve().parent
T = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T / 'shuolee/current/robust-portrait-122'
O = R / 'native-capacity'
G = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')

def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
def mount(s):
    s = s.replace('\\', '/')
    return Path('/mnt/'+s[0].lower()+s[2:]) if len(s)>2 and s[1]==':' else Path(s)

def main():
    if len(sys.argv)!=1: raise ValueError('No runtime options or sweep supported')
    if (O/'progress.json').exists() or (O/'result.json').exists():
        raise RuntimeError('T7 already attempted; inspect evidence, never duplicate')
    plan=read(O/'plan.json'); registration=read(O/'execution-registration.json')
    if registration['driver_sha256']!=sha(__file__) or registration['plan_sha256']!=sha(O/'plan.json'):
        raise RuntimeError('Unregistered driver or changed plan')
    if registration['wrapper_sha256']!=sha(H/'run_native_capacity122.sh'):
        raise RuntimeError('Unregistered wrapper')
    if os.environ.get('SHUOLEE_T7_SHARED_LOCK')!='held-by-flock':
        raise RuntimeError('Use the registered shared-lock wrapper')
    start=time.monotonic(); cases=[]; checks=[]; stage='input_hashes'
    def progress(value):
        nonlocal stage
        stage=value
        save(O/'progress.json',dict(state='running',stage=stage,pid=os.getpid(),updated=stamp(),elapsed=time.monotonic()-start))
        print(stage, flush=True)
    def check(name, condition, **details):
        checks.append(dict(name=name,passed=bool(condition),**details))
        if not condition: raise RuntimeError('Control failed: '+name)
    def timeout(signum, frame): raise TimeoutError('Registered 300-second T7 wall budget exhausted')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(plan['limits']['wall_seconds'])
    progress(stage)
    try:
        check('all_frozen_sources_match',all(sha(mount(k))==v for k,v in plan['input_sha256'].items()))
        sys.path.insert(0,str(G))
        import numpy as np
        import torch
        from PIL import Image
        from native_portrait_renderer122_v2 import NativeCameraContractV2, render_native
        from native_training122_v2 import FreshNativeGaussians, export_native_checkpoint
        from native_scene_package122_v2 import load_scene, render_recorded, runtime_record
        check('exact_runtime',torch.__version__=='2.7.1+cu128' and runtime_record()['version']=='1.5.3')
        fixture=R/'native-training-integration/subpixel-culling-diagnosis'
        with np.load(fixture/'frozen-large-arrays.npz') as z: initial={k:z[k] for k in z.files}
        check('full_200k_complete_SH3_fixture',initial['means'].shape==(200000,3) and initial['sh'].shape==(200000,16,3))
        E_np=np.asarray(read(fixture/'camera.json')['w2c'],dtype=np.float32)
        contract=NativeCameraContractV2('pinhole','synthetic_known_camera')
        runtime=runtime_record()
        save(O/'runtime.json',dict(updated=stamp(),backend=runtime,device=torch.cuda.get_device_name(),total_bytes=torch.cuda.get_device_properties(0).total_memory))
        for camera in plan['cameras']:
            cid=camera['id']; target=O/cid; target.mkdir(exist_ok=False)
            progress(cid+':fresh_forward')
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            case_start=time.monotonic()
            t=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32,device='cuda')
            E=t(E_np);K=t([camera['K']]);E_before=E.clone();K_before=K.clone()
            model=FreshNativeGaussians(initial,device='cuda');optimizer=model.optimizer(plan['optimizer'])
            check(cid+':empty_optimizer',len(optimizer.state)==0)
            before=model.means.detach().clone()
            def render():
                p=model.parameters_linear()
                return render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],E,K,camera['width'],camera['height'],contract=contract,sh_degree=3)
            torch.cuda.synchronize(); tick=time.monotonic()
            rgb,alpha,meta=render();torch.cuda.synchronize();forward_seconds=time.monotonic()-tick
            check(cid+':native_shape',list(rgb.shape)==[1,camera['height'],camera['width'],3] and list(alpha.shape)==[1,camera['height'],camera['width'],1])
            check(cid+':finite_nonempty',bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all() and alpha.max()>0 and rgb.abs().max()>0))
            visible=int((meta['radii']>0).all(-1).sum())
            intersections=int(meta['flatten_ids'].numel())
            progress(cid+':backward')
            loss=rgb.square().mean()+alpha.square().mean()
            tick=time.monotonic();loss.backward();torch.cuda.synchronize();backward_seconds=time.monotonic()-tick
            gradients={}
            for name, parameter in model.named_parameters():
                grad=parameter.grad
                finite=grad is not None and bool(torch.isfinite(grad).all())
                nonzero=0 if grad is None else int(torch.count_nonzero(grad))
                gradients[name]=dict(finite=finite,nonzero_elements=nonzero,max_abs=None if grad is None else float(grad.abs().max()))
                check(cid+':gradient_'+name,finite and nonzero>0)
            optimizer.step();torch.cuda.synchronize()
            moved=int((model.means.detach()!=before).any(-1).sum())
            check(cid+':actual_fresh_center_update',moved>0)
            check(cid+':K_and_E_unchanged',torch.equal(K,K_before) and torch.equal(E,E_before))
            loss_value=float(loss.detach())
            del rgb,alpha,meta,loss,before
            optimizer.zero_grad(set_to_none=True)
            progress(cid+':updated_reference_and_export')
            with torch.no_grad():
                rr,aa,mm=render();torch.cuda.synchronize()
                reference_rgb=rr[0].cpu().numpy();reference_alpha=aa[0].cpu().numpy()
            del rr,aa,mm
            record=dict(camera,id=cid,image_name=cid+'.png',w2c=E_np[0].tolist())
            scene_path=export_native_checkpoint(target/'package',model,[record],contract,scope='synthetic-control-only')
            parameters=model.numpy_linear()
            # Keep the peak from update and direct rendering; release training state
            # before independently loading the package. This is the intended export flow.
            training_peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved())
            del model,optimizer,E,K,E_before,K_before,parameter,grad
            gc.collect();torch.cuda.empty_cache()
            progress(cid+':recorded_camera_roundtrip')
            loaded=load_scene(scene_path)
            check(cid+':PLY_full_count_and_SH',len(loaded.parameters['means'])==200000 and loaded.parameters['sh_degree']==3)
            attr_delta={k:float(np.max(np.abs(loaded.parameters[k]-parameters[k]))) for k in ('means','scales','opacities','sh')}
            # The codec uses normalized wxyz quaternions; compare covariance,
            # rather than a quaternion sign, when validating the ellipsoid.
            def covariance(p):
                q=p['quats'].astype(np.float64);q/=np.linalg.norm(q,axis=1,keepdims=True)
                w,x,y,z=q.T
                rot=np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
                              2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
                              2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(-1,3,3)
                return (rot*p['scales'].astype(np.float64)[:,None,:]**2)@rot.transpose(0,2,1)
            covariance_delta=float(np.max(np.abs(covariance(parameters)-covariance(loaded.parameters))))
            actual_rgb,actual_alpha,receipt=render_recorded(loaded,cid,'pinhole')
            delta_rgb=float(np.max(np.abs(actual_rgb-reference_rgb)))
            delta_alpha=float(np.max(np.abs(actual_alpha-reference_alpha)))
            Image.fromarray(np.uint8(np.clip(actual_rgb,0,1)*255+.5)).save(target/'rgb.png')
            Image.fromarray(np.uint8(np.clip(actual_alpha[...,0],0,1)*255+.5)).save(target/'alpha.png')
            np.savez_compressed(target/'native-float-response.npz',rgb=actual_rgb,alpha=actual_alpha)
            row=dict(id=cid,width=camera['width'],height=camera['height'],points=200000,steps=1,loss=loss_value,
                forward_seconds=forward_seconds,backward_seconds=backward_seconds,elapsed=time.monotonic()-case_start,
                visible_gaussians=visible,tile_intersections=intersections,gradients=gradients,moved_centers=moved,
                training_peak=training_peak,whole_case_peak=dict(allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved()),
                attribute_max_abs=attr_delta,covariance_max_abs=covariance_delta,rgb_roundtrip_max_abs=delta_rgb,alpha_roundtrip_max_abs=delta_alpha,
                scene=str(scene_path),scene_sha256=sha(scene_path),receipt=receipt,physical_quality_accepted=False)
            cases.append(row);save(target/'result.json',row)
            check(cid+':native_fullfloat_roundtrip',max(delta_rgb,delta_alpha)<=plan['limits']['roundtrip_max_abs'],rgb=delta_rgb,alpha=delta_alpha)
            check(cid+':complete_covariance_and_SH',covariance_delta<1e-8 and attr_delta['sh']<1e-7)
            del loaded,parameters,reference_rgb,reference_alpha,actual_rgb,actual_alpha
            gc.collect();torch.cuda.empty_cache()
        check('sources_unchanged_after_run',all(sha(mount(k))==v for k,v in plan['input_sha256'].items()))
        result=dict(state='complete_pending_review',updated=stamp(),elapsed=time.monotonic()-start,cases=cases,checks=checks,
                    renderer=contract.manifest(),physical_training_admitted=False,person_training=False,
                    scope=plan['scope'],source_hashes_unchanged=True)
        save(O/'result.json',result)
        save(O/'progress.json',dict(state='complete',pid=None,updated=stamp(),elapsed=time.monotonic()-start))
        print(json.dumps(dict(state=result['state'],cases=len(cases),checks=len(checks),elapsed=result['elapsed'])),flush=True)
    except BaseException as exc:
        failure=dict(state='failed',updated=stamp(),stage=stage,elapsed=time.monotonic()-start,error=repr(exc),traceback=traceback.format_exc(),cases=cases,checks=checks,
                     physical_training_admitted=False,person_training=False,no_resolution_or_count_fallback=True)
        save(O/'result.json',failure);save(O/'progress.json',dict(state='failed',pid=None,updated=stamp(),stage=stage,error=repr(exc)))
        raise
    finally: signal.alarm(0)

if __name__=='__main__':main()
