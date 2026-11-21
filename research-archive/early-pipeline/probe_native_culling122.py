"""T0b: one frozen synthetic diagnosis of UT bounds versus ray-3D response.

This does not modify gsplat, the live native viewer, camera parameters, or T0a.
The low-level path is a diagnostic, not a production renderer replacement.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, math, os, signal, subprocess, sys, time, traceback
import numpy as np

H=Path(__file__).resolve().parent
PREFIX='C:/' if os.name=='nt' else '/mnt/c/'
R=Path(PREFIX+'Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
G=Path(PREFIX+'Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
O=R/'native-training-integration'/'subpixel-culling-diagnosis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def register():
    old=R/'native-training-integration'/'existing-backend-path'/'result.json'
    assert read(old)['state']=='failed'
    assert read(R/'queue-state.json')['running'] is None
    O.mkdir(exist_ok=False)
    local=['probe_native_culling122.py','run_native_culling122.sh','probe_native_training122.py','native_training122.py','native_portrait_renderer122.py']
    backend=['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/cuda/csrc/ProjectionUT3DGSFused.cu','gsplat/cuda/csrc/RasterizeToPixelsFromWorld3DGSFwd.cu']
    save(O/'plan.json',dict(stage='T0b',registered=now(),prior_failure_sha256=sha(old),
        hypothesis='UT bounds multiply opacity by 2D blur compensation even for classic ray-3D response, potentially rejecting subpixel Gaussians before a nonzero 3D response is evaluated.',
        variable='Only the optional opacity argument to fully_fused_projection_with_ut: original opacity versus None. The actual ray-3D rasterizer always receives unchanged original opacity, exactly as classic mode does.',
        fixed='Same T0a 200000-point generator/seed/geometry/scales/opacities/SH, camera pose/K, eps2d=.3, clipping and all CUDA kernels. No resolution or parameter sweep.',
        controls=['Direct original-bounds path must reproduce stock gsplat RGB/alpha and position gradient.',
            'Record visible bounds, compensation*opacity distribution, alpha and gradient counts for both paths.',
            'One fresh Adam update with the registered T0a learning rates after the no-opacity-cull backward; never resume T0a.',
            'One anisotropic subpixel Gaussian centered on an exactly known pixel ray, compared with independent NumPy Mahalanobis ray response at every pixel. Full K and the original off-axis camera are retained.'],
        limits=dict(wall_seconds=180,stock_parity=3e-6,oracle_max_abs=3e-5,minimum_moving_centers=100),
        budget=dict(large_geometry=200000,large_forwards=3,large_backwards=3,optimizer_steps=1,analytic_fixture=1,GPU_jobs=1),
        stopping='Keep all failures. Do not alter physical calibration, old source, live renderer, point scale/opacity or thresholds to obtain a pass.',
        scope='Synthetic implementation diagnosis only. No human training, model promotion, physical calibration or actual-input ghosting attribution.',
        source_hashes={n:sha(H/n) for n in local},backend_hashes={n:sha(G/n) for n in backend}))
    print('T0B_REGISTERED',flush=True)

def low_level(p,E,K,width,height,opacity_bounds):
    import torch
    from gsplat.cuda._wrapper import fully_fused_projection_with_ut,isect_tiles,isect_offset_encode,rasterize_to_pixels_eval3d,spherical_harmonics
    radii,means2d,depths,conics,comp=fully_fused_projection_with_ut(
        p['means'],p['quats'],p['scales'],p['opacities'] if opacity_bounds else None,E,K,width,height,
        eps2d=.3,near_plane=.01,far_plane=1e10,radius_clip=0.,calc_compensations=True,camera_model='pinhole')
    C=len(E);tile=16;tw=math.ceil(width/tile);th=math.ceil(height/tile)
    _,ids,flat=isect_tiles(means2d,radii,depths,tile,tw,th,packed=False)
    offsets=isect_offset_encode(ids,C,tw,th).reshape(C,th,tw)
    center=torch.inverse(E)[:,:3,3]
    directions=p['means'][None]-center[:,None]
    rgb=(spherical_harmonics(3,directions,p['sh'][None].expand(C,-1,-1,-1),masks=(radii>0).all(-1))+.5).clamp_min(0)
    color,alpha=rasterize_to_pixels_eval3d(p['means'],p['quats'],p['scales'],rgb,p['opacities'][None].expand(C,-1),
        E,K,width,height,tile,offsets,flat,camera_model='pinhole')
    return color,alpha,dict(radii=radii,compensations=comp,intersections=len(flat))

def run():
    assert not (O/'progress.json').exists(),'Do not repeat this registered run'
    plan=read(O/'plan.json')
    for n,d in plan['source_hashes'].items():assert sha(H/n)==d,n
    for n,d in plan['backend_hashes'].items():assert sha(G/n)==d,n
    start=time.monotonic();result=dict(stage='T0b',checks=[],physical_admission=False,human_training_started=False)
    def check(name,ok,**data):
        result['checks'].append(dict(name=name,passed=bool(ok),**data))
        if not ok:raise AssertionError(name)
    def stop(*_):raise TimeoutError('T0b 180-second budget')
    signal.signal(signal.SIGALRM,stop);signal.alarm(180)
    save(O/'progress.json',dict(state='running',pid=os.getpid(),updated=now(),command=sys.argv))
    try:
        busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
        if busy:raise RuntimeError('GPU busy: '+busy)
        sys.path.insert(0,str(G))
        import torch,gsplat
        from PIL import Image
        from scipy.spatial.transform import Rotation
        from probe_native_training122 import fixture,pose
        from native_training122 import FreshNativeGaussians
        torch.set_num_threads(2)
        original=read(R/'native-training-integration'/'plan.json');f=original['fixture']
        arrays=fixture(200000,f['axes'],[.004,.005,.003],.1,f['seed'])
        np.savez_compressed(O/'frozen-large-arrays.npz',**arrays)
        E=torch.tensor(pose(-45,3.,.2)[None],dtype=torch.float32,device='cuda')
        K=torch.tensor([[[60.,0.,33.5],[0.,62.,30.5],[0.,0.,1.]]],dtype=torch.float32,device='cuda')
        save(O/'camera.json',dict(w2c=E.cpu().tolist(),K=K.cpu().tolist(),width=64,height=64,pixel_convention='pixel_corners'))
        model=FreshNativeGaussians(arrays,device='cuda');initial=model.means.detach().clone()
        diagnostics={};images={};gradients={}
        for name in ['stock','same_bounds_low_level','no_opacity_bounds']:
            model.zero_grad(set_to_none=True);p=model.parameters_linear()
            if name=='stock':
                color,alpha,meta=gsplat.rasterization(means=p['means'],quats=p['quats'],scales=p['scales'],opacities=p['opacities'],colors=p['sh'],
                    viewmats=E,Ks=K,width=64,height=64,packed=False,with_ut=True,with_eval3d=True,camera_model='pinhole',sh_degree=3,eps2d=.3)
            else:color,alpha,meta=low_level(p,E,K,64,64,name=='same_bounds_low_level')
            objective=color.square().mean()+alpha.square().mean();objective.backward();torch.cuda.synchronize()
            finite=all(q.grad is not None and bool(torch.isfinite(q.grad).all()) for q in model.parameters())
            images[name]=(color.detach().cpu().numpy(),alpha.detach().cpu().numpy())
            gradients[name]=model.means.grad.detach().clone()
            row=dict(visible_gaussians=int((meta['radii']>0).all(-1).sum()),alpha_max=float(alpha.max()),alpha_sum=float(alpha.sum()),
                objective=float(objective.detach()),nonzero_position_gradients=int((model.means.grad.norm(dim=-1)>0).sum()),all_gradients_finite=finite)
            if name=='no_opacity_bounds':
                compensated=(meta['compensations']*p['opacities'][None]).detach().cpu().numpy()
                row.update(compensated_opacity_quantiles=np.quantile(compensated,[0,.5,1]).tolist(),below_kernel_threshold=int((compensated<1/255).sum()),intersections=meta['intersections'])
                optimizer=model.optimizer(original['optimizer']);row['initial_optimizer_states']=len(optimizer.state);optimizer.step()
                row['moving_centers']=int((torch.linalg.norm(model.means-initial,dim=-1)>0).sum())
            diagnostics[name]=row
            Image.fromarray(np.rint(images[name][0][0].clip(0,1)*255).astype('uint8')).save(O/(name+'.png'))
            Image.fromarray(np.rint(images[name][1][0,:,:,0].clip(0,1)*255).astype('uint8')).save(O/(name+'-alpha.png'))
        result['large']=diagnostics
        parity=max(float(np.max(np.abs(a-b))) for a,b in zip(images['stock'],images['same_bounds_low_level']))
        gradient_parity=float((gradients['stock']-gradients['same_bounds_low_level']).abs().max())
        check('Original-bounds low-level path reproduces stock result and gradients',max(parity,gradient_parity)<plan['limits']['stock_parity'],image_max_abs=parity,gradient_max_abs=gradient_parity)
        check('Frozen fixture reproduces all-point stock culling',diagnostics['stock']['visible_gaussians']==0 and diagnostics['stock']['nonzero_position_gradients']==0)
        d=diagnostics['no_opacity_bounds']
        check('Removing only opacity bounds restores actual 3D response and finite geometry update',d['visible_gaussians']==200000 and d['nonzero_position_gradients']>=100 and d['moving_centers']>=100 and d['all_gradients_finite'])
        check('Compensated opacity explains every culled point',d['below_kernel_threshold']==200000)
        # Independent NumPy oracle: one unchanged-size anisotropic Gaussian exactly on a pixel ray.
        ee=E[0].cpu().numpy().astype('float64');kk=K[0].cpu().numpy().astype('float64')
        local=np.linalg.solve(kk,np.array([32.5,31.5,1.]))*3.
        world=ee[:3,:3].T@(local-ee[:3,3])
        one={k:v[:1].copy() for k,v in arrays.items()};one['means'][0]=world;one['sh'][:]=0
        one['sh'][:,0]=(np.array([.3,.6,.8])-.5)/.28209479177387814
        one_model=FreshNativeGaussians(one,device='cuda')
        with torch.no_grad():
            rgb,aa,mm=low_level(one_model.parameters_linear(),E,K,64,64,False)
            oldrgb,oldaa,oldmm=low_level(one_model.parameters_linear(),E,K,64,64,True)
        linear=one_model.numpy_linear();q=linear['quats'][0]
        rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();A=np.diag(1/linear['scales'][0].astype('float64'))@rot.T
        uv=np.stack(np.meshgrid(np.arange(64)+.5,np.arange(64)+.5),axis=-1)
        camera_rays=np.c_[uv.reshape(-1,2),np.ones(4096)]@np.linalg.inv(kk).T
        rays=camera_rays@ee[:3,:3];origin=-ee[:3,:3].T@ee[:3,3]
        gro=A@(origin-linear['means'][0]);grd=rays@A.T;grd/=np.linalg.norm(grd,axis=1,keepdims=True)
        power=-.5*np.sum(np.cross(grd,gro)**2,axis=1)
        oracle=np.minimum(.999,float(linear['opacities'][0])*np.exp(power));oracle[oracle<1/255]=0
        oracle=oracle.reshape(64,64,1);expected_rgb=oracle*np.array([.3,.6,.8])
        difference=max(float(np.abs(aa.cpu().numpy()[0]-oracle).max()),float(np.abs(rgb.cpu().numpy()[0]-expected_rgb).max()))
        result['oracle']=dict(max_abs=difference,expected_alpha_max=float(oracle.max()),native_alpha_max=float(aa.max()),stock_alpha_max=float(oldaa.max()),point=linear['means'][0].tolist(),pixel=[32,31])
        check('Independent anisotropic ray formula confirms response that stock bounds reject',difference<plan['limits']['oracle_max_abs'] and float(aa.max())>.09 and float(oldaa.max())==0,**result['oracle'])
        panels=[]
        for name,im in [('stock-ray',oldrgb.cpu().numpy()[0]),('unculled-ray',rgb.cpu().numpy()[0]),('numpy-oracle',expected_rgb)]:
            image=Image.fromarray(np.rint(im.clip(0,1)*255).astype('uint8'));image.save(O/(name+'.png'));panels.append(image)
        combined=Image.new('RGB',(192,128))
        for col,name in enumerate(['stock','same_bounds_low_level','no_opacity_bounds']):combined.paste(Image.open(O/(name+'.png')),(64*col,0))
        for col,image in enumerate(panels):combined.paste(image,(64*col,64))
        combined.save(O/'culling_comparison.png')
        save(O/'figure-index.json',dict(native_tile=[64,64],rows=['200000-point unchanged fixture: stock / original-bounds primitive path / opacity bounds disabled','single-Gaussian independent control: stock / opacity bounds disabled / NumPy ray oracle'],text_on_image=False,rescaled=False))
        for n,digest in plan['source_hashes'].items():assert sha(H/n)==digest,n
        for n,digest in plan['backend_hashes'].items():assert sha(G/n)==digest,n
        result.update(state='complete_pending_review',all_controls_passed=True,seconds=time.monotonic()-start,updated=now(),runtime=dict(torch=torch.__version__,gsplat=gsplat.__version__,gpu=torch.cuda.get_device_name()))
        save(O/'result.json',result);save(O/'progress.json',dict(state='complete_pending_review',pid=None,updated=now()))
        print(json.dumps(result),flush=True)
    except BaseException as exc:
        result.update(state='failed',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-start,updated=now())
        save(O/'result.json',result);save(O/'progress.json',dict(state='failed',pid=None,updated=now(),error=repr(exc)));raise
    finally:signal.alarm(0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);a=p.parse_args()
    register() if a.action=='register' else run()
