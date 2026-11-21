"""One small synthetic check of installed gsplat distortion and gradients.

No human images/cameras, no optimizer, no coefficient estimation or new weights.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,time,os,sys,traceback,signal,hashlib
import numpy as np
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/gsplat-distortion-research')
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def run():
    if (O/'probe-progress.json').exists():raise FileExistsError('Existing attempt; do not repeat')
    plan=json.loads((O/'probe-plan.json').read_text(encoding='utf-8'))
    for p,h in plan['source_hashes'].items():
        if sha(p)!=h:raise RuntimeError('Registered source changed: '+p)
    started=time.monotonic();checks=[]
    save(O/'probe-progress.json',dict(state='running',pid=os.getpid(),updated=now()))
    def check(name,value,**detail):
        checks.append(dict(name=name,passed=bool(value),**detail));save(O/'probe-checks.json',checks)
        if not value:raise AssertionError(name)
    def timeout(*_):raise TimeoutError('120s budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    try:
        import torch,cv2,gsplat
        from gsplat import rasterization
        from PIL import Image
        check('Pinned installed-source release',gsplat.__version__=='1.5.3',version=gsplat.__version__,path=gsplat.__file__)
        # Explicit OpenCV packing: radial(k1,k2,k3,k4,k5,k6), tangential(p1,p2), prism(s1..s4).
        K=np.array([[55.,0.,43.2],[0.,57.,30.1],[0.,0.,1.]])
        r=np.array([.07,-.015,.002,.003,0.,0.]);t=np.array([.002,-.001]);s=np.array([.0005,-.0002,.0003,.0001])
        xyz=np.array([[-.55,-.30,2.],[.45,-.25,2.1],[-.35,.30,2.],[.55,.35,2.2],[0.,0.,1.8],[.1,.2,1.8]])
        cvd=np.array([r[0],r[1],t[0],t[1],r[2],r[3],r[4],r[5],*s])
        uv=cv2.projectPoints(xyz,np.zeros(3),np.zeros(3),K,cvd)[0][:,0,:]
        q=xyz[:,:2]/xyz[:,2:];x,y=q.T;r2=x*x+y*y
        radial=(1+r[0]*r2+r[1]*r2**2+r[2]*r2**3)/(1+r[3]*r2+r[4]*r2**2+r[5]*r2**3)
        pred=np.stack([x*radial+2*t[0]*x*y+t[1]*(r2+2*x*x)+s[0]*r2+s[1]*r2**2,
                       y*radial+t[0]*(r2+2*y*y)+2*t[1]*x*y+s[2]*r2+s[3]*r2**2],-1)
        pred=pred*np.array([K[0,0],K[1,1]])+K[:2,2]
        err=float(np.max(np.abs(uv-pred)))
        check('Documented coefficient order matches independent OpenCV',err<1e-10,max_abs_px=err)
        def tensor(x,grad=False):return torch.tensor(x,dtype=torch.float32,device='cuda',requires_grad=grad)
        means=tensor(xyz,True);quats=tensor(np.tile([1.,0,0,0],(6,1)),True)
        scales=tensor(np.tile([.09,.13,.07],(6,1)),True);opacity=tensor(np.full(6,.7),True)
        colors=tensor([[1,.1,.1],[.1,1,.1],[.1,.1,1],[1,1,.1],[1,.2,1],[.1,1,1]],True)
        Ks=tensor(K[None],True);E=tensor(np.eye(4)[None]);rr=tensor(r[None],True);tt=tensor(t[None],True);ss=tensor(s[None],True)
        kw=dict(packed=False,with_ut=True,with_eval3d=True,camera_model='pinhole',radial_coeffs=rr,tangential_coeffs=tt,thin_prism_coeffs=ss)
        rgb,a,meta=rasterization(means,quats,scales,opacity,colors,E,Ks,96,64,**kw)
        check('Brown6 plus tangential/prism renders finite nonempty image',torch.isfinite(rgb).all() and torch.isfinite(a).all() and a.max()>.05,
              alpha_max=float(a.max().detach()))
        weight=1+torch.arange(96,device='cuda')[None,None,:,None]/96
        loss=(rgb*weight).sum()+.2*a.sum();loss.backward();torch.cuda.synchronize()
        gaussians=dict(means=means,quats=quats,scales=scales,opacity=opacity,colors=colors)
        check('Gaussian geometry and appearance gradients are finite',all(p.grad is not None and torch.isfinite(p.grad).all() for p in gaussians.values()),
              gradient_abs_sums={n:float(p.grad.abs().sum()) for n,p in gaussians.items()})
        no_grad={n:p.grad is None for n,p in dict(K=Ks,radial=rr,tangential=tt,thin_prism=ss).items()}
        check('Installed ray3D branch does not backpropagate K or distortion',all(no_grad.values()),gradient_is_none=no_grad)
        with torch.no_grad():
            base,ba,_=rasterization(means,quats,scales,opacity,colors,E,Ks,96,64,packed=False,with_ut=True,with_eval3d=True,camera_model='pinhole')
        delta=float((rgb.detach()-base).abs().max())
        check('Nonzero coefficients change rendered pixels',delta>1e-5,max_abs_rgb_difference=delta)
        for name,im in [('brown-distorted',rgb[0]),('no-distortion',base[0])]:
            Image.fromarray((im.detach().clamp(0,1).cpu().numpy()*255+.5).astype(np.uint8)).save(O/(name+'.png'))
        np.savez_compressed(O/'synthetic-response.npz',rgb=rgb.detach().cpu().numpy(),alpha=a.detach().cpu().numpy(),
                            baseline_rgb=base.cpu().numpy(),baseline_alpha=ba.cpu().numpy(),K=K,radial=r,tangential=t,thin_prism=s)
        # These are documented unsupported camera gradients, not an optimizer trial.
        EE=E.detach().clone().requires_grad_(True)
        try:
            er,ea,_=rasterization(means,quats,scales,opacity,colors,EE,Ks,96,64,**kw)
            (er.sum()+ea.sum()).backward()
        except NotImplementedError:pose_unsupported=True
        else:pose_unsupported=False
        check('Installed eval3d extrinsic backward explicitly unsupported',pose_unsupported)
        # The user's current local adapter remains unchanged: it does not expose this Brown branch.
        from native_portrait_renderer122_v2 import render_native,NativeCameraContractV2
        try:render_native(means,quats,scales,opacity,colors,E,Ks.detach(),96,64,
                          contract=NativeCameraContractV2('pinhole','rectified'),radial_coeffs=rr.detach())
        except ValueError as e:rejected='Rectified pinhole accepts no distortion' in str(e)
        else:rejected=False
        check('Current local adapter rejects residual Brown coefficients',rejected)
        result=dict(state='completed',updated=now(),seconds=time.monotonic()-started,checks=checks,
            synthetic_only=True,optimizer_steps=0,human_training_started=False,estimated_calibration=False,
            main_branch_installed=False,person_quality_accepted=False)
        save(O/'probe-result.json',result);save(O/'probe-progress.json',dict(state='completed',updated=now(),pid=os.getpid()))
        print(json.dumps(result))
    except Exception as e:
        save(O/'probe-failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),checks=checks))
        save(O/'probe-progress.json',dict(state='failed',updated=now(),pid=os.getpid()));raise
    finally:signal.alarm(0)
if __name__=='__main__':run()
