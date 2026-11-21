"""Frozen24 native patches: texture information and one known subpixel control."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,signal,time,traceback
import numpy as np
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'video10-correspondence-contrast/head-roi-contrast';OUT=D/'native-patch-control'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def local(s):return Path('/mnt/'+s[0].lower()+s[2:].replace('\\','/')) if len(s)>1 and s[1]==':' else Path(s)
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');t.replace(p)
def gray(path):return np.asarray(Image.open(path).convert('RGB'),dtype=np.float64)@np.array([.299,.587,.114])/255
def make_spline(im):return RectBivariateSpline(np.arange(im.shape[0]),np.arange(im.shape[1]),im,kx=3,ky=3,s=0)
def crop_context(im,uv):
    center=np.floor(np.array(uv)+.5).astype(int);x,y=center-48
    assert x>=0 and y>=0 and x+97<=im.shape[1] and y+97<=im.shape[0]
    return im[y:y+97,x:x+97].copy(),np.array(uv)-np.array([x,y]),[int(x),int(y),int(x+97),int(y+97)]

class TranslationPhotometry:
    """T(x+dx,y+dy) = gain*S(x,y)+bias. Affine brightness, translation geometry."""
    def __init__(self,source,target,source_center,target_center):
        grid=np.arange(-15,16,dtype=np.float64);xx,yy=np.meshgrid(grid,grid)
        self.x=xx.ravel()+target_center[0];self.y=yy.ravel()+target_center[1]
        self.s=source.ev(yy.ravel()+source_center[1],xx.ravel()+source_center[0]);self.target=target
    def fun(self,p):return self.target.ev(self.y+p[1],self.x+p[0])-p[2]*self.s-p[3]
    def jac(self,p):
        x=self.x+p[0];y=self.y+p[1]
        return np.column_stack([self.target.ev(y,x,dx=0,dy=1),self.target.ev(y,x,dx=1,dy=0),-self.s,-np.ones_like(self.s)])
    def fit(self):return least_squares(self.fun,[0.,0.,1.,0.],jac=self.jac,bounds=([-3,-3,.25,-.5],[3,3,4,.5]),max_nfev=30,ftol=1e-9,xtol=1e-9,gtol=1e-9)

def stats(v):
    a=np.asarray(v,float);return dict(n=len(a),minimum=float(a.min()),median=float(np.median(a)),mean=float(a.mean()),maximum=float(a.max()))

def main():
    assert not OUT.exists(),'The one registered24 patch attempt already exists'
    plan=read(D/'native-patch-control-plan.json');p=read(D/'plan.json');alloc=read(D/'review-allocation.json')['rows']
    assert sha(D/'review-allocation.json')==plan['queries_sha256'] and len(alloc)==24
    images={}
    for f,c in p['crops'].items():
        path=local(c['source']);assert sha(path)==p['source_sha256'][c['source']]
        images[int(f)]=gray(path)
    OUT.mkdir();started=time.monotonic()
    admission=dict(updated=now(),plan_sha256=sha(D/'native-patch-control-plan.json'),source_sha256=sha(Path(__file__)),query_sha256=sha(D/'review-allocation.json'),
        parameters='dx,dy,gain,bias. Cubic interpolation,31x31 active samples,97x97 native context. Unclipped float target, no quantization; gamma-encoded grayscale, not a camera noise model.',
        bounds=[[-3,-3,.25,-.5],[3,3,4,.5]],initial=[0,0,1,0],max_nfev=30,
        control_acceptance='Every patch finite, optimizer success, translation Euclidean error<0.1px. This numerical synthetic criterion is not an image/camera admission.',
        finite_difference='4columns on rows0,8,16,centralstep1e-5,maxabsolute error<1e-6.',real_observations_changed=False)
    save(OUT/'execution-admission.json',admission)
    def state(s,**kw):save(OUT/'progress.json',dict(updated=now(),state=s,pid=os.getpid(),seconds=time.monotonic()-started,**kw))
    def alarm(*a):raise TimeoutError('Registered60second CPU control budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(60);rng=np.random.default_rng(12313);rows=[];arrays={};fds=[]
    try:
        for i,r in enumerate(alloc):
            state('control',row=i);context,center,box=crop_context(images[r['frame']],r['uv']);src=make_spline(context)
            yy,xx=np.mgrid[:97,:97];truth=np.array([.35,-.40,1.05,2/255]);target=truth[2]*src.ev((yy-truth[1]).ravel(),(xx-truth[0]).ravel()).reshape(97,97)+truth[3]
            target+=rng.normal(0,.5/255,target.shape);matcher=TranslationPhotometry(src,make_spline(target),center,center)
            if i in [0,8,16]:
                pt=np.array([0.,0.,1.,0.]);j=matcher.jac(pt);err=[]
                for k in range(4):
                    delta=np.zeros(4);delta[k]=1e-5;fd=(matcher.fun(pt+delta)-matcher.fun(pt-delta))/(2e-5);err.append(float(np.max(np.abs(fd-j[:,k]))))
                assert max(err)<1e-6;fds.append(dict(row=i,max_abs_by_column=err))
            j0=matcher.jac([0,0,1,0]);gx=-src.ev(matcher.y,matcher.x,dx=0,dy=1);gy=-src.ev(matcher.y,matcher.x,dx=1,dy=0)
            jt=np.column_stack([gx,gy]);tensor=jt.T@jt;ev=np.linalg.eigvalsh(tensor)
            nuisance=np.column_stack([matcher.s,np.ones_like(matcher.s)]);projected=jt-nuisance@np.linalg.lstsq(nuisance,jt,rcond=None)[0];schur=projected.T@projected;se=np.linalg.eigvalsh(schur)
            opt=matcher.fit();error=float(np.linalg.norm(opt.x[:2]-truth[:2]));cov=np.linalg.pinv(schur)*(.5/255)**2
            row=dict(row=i,frame=r['frame'],feature=r['feature'],family=r['family'],split=r['split'],uv=r['uv'],context_box=box,
                brightness_std=float(matcher.s.std()),gradient_tensor_eigenvalues=ev.tolist(),brightness_marginalized_eigenvalues=se.tolist(),
                anisotropy=float(ev[1]/max(ev[0],1e-16)),conditional_noise_sigma_px=np.sqrt(np.maximum(np.diag(cov),0)).tolist(),
                estimate=opt.x.tolist(),truth=truth.tolist(),translation_error_px=error,optimizer_success=bool(opt.success),status=int(opt.status),nfev=int(opt.nfev),
                initial_RMS=float(np.sqrt(np.mean(matcher.fun([0,0,1,0])**2))),final_RMS=float(np.sqrt(np.mean(opt.fun**2))),
                passed=bool(opt.success and np.isfinite(opt.x).all() and error<.1))
            rows.append(row);arrays[f'source_{i:02d}']=context;arrays[f'target_{i:02d}']=target
        np.savez_compressed(OUT/'synthetic-inputs.npz',**arrays);save(OUT/'finite-difference.json',fds)
        result=dict(updated=now(),state='complete',seconds=time.monotonic()-started,passed=all(x['passed'] for x in rows),passed_controls=sum(x['passed'] for x in rows),total=24,
            errors=stats([x['translation_error_px'] for x in rows]),rows=rows,
            synthetic_only=True,real_correspondences_refined=False,physical_subpixel_accuracy_verified=False,cameras_changed=False,training_started=False,
            limitation='Known translated copies with affine brightness and fixed independent noise only. No cross-view occlusion, material ambiguity, nonrigid motion, specularity or camera calibration truth tested. Conditional texture sigma assumes an ideal noise model and is not real uncertainty.')
        save(OUT/'result.json',result);state('complete');v=read(OUT/'progress.json');v['pid']=None;save(OUT/'progress.json',v)
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    except BaseException as ex:
        save(OUT/'failure.json',dict(updated=now(),error=repr(ex),traceback=traceback.format_exc()));state('failed');v=read(OUT/'progress.json');v['pid']=None;save(OUT/'progress.json',v);raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
