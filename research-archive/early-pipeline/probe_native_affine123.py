from pathlib import Path
import numpy as np
import signal,time,json,os
from native_patch_control123 import R,read,save,sha,local,now,gray,crop_context,make_spline,stats
from native_affine_patch123 import AffinePhotometry
D=R/'video10-correspondence-contrast/head-roi-contrast';O=D/'native-affine-contrast'
def main():
    assert not (O/'control-result.json').exists() and not (O/'control-progress.json').exists()
    plan=read(O/'plan.json');p=read(D/'plan.json');alloc=read(D/'review-allocation.json')['rows']
    for s,h in plan['source_sha256'].items():assert sha(local(s))==h,s
    images={int(f):gray(local(c['source'])) for f,c in p['crops'].items()};rng=np.random.default_rng(12313);rows=[];fdchecks=[];started=time.monotonic()
    save(O/'control-progress.json',dict(updated=now(),state='running',pid=os.getpid(),source_sha256=sha(Path(__file__))))
    def alarm(*a):raise TimeoutError('60second synthetic control budget')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(60)
    try:
        for i,r in enumerate(alloc):
            si,center,_=crop_context(images[r['frame']],r['uv']);src=make_spline(si);yy,xx=np.mgrid[:97,:97];A=np.array([[1.03,.02],[-.015,.98]]);shift=np.array([.35,-.40]);truth=np.array([.35,-.40,.03,.02,-.015,-.02,1.05,2/255])
            back=(np.stack([xx,yy],-1).reshape(-1,2)-center-shift)@np.linalg.inv(A).T+center
            target=(1.05*src.ev(back[:,1],back[:,0])+2/255).reshape(97,97)+rng.normal(0,.5/255,(97,97));m=AffinePhotometry(src,make_spline(target),center,center)
            if i in [0,8,16]:
                p0=np.array([0,0,0,0,0,0,1,0.]);j=m.jac(p0);errs=[]
                for k in range(8):
                    dt=np.zeros(8);dt[k]=1e-6;fd=(m.fun(p0+dt)-m.fun(p0-dt))/2e-6;errs.append(float(np.max(np.abs(j[:,k]-fd))))
                assert max(errs)<1e-6;fdchecks.append(dict(row=i,column_max_abs=errs))
            opt=m.fit();centererr=float(np.linalg.norm(opt.x[:2]-shift));corners=np.array([[-15,-15],[-15,15],[15,-15],[15,15]]);dA=np.array([[opt.x[2]-.03,opt.x[3]-.02],[opt.x[4]+.015,opt.x[5]+.02]])
            cornererr=float(np.linalg.norm(corners@dA.T+(opt.x[:2]-shift),axis=-1).max());passed=bool(opt.success and centererr<.1 and cornererr<.5)
            rows.append(dict(row=i,frame=r['frame'],family=r['family'],split=r['split'],center_error_px=centererr,corner_error_px=cornererr,estimate=opt.x.tolist(),truth=truth.tolist(),success=bool(opt.success),nfev=int(opt.nfev),passed=passed))
        result=dict(updated=now(),state='complete',seconds=time.monotonic()-started,passed=all(x['passed'] for x in rows),passed_controls=sum(x['passed'] for x in rows),rows=rows,center_error=stats([x['center_error_px'] for x in rows]),corner_error=stats([x['corner_error_px'] for x in rows]),synthetic_only=True,physical_accuracy=False)
        save(O/'control-result.json',result);save(O/'control-finite-difference.json',fdchecks);save(O/'control-progress.json',dict(updated=now(),state='complete',pid=None))
        print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    finally:signal.alarm(0)
if __name__=='__main__':main()
