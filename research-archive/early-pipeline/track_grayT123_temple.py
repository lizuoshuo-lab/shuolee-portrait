"""One registered native LK track of M1; no access to cameras or head transforms."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib,os
import cv2
import numpy as np
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'grayT-temple-validation-review';ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    started=time.monotonic()
    if (OUT/'tracking-started.json').exists():raise RuntimeError('Tracking already started; inspect result, no automatic retry.')
    plan=read(OUT/'plan.json');review=read(OUT/'seed-review.json');assert review['actual_seed_review_passed'] and review['plan_sha256']==sha(OUT/'plan.json')
    rows=read(ROOT/'data/training-inputs.json')['records'];frames=plan['frames'];inds=[i-1 for i in frames]
    assert frames==list(range(40,81));seed=53;xy=np.array(plan['seed']['native_xy'],np.float32).reshape(1,1,2)
    save(OUT/'tracking-started.json',dict(created=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),code_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'plan.json')))
    positions={seed:xy};metrics={seed:dict(seed=True,status=True,reverse_status=True,forward_backward_px=0)};cache={}
    def frame(i):
        if i not in cache:
            path=Path(rows[i]['image']);assert sha(path)==rows[i]['source_sha256'];cache[i]=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            assert cache[i] is not None and cache[i].shape==(2704,1520)
        return cache[i]
    cv2.setNumThreads(2);cv2.ocl.setUseOpenCL(False)
    params=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,minEigThreshold=.0001)
    for direction in [-1,1]:
        i=seed
        while i+direction in inds:
            j=i+direction;nxt,status,eig=cv2.calcOpticalFlowPyrLK(frame(i),frame(j),positions[i],None,**params)
            if nxt is None or not np.isfinite(nxt).all():raise RuntimeError(f'No finite track at{j+1}')
            back,reverse,_=cv2.calcOpticalFlowPyrLK(frame(j),frame(i),nxt,None,**params)
            if back is None:raise RuntimeError(f'No reverse track at{j+1}')
            positions[j]=nxt;metrics[j]=dict(from_frame=i+1,seed=False,status=bool(status[0,0]),reverse_status=bool(reverse[0,0]),
                forward_backward_px=float(np.linalg.norm(back-positions[i])),minimum_eigenvalue=float(eig[0,0]))
            i=j
            if time.monotonic()-started>120:raise RuntimeError('120-second tracking budget exhausted; no repeat.')
    obs=[];raw=np.full((6*96,8*96,3),245,np.uint8);marked=raw.copy()
    for n,i in enumerate(inds):
        p=positions[i].reshape(2).astype(float);cx,cy=map(int,np.rint(p));crop=[cx-48,cy-48,cx+48,cy+48]
        original=cv2.imread(rows[i]['image']);tile=original[crop[1]:crop[3],crop[0]:crop[2]].copy();assert tile.shape==(96,96,3)
        raw[(n//8)*96:(n//8+1)*96,(n%8)*96:(n%8+1)*96]=tile;cv2.imwrite(str(OUT/f'frame_{i+1:03d}_native.png'),tile)
        uv=p-crop[:2];cv2.circle(tile,tuple(np.rint(uv*256).astype(int)),4*256,(255,255,0),1,cv2.LINE_AA,8)
        marked[(n//8)*96:(n//8+1)*96,(n%8)*96:(n%8+1)*96]=tile
        obs.append(dict(frame=rows[i]['frame'],frame_number=i+1,xy=p.tolist(),crop=crop,metrics=metrics[i],
            role='ANCHOR' if i+1 in plan['anchors'] else 'HELD',source_sha256=rows[i]['source_sha256']))
    cv2.imwrite(str(OUT/'tracking_all41_native.png'),raw);cv2.imwrite(str(OUT/'tracking_all41_locator.png'),marked)
    m=[o['metrics'] for o in obs if not o['metrics']['seed']];passed=all(o['status'] and o['reverse_status'] and o['forward_backward_px']<.5 for o in m)
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,observations=obs,
        tracking_runs=1,algorithmic_checks_passed=passed,max_forward_backward_px=max(o['forward_backward_px'] for o in m),
        no_geometry_access=True,physical_accuracy_certified=False,quality_accepted=False)
    save(OUT/'tracking-result.json',result);print(json.dumps({k:v for k,v in result.items() if k!='observations'}))
if __name__=='__main__':
    try:main()
    except Exception as e:save(OUT/'tracking-failure.json',dict(error=repr(e),quality_accepted=False));raise
