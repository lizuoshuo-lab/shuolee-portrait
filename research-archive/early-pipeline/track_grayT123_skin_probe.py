"""One preregistered native skin-mark LK track; no camera/geometry access."""
from pathlib import Path
from datetime import datetime,timezone
import json,time,hashlib
import cv2
import numpy as np

REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
OUT=REC/'grayT-skin-texture-probe'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grayT')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    started=time.monotonic();target=OUT/'tracking-result.json'
    if target.exists():raise RuntimeError('Tracking has already run; no retry.')
    plan=read(OUT/'plan.json');assert plan['status']=='seed_verified_before_first_tracking'
    rows=read(ROOT/'data/training-inputs.json')['records'];frame_numbers=plan['native_frame_numbers'];inds=[f-1 for f in frame_numbers]
    seed=53;xy=np.array(plan['seed']['xy'],np.float32).reshape(1,1,2)
    assert frame_numbers==list(range(40,81)) and seed in inds
    cache={};positions={seed:xy};metrics={seed:dict(status=True,reverse_status=True,forward_backward_px=0.,seed=True)}
    def frame(i):
        if i not in cache:
            p=Path(rows[i]['image']);cache[i]=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
            if cache[i] is None:raise RuntimeError('Missing image:'+str(p))
            assert cache[i].shape==(2704,1520)
        return cache[i]
    params=dict(winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01),
        flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,minEigThreshold=.0001)
    cv2.setNumThreads(2)
    for direction in [-1,1]:
        i=seed
        while i+direction in inds:
            j=i+direction
            nxt,status,eig=cv2.calcOpticalFlowPyrLK(frame(i),frame(j),positions[i],None,**params)
            if nxt is None or not np.all(np.isfinite(nxt)):
                raise RuntimeError(f'No finite LK observation at frame{j+1}; preserve failure and stop.')
            back,rstatus,reig=cv2.calcOpticalFlowPyrLK(frame(j),frame(i),nxt,None,**params)
            fb=float(np.linalg.norm(back-positions[i])) if back is not None else 1e10
            positions[j]=nxt;metrics[j]=dict(from_frame=i+1,status=bool(status[0,0]),reverse_status=bool(rstatus[0,0]),
                forward_backward_px=fb,min_eigenvalue=float(eig[0,0]),reverse_min_eigenvalue=float(reig[0,0]),seed=False)
            i=j
            if time.monotonic()-started>120:raise RuntimeError('CPU budget exceeded; no retry.')
    entries=[];rawcanvas=np.full((6*96,8*96,3),245,np.uint8);marked=rawcanvas.copy()
    for n,i in enumerate(inds):
        p=positions[i].reshape(2).astype(float);cx,cy=np.round(p).astype(int);crop=[int(cx-48),int(cy-48),int(cx+48),int(cy+48)]
        original=cv2.imread(rows[i]['image']);assert original is not None
        tile=original[crop[1]:crop[3],crop[0]:crop[2]].copy();assert tile.shape==(96,96,3)
        y=(n//8)*96;x=(n%8)*96;rawcanvas[y:y+96,x:x+96]=tile
        uv=p-np.array(crop[:2]);center=tuple(np.round(uv*256).astype(int))
        cv2.circle(tile,center,4*256,(255,255,0),1,cv2.LINE_AA,8)
        marked[y:y+96,x:x+96]=tile
        cv2.imwrite(str(OUT/f'skin_{i+1:03d}_native.png'),original[crop[1]:crop[3],crop[0]:crop[2]])
        entries.append(dict(frame=rows[i]['frame'],frame_number=i+1,source_frame_index=rows[i]['source_frame_index'],
            source_sha256=rows[i]['source_sha256'],xy=p.tolist(),crop=crop,metrics=metrics[i],
            role='ANCHOR' if i+1 in plan['anchor_frame_numbers'] else 'HELD'))
    cv2.imwrite(str(OUT/'all41_native.png'),rawcanvas);cv2.imwrite(str(OUT/'all41_locator.png'),marked)
    vals=[x['metrics'] for x in entries if not x['metrics']['seed']]
    passed=all(m['status'] and m['reverse_status'] and m['forward_backward_px']<.5 for m in vals)
    result=dict(created=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started,tracking_runs=1,
        method='Native LK, fixed parameters; no camera projection used.',algorithmic_checks_passed=passed,
        max_forward_backward_px=max(m['forward_backward_px'] for m in vals),
        min_eigenvalue=min(m['min_eigenvalue'] for m in vals),observations=entries,
        actual_material_identity_review_pending=True,geometry_not_computed=True,quality_accepted=False)
    save(target,result)
    save(OUT/'index.json',dict(subject='grayT',images=[dict(frame='frame_000040..frame_000080',region='皮肤色素点：原生96px裁片按时间排序，每行8幅',
        file='all41_locator.png',raw='all41_native.png',columns=['same point candidate; cyan ring'],observations=entries)],actual_review_pending=True))
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}))

if __name__=='__main__':
    try:main()
    except Exception as ex:
        save(OUT/'tracking-failure.json',dict(created=datetime.now(timezone.utc).isoformat(),error=repr(ex),no_retry=True))
        raise
