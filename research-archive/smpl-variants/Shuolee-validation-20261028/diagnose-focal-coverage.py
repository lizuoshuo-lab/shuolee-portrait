"""Compare fixed, evenly spaced temporal anchors on the saved new-video FIT tracks."""
from pathlib import Path
import json
import cv2
import numpy as np
from scipy.optimize import minimize_scalar

root=Path('/opt/gs-slam-lab/smpl-x-validation-20261028/runs/portrait-32be30636adabfc4')
xy=[np.load(p)['xy'] for p in sorted((root/'camera').glob('features-*.npz'))]
records=json.loads((root/'camera/appearance-tracks.json').read_text())['records']
train=[dict(r['nodes']) for r in records if not r['held']]
wh=json.loads((root/'decoded.json').read_text())['records'][0]['wh']
low,high=np.log([.35*max(wh),5*max(wh)])
def matrix(logf):
    f=np.exp(logf)
    return np.array([[f,0,(wh[0]-1)/2],[0,f,(wh[1]-1)/2],[0,0,1.]])
def cost(F,logf):
    K=matrix(logf)
    s=np.linalg.svd(K.T@F@K,compute_uv=False)
    return ((s[0]-s[1])/max(s[0]+s[1],1e-15))**2
cv2.setRNGSeed(119)
rows=[];fundamentals=[]
for i in np.linspace(0,len(xy)-17,6).astype(int):
    for hop in (1,2,4,8,16):
        j=int(i+hop)
        shared=[t for t in train if i in t and j in t]
        if len(shared)<160: continue
        a=np.asarray([xy[i][t[i]] for t in shared]);b=np.asarray([xy[j][t[j]] for t in shared])
        F,inliers=cv2.findFundamentalMat(a,b,cv2.FM_RANSAC,1.5,.999,10000)
        if F is None or F.shape!=(3,3) or inliers is None or int(inliers.sum())<80:continue
        result=minimize_scalar(lambda x:cost(F,x),bounds=(low,high),method='bounded',options={'maxiter':80,'xatol':1e-6})
        fundamentals.append(F)
        rows.append(dict(pair=[int(i),j],matches=len(shared),inliers=int(inliers.sum()),focal=float(np.exp(result.x)),cost=float(result.fun)))
samples=np.linspace(low,high,81)
curves={name:[float(aggregate([cost(F,x) for F in fundamentals])) for x in samples] for name,aggregate in [('mean',np.mean),('median',np.median)]}
report=dict(scope='CPU diagnosis of temporal coverage only; no camera candidate or training',pairs=rows,aggregate={key:dict(grid_best_focal=float(np.exp(samples[np.argmin(vals)])),grid_cost=float(min(vals))) for key,vals in curves.items()},held_tracks_used=False)
Path('/mnt/d/Projects/Shuolee-validation-20261028/focal-coverage-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
