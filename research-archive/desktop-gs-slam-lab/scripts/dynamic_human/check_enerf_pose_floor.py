"""Check wall-fitted camera candidates against independent floor-band source matches."""
from pathlib import Path
import json,cv2,numpy as np
from PIL import Image

LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-matte-background-034';seed=LAB/'outputs/dynamic-human-enerf-background-032';r=json.loads((out/'static-pose-diagnostic.json').read_text());tracks=json.loads((seed/'triangulation.json').read_text())['tracks'];xyz=np.load(seed/'background-init.npz')['points'].astype(np.float64);features=np.load(seed/'features-08.npz');mapping={o['feature']:i for i,t in enumerate(tracks) for o in t['observations'] if o['camera']=='08'}
cv2.setNumThreads(8);cv2.setRNGSeed(78);sift=cv2.SIFT_create(nfeatures=8000);checks=[]
def project(X,w,K):
    camera=X@w[:3,:3].T+w[:3,3];uv=camera@K.T;return uv[:,:2]/uv[:,2:]
for row in r['rows']:
    t=row['frame'];p=np.load(out/f'camera08-frame{t:06d}-static-pnp.npz');before=project(p['X'],p['original_w2c'],p['K']);after=project(p['X'],p['proposed_w2c'],p['K']);check=p['check']
    assert abs(np.median(np.linalg.norm(before[check]-p['y'][check],axis=1))-row['original_check_median_px'])<1e-8
    assert abs(np.median(np.linalg.norm(after[check]-p['y'][check],axis=1))-row['pnp_check_median_px'])<1e-8
    rgb=np.array(Image.open(out/f'camera08-frame{t:06d}-source.png'));m=np.array(Image.open(out/f'camera08-frame{t:06d}-matte.png').convert('L'));mask=cv2.erode((m<3).astype('uint8'),np.ones((11,11),np.uint8));mask[:900]=0;mask[-5:]=0;mask[:,:10]=0;mask[:,-10:]=0
    kp,ds=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask*255);ys=np.array([k.pt for k in kp]);db=features['descriptors'];matcher=cv2.BFMatcher(cv2.NORM_L2);ab=matcher.knnMatch(db,ds,k=2);ba=matcher.knnMatch(ds,db,k=2);rev={a.queryIdx:a.trainIdx for a,b in ba if a.distance<.7*b.distance};pairs=[(mapping[a.queryIdx],a.trainIdx) for a,b in ab if a.queryIdx in mapping and features['xy'][a.queryIdx,1]>=900 and a.distance<.7*b.distance and rev.get(a.trainIdx)==a.queryIdx]
    X=xyz[[i for i,j in pairs]];y=ys[[j for i,j in pairs]];before=np.linalg.norm(project(X,p['original_w2c'],p['K'])-y,axis=1);after=np.linalg.norm(project(X,p['proposed_w2c'],p['K'])-y,axis=1)
    np.savez_compressed(out/f'camera08-frame{t:06d}-floor-check.npz',X=X,y=y,original_errors=before,proposed_errors=after)
    checks.append(dict(frame=t,wall_median_recomputed=True,floor_correspondences=len(pairs),floor_original_median_px=float(np.median(before)) if len(pairs) else None,floor_proposed_median_px=float(np.median(after)) if len(pairs) else None,floor_improved=int((after<before).sum())))
report=dict(status='wall_reprojection_recomputed_and_separate_bottom_band_checked',camera='08',checks=checks,scope='No pose refitting using floor matches. Bottom900:1080 pixels are an image band; inspected source shows paving but the mask is not a semantic floor annotation. Few matches or regression limit transfer beyond wall. No cameras changed.')
(out/'pose-floor-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report,indent=2))
