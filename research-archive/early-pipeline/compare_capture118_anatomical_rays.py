"""Same frozen anatomical labels, two camera candidates; no camera or body fitting."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import cv2,numpy as np,pycolmap
from scipy.optimize import least_squares
B=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
S=B.parent/'shuolee-new-capture-118-support'
parser=argparse.ArgumentParser()
parser.add_argument('--candidate-root',default=str(B.parent/'shuolee-new-capture-118-fixedk-opencv-observationrepair'))
C=Path(parser.parse_args().candidate_root)
O=C/'anatomical-ray-review'
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def stats(a):
    a=np.asarray(a);return dict(n=int(a.size),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if a.size else dict(n=0)
def main():
    O.mkdir(exist_ok=False)
    rows=read(S/'data/training-inputs.json')['records'];obs=np.load(S/'data/observations.npz');weights=np.load(S/'fit/observation-weights.npz')
    previous=read(S/'alignment/nose-foot-calibration-audit/result.json')
    raw=pycolmap.Reconstruction(str(B/'sfm/models/0'));cam=next(iter(raw.cameras.values()));f,cx,cy,k=cam.params
    K0=np.array([[f,0,cx-.5],[0,f,cy-.5],[0,0,1.]]);D0=np.array([k,0,0,0,0.]);E0=np.array([r['w2c'] for r in rows])
    candidate=read(C/'candidate-cameras.json')['records'];K1=np.array(candidate[0]['K_integer_centers']);E1=np.array([r['w2c'] for r in candidate]);D1=np.zeros(5)
    uv_rect=obs['keypoints'];Ks=np.array([r['K'] for r in rows]);uv_raw=np.empty_like(uv_rect,dtype=float)
    for i in range(len(rows)):
        q=np.c_[(uv_rect[i,:,0]+.5-Ks[i,0,2])/Ks[i,0,0],(uv_rect[i,:,1]+.5-Ks[i,1,2])/Ks[i,1,1],np.ones(uv_rect.shape[1])]
        uv_raw[i]=cv2.projectPoints(q,np.zeros(3),np.zeros(3),K0,D0)[0].reshape(-1,2)
    selections=[dict(name=t['name'],joint=40+t['face51_index'],ids=t['view_ids']) for t in previous['triangulation']]
    for j,name in enumerate(['left_big_toe','left_small_toe','left_heel','right_big_toe','right_small_toe','right_heel']):
        selections.append(dict(name=name,joint=17+j,ids=np.flatnonzero(weights['feet'][:,j]>.25).tolist()))
    plan=dict(created=datetime.now(timezone.utc).isoformat(),hypothesis='Same detected anatomical points should intersect a shared static 3D point under correct cameras, but detector identity, occlusion and micro-motion remain competing explanations.',
        view_selection='Frozen prior-review face view IDs and feet weights; not selected by new-camera residual. Face IDs were conditional on the old template normals, not independent ground truth.',
        fit_holdout_split='Alternating ordered available views, unchanged for both cameras.',
        pixel_conversion='Existing rectified integer-centre observations -> known baseline SIMPLE_RADIAL raw integer-centre pixels; no image warp and no reinterpretation of FISHEYE_8.',
        same_source_observations=True,no_camera_or_body_optimization=True,selection=selections)
    save(O/'plan.json',plan)
    results=[]
    for item in selections:
        ids=np.array(item['ids'],int);a=ids[::2];b=ids[1::2];u=uv_raw[:,item['joint']]
        rec=dict(**item,fit_views=a.tolist(),held_views=b.tolist(),methods={})
        if len(a)<3 or len(b)<2:rec['state']='insufficient_support';results.append(rec);continue
        for label,K,D,E in [('baseline',K0,D0,E0),('fixed_factory_K',K1,D1,E1)]:
            rays=cv2.undistortPoints(u[a].reshape(-1,1,2),K,D).reshape(-1,2);A=[]
            for i,v in zip(a,rays):A.extend([v[0]*E[i,2]-E[i,0],v[1]*E[i,2]-E[i,1]])
            _,_,vh=np.linalg.svd(np.asarray(A));X=vh[-1,:3]/vh[-1,3]
            def project(X,ii):
                q=np.einsum('nij,j->ni',E[ii,:3,:3],X)+E[ii,:3,3]
                return cv2.projectPoints(q,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
            solved=least_squares(lambda x:(project(x,a)-u[a]).ravel(),X,loss='soft_l1',f_scale=2.,max_nfev=100)
            pred=project(solved.x,ids);err=np.linalg.norm(pred-u[ids],axis=1)
            depth=(np.einsum('nij,j->ni',E[ids,:3,:3],solved.x)+E[ids,:3,3])[:,2]
            rec['methods'][label]=dict(held_error=stats(err[1::2]),fit_error=stats(err[::2]),all_depths_positive=bool((depth>0).all()),solver_success=bool(solved.success),predicted_raw_uv=pred.tolist())
        selected=np.unique(np.linspace(0,len(ids)-1,min(6,len(ids))).astype(int));tiles=[];shown=[]
        for z in selected:
            i=ids[z];rawim=cv2.imread(str(B/'raw'/rows[i]['name']));x,y=np.rint(u[i]).astype(int)
            # Crops beyond the image are explicitly padded; they are not evidence of foot visibility.
            pad=cv2.copyMakeBorder(rawim,256,256,256,256,cv2.BORDER_CONSTANT,value=(255,255,255));x0=x+160;y0=y+160
            if x0<0 or y0<0 or x0+192>pad.shape[1] or y0+192>pad.shape[0]:continue
            crop=pad[y0:y0+192,x0:x0+192].copy();panels=[crop.copy(),crop.copy(),crop.copy()]
            for p,label in zip(panels[1:],['baseline','fixed_factory_K']):
                uv=np.array(rec['methods'][label]['predicted_raw_uv'][z]);centre=np.array([96,96])+uv-u[i]
                cv2.drawMarker(p,(96,96),(0,255,0),cv2.MARKER_CROSS,11,1)
                if np.isfinite(centre).all() and np.max(abs(centre))<10000:cv2.drawMarker(p,tuple(np.rint(centre).astype(int)),(0,0,255),cv2.MARKER_TILTED_CROSS,11,1)
            tiles.append(np.hstack(panels));shown.append(rows[i]['frame'])
        if tiles:
            filename=item['name']+'_raw_projection.png';cv2.imwrite(str(O/filename),np.vstack(tiles));rec.update(review_image=filename,review_frames=shown,columns=['原像素','上一轮相机','固定内参候选'])
        rec['state']='diagnostic_complete';results.append(rec)
    out=dict(created=datetime.now(timezone.utc).isoformat(),landmarks=results,physical_identity_verified=False,training_admitted=False,quality_accepted=False)
    save(O/'result.json',out)
    print(json.dumps([{k:v for k,v in t.items() if k in ['name','state']}|{'errors':{n:m['held_error'] for n,m in t['methods'].items()}} for t in results],indent=2))
if __name__=='__main__':main()
