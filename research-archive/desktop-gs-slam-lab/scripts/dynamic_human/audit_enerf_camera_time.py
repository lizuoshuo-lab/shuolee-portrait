"""Train-only temporal-offset diagnostic from fixed cross-camera tracks.

Matches are established at zero offset WITHOUT an epipolar gate. The same
matched target points are tracked to every candidate offset, so the score
cannot improve merely by changing the matched feature subset for each lag.
This tests relative timing consistency; it does not establish physical clocks.
"""
from pathlib import Path
from functools import lru_cache
import argparse,json,time
import cv2,numpy as np
from PIL import Image,ImageDraw
from enerf_native_loader import EnerfNativeDataset
from audit_enerf_input import digest

def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def project_F(c1,c2):
    a=np.array(c1['w2c']);b=np.array(c2['w2c']);R=b[:3,:3]@a[:3,:3].T;t=b[:3,3]-R@a[:3,3]
    cross=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])
    return np.linalg.inv(np.array(c2['K'])).T@cross@R@np.linalg.inv(np.array(c1['K']))
def sampson(x,y,F):
    a=np.c_[x,np.ones(len(x))];b=np.c_[y,np.ones(len(y))];l2=a@F.T;l1=b@F
    return (a*l1).sum(1)/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1))
def flow(a,b,points):
    p=np.float32(points).reshape(-1,1,2)
    q,ok,e=cv2.calcOpticalFlowPyrLK(a,b,p,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    back,ok2,_=cv2.calcOpticalFlowPyrLK(b,a,q,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    valid=(ok.ravel()>0)&(ok2.ravel()>0)&(np.linalg.norm(back-p,axis=2).ravel()<.5)&(e.ravel()<20)
    return q[:,0].astype(float),valid

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);ap.add_argument('--neighbors',type=int,default=1);ap.add_argument('--anchors',type=int,nargs=3,default=[9,29,49]);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'timing-diagnostic.json').exists():raise FileExistsError('Existing completed diagnostic')
    cv2.setNumThreads(8);ds=EnerfNativeDataset(a.manifest,'train',allow_calibration_only=True);lookup={(r['camera'],r['frame']):i for i,r in enumerate(ds.rows)}
    cams=sorted({r['camera'] for r in ds.rows},key=lambda c:ds.cameras[c]['camera_center_world'][0]);pairs=[(cams[i],cams[j]) for i in range(len(cams)) for j in range(i+1,min(len(cams),i+1+a.neighbors))];anchors=a.anchors;offsets=list(range(-3,4))
    protocol=dict(anchor_frames=anchors,fit_anchor_frames=[9,29],check_anchor_frames=[49],camera_pairs=pairs,offsets=offsets,selection='All training cameras ordered by world center X; adjacent pairs, fixed source anchors. No heldout RGB/matte opened.',matches='Mutual SIFT ratio<0.7, foreground matte>0.9. No initial epipolar filtering. Same accepted points in every lag score.',tracking='Native rectified OpenCV pyramidal LK, forward-backward<0.5px and patch error<20 for ALL candidate offsets',motion='Source flow to previous and next frame; mean displacement>1.0px/frame is moving, <0.3 is low-motion',score='Median absolute Sampson distance in native pixels, also all point values saved; no re-estimated F',meaning='Candidate source(c,t) to target(d,t+lag) alignment. Nonzero optimum alone is not proof of clock offset; check motion-point coverage, track identity, and independent anchor49.')
    protocol['selection']=f'All training cameras ordered by world center X; up to {a.neighbors} adjacent camera positions, fixed source anchors. No heldout RGB/matte opened.'
    protocol['fit_anchor_frames']=anchors[:2];protocol['check_anchor_frames']=anchors[2:]
    protocol['meaning']=f'Candidate source(c,t) to target(d,t+lag) alignment. Nonzero optimum alone is not proof of clock offset; check coverage, track identity, and independent anchor{anchors[2]}.'
    source_spec=json.loads(Path(a.manifest).read_text());protocol['dataset_role']=source_spec.get('purpose','original_appearance_training_observations')
    if source_spec.get('appearance_training_permitted') is False:
        protocol['selection']=source_spec['split_scope']+f' Camera graph connects up to {a.neighbors} adjacent positions.'
    save(out/'protocol.json',protocol)
    @lru_cache(maxsize=64)
    def obs(cam,t):return ds[lookup[(cam,t)]]
    @lru_cache(maxsize=96)
    def gray(cam,t):return cv2.cvtColor(obs(cam,t)['rgb'],cv2.COLOR_RGB2GRAY)
    sift=cv2.SIFT_create(nfeatures=6000);features={};matcher=cv2.BFMatcher();reports=[];began=time.time()
    for t in anchors:
        for c in cams:
            o=obs(c,t);kp,de=sift.detectAndCompute(gray(c,t),((o['foreground_matte']>.9)&o['valid_pixels']).astype('uint8')*255)
            xy=np.float64([k.pt for k in kp]);before,okb=flow(gray(c,t),gray(c,t-1),xy);after,oka=flow(gray(c,t),gray(c,t+1),xy)
            speed=(np.linalg.norm(before-xy,axis=1)+np.linalg.norm(after-xy,axis=1))*.5
            tracks={0:xy};valid=okb&oka
            for lag in offsets:
                if lag==0:continue
                points,ok=flow(gray(c,t),gray(c,t+lag),xy);tracks[lag]=points;valid&=ok
            features[(c,t)]=(xy,de,speed,tracks,valid)
            print('FEATURES',c,t,len(xy),'valid',int(valid.sum()),flush=True)
        for c,d in pairs:
            x,dx,speed,_,vx=features[c,t];y,dy,_,tracks,vy=features[d,t]
            forward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dx,dy,k=2) if m.distance<.7*n.distance};backward={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(dy,dx,k=2) if m.distance<.7*n.distance}
            matched=[(i,j) for i,j in forward.items() if backward.get(j)==i and vx[i] and vy[j]]
            if len(matched)<4:continue
            ix,iy=np.array(matched).T;F=project_F(ds.cameras[c],ds.cameras[d]);errors=np.stack([sampson(x[ix],tracks[lag][iy],F) for lag in offsets],1)
            moving=speed[ix]>1.;still=speed[ix]<.3;row=dict(source=c,target=d,anchor=t,mutual_tracked_matches=len(ix),moving_count=int(moving.sum()),low_motion_count=int(still.sum()))
            for label,mask in [('moving',moving),('low_motion',still),('all',np.ones(len(ix),bool))]:
                scores=np.median(np.abs(errors[mask]),axis=0) if mask.any() else None
                row[label]=dict(scores_px=scores.tolist() if scores is not None else None,best_lag=int(offsets[int(scores.argmin())]) if scores is not None else None,zero_px=float(scores[3]) if scores is not None else None,min_px=float(scores.min()) if scores is not None else None)
            np.savez_compressed(out/f'tracks-{c}-{d}-t{t:03d}.npz',source_xy=x[ix],target_xy=np.stack([tracks[lag][iy] for lag in offsets],1),source_speed=speed[ix],signed_sampson=errors,source_feature=ix,target_feature=iy,offsets=offsets)
            reports.append(row);print('PAIR',json.dumps(row),flush=True)
    summary=[]
    for c,d in pairs:
        rows=[r for r in reports if r['source']==c and r['target']==d];fit=[r for r in rows if r['anchor'] in anchors[:2] and r['moving_count']>=12];check=next((r for r in rows if r['anchor']==anchors[2]),None)
        if not fit:summary.append(dict(source=c,target=d,status='insufficient_moving_fit_matches'));continue
        scores=np.mean([r['moving']['scores_px'] for r in fit],0);best=int(np.argmin(scores));summary.append(dict(source=c,target=d,status='train_anchor_candidate_needs_identity_review',fit_anchors=[r['anchor'] for r in fit],selected_lag=offsets[best],fit_zero_px=float(scores[3]),fit_selected_px=float(scores[best]),check_moving_count=check['moving_count'] if check else 0,check_zero_px=check['moving']['zero_px'] if check else None,check_selected_px=check['moving']['scores_px'][best] if check and check['moving']['scores_px'] else None))
    result=dict(status='train_only_fixed_track_relative_time_diagnostic_computed_identity_review_pending',pairs=reports,selected_candidates=summary,source_manifest_sha256=digest(a.manifest),elapsed_seconds=time.time()-began,input_files_verified=len(ds._verified),limits=['LK drift/occlusion may survive numerical thresholds','Texture matches need visual identity review','Low motion has weak timing observability','All offsets here are frame units, physical fps not independently established','No dataset retiming or model fitting performed'])
    result['dataset_role']=protocol['dataset_role']
    if source_spec.get('appearance_training_permitted') is False:result['status']='separate_calibration_only_fixed_track_relative_time_diagnostic_computed_identity_review_pending'
    save(out/'timing-diagnostic.json',result);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
