"""Read-only comparison on the frozen pre-SfM raw-image holdout."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
import cv2
import numpy as np
import pycolmap

B=Path('/opt/gs-slam-lab/shuolee-new-capture-118')
parser=argparse.ArgumentParser()
parser.add_argument('--candidate-root',default='/opt/gs-slam-lab/shuolee-new-capture-118-fixedk-opencv-seedrepair')
C=Path(parser.parse_args().candidate_root)
O=C/'evaluation'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,o):p.write_text(json.dumps(o,indent=2,ensure_ascii=False)+'\n')
def stats(a):
    a=np.asarray(a,float)
    return dict(n=int(a.size),mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),maximum=float(a.max())) if a.size else dict(n=0)
def main():
    O.mkdir(exist_ok=False)
    tracks=read(B/'sfm/holdout/split.json')['tracks']
    annotations={t['track_id']:t for t in read(B/'data/alignment-diagnostics/annotated-tracks.json')['tracks']}
    frames=read(B/'decoded.json')['records'];names=[r['frame'] for r in frames];lookup={s:i for i,s in enumerate(names)}
    sfm=read(B/'data/sfm-inputs.json')['records'];bm={r['frame']:r for r in sfm}
    raw=pycolmap.Reconstruction(str(B/'sfm/models/0'));cam=next(iter(raw.cameras.values()))
    assert len(cam.params)==4 and (cam.width,cam.height)==(1600,899)
    f,cx,cy,k=cam.params;K0=np.array([[f,0,cx-.5],[0,f,cy-.5],[0,0,1.]])
    E0=np.array([bm[n]['w2c'] for n in names]);D0=np.array([k,0,0,0,0.])
    candidate=read(C/'candidate-cameras.json')['records'];assert [r['frame'] for r in candidate]==names
    K1=np.array(candidate[0]['K_integer_centers']);E1=np.array([r['w2c'] for r in candidate])
    D1=np.zeros(5)
    save(O/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),
      purpose='Same frozen pre-camera-solve tracks and first/last triangulation frames, evaluated in original decoded integer-centre pixels. No re-fit to held points.',
      tracks=len(tracks),min_parallax_degrees=1.0,no_residual_rejection=True,
      regions='Existing automatic annotations, not new physical identity truth.',
      camera_comparison='Original SIMPLE_RADIAL baseline vs fixed factory-K/center-crop pinhole hypothesis; this changes both camera model and pose solver, so it cannot isolate a single physical cause.',
      gates=['All 89 poses finite/rigid','At least 50 fitting observations in every view','Fitting mean error <=2px','No held face/body/background median or P90 degradation','No loss of held observability','Solver converged within predeclared budget'],
      physical_calibration_and_subject_staticity_verified=False,
      hashes={str(p):sha(p) for p in [B/'sfm/holdout/split.json',B/'data/alignment-diagnostics/annotated-tracks.json',C/'candidate-cameras.json',C/'solver-output.npz']}))
    methods={'baseline':(K0,D0,E0),'fixed_factory_K':(K1,D1,E1)}
    all_results={};measurements={}
    for method,(K,D,Es) in methods.items():
        results=[];byregion={}
        for t in tracks:
            region=annotations.get(t['track_id'],{}).get('region','unknown')
            obs=t['observations'];om={o['frame']:o for o in obs};seeds=t['triangulation_frames']
            record=dict(track_id=t['track_id'],region=region,seed_frames=seeds,observations=[],observable=False)
            rays=[]
            for name in seeds:
                uv=np.array(om[name]['raw_colmap_xy'],float)-.5
                rays.append(cv2.undistortPoints(uv.reshape(1,1,2),K,D).reshape(2))
            i,j=map(lookup.get,seeds);x=cv2.triangulatePoints(Es[i,:3],Es[j,:3],rays[0].reshape(2,1),rays[1].reshape(2,1)).ravel()
            if abs(x[3])<1e-12:record['reason']='point_at_infinity';results.append(record);continue
            X=x[:3]/x[3];centres=[-Es[z,:3,:3].T@Es[z,:3,3] for z in (i,j)]
            a,b=X-centres[0],X-centres[1];angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b)))
            depths=[(Es[z,:3,:3]@X+Es[z,:3,3])[2] for z in (i,j)]
            record['parallax_degrees']=float(angle)
            if not np.isfinite(X).all() or min(depths)<=0 or angle<1.:
                record['reason']='negative_or_nonfinite_depth' if not np.isfinite(X).all() or min(depths)<=0 else 'parallax_below_1_degree'
                results.append(record);continue
            record['observable']=True
            for o in obs:
                idx=lookup[o['frame']];E=Es[idx];uv=np.array(o['raw_colmap_xy'])-.5
                pred=cv2.projectPoints(X.reshape(1,3),cv2.Rodrigues(E[:3,:3])[0],E[:3,3],K,D)[0].reshape(2)
                held=o['frame'] not in seeds;error=float(np.linalg.norm(pred-uv));depth=float((E[:3,:3]@X+E[:3,3])[2])
                record['observations'].append(dict(frame=o['frame'],held=held,uv=uv.tolist(),pred=pred.tolist(),error_px=error,depth=depth,observation_id=o['observation_id']))
                if held:byregion.setdefault(region,[]).append(error)
            results.append(record)
        region_results={r:dict(tracks=sum(t['region']==r for t in results),observable_tracks=sum(t['region']==r and t['observable'] for t in results),held_error=stats(byregion.get(r,[]))) for r in sorted({t['region'] for t in results})}
        all_results[method]=dict(regions=region_results,observable_tracks=sum(t['observable'] for t in results),total_tracks=len(results),unobservable=[dict(track_id=t['track_id'],reason=t.get('reason')) for t in results if not t['observable']])
        measurements[method]=results
    # Paired IDs prevent apparent improvement through a change in observable subset.
    common={};baseline={t['track_id']:t for t in measurements['baseline']};trial={t['track_id']:t for t in measurements['fixed_factory_K']}
    for region in sorted({t['region'] for t in baseline.values()}):
        e0=[];e1=[];ids=[]
        for tid,t in baseline.items():
            v=trial[tid]
            if t['region']!=region or not t['observable'] or not v['observable']:continue
            obs1={o['observation_id']:o for o in v['observations']}
            for o in t['observations']:
                if o['held']:e0.append(o['error_px']);e1.append(obs1[o['observation_id']]['error_px']);ids.append(o['observation_id'])
        common[region]=dict(baseline=stats(e0),fixed_factory_K=stats(e1),same_observation_ids=True,observations=len(ids))
    sol=np.load(C/'solver-output.npz');Es=sol['w2c'];X=sol['points'];ci=sol['obs_cam'];pi=sol['obs_point']
    q=np.einsum('nij,nj->ni',Es[ci,:3,:3],X[pi])+Es[ci,:3,3]
    errors=sol['errors'];perview=[dict(frame=names[i],observations=int((ci==i).sum()),error=stats(errors[ci==i]),behind_camera=int((q[ci==i,2]<=0).sum())) for i in range(len(names))]
    fit=stats(errors);fit['negative_depth_observations']=int((q[:,2]<=0).sum())
    solved=read(C/'solver-result.json')
    gates=dict(poses_finite_rigid=bool(np.isfinite(Es).all() and np.max(abs(Es[:,:3,:3]@Es[:,:3,:3].transpose(0,2,1)-np.eye(3)))<1e-6),
       minimum_50_observations_per_view=min(r['observations'] for r in perview)>=50,
       fitting_mean_below_2px=fit['mean']<=2,
       solver_converged=solved['success'],
       no_observability_loss=all_results['fixed_factory_K']['observable_tracks']>=all_results['baseline']['observable_tracks'],
       all_required_held_regions_supported=all(common.get(r,{}).get('observations',0)>0 for r in ['face','body','static_scene_candidate']))
    for r in ['face','body','static_scene_candidate']:
        z=common.get(r,{});gates[r+'_not_degraded']=bool(z.get('observations',0)>0 and z['fixed_factory_K']['median']<=z['baseline']['median'] and z['fixed_factory_K']['p90']<=z['baseline']['p90'])
    result=dict(state='candidate_evaluated',methods=all_results,paired_held_comparison=common,fitting_error=fit,gates=gates,
       passed=all(gates.values()),training_admitted=False,quality_accepted=False,new_gaussian_training_started=False,
       conclusion='Measured gates evaluated; no physical calibration or subject-staticity truth inferred.',
       limitations=['D8 formula/export mapping unverified','Appearance tracks may contain wrong materials','Only 8 automatic face ROI tracks; not sufficient nose/mouth coverage','Changed lens model and pose solver together; not a single-cause attribution'])
    save(O/'result.json',result);save(O/'held-measurements.json',measurements);save(O/'fit-per-camera.json',perview)
    # Original images and native-size, unwarped patches; overlay symbols only, no text burned in.
    reviews=O/'review';reviews.mkdir(exist_ok=True)
    selected=[t['track_id'] for t in measurements['baseline'] if t['region']=='face']
    selected += [t['track_id'] for t in measurements['baseline'] if t['region']=='body'][:4]
    selected += [t['track_id'] for t in measurements['baseline'] if t['region']=='static_scene_candidate'][:4]
    index=[]
    for k,tid in enumerate(selected):
        t=baseline[tid];v=trial[tid]
        if not t['observable'] or not v['observable']:continue
        vm={o['frame']:o for o in v['observations']};ob=[o for o in t['observations'] if o['held']]
        if not ob:continue
        chosen=[ob[i] for i in sorted(set([0,len(ob)//2,len(ob)-1]))]
        tiles=[]
        for o in chosen:
            raw=cv2.imread(str(B/'raw'/f"{o['frame']}.png"));assert raw is not None
            x,y=np.round(o['uv']).astype(int);pad=cv2.copyMakeBorder(raw,96,96,96,96,cv2.BORDER_CONSTANT,value=(255,255,255))
            crop=pad[y:y+192,x:x+192].copy();panels=[crop.copy(),crop.copy(),crop.copy()]
            for panel,measurement in zip(panels[1:],[o,vm[o['frame']]]):
                centre=np.array([96.,96.])+np.array(measurement['pred'])-np.array(o['uv'])
                cv2.drawMarker(panel,(96,96),(0,255,0),cv2.MARKER_CROSS,11,1)
                if np.max(abs(centre))<10000:cv2.drawMarker(panel,tuple(np.round(centre).astype(int)),(0,0,255),cv2.MARKER_TILTED_CROSS,11,1)
            tiles.append(np.hstack(panels))
        name=f"{k+1:02d}_{t['region']}_{tid[-8:]}.png";cv2.imwrite(str(reviews/name),np.vstack(tiles))
        index.append(dict(file=name,track_id=tid,region=t['region'],frames=[o['frame'] for o in chosen],columns=['Original raw patch','Baseline: green observation, red prediction','Fixed K OpenCV: green observation, red prediction'],pixels_per_tile=192,warped=False))
    save(O/'review-index.json',index)
    print(json.dumps(dict(paired_held_comparison=common,fitting=fit,gates=gates,passed=result['passed']),indent=2))
if __name__=='__main__':main()
