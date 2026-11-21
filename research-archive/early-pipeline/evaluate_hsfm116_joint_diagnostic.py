"""CPU hold-middle-view evaluation of frozen real-image material tracks."""
from pathlib import Path
import datetime as dt,hashlib,json
import cv2
import numpy as np
from PIL import Image,ImageDraw
B=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
H=B/'current/portrait-alignment-open-source-116/hsfm';J=H/'joint-diagnostic-reserved';O=J/'native-fit-cached';E=O/'evaluation'
LABELS=['background','face_lower','face_upper','torso','legs_shoes']
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,v):(E/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def evaluate(frames,xy,K,c2w,ids):
    index={f:i for i,f in enumerate(ids)};w2c=np.linalg.inv(c2w);P=K@w2c[:,:3,:]
    errors=[];pred=[];valid=[];end=[]
    for fs,ps in zip(frames,xy):
        ii=[index[int(f)] for f in fs]
        X=cv2.triangulatePoints(P[ii[0]],P[ii[2]],ps[0].reshape(2,1),ps[2].reshape(2,1))[:,0]
        ok=np.isfinite(X).all() and abs(X[3])>1e-12
        if ok:
            X=X/X[3];z=(w2c[ii]@X)[:,2];pr=P[ii]@X;uv=pr[:,:2]/pr[:,2:]
            ok=np.isfinite(uv).all() and (z>0).all()
        if not ok:errors.append(np.nan);pred.append([np.nan,np.nan]);end.append(np.nan);valid.append(False);continue
        errors.append(np.linalg.norm(uv[1]-ps[1]));pred.append(uv[1]);end.append(np.linalg.norm(uv[[0,2]]-ps[[0,2]],axis=-1).mean());valid.append(True)
    return np.array(errors),np.array(pred),np.array(valid),np.array(end)
def summary(err,valid,groups):
    out={}
    for g,n in enumerate(LABELS):
        ix=groups==g;a=err[ix & valid]
        out[n]=dict(count=int(ix.sum()),invalid=int((ix&~valid).sum()),median=float(np.median(a)) if len(a) else None,p90=float(np.quantile(a,.9)) if len(a) else None,
                    mean=float(np.mean(a)) if len(a) else None,max=float(np.max(a)) if len(a) else None)
    return out
def main():
    assert not E.exists(),'Do not overwrite fixed acceptance results'
    assert read(O/'result.json')['state']=='paired_native_fit_complete_evaluation_pending'
    E.mkdir();val=np.load(J/'frozen-validation.npz');frames,xy,groups=[val[k] for k in ['frames','xy','groups']]
    assert sha(J/'frozen-validation.npz')==read(J/'frozen-validation.json')['sha256']
    rows=read(H/'input-pilot/inputs.json')['records'];ids=[r['raw_frame'] for r in rows];rr={r['raw_frame']:r for r in rows}
    old={r['source_frame_index']:r for r in read(B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json')['records']}
    K=np.array([old[f]['K'] for f in ids]);K[:,:2,2]-=.5
    cams={'original074':(K,np.linalg.inv(np.array([old[f]['w2c'] for f in ids])))}
    for arm in ['scene_only','scene_human']:
        c=np.load(O/f'{arm}-cameras.npz');assert np.array_equal(c['raw_frame_ids'],ids);cams[arm]=(c['K_centers'],c['c2w'])
    results={};arrays={};allbg=np.load(J/'background-tracks.npz');bgs={}
    for arm,(K,c2w) in cams.items():
        error,pred,valid,endpoint=evaluate(frames,xy,K,c2w,ids);results[arm]=summary(error,valid,groups)
        arrays[arm]=dict(errors=error,projected=pred,valid=valid,endpoint_fit_errors=endpoint)
        np.savez_compressed(E/f'{arm}-frozen-errors.npz',**arrays[arm])
        be,bp,bv,bd=evaluate(allbg['frames'],allbg['xy'],K,c2w,ids)
        bgs[arm]=dict(all142_count=len(be),invalid=int((~bv).sum()),median=float(np.nanmedian(be)),p90=float(np.nanquantile(be,.9)))
        np.savez_compressed(E/f'{arm}-all-background-errors.npz',errors=be,projected=bp,valid=bv)
    base=results['original074'];treat=results['scene_human'];checks={}
    checks['all_frozen_tracks_positive_depth_and_finite']=all(treat[n]['invalid']==0 for n in LABELS)
    checks['lower_face_median_improves20pct']=treat['face_lower']['median']<=.8*base['face_lower']['median']
    checks['lower_face_p90_not_worse']=treat['face_lower']['p90']<=base['face_lower']['p90']
    checks['upper_face_p90_not_worse']=treat['face_upper']['p90']<=base['face_upper']['p90']
    checks['background_p90_within_plus0p2']=treat['background']['p90']<=base['background']['p90']+.2
    for n in ['torso','legs_shoes']:checks[n+'_p90_within_plus0p5']=treat[n]['p90']<=base[n]['p90']+.5
    comparisons={}
    for name in LABELS:
        b=base[name];sc=results['scene_only'][name];hc=treat[name]
        comparisons[name]=dict(joint_minus074_median=hc['median']-b['median'],joint_minus_scene_median=hc['median']-sc['median'],
                              joint_minus074_p90=hc['p90']-b['p90'],joint_minus_scene_p90=hc['p90']-sc['p90'])
    # Identical source crops and aspect ratio in all three columns. No image warping.
    obs=np.load(H/'input-pilot/observations.npz');oi={int(f):i for i,f in enumerate(obs['raw_frame_ids'])}
    for f in [50,75,100,125,150]:
        r=rr[f];image=Image.open(r['image']).convert('RGB')
        for part in ['face','whole']:
            if part=='face':
                p=obs['keypoints'][oi[f],23:91];lo=p.min(0);hi=p.max(0);lo-=np.array([45,70]);hi+=np.array([45,45]);box=np.round(np.r_[lo,hi]).astype(int)
                track_ids=np.flatnonzero((frames[:,1]==f)&np.isin(groups,[1,2]))
                scale=min(360/(box[2]-box[0]),450/(box[3]-box[1]))
            else:
                box=np.round(np.array(r['bbox_xyxy'])+np.array([-55,-65,55,40])).astype(int);track_ids=np.flatnonzero((frames[:,1]==f)&(groups>0));scale=650/(box[3]-box[1])
            size=tuple(np.maximum(1,np.round((box[2:]-box[:2])*scale).astype(int)));crop=image.crop(tuple(box)).resize(size,Image.Resampling.LANCZOS)
            panel=Image.new('RGB',(size[0]*3,size[1]+45),'white');d=ImageDraw.Draw(panel)
            for col,arm in enumerate(cams):
                panel.paste(crop,(col*size[0],45));d.text((col*size[0]+5,5),f'raw{f:06d} | {arm}',fill='black')
                d.text((col*size[0]+5,23),'cyan: observed / red: projected',fill='black')
                for ti in track_ids:
                    p=(xy[ti,1]-box[:2])*scale+[col*size[0],45]
                    d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),outline='#00b6ff',width=2)
                    if not arrays[arm]['valid'][ti]:continue
                    q=(arrays[arm]['projected'][ti]-box[:2])*scale+[col*size[0],45]
                    # Clip diagnostic lines to this column; never draw across adjacent comparisons.
                    if col*size[0]<=q[0]<(col+1)*size[0] and 45<=q[1]<panel.height:
                        d.line([tuple(p),tuple(q)],fill='yellow',width=1);d.ellipse((q[0]-3,q[1]-3,q[0]+3,q[1]+3),outline='red',width=2)
            panel.save(E/f'raw{f:06d}-{part}-held-projections.jpg',quality=96)
    result=dict(updated=now(),state='numerical_evaluation_complete_visual_review_pending',frames=13,required_final_frames=175,
        coordinate_unit='Pixels at original rectified1514x2694 resolution; same source and no crop-dependent K changes',
        protocol='One shared3Dpoint per frozen material track: fit using endpoint images, validate on held middle image. Cameras never use these231points as supervision.',
        observations=len(frames),by_method=results,differences=comparisons,all_background_secondary=bgs,
        gates=checks,pilot_numeric_passed=all(checks.values()),quality_accepted=False,
        full175_evaluated=False,gaussian_training=False,root_cause_resolved=False,
        inference='Local adapted HSfM-SMPL camera diagnostic, not an avatar training result or physical calibration ground truth.',
        limitations=['Semantics/cycle checks do not make image correspondences exact physicalGT.',
                    'Only6front/three-quarter frames supply visible body joints; side frames remain scene-constrained.',
                    'Both arms fix the same initial global metric gauge; absolute physicalscale remains unverified.',
                    'All142background candidates retained as a secondary stratum; primary65exclude visible bystanders before fitting.'],
        evidence=dict(frozen_sha256=sha(J/'frozen-validation.npz'),source_manifest_sha256=sha(H/'input-pilot/inputs.json')))
    save('result.json',result);print(json.dumps(dict(gates=checks,by_method=results,pilot_numeric_passed=all(checks.values()))),flush=True)
if __name__=='__main__':main()
