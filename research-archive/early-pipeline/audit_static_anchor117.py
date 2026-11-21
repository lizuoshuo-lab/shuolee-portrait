"""Read-only fixed074 diagnostics. No camera/model optimization or image warping."""
from pathlib import Path
import datetime as dt, hashlib, json, os
import cv2
import numpy as np
from PIL import Image, ImageDraw

def local(p):
    p=str(p)
    if os.name=='nt':return Path(p.replace('/mnt/c/','C:/'))
    return Path(p.replace('C:/','/mnt/c/').replace('C:\\','/mnt/c/').replace('\\','/'))

B=local('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/portrait-causal-diagnosis-117'
J=B/'current/portrait-alignment-open-source-116/hsfm/joint-diagnostic-reserved'
LABELS=['background','face_lower','face_upper','torso','legs_shoes']

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(count=len(x),median=float(np.median(x)) if len(x) else None,
                p90=float(np.quantile(x,.9)) if len(x) else None,max=float(np.max(x)) if len(x) else None)
def skew(t):
    x,y,z=t;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def sampson(pa,pb,Ka,Kb,A,Bb):
    rel=Bb@np.linalg.inv(A)
    F=np.linalg.inv(Kb).T@skew(rel[:3,3])@rel[:3,:3]@np.linalg.inv(Ka)
    F/=np.linalg.norm(F)
    x=np.r_[pa,1.];y=np.r_[pb,1.];l=F@x;r=F.T@y
    return abs(y@l)/np.sqrt(l[0]**2+l[1]**2+r[0]**2+r[1]**2)

def main():
    assert (O/'protocol.json').is_file() and not (O/'D0').exists()
    D=O/'D0';D.mkdir()
    protocol=read(O/'protocol.json')
    for row in protocol['source_files']:
        assert hashlib.sha256(local(row['path']).read_bytes()).hexdigest()==row['sha256']
    rows=read(B/'current/avatar-repository-pair-114/smplx/data/training-inputs.json')['records']
    rr={r['source_frame_index']:r for r in rows};assert sorted(rr)==list(range(1,176))
    K={f:np.array(r['K'],dtype=float) for f,r in rr.items()}
    for k in K.values():k[:2,2]-=.5
    W={f:np.array(r['w2c'],dtype=float) for f,r in rr.items()}
    assert max(np.linalg.norm(w[:3,:3].T@w[:3,:3]-np.eye(3)) for w in W.values())<1e-8
    a=np.load(J/'frozen-validation.npz');frames,xy,groups=[a[k] for k in ['frames','xy','groups']]
    n=len(frames);points=np.full((n,3),np.nan);pred=np.full((n,2),np.nan)
    err=np.full(n,np.nan);epi=np.full((n,3),np.nan);angle=np.full(n,np.nan);depth=np.full((n,3),np.nan)
    pairs=[(0,1),(1,2),(0,2)]
    for i,(fs,ps) in enumerate(zip(frames,xy)):
        fs=[int(f) for f in fs];P=[K[f]@W[f][:3] for f in fs]
        Xh=cv2.triangulatePoints(P[0],P[2],ps[0].reshape(2,1),ps[2].reshape(2,1))[:,0]
        for j,(u,v) in enumerate(pairs):epi[i,j]=sampson(ps[u],ps[v],K[fs[u]],K[fs[v]],W[fs[u]],W[fs[v]])
        if not np.isfinite(Xh).all() or abs(Xh[3])<1e-12:continue
        Xh/=Xh[3];z=np.array([(W[f]@Xh)[2] for f in fs]);depth[i]=z
        if (z<=0).any():continue
        points[i]=Xh[:3];q=P[1]@Xh;pred[i]=q[:2]/q[2];err[i]=np.linalg.norm(pred[i]-ps[1])
        c=[np.linalg.inv(W[f])[:3,3] for f in [fs[0],fs[2]]]
        rays=[Xh[:3]-cc for cc in c];rays=[r/np.linalg.norm(r) for r in rays]
        angle[i]=np.degrees(np.arccos(np.clip(np.dot(*rays),-1,1)))
    # Independent algebraic fixture: exact projections of the triangulated point must satisfy F.
    fixtures=[]
    for i in np.flatnonzero(np.isfinite(points).all(axis=1))[:20]:
        fs=frames[i];X=np.r_[points[i],1.];p=[]
        for f in fs:
            q=K[int(f)]@W[int(f)][:3]@X;p.append(q[:2]/q[2])
        fixtures.append(sampson(p[0],p[2],K[int(fs[0])],K[int(fs[2])],W[int(fs[0])],W[int(fs[2])]))
    assert max(fixtures)<1e-8
    prior=np.load(J/'native-fit-cached/evaluation/original074-frozen-errors.npz')
    assert np.array_equal(np.isfinite(err),np.isfinite(prior['errors']))
    assert np.nanmax(np.abs(err-prior['errors']))<1e-8
    summaries={}
    for g,label in enumerate(LABELS):
        ix=groups==g
        summaries[label]=dict(tracks=int(ix.sum()),invalid=int((ix&~np.isfinite(err)).sum()),
            held_middle_px=stats(err[ix]),max_pair_sampson_px=stats(np.max(epi[ix],axis=1)),
            endpoint_ray_angle_deg=stats(angle[ix]))
    bg=groups==0;bgids=np.flatnonzero(bg);covered=sorted(set(frames[bg].ravel().tolist()));wh=np.array(rows[0]['wh'])
    coverage=[]
    for f in range(1,176):
        ti,vi=np.where(frames[bg]==f);pts=xy[bg][ti,vi]
        cells=np.clip(np.floor(pts/wh*4).astype(int),0,3)
        hullarea=cv2.contourArea(cv2.convexHull(pts.astype(np.float32))) if len(pts)>=3 else 0
        coverage.append(dict(raw_frame=f,unique_tracks=len(ti),grid_cells=len(set(map(tuple,cells))),
            image_hull_fraction=float(hullarea/np.prod(wh))))
    X=points[bg];sv=np.linalg.svd(X-X.mean(0),compute_uv=False)
    positive_depth=depth[bg][np.isfinite(depth[bg]) & (depth[bg]>0)]
    result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='D0_complete_visual_review_pending',
        calibration_changed=False,training_started=False,new_gaussian=False,root_cause_resolved=False,
        image_units='Original rectified1514x2694 pixels; integer pixel-center convention',
        observations='Reanalysis of frozen116validation. No new independent measurement or physicalGT.',
        by_region=summaries,static_anchor_coverage=dict(tracks=int(bg.sum()),covered_frames=covered,
            covered_frame_count=len(covered),total_actual_cameras=175,uncovered_validation_frames=175-len(covered),
            frames_meeting_20tracks_4cells=sum(r['unique_tracks']>=20 and r['grid_cells']>=4 for r in coverage),
            depth_q90_over_q10=float(np.quantile(positive_depth,.9)/np.quantile(positive_depth,.1)),
            xyz_singular_values=sv.tolist(),smallest_over_largest_singular=float(sv[-1]/sv[0]),
            by_frame=coverage),
        checks=dict(original074_held_errors_reproduced=True,exact_projection_sampson_max_px=max(fixtures),
            all175_camera_ids_present=True,all175_intrinsics_identical=all(np.array_equal(K[1],k) for k in K.values()),
            sources_unchanged=True),
        interpretation=['175actual cameras exist. Only13frames have observations in this manually reviewed independent-from074 static-object validation bank.',
            'Epipolar residual is depth-independent but cannot by itself distinguish wrong match, subject motion or wrong calibration.',
            'Small background residual is a local regression check; it does not prove physical calibration or close-depth face accuracy.',
            'Coverage/depth metrics derive from estimated074geometry and reused tracks, not surveyed control points. No camera optimization or unique rootcause conclusion.'])
    np.savez_compressed(D/'track-measurements.npz',frames=frames,xy=xy,groups=groups,xyz=points,
        predicted_middle=pred,held_error_px=err,pair_sampson_px=epi,endpoint_angle_deg=angle,depth=depth)
    save(D/'result.json',result)
    # Source-image evidence, same crop and aspect. Markers are diagnostics, not synthetic theory figures.
    for f in [50,75,100]:
        p=local(rr[f]['image']);assert hashlib.sha256(p.read_bytes()).hexdigest()==rr[f]['source_sha256']
        image=Image.open(p).convert('RGB');s=900/image.height
        full=image.resize((round(image.width*s),900),Image.Resampling.LANCZOS)
        board=Image.new('RGB',(full.width,946),'white');board.paste(full,(0,46));draw=ImageDraw.Draw(board)
        draw.text((8,6),f'raw{f:06d}: fixed074, held middle view',fill='black')
        draw.text((8,24),'cyan=observed / red=projected; same source pixels',fill='black')
        for ti in np.flatnonzero(frames[:,1]==f):
            q=xy[ti,1]*s+[0,46];r=pred[ti]*s+[0,46]
            if np.isfinite(r).all():draw.line([tuple(q),tuple(r)],fill='yellow',width=1);draw.ellipse((*tuple(r-2),*tuple(r+2)),outline='red',width=1)
            draw.ellipse((*tuple(q-3),*tuple(q+3)),outline='#00cfff',width=1)
            if groups[ti]==0:draw.text((q[0]+4,q[1]-6),f'B{ti}',fill='#00ffff')
        board.save(D/f'raw{f:06d}-source-overlay.jpg',quality=96)
    q=read(O/'queue-state.json');q.update(updated=result['updated'],state='D0_complete_visual_review_pending',stage='D0',
        next_action='Review three original-image overlays. Reused65static tracks cover13frames; prepare fresh175-frame static-object fit/held inventory before any camera fit.',
        result='D0/result.json',active_pid=None);save(O/'queue-state.json',q)
    print(json.dumps(dict(by_region=summaries,static_covered_frames=len(covered),calibration_changed=False),ensure_ascii=False))

if __name__=='__main__':main()
