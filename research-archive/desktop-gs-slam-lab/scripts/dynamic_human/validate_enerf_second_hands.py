"""Independent-view geometry and even-time hand trajectory qualification."""
from pathlib import Path
import json,hashlib,itertools,time,shutil
import numpy as np
from scipy.optimize import least_squares
from PIL import Image,ImageDraw,ImageFont

L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-motion-082'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'trajectory-summary.json').exists(), 'Preserve completed qualification'
rows=read(O/'observations.json')['records'];by={(r['camera'],r['frame']):r for r in rows};cams=sorted({r['camera'] for r in rows});times=[722,724,726,728,730]
clock=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz'
with np.load(clock) as z:offsets={str(c):float(d) for c,d in zip(z['cameras'],z['offsets'])}
P=np.stack([np.array(by[c,726]['K'])@np.array(by[c,726]['w2c'])[:3] for c in cams]);centers=np.array([-np.array(by[c,726]['w2c'])[:3,:3].T@np.array(by[c,726]['w2c'])[:3,3] for c in cams])
protocol=dict(status='fixed_before_geometry_results',frames_physical=times,
    methods=['wholebody','hand','hand_stable_crop'],scores='Report SimCC values but do not transfer old whole-body0.4threshold; different score distributions, neither is calibrated visibility.',
    crop_gate_px=3,consensus_gate_px=3,min_other_support_cameras=4,min_ray_angle_deg=2,
    interpolation='Only even-time observations interpolated at physical_time minus frozen camera clock; no temporal extrapolation.',
    evaluation='For each target camera/time/joint, exclude that camera from every hypothesis, inlier decision and refinement. All15 targets counted even if triangulation fails. Stable variant filters only supporting observations, target stability separately reported.',
    trajectories='Fit independent cubicXYZ only to even observations outside camera04. Consensus initialization at722..730; exclude crop-unstable observations for stable variant; initial projection gate6px; softL1 scale1.5px. No kinematic constraints or Gaussian training.',
    clock_sha256=sha(clock),offsets=offsets,goal_complete=False)
if (O/'geometry-protocol.json').exists():assert read(O/'geometry-protocol.json')==protocol
else:save(O/'geometry-protocol.json',protocol)

def project(X,ps=P):
    q=np.einsum('nij,j->ni',ps[:,:,:3],X)+ps[:,:,3]
    return q[:,:2]/q[:,2:],q[:,2]
def robust(xy,valid):
    ids=np.where(valid)[0]
    if len(ids)<4:return None
    pairs=np.array(list(itertools.combinations(ids,2)))
    ps=P[pairs];uv=xy[pairs];A=(uv[:,:,:,None]*ps[:,:,None,2,:]-ps[:,:,:2,:]).reshape(-1,4,4)
    _,_,v=np.linalg.svd(A);xh=v[:,-1];good=abs(xh[:,3])>1e-10
    X=xh[:,:3]/np.where(good,xh[:,3],1)[:,None]
    q=np.einsum('nij,kj->kni',P[:,:,:3],X)+P[None,:,:,3];errs=np.linalg.norm(q[:,:,:2]/q[:,:,2:]-xy[None],axis=-1)
    inl=(errs<3)&(q[:,:,2]>0)&valid[None]&good[:,None];count=inl.sum(1);cost=np.minimum(errs[:,valid],3).sum(1)
    k=np.lexsort((cost,-count))[0]
    if count[k]<4:return None
    x=X[k];ins=inl[k]
    for _ in range(3):
        inds=np.where(ins)[0]
        x=least_squares(lambda y:(project(y,P[inds])[0]-xy[inds]).ravel(),x,loss='soft_l1',f_scale=1.5,max_nfev=50).x
        uv,z=project(x);err=np.linalg.norm(uv-xy,axis=1);new=(err<3)&(z>0)&valid
        if np.array_equal(new,ins):break
        ins=new
        if ins.sum()<4:return None
    rays=x-centers[ins];rays/=np.linalg.norm(rays,axis=1,keepdims=True)
    angle=float(np.degrees(np.arccos(np.clip((rays@rays.T).min(),-1,1))))
    if angle<=2:return None
    return dict(point=x,support=ins,error=err,ray_angle=angle)

resume_geometry=(O/'geometry-summary.json').exists()
interp={};georows=read(O/'cross-camera-records.json')['records'] if resume_geometry else [];start=time.monotonic()
for t in times:
    inp={}
    for c in cams:
        nominal=t-offsets[c];lo=2*int(np.floor(nominal/2));hi=lo+2;assert 720<=lo<hi<=732
        a,b=by[c,lo],by[c,hi];f=(nominal-lo)/2
        inp[c]=dict(wholebody=(1-f)*np.array(a['wholebody_keypoints'])+f*np.array(b['wholebody_keypoints']),
            hand=(1-f)*np.array(a['keypoints'])+f*np.array(b['keypoints']),
            stable=np.maximum(a['crop_deviation_px'],b['crop_deviation_px'])<=3,source_even_frames=[lo,hi],fraction=f)
    interp[t]=inp
    for method in ([] if resume_geometry else ['wholebody','hand','hand_stable_crop']):
        arr=np.stack([inp[c]['wholebody' if method=='wholebody' else 'hand'] for c in cams])
        stable=np.stack([inp[c]['stable'] for c in cams]) if method=='hand_stable_crop' else np.ones((15,21),bool)
        for j in range(21):
            for ci,c in enumerate(cams):
                valid=stable[:,j].copy();valid[ci]=False;fit=robust(arr[:,j],valid)
                rr=dict(method=method,time=t,joint=j,target_camera=c,available_support_views=int(valid.sum()),target_crop_stable=bool(inp[c]['stable'][j]),valid=False)
                if fit is not None:
                    uv,z=project(fit['point']);rr.update(valid=bool(z[ci]>0),error_px=float(np.linalg.norm(uv[ci]-arr[ci,j])),support_cameras=[cams[i] for i in np.where(fit['support'])[0]],point=fit['point'].tolist(),max_ray_angle_deg=fit['ray_angle'])
                    assert c not in rr['support_cameras']
                georows.append(rr)
    print('HAND_GEOMETRY',t,len(georows),flush=True)
if not resume_geometry:save(O/'cross-camera-records.json',dict(records=georows))
def metrics(rs):
    e=np.array([r['error_px'] for r in rs if r['valid']]);n=len(rs)
    return dict(total_targets=n,valid_targets=len(e),valid_fraction=len(e)/n if n else None,median_px=float(np.median(e)) if len(e) else None,p90_px=float(np.quantile(e,.9)) if len(e) else None,under3px_all_targets=int((e<3).sum()),under3px_fraction_all_targets=float((e<3).sum()/n) if n else None)
methods=['wholebody','hand','hand_stable_crop'];categories={'all':list(range(21)),'wrist':[0],'thumb':[1,2,3,4],'other_fingers':list(range(5,21)),'tips':[4,8,12,16,20]}
summary={m:{category:metrics([r for r in georows if r['method']==m and r['joint'] in js]) for category,js in categories.items()} for m in methods}
if not resume_geometry:save(O/'geometry-summary.json',dict(methods=summary,by_time={str(t):{m:metrics([r for r in georows if r['method']==m and r['time']==t]) for m in methods} for t in times},meaning='Leave-one-camera-out consistency with interpolated predictions, not hand GT accuracy.',seconds=time.monotonic()-start))

# Fit a trajectory while camera04 remains entirely excluded. No odd-time fitting.
coeffs={};trajrows=[];fitstats=[]
for method in methods:
    key='wholebody_keypoints' if method=='wholebody' else 'keypoints'
    for j in range(21):
        init=[]
        for t in times:
            arr=np.stack([interp[t][c]['wholebody' if method=='wholebody' else 'hand'][j] for c in cams])
            valid=np.array([interp[t][c]['stable'][j] if method=='hand_stable_crop' else True for c in cams]);valid[cams.index('04')]=False
            f=robust(arr,valid)
            if f is not None:init.append((t,f['point']))
        rr=dict(method=method,joint=j,accepted=False,initial_supported_times=[t for t,x in init])
        if len(init)<4:rr['reason']='fewer_than_four_triangulated_times';fitstats.append(rr);continue
        ts=np.array([(t-726)/6 for t,x in init]);basis=ts[:,None]**np.arange(4);coef=np.linalg.lstsq(basis,np.array([x for t,x in init]),rcond=None)[0]
        fit=[r for r in rows if r['camera']!='04' and r['frame']%2==0 and (method!='hand_stable_crop' or r['crop_deviation_px'][j]<=3)]
        ps=np.stack([P[cams.index(r['camera'])] for r in fit]);uv=np.array([r[key][j] for r in fit]);taus=np.array([(r['frame']+offsets[r['camera']]-726)/6 for r in fit]);B=taus[:,None]**np.arange(4)
        def forward(v):
            X=B@v.reshape(4,3);q=np.einsum('nij,nj->ni',ps[:,:,:3],X)+ps[:,:,3];return q[:,:2]/q[:,2:]
        errs=np.linalg.norm(forward(coef.ravel())-uv,axis=1);mask=errs<6
        if np.linalg.matrix_rank(B[mask])<4 or mask.sum()<16:rr['reason']='insufficient_temporal_support';fitstats.append(rr);continue
        ps=ps[mask];uv=uv[mask];B=B[mask]
        A=(uv[:,:,None]*ps[:,None,2,:3]-ps[:,:2,:3])[:,:,None,:]*B[:,None,:,None]
        rank=int(np.linalg.matrix_rank(A.reshape(-1,12)))
        if rank<12:rr['reason']='rank_deficient_projection_system';fitstats.append(rr);continue
        fitted=least_squares(lambda v:(forward(v)-uv).ravel(),coef.ravel(),loss='soft_l1',f_scale=1.5,max_nfev=200,x_scale='jac')
        coef=fitted.x.reshape(4,3);coeffs[method,j]=coef;rr.update(accepted=True,observations=int(mask.sum()),rank=rank,solver_success=bool(fitted.success),fit_median_px=float(np.median(np.linalg.norm(forward(fitted.x)-uv,axis=1))));fitstats.append(rr)
        for r in rows:
            tau=(r['frame']+offsets[r['camera']]-726)/6;X=(tau**np.arange(4))@coef;uv0,z=project(X,P[cams.index(r['camera']):cams.index(r['camera'])+1]);err=float(np.linalg.norm(uv0[0]-r[key][j]))
            trajrows.append(dict(method=method,joint=j,camera=r['camera'],frame=r['frame'],split='excluded_camera04' if r['camera']=='04' else ('diagnostic_odd' if r['frame']%2 else 'training_even'),valid=bool(z[0]>0),error_px=err,xy=uv0[0].tolist(),crop_stable=r['crop_deviation_px'][j]<=3))
    print('HAND_TRAJECTORY',method,sum((method,j) in coeffs for j in range(21)),flush=True)
save(O/'trajectory-fit.json',dict(records=fitstats,coefficients={f'{m}/{j}':c.tolist() for (m,j),c in coeffs.items()},fit_excludes_camera04=True,fit_excludes_odd_times=True))
save(O/'trajectory-records.json',dict(records=trajrows))
save(O/'trajectory-summary.json',dict(methods={m:{split:metrics([r for r in trajrows if r['method']==m and r['split']==split]) for split in ['training_even','diagnostic_odd','excluded_camera04']} for m in methods},accepted_joint_counts={m:sum((m,j) in coeffs for j in range(21)) for m in methods},meaning='Consistency of independent cubic3D predictions; not a physically constrained articulated hand or appearance result.'))
(O/'motion-panels').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12);colors=[(235,85,30),(210,40,160),(30,150,210),(65,180,55),(165,75,210)]
for c in ['01','04','08','09','13']:
    sheet=Image.new('RGB',(1024,1030),'white');d=ImageDraw.Draw(sheet)
    d.text((8,4),f'Camera{c} | Native / hand detection / 3D wholebody / 3D hand / 3D stable hand',font=font,fill='black')
    d.text((8,20),'All 3D fits exclude camera04 and all odd frames; missing joints are not drawn. Native 1:1 pixels.',font=font,fill='black')
    for ti,t in enumerate([721,723,727,729,731]):
        r=by[c,t];crop=r['crop']
        with np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz') as z:raw=Image.fromarray(z['rgb']).crop(crop)
        py=62+ti*190;d.text((8,py-18),f'Frame{t} (not fitted)',font=font,fill='black');sheet.paste(raw,(8,py))
        pts=[np.array(r['keypoints'])]
        for m in methods:
            arr=np.full((21,2),np.nan)
            for rr in trajrows:
                if rr['method']==m and rr['camera']==c and rr['frame']==t:arr[rr['joint']]=rr['xy']
            pts.append(arr)
        for mi,p in enumerate(pts):
            im=raw.copy();draw=ImageDraw.Draw(im);p=p-np.array(crop[:2])
            for f in range(5):
                chain=[0]+list(range(1+4*f,5+4*f))
                for i,j in zip(chain[:-1],chain[1:]):
                    if np.isfinite(p[[i,j]]).all():draw.line([tuple(p[i]),tuple(p[j])],fill=colors[f],width=1)
            for x,y in p:
                if np.isfinite(x+y):draw.ellipse((x-1,y-1,x+1,y+1),fill=(255,20,20))
            sheet.paste(im,(178+mi*170,py))
    sheet.crop((0,0,860,1015)).save(O/'motion-panels'/f'camera{c}-motion.png')
shutil.copy2(__file__,O/Path(__file__).name);print('HAND_QUALIFICATION_COMPLETE',time.monotonic()-start,flush=True)
