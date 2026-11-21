"""Training-only geometric consensus and fixed-clock cubic joint trajectories."""
from pathlib import Path
import itertools, json, hashlib, time, shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
from PIL import Image, ImageDraw, ImageFont

L=Path(__file__).resolve().parents[2]; R=L/'outputs/dynamic-human-enerf-second-interval-079'; O=R/'motion'; O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text()); save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'trajectory-summary.json').exists()
types=['person_dark_clothing','person_light_clothing']; obs=read(R/'pose/observations.json')['records']; proto=read(R/'protocol.json')
cameras=proto['training_cameras']; frames=proto['appearance_train_frames']; ncoef=36*12
clockpath=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz'
with np.load(clockpath) as z: clocks={str(c):float(d) for c,d in zip(z['cameras'],z['offsets'])}
offsets=np.array([clocks[c] for c in cameras]);assert set(clocks)==set(cameras)
save(O/'trajectory-protocol.json',dict(status='fixed_before_new_interval_geometric_fitting',reference_frame=726,time_scale_frames=6,
 fit_frames=frames,validation_frames=proto['validation_time_frames'],
 clock='Freeze074training-camera clock estimates; same capture, short gap. Refit all joint trajectories. No new clock optimization, no old appearance or motion extrapolation.',
 clock_sha256=sha(clockpath),offsets_frames=clocks,
 targets='12COCObody+6feet, predicted RGB keypoints>=0.4 after photo-based identity exclusions; not GT. No face/finger fit.',
 consensus='For each even frame and joint: all-pair DLT, positive depth,8px consensus,min4views,maxrayangle>2deg; softL1 f_scale3 three support refinements. Raw-frame consensus initialization approximates synchronization; final trajectory uses each fixed capturetime.',
 cubic='Independent XYZ cubic per joint in tau=(frame+clock-726)/6; robust reprojection fit only to new even-time consensus inliers; float64 analytic sparseJacobian,softL1 f_scale3,max300nfev.',
 validation='Oddtime observations withcamera/joint supported in>=3training frames, fixed before solving. Report all eligible training errors too, not only consensus inliers.',
 goal_complete=False))
rows=[]
for r in obs:
 for d in r['detections']:
  pi=types.index(d['person'])
  for j in range(5,23):
   if j in d['disabled_fitting_indices'] or d['scores'][j]<.4:continue
   K=np.array(r['K']);W=np.array(r['w2c'])
   rows.append(dict(camera=r['camera'],frame=r['frame'],person=pi,index=j,joint=pi*18+j-5,xy=np.array(d['keypoints'][j]),score=min(1.,d['scores'][j]),K=K,W=W,P=K@W[:3]))
def projection(x,o):
 q=o['P'][:,:3]@x+o['P'][:,3];return q[:2]/q[2],q[2]
def errors(x,os):return np.array([np.linalg.norm(q-o['xy']) if z>0 else 1e9 for o in os for q,z in [projection(x,o)]])
tri=[];accepted=set();started=time.monotonic()
for f in frames:
 for ji in range(36):
  ids=[i for i,r in enumerate(rows) if r['frame']==f and r['joint']==ji];os=[rows[i] for i in ids];best=None
  for a,b in itertools.combinations(os,2):
   A=np.stack([xy*p[2]-p[k] for o in [a,b] for p in [o['P']] for k,xy in enumerate(o['xy'])]);_,_,v=np.linalg.svd(A)
   if abs(v[-1,3])<1e-12:continue
   x=v[-1,:3]/v[-1,3];e=errors(x,os);ins=e<8;key=(int(ins.sum()),-float(np.median(e[ins]))) if ins.any() else (0,-1e9)
   if best is None or key>best[0]:best=(key,x,ins)
  rr=dict(frame=f,joint=ji,person=types[ji//18],index=ji%18+5,eligible_views=len(os),accepted=False)
  if best is None or best[0][0]<4:rr['reason']='fewer_than4_consistent_views';tri.append(rr);continue
  _,x,ins=best
  for _ in range(3):
   x=least_squares(lambda y:np.concatenate([projection(y,o)[0]-o['xy'] for o,b in zip(os,ins) if b]),x,loss='soft_l1',f_scale=3,max_nfev=100).x
   e=errors(x,os);newins=e<8
   if np.array_equal(newins,ins) or newins.sum()<4:ins=newins;break
   ins=newins
  dirs=[]
  for o,b in zip(os,ins):
   if b:
    center=-o['W'][:3,:3].T@o['W'][:3,3];v=x-center;dirs.append(v/np.linalg.norm(v))
  angle=max([float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for a,b in itertools.combinations(dirs,2)],default=0.)
  ok=bool(ins.sum()>=4 and angle>2)
  rr.update(accepted=ok,point=x.tolist(),inlier_count=int(ins.sum()),max_ray_angle_deg=angle,observations=[dict(camera=o['camera'],error_px=float(err),inlier=bool(b)) for o,err,b in zip(os,e,ins)])
  if ok:accepted.update(i for i,b in zip(ids,ins) if b)
  tri.append(rr)
 print('SECOND_CONSENSUS',f,sum(r['accepted'] for r in tri if r['frame']==f),flush=True)
save(O/'triangulation.json',dict(records=tri,accepted=sum(r['accepted'] for r in tri),total=len(tri),seconds=time.monotonic()-started,goal_complete=False))
ci=np.array([cameras.index(r['camera']) for r in rows]);ti=np.array([r['frame'] for r in rows]);ji=np.array([r['joint'] for r in rows]);xy=np.array([r['xy'] for r in rows]);P=np.stack([r['P'] for r in rows]);weight=np.sqrt([r['score'] for r in rows])
fitmask=np.array([i in accepted for i in range(len(rows))]);fitidx=np.where(fitmask)[0]
support={(c,j):sum(fitmask&(ci==c)&(ji==j)) for c in range(len(cameras)) for j in range(36)}
valmask=(ti%2==1)&np.array([support[c,j]>=3 for c,j in zip(ci,ji)])
coef=np.zeros((36,4,3));ranks=[]
for j in range(36):
 ids=np.where(fitmask&(ji==j))[0];tau=(ti[ids]+offsets[ci[ids]]-726)/6;basis=np.stack([tau**k for k in range(4)],1)
 lhs=xy[ids,:,None]*P[ids,None,2,:3]-P[ids,:2,:3];rhs=P[ids,:2,3]-xy[ids]*P[ids,None,2,3]
 A=(basis[:,None,:,None]*lhs[:,:,None,:]).reshape(-1,12);b=rhs.reshape(-1);rank=int(np.linalg.matrix_rank(A));ranks.append(dict(joint=j,rank=rank,observations=len(ids),frames=sorted(set(ti[ids].tolist()))))
 if rank==12:coef[j]=np.linalg.lstsq(A,b,rcond=None)[0].reshape(4,3)
save(O/'trajectory-rank-audit.json',dict(records=ranks,all_full_rank=all(r['rank']==12 for r in ranks)))
assert all(r['rank']==12 for r in ranks),'Insufficient geometric support; inspect before continuing'
def pack(v,ids):
 co=v.reshape(36,4,3);tau=(ti[ids]+offsets[ci[ids]]-726)/6;basis=np.stack([tau**k for k in range(4)],1);X=np.einsum('nk,nkd->nd',basis,co[ji[ids]]);pp=np.einsum('nij,nj->ni',P[ids,:,:3],X)+P[ids,:,3]
 return basis,pp
def residual(v):
 basis,pp=pack(v,fitidx);return ((pp[:,:2]/pp[:,2:]-xy[fitidx])*weight[fitidx,None]).ravel()
def jacobian(v):
 basis,pp=pack(v,fitidx);n=len(fitidx);D=(P[fitidx,:2,:3]*pp[:,None,2,None]-pp[:,:2,None]*P[fitidx,None,2,:3])/pp[:,None,2,None]**2
 block=D[:,:,None,:]*basis[:,None,:,None]*weight[fitidx,None,None,None]
 rr=np.repeat(np.arange(2*n),12);cc=np.broadcast_to(ji[fitidx,None,None]*12+np.arange(12)[None,None,:],(n,2,12)).ravel()
 return coo_matrix((block.ravel(),(rr,cc)),shape=(2*n,ncoef)).tocsr()
v0=coef.ravel();rng=np.random.default_rng(73);checks=[]
for _ in range(3):
 d=rng.normal(size=len(v0));d/=np.linalg.norm(d);numeric=(residual(v0+1e-6*d)-residual(v0-1e-6*d))/2e-6;analytic=jacobian(v0)@d;err=float(np.max(abs(numeric-analytic)));assert err<1e-3;checks.append(err)
save(O/'jacobian-check.json',dict(max_absolute_errors=checks))
result=least_squares(residual,v0,jac=jacobian,loss='soft_l1',f_scale=3,max_nfev=300,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
basis,pp=pack(result.x,np.arange(len(rows)));assert np.all(pp[:,2]>0);e=np.linalg.norm(pp[:,:2]/pp[:,2:]-xy,axis=1)
def metrics(mask):
 a=e[mask];return dict(count=len(a),median_px=float(np.median(a)),rmse_px=float(np.sqrt(np.mean(a*a))),p90_px=float(np.quantile(a,.9)),fraction_below8px=float(np.mean(a<8))) if len(a) else dict(count=0)
np.savez_compressed(O/'fixed_clock-fit.npz',coefficients=result.x.reshape(36,4,3),offsets=offsets,cameras=np.array(cameras),error_px=e,xy=xy,P=P,camera_index=ci,frame=ti,joint_index=ji,fitmask=fitmask,validationmask=valmask)
out=dict(status='new_interval_cubic_trajectories_fitted_projection_review_pending',fit_consensus=metrics(fitmask),all_eligible_even=metrics(ti%2==0),validation_fixed_support=metrics(valmask),validation_all_eligible=metrics(ti%2==1),
 per_person={typ:dict(train=metrics(fitmask&(ji//18==i)),validation=metrics(valmask&(ji//18==i))) for i,typ in enumerate(types)},
 per_camera={c:metrics(valmask&(ci==i)) for i,c in enumerate(cameras)},
 solver=dict(status=int(result.status),message=result.message,nfev=result.nfev,cost=float(result.cost),optimality=float(result.optimality)),
 source_observations_sha256=sha(R/'pose/observations.json'),seconds=time.monotonic()-started,
 qualification='Reprojection consistency of predicted landmarks, not GT accuracy or human appearance clarity.',goal_complete=False)
save(O/'trajectory-summary.json',out)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf' if __import__('os').name=='nt' else '/mnt/c/Windows/Fonts/arial.ttf',13)
co=result.x.reshape(36,4,3)
for c in ['01','02','04','05','07','13','17']:
 for f in [720,726,732]:
  z=np.load(R/'native-input'/f'camera{c}-frame{f:06d}.npz');raw=Image.fromarray(z['rgb']);im=raw.copy();draw=ImageDraw.Draw(im);t=(f+clocks[c]-726)/6;b=np.array([t**k for k in range(4)])
  for j in range(36):
   X=b@co[j];q=z['K']@(z['w2c'][:3,:3]@X+z['w2c'][:3,3]);x,y=q[:2]/q[2];col=(255,100,20) if j//18==0 else (0,160,255)
   draw.line([(x-4,y-4),(x+4,y+4)],fill=col,width=2);draw.line([(x-4,y+4),(x+4,y-4)],fill=col,width=2);draw.text((x+4,y-10),str(j%18+5),font=font,fill=col,stroke_width=1,stroke_fill='white')
  bb=z['bbox'];raw=raw.crop(bb);im=im.crop(bb);sheet=Image.new('RGB',(raw.width*2,raw.height+30),'white');sheet.paste(raw,(0,30));sheet.paste(im,(raw.width,30));ImageDraw.Draw(sheet).text((5,5),f'Camera{c} frame{f}: original | new cubic 3D projection',font=font,fill='black');sheet.save(O/f'camera{c}-frame{f:06d}-trajectory.png')
shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(out),flush=True)
