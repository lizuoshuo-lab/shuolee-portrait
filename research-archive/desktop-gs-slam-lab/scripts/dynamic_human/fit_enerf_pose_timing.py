"""Training-camera cubic3D tracks: even-time fit, odd-time validation."""
from pathlib import Path
import json,time,shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';OLD=LAB/'outputs/dynamic-human-enerf-wholebody-059'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (OUT/'timing-summary.json').exists()
data=read(OUT/'observations.json');types=['person_dark_clothing','person_light_clothing'];cameras=sorted({r['camera'] for r in data['records']});assert cameras[0]=='01' and not {'00','06','12'}&set(cameras)
support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']}
save(OUT/'timing-protocol.json',dict(status='fixed_before_fitting',fit_frames=list(range(744,757,2)),validation_frames=list(range(745,756,2)),joints='12COCObody+6feet perperson; noface/fingers',
 camera_joint_support='059frame750robustinliermembership fixed for alltimes, plusRTMWscore>=0.4 at each observation. Frame750 belongs to fitting split.',
 trajectory='Eachjoint independent cubicXYZ polynomial in normalized(frame750difference+cameraoffset)/6. NoSMPLorimageRGBloss.',
 optimized_offsets='14 offsets, camera01fixed0; bound[-2,2]frames; residual0.1*offset regularizer. Original camera calibration fixed.',
 solver='float64 analytic sparseJacobian, softL1 loss f_scale3px, max_nfev300. Baselinecubicfitwithzerooffset, then samecubicplusoffsets. Fixed stopping; oddtimesexcludedfromfitting.',
 decision='Timinghypothesis only. Check oddtime/percamera/perperson improvement and boundaryhits beforeanyRGBresampling. Posepredictionbias andshortcubicmotion approximation can mimic offsets.',goal_complete=False))
rows=[]
for r in data['records']:
 for pi,typ in enumerate(types):
  d=next(d for d in r['detections'] if d['person']==typ)
  for j in range(5,23):
   if r['camera'] not in support[typ,j] or d['scores'][j]<.4:continue
   rows.append(dict(camera=cameras.index(r['camera']),frame=r['frame'],person=pi,joint=pi*18+j-5,xy=d['keypoints'][j],P=np.array(r['K'])@np.array(r['w2c'])[:3],score=min(1,d['scores'][j])))
ci=np.array([r['camera'] for r in rows]);ti=np.array([r['frame'] for r in rows]);ji=np.array([r['joint'] for r in rows]);xy=np.array([r['xy'] for r in rows]);P=np.stack([r['P'] for r in rows]);weight=np.sqrt([r['score'] for r in rows]);fitmask=ti%2==0
fitidx=np.where(fitmask)[0];ncoef=36*4*3
coef=np.zeros((36,4,3))
for j in range(36):
 ids=np.where(fitmask&(ji==j))[0];tau=(ti[ids]-750)/6;basis=np.stack([tau**k for k in range(4)],1)
 lhs=xy[ids,:,None]*P[ids,None,2,:3]-P[ids,:2,:3];rhs=P[ids,:2,3]-xy[ids]*P[ids,None,2,3]
 A=(basis[:,None,:,None]*lhs[:,:,None,:]).reshape(-1,12);b=rhs.reshape(-1)
 assert np.linalg.matrix_rank(A)==12
 coef[j]=np.linalg.lstsq(A,b,rcond=None)[0].reshape(4,3)
def pack_data(v,offsets,ids):
 co=v[:ncoef].reshape(36,4,3);delta=np.r_[0,v[ncoef:]] if offsets else np.zeros(15);tau=(ti[ids]-750+delta[ci[ids]])/6;basis=np.stack([tau**k for k in range(4)],1);X=np.einsum('nk,nkd->nd',basis,co[ji[ids]]);pp=np.einsum('nij,nj->ni',P[ids,:,:3],X)+P[ids,:,3]
 return co,delta,tau,basis,pp
def residual(v,offsets):
 co,delta,tau,basis,pp=pack_data(v,offsets,fitidx);r=(pp[:,:2]/pp[:,2:]-xy[fitidx])*weight[fitidx,None]
 return np.r_[r.ravel(),.1*delta[1:]] if offsets else r.ravel()
def jacobian(v,offsets):
 co,delta,tau,basis,pp=pack_data(v,offsets,fitidx);n=len(fitidx);D=(P[fitidx,:2,:3]*pp[:,None,2,None]-pp[:,:2,None]*P[fitidx,None,2,:3])/pp[:,None,2,None]**2
 block=D[:,:,None,:]*basis[:,None,:,None]*weight[fitidx,None,None,None]
 rr=np.repeat(np.arange(2*n),12);cc=np.broadcast_to(ji[fitidx,None,None]*12+np.arange(12)[None,None,:],(n,2,12)).ravel();vv=block.ravel()
 if offsets:
  dx=(co[ji[fitidx],1]+2*tau[:,None]*co[ji[fitidx],2]+3*tau[:,None]**2*co[ji[fitidx],3])/6
  deriv=np.einsum('nij,nj->ni',D,dx)*weight[fitidx,None];has=ci[fitidx]>0
  rr=np.r_[rr,np.arange(2*n).reshape(n,2)[has].ravel(),2*n+np.arange(14)];cc=np.r_[cc,np.repeat(ncoef+ci[fitidx][has]-1,2),ncoef+np.arange(14)];vv=np.r_[vv,deriv[has].ravel(),np.full(14,.1)]
 return coo_matrix((vv,(rr,cc)),shape=(2*n+(14 if offsets else 0),ncoef+(14 if offsets else 0))).tocsr()
v0=np.r_[coef.ravel(),np.zeros(14)];rng=np.random.default_rng(73);checks=[]
for _ in range(3):
 d=rng.normal(size=len(v0));d/=np.linalg.norm(d);numeric=(residual(v0+1e-6*d,True)-residual(v0-1e-6*d,True))/2e-6;analytic=jacobian(v0,True)@d;err=float(np.max(abs(numeric-analytic)));assert err<1e-3,err;checks.append(err)
save(OUT/'timing-jacobian-check.json',dict(status='3random_direction_central_difference_checks_passed',max_absolute_error=checks))
results={};start=time.monotonic()
for offsets,label in [(False,'fixed_offsets'),(True,'estimated_offsets')]:
 initial=coef.ravel() if not offsets else np.r_[results['fixed_offsets'].x,np.zeros(14)]
 low=np.r_[np.full(ncoef,-np.inf),np.full(14,-2)] if offsets else -np.inf;high=np.r_[np.full(ncoef,np.inf),np.full(14,2)] if offsets else np.inf
 result=least_squares(lambda v:residual(v,offsets),initial,jac=lambda v:jacobian(v,offsets),bounds=(low,high),loss='soft_l1',f_scale=3,max_nfev=300,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
 results[label]=result;save(OUT/f'{label}-solver.json',dict(status=int(result.status),message=result.message,nfev=result.nfev,cost=float(result.cost),optimality=float(result.optimality),elapsed_seconds=time.monotonic()-start));print(label,result.nfev,result.cost,flush=True)
errors={};reports={}
def metrics(e):return dict(count=len(e),median_px=float(np.median(e)),rmse_px=float(np.sqrt(np.mean(e**2))),p90_px=float(np.quantile(e,.9)),fraction_below8px=float(np.mean(e<8)))
for label,result in results.items():
 co,delta,tau,basis,pp=pack_data(result.x,label=='estimated_offsets',np.arange(len(rows)));assert np.all(pp[:,2]>0)
 e=np.linalg.norm(pp[:,:2]/pp[:,2:]-xy,axis=1);errors[label]=e
 reports[label]=dict(fit=metrics(e[fitmask]),validation=metrics(e[~fitmask]),offsets_frames={c:float(d) for c,d in zip(cameras,delta)},validation_per_camera={c:metrics(e[(ci==i)&~fitmask]) for i,c in enumerate(cameras)},validation_per_person={typ:metrics(e[(ji//18==pi)&~fitmask]) for pi,typ in enumerate(types)})
 np.savez_compressed(OUT/f'{label}-fit.npz',coefficients=co,offsets=delta,cameras=np.array(cameras),error_px=e,xy=xy,P=P,camera_index=ci,frame=ti,joint_index=ji,fitmask=fitmask)
delta=reports['estimated_offsets']['offsets_frames'];boundary=[c for c,x in delta.items() if abs(x)>1.98]
save(OUT/'timing-summary.json',dict(status='fixed_and_estimated_camera_offset_trajectories_fitted_reserved_time_evaluated_review_pending',records=reports,boundary_cameras=boundary,observations=len(rows),fit_observations=int(fitmask.sum()),validation_observations=int((~fitmask).sum()),qualification='Camera01defineszero. Predictedlandmarkvalidation, noRGBretimingorclarityacceptance.',goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(dict(validation_before=reports['fixed_offsets']['validation'],validation_after=reports['estimated_offsets']['validation'],offsets=delta,boundaries=boundary)))
