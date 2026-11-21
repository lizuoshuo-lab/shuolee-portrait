"""Estimate three new-camera clock offsets; all existing geometry remains frozen."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from scipy.optimize import least_squares
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/heldout';T=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'timing-summary.json').exists();assert sha(T)==read(R/'protocol.json')['trajectory_sha256'];coef=np.load(T)['coefficients'];assert coef.shape==(36,4,3)
rows=[]
for r in read(R/'observations.json')['records']:
 for d in r['detections']:
  for j in range(5,23):
   if d['scores'][j]<.4:continue
   rows.append(dict(camera=r['camera'],frame=r['frame'],person=d['person'],joint=d['person']*18+j-5,xy=d['keypoints'][j],P=(np.array(r['K'])@np.array(r['w2c'])[:3]).tolist(),score=min(1.,d['scores'][j])))
ci=np.array([r['camera'] for r in rows]);ti=np.array([r['frame'] for r in rows]);pi=np.array([r['person'] for r in rows]);ji=np.array([r['joint'] for r in rows]);xy=np.array([r['xy'] for r in rows]);P=np.array([r['P'] for r in rows]);w=np.sqrt([r['score'] for r in rows]);even=ti%2==0
def project(delta,ids):
 tau=(ti[ids]-750+float(delta))/6;basis=np.stack([tau**k for k in range(4)],1);X=np.einsum('nk,nkd->nd',basis,coef[ji[ids]]);q=np.einsum('nij,nj->ni',P[ids,:,:3],X)+P[ids,:,3];assert np.all(q[:,2]>0)
 derivative=(coef[ji[ids],1]+2*tau[:,None]*coef[ji[ids],2]+3*tau[:,None]**2*coef[ji[ids],3])/6
 dq=np.einsum('nij,nj->ni',P[ids,:,:3],derivative);J=(dq[:,:2]*q[:,2:]-q[:,:2]*dq[:,2:])/(q[:,2:]**2)
 return q[:,:2]/q[:,2:],J
zero=np.linalg.norm(project(0,np.arange(len(rows)))[0]-xy,axis=1);support=np.zeros(len(rows),bool);gates=[]
for c in ['00','06','12']:
 for j in range(36):
  ids=np.flatnonzero((ci==c)&(ji==j)&even);med=float(np.median(zero[ids])) if len(ids) else None;accepted=len(ids)>=4 and med<=40
  gates.append(dict(camera=c,joint=j,even_observations=len(ids),zero_offset_median_px=med,accepted=bool(accepted)))
  if accepted:support|=(ci==c)&(ji==j)
def metric(error):
 return dict(count=len(error),median_px=float(np.median(error)),rmse_px=float(np.sqrt(np.mean(error**2))),p90_px=float(np.percentile(error,90)))
reports={};checks=[];offsets={}
for c in ['00','06','12']:
 mask=(ci==c)&support;fit=np.flatnonzero(mask&even);val=np.flatnonzero(mask&~even);assert len(fit)>=30 and len(val)>=20
 def solve(ids):
  def fun(v):return np.r_[((project(v[0],ids)[0]-xy[ids])*w[ids,None]).ravel(),.1*v[0]]
  def jac(v):return np.r_[(project(v[0],ids)[1]*w[ids,None]).ravel(),.1][:,None]
  numeric=(fun([1e-6])-fun([-1e-6]))/2e-6;error=float(np.max(abs(numeric-jac([0])[:,0])));assert error<1e-3;checks.append(error)
  result=least_squares(fun,[0.],jac=jac,bounds=([-2.],[2.]),loss='soft_l1',f_scale=3,max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  profile=[]
  for d in np.linspace(-2,2,81):
   r=fun([d]);cost=float(np.sum(9*(np.sqrt(1+(r/3)**2)-1)));profile.append(dict(offset=float(d),cost=cost))
  return dict(offset=float(result.x[0]),nfev=result.nfev,cost=float(result.cost),profile=profile)
 combined=solve(fit);own={str(p):solve(fit[pi[fit]==p]) for p in [0,1]};d=combined['offset'];offsets[c]=d
 scores={}
 for label,delta in [('zero',0),('estimated',d),('person0_clock',own['0']['offset']),('person1_clock',own['1']['offset'])]:
  pred,_=project(delta,np.flatnonzero(mask));err=np.linalg.norm(pred-xy[mask],axis=1);indices=np.flatnonzero(mask)
  scores[label]=dict(fit=metric(err[even[indices]]),validation=metric(err[~even[indices]]),validation_per_person={str(p):metric(err[(~even[indices])&(pi[indices]==p)]) for p in [0,1]})
 reports[c]=dict(combined_fit=combined,person_fits=own,person_offset_disagreement=abs(own['0']['offset']-own['1']['offset']),boundary_hit=abs(d)>1.98,metrics=scores,fit_observations=len(fit),validation_observations=len(val))
 print(c,json.dumps(dict(offset=d,person_offsets=[own[str(p)]['offset'] for p in [0,1]],validation_before=scores['zero']['validation'],validation_after=scores['estimated']['validation'])),flush=True)
save(R/'timing-summary.json',dict(status='3newcamera_clocks_fitted_against_frozen_training_trajectories',offsets_frames=offsets,reports=reports,trajectory_sha256=sha(T),qualification='Appearanceheldout cameras contribute even-time predictedkeypoints toclockcalibration only; no geometry/appearance gradients. Person disagreement and boundary hits invalidate strong synchronization claims. Validation is against predictions, notGT.',goal_complete=False))
save(R/'timing-support.json',dict(gates=gates,fit_observations=[{k:v for k,v in r.items() if k not in ['xy','P']} for i,r in enumerate(rows) if support[i] and even[i]],validation_observations=int(np.sum(support&~even)),gate_uses_validation=False,goal_complete=False))
save(R/'timing-jacobian-check.json',dict(status='nine_offset_jacobians_match_central_differences',max_absolute_error=checks));shutil.copy2(__file__,R/Path(__file__).name)
