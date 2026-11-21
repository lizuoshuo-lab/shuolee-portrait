"""One person's camera offsets must transfer to the other person's reserved times."""
from pathlib import Path
import json,shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (OUT/'person-transfer-summary.json').exists()
z=np.load(OUT/'fixed_offsets-fit.npz');ci=z['camera_index'];ti=z['frame'];ji=z['joint_index'];xy=z['xy'];P=z['P'];fitmask=z['fitmask'];cameras=z['cameras'];initial=z['coefficients'];types=['person_dark_clothing','person_light_clothing']
ob={(r['camera'],r['frame']):r for r in read(OUT/'observations.json')['records']}
weight=np.sqrt([min(1,next(d for d in ob[str(cameras[c]),int(t)]['detections'] if d['person']==types[int(j//18)])['scores'][int(j%18+5)]) for c,t,j in zip(ci,ti,ji)])
save(OUT/'person-transfer-protocol.json',dict(status='fixed_before_transfer_fits',source='062commonfixedsupport andpre-reservedoddtimeobservations',design='Fit separate offsets using eachperson on evenframes; transfer these fixed offsets to the otherperson, fitting onlythatpersons cubictrajectory on evenframes; evaluate oddframes.',acceptance='Requireconsistentoffsets andcrosspersonbenefit beforetreatingparametersasclockcorrection. This isasequentialdiagnostic aftercombined062fit, nota freshfinaltest.',goal_complete=False))
def solve(person,free_offsets=False,fixed_offsets=None):
 ids=np.where(fitmask&(ji//18==person))[0];localj=ji[ids]%18;n=len(ids);ncoef=18*12;fixed=np.zeros(15) if fixed_offsets is None else fixed_offsets
 def unpack(v,sel):
  co=v[:ncoef].reshape(18,4,3);d=np.r_[0,v[ncoef:]] if free_offsets else fixed;t=(ti[sel]-750+d[ci[sel]])/6;b=np.stack([t**k for k in range(4)],1);X=np.einsum('nk,nkd->nd',b,co[ji[sel]%18]);p=np.einsum('nij,nj->ni',P[sel,:,:3],X)+P[sel,:,3];return co,d,t,b,p
 def fun(v):
  co,d,t,b,p=unpack(v,ids);r=((p[:,:2]/p[:,2:]-xy[ids])*weight[ids,None]).ravel();return np.r_[r,.1*d[1:]] if free_offsets else r
 def jac(v):
  co,d,t,b,p=unpack(v,ids);D=(P[ids,:2,:3]*p[:,None,2,None]-p[:,:2,None]*P[ids,None,2,:3])/p[:,None,2,None]**2
  block=D[:,:,None,:]*b[:,None,:,None]*weight[ids,None,None,None];rr=np.repeat(np.arange(n*2),12);cc=np.broadcast_to(localj[:,None,None]*12+np.arange(12)[None,None,:],(n,2,12)).ravel();vv=block.ravel()
  if free_offsets:
   dx=(co[localj,1]+2*t[:,None]*co[localj,2]+3*t[:,None]**2*co[localj,3])/6;deriv=np.einsum('nij,nj->ni',D,dx)*weight[ids,None];has=ci[ids]>0
   rr=np.r_[rr,np.arange(2*n).reshape(n,2)[has].ravel(),2*n+np.arange(14)];cc=np.r_[cc,np.repeat(ncoef+ci[ids][has]-1,2),ncoef+np.arange(14)];vv=np.r_[vv,deriv[has].ravel(),np.full(14,.1)]
  return coo_matrix((vv,(rr,cc)),shape=(2*n+(14 if free_offsets else 0),ncoef+(14 if free_offsets else 0))).tocsr()
 v=initial[person*18:(person+1)*18].ravel();v=np.r_[v,np.zeros(14)] if free_offsets else v
 lo=np.r_[np.full(ncoef,-np.inf),np.full(14,-2)] if free_offsets else -np.inf;hi=np.r_[np.full(ncoef,np.inf),np.full(14,2)] if free_offsets else np.inf
 r=least_squares(fun,v,jac=jac,bounds=(lo,hi),loss='soft_l1',f_scale=3,max_nfev=300,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
 sel=np.where(ji//18==person)[0];co,d,t,b,p=unpack(r.x,sel);e=np.linalg.norm(p[:,:2]/p[:,2:]-xy[sel],axis=1);v=e[~fitmask[sel]]
 return dict(coefficients=co,offsets=d,validation_errors=v,validation_camera=ci[sel][~fitmask[sel]],solver_nfev=r.nfev,solver_status=int(r.status))
def summary(r):
 e=r['validation_errors'];return dict(median_px=float(np.median(e)),rmse_px=float(np.sqrt(np.mean(e**2))),p90_px=float(np.quantile(e,.9)),count=len(e),offsets={str(c):float(d) for c,d in zip(cameras,r['offsets'])},nfev=r['solver_nfev'],solver_status=r['solver_status'])
fixed=[solve(p) for p in range(2)];own=[solve(p,True) for p in range(2)];transfer=[solve(p,False,own[1-p]['offsets']) for p in range(2)]
diff=own[0]['offsets']-own[1]['offsets'];reports={}
for p,typ in enumerate(types):
 reports[typ]=dict(fixed=summary(fixed[p]),own_offsets=summary(own[p]),other_person_offsets=summary(transfer[p]))
 for label,r in [('fixed',fixed[p]),('own',own[p]),('transfer',transfer[p])]:np.savez_compressed(OUT/f'person{p}-{label}-fit.npz',**r)
save(OUT/'person-transfer-summary.json',dict(status='crossperson_offset_transfer_diagnostic_completed',records=reports,offset_difference_dark_minus_light_frames={str(c):float(d) for c,d in zip(cameras,diff)},offset_disagreement_rmse_frames=float(np.sqrt(np.mean(diff[1:]**2))),goal_complete=False))
shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(dict(offset_difference_rmse=float(np.sqrt(np.mean(diff[1:]**2))),records={typ:{k:v['median_px'] for k,v in rr.items()} for typ,rr in reports.items()})))
