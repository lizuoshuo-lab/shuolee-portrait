"""Check that time calibration improves moving tracks, not only static regions."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';Q=O/'evaluation-time'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
s=read(Q/'summary.json');cams=s['calibration_cameras'];ref=cams.index('04');variables=[i for i in range(18) if i!=ref];old=np.array([s['clock_fit']['offsets'][cams[i]] for i in variables]);rows=[]
def interp(x,t):return x[:,1]+t/4*(x[:,2]-x[:,0])+t*t/8*(x[:,2]-2*x[:,1]+x[:,0])
def dvec(v):
    d=np.zeros(18);d[variables]=v;return d
def errors(v,rr):
    d=dvec(v);out=[]
    for r in rr:
        x=interp(r['a'],-d[r['i']]);y=interp(r['b'],-d[r['j']]);xx=np.c_[x,np.ones(len(x))];yy=np.c_[y,np.ones(len(y))];l2=xx@r['F'].T;l1=yy@r['F'];out.append((yy*l2).sum(1)/np.sqrt((l1[:,:2]**2).sum(1)+(l2[:,:2]**2).sum(1)+1e-30))
    return np.concatenate(out)
def stats(x):return dict(count=len(x),median=float(np.median(abs(x))),p90=float(np.quantile(abs(x),.9)),fraction_below1p5=float(np.mean(abs(x)<1.5)))
for p in sorted(Q.glob('pair*-frame*.npz')):
    z=np.load(p);motion=np.maximum(np.linalg.norm(z['a'][:,2]-z['a'][:,0],axis=1),np.linalg.norm(z['b'][:,2]-z['b'][:,0],axis=1));mask=motion>=2
    if mask.sum()<8:continue
    ca,cb=z['cameras'].tolist();rows.append(dict(i=cams.index(ca),j=cams.index(cb),frame=int(z['anchor']),a=z['a'][mask],b=z['b'][mask],F=z['F'],count=int(mask.sum())))
fit=[r for r in rows if r['frame'] in [584,592]];val=[r for r in rows if r['frame']==588];zero=np.zeros(17)
sol=least_squares(lambda v:errors(v,fit),zero,bounds=(-2,2),loss='cauchy',f_scale=1,max_nfev=150,xtol=1e-9,ftol=1e-9,gtol=1e-9)
sv=np.linalg.svd(sol.jac,compute_uv=False);branches={'zero':zero,'all_track_fit':old,'moving_track_fit':sol.x};result=dict(status='moving_track_diagnostic_no_training_or_camera_change',motion_threshold_px_over4frame_indices=2,moving_training_pairs=len(fit),moving_validation_pairs=len(val),fit={k:stats(errors(v,fit)) for k,v in branches.items()},validation={k:stats(errors(v,val)) for k,v in branches.items()},moving_offsets=dict(zip(cams,dvec(sol.x).tolist())),jacobian_singular_values=sv.tolist(),jacobian_rank=int(np.linalg.matrix_rank(sol.jac)),normal_equation_condition_proxy=float((sv[0]/sv[-1])**2),max_difference_from_all_track_offsets=float(abs(sol.x-old).max()),visual_scope='Disjoint calibration-only correspondence panels have not yet been visually reviewed; this is a numerical moving-track check.',interpretation='Motion-conditioned track errors qualify empirical timing only. Fitted Jacobian rank is a local numerical observability check, not timing ground truth or uncertainty calibration.',goal_complete=False)
save(Q/'moving-track-clock-audit.json',result);print(json.dumps(dict(validation=result['validation'],rank=result['jacobian_rank'],max_offset_difference=result['max_difference_from_all_track_offsets'])))
