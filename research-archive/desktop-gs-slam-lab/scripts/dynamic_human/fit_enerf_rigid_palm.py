"""Bounded rigid-palm diagnostic: hold palm shape fixed across a short interval."""
from pathlib import Path
import json,hashlib,time,shutil
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont

L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-motion-082';S=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'palm-summary.json').exists()
obs=read(O/'observations.json')['records'];old=read(O/'trajectory-fit.json');joints=[0,5,9,13,17]
coeff=np.array([old['coefficients'][f'hand_stable_crop/{j}'] for j in joints])
reference=coeff[:,0];origin=reference.mean(0);local=reference-origin
with np.load(L/'outputs/dynamic-human-enerf-rgb-pose-correction-074/estimated_offsets-fit.npz') as z:offsets={str(c):float(d) for c,d in zip(z['cameras'],z['offsets'])}
save(O/'palm-protocol.json',dict(status='fixed_before_rigid_palm_fit',joints=joints,template='Wrist and four MCP bases, initialized from crop-stable independent trajectories at physical726. Template shape frozen; not anatomical GT.',
   fit='Only even frames, exclude camera04. Crop deviation<=3px; initial independent-trajectory residual<6px; equal weights, softL1 scale1.5px. All eligible targets reported afterward.',
   motion='Rotation vector cubic without constant term (R726=I); translation cubic.9rotation+12translation=21parameters; no scale or shape changes.',
   frozen='Cameras, clocks, images and Gaussian models remain unchanged.',source_sha256=sha(O/'trajectory-fit.json'),goal_complete=False))
allrows=[]
for r in obs:
    tau=(r['frame']+offsets[r['camera']]-726)/6;P=np.array(r['K'])@np.array(r['w2c'])[:3]
    for ji,j in enumerate(joints):
        X=(tau**np.arange(4))@coeff[ji];q=P@np.r_[X,1];xy=np.array(r['keypoints'][j]);old_error=float(np.linalg.norm(q[:2]/q[2]-xy))
        allrows.append(dict(camera=r['camera'],frame=r['frame'],joint=j,ji=ji,tau=tau,P=P,xy=xy,old_error=old_error,fit=bool(r['camera']!='04' and r['frame']%2==0 and r['crop_deviation_px'][j]<=3 and old_error<6)))
fit=[r for r in allrows if r['fit']];B=np.array([r['tau'] for r in fit])[:,None]**np.arange(4);P=np.array([r['P'] for r in fit]);xy=np.array([r['xy'] for r in fit]);ji=np.array([r['ji'] for r in fit])
assert np.linalg.matrix_rank(B)==4 and len(fit)>100
initial=np.r_[np.zeros(9),coeff.mean(0).ravel()]
def forward(v,basis,ps,jidx):
    rv=basis[:,1:]@v[:9].reshape(3,3);R=Rotation.from_rotvec(rv).as_matrix();T=basis@v[9:].reshape(4,3)
    X=np.einsum('nij,nj->ni',R,local[jidx])+T;q=np.einsum('nij,nj->ni',ps[:,:,:3],X)+ps[:,:,3]
    return q[:,:2]/q[:,2:],X
start=time.monotonic();solution=least_squares(lambda v:(forward(v,B,P,ji)[0]-xy).ravel(),initial,loss='soft_l1',f_scale=1.5,max_nfev=500,x_scale='jac')
results=[]
for r in allrows:
    b=(r['tau']**np.arange(4))[None];uv,X=forward(solution.x,b,r['P'][None],np.array([r['ji']]))
    split='excluded_camera04' if r['camera']=='04' else ('diagnostic_odd' if r['frame']%2 else 'training_even')
    results.append(dict(camera=r['camera'],frame=r['frame'],joint=r['joint'],split=split,used_for_fit=r['fit'],old_error_px=r['old_error'],error_px=float(np.linalg.norm(uv[0]-r['xy'])),xy=uv[0].tolist(),point=X[0].tolist()))
def metric(rr,key):
    a=np.array([r[key] for r in rr]);return dict(count=len(a),median_px=float(np.median(a)),p90_px=float(np.quantile(a,.9)),under3px_count=int((a<3).sum()))
co=solution.x;lengthchecks=[]
for a,b in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4)]:
    values=[];free=[]
    for t in np.linspace(-1.35,1.35,101):
        ba=np.tile(t**np.arange(4),(5,1));ps=np.tile(P[:1],(5,1,1));_,X=forward(co,ba,ps,np.arange(5));values.append(float(np.linalg.norm(X[a]-X[b])))
        X0=np.einsum('k,jkd->jd',t**np.arange(4),coeff);free.append(float(np.linalg.norm(X0[a]-X0[b])))
    assert max(values)-min(values)<1e-12
    lengthchecks.append(dict(joints=[joints[a],joints[b]],rigid_range=max(values)-min(values),free_range_over_median=(max(free)-min(free))/np.median(free)))
summary=dict(status='rigid_palm_fit_complete_not_applied_to_Gaussians',fit_count=len(fit),solver_success=bool(solution.success),solver_message=solution.message,seconds=time.monotonic()-start,
    metrics={split:{name:metric([r for r in results if r['split']==split],key) for name,key in [('free_joints','old_error_px'),('rigid_palm','error_px')]} for split in ['training_even','diagnostic_odd','excluded_camera04']},
    length_check=lengthchecks,parameters=dict(reference=reference.tolist(),rotation_coefficients=co[:9].reshape(3,3).tolist(),translation_coefficients=co[9:].reshape(4,3).tolist()),
    scope='Five palm landmarks only; no finger articulation, mesh, Gaussian updates, appearance or accuracy acceptance.',goal_complete=False)
save(O/'palm-summary.json',summary);save(O/'palm-records.json',dict(records=results))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
for c in ['04','09','13']:
    sheet=Image.new('RGB',(700,1015),'white');d=ImageDraw.Draw(sheet)
    d.text((8,4),f'Camera{c}: Native / hand model / free 3D palm / rigid 3D palm',font=font,fill='black');d.text((8,20),'Five landmarks only; camera04 and every odd frame excluded from all fitting. Native 1:1.',font=font,fill='black')
    for ti,t in enumerate([721,723,727,729,731]):
        r=next(r for r in obs if r['camera']==c and r['frame']==t);crop=r['crop']
        with np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz') as z:raw=Image.fromarray(z['rgb']).crop(crop)
        y=60+190*ti;d.text((8,y-18),f'Frame{t}',font=font,fill='black');sheet.paste(raw,(8,y))
        tau=(t+offsets[c]-726)/6;ps=np.array(r['K'])@np.array(r['w2c'])[:3];points=np.einsum('k,jkd->jd',tau**np.arange(4),coeff);q=(ps[:,:3]@points.T).T+ps[:,3]
        uv=[np.array(r['keypoints'])[joints],q[:,:2]/q[:,2:],np.array([next(rr['xy'] for rr in results if rr['camera']==c and rr['frame']==t and rr['joint']==j) for j in joints])]
        for k,p in enumerate(uv):
            im=raw.copy();draw=ImageDraw.Draw(im);p=p-np.array(crop[:2])
            for a,b in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4)]:draw.line([tuple(p[a]),tuple(p[b])],fill=(0,175,220),width=1)
            for x,y0 in p:draw.ellipse((x-1,y0-1,x+1,y0+1),fill=(250,40,20))
            sheet.paste(im,(178+k*170,y))
    sheet.save(O/'motion-panels'/f'camera{c}-palm.png')
shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k in ['fit_count','solver_success','seconds','metrics']}))
