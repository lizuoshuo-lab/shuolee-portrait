"""New even-time face/ear consensus and a head-only fixed-budget fit script."""
from pathlib import Path
import itertools,json,hashlib
import numpy as np
from scipy.optimize import least_squares
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'head';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (O/'head-support.json').exists()
obs=read(R/'pose/observations.json')['records'];types=['person_dark_clothing','person_light_clothing'];rows=[]
save(O/'support-protocol.json',dict(status='fixed_before_consensus',frames=list(range(720,733,2)),
 targets='dark nose/eyes/ears; light ears only because most views are rear-facing. Training observations only. Same confidence0.4,8px,min4views,>2degree ray angle as body consensus.',
 limitation='Geometric support for predicted landmarks is not ground truth or visibility certification. No face details or fingers synthesized by this fitting.',goal_complete=False))
def proj(x,o):
 q=o['P'][:,:3]@x+o['P'][:,3];return q[:2]/q[2],q[2]
def err(x,os):return np.array([np.linalg.norm(q-o['xy']) if z>0 else 1e9 for o in os for q,z in [proj(x,o)]])
for f in range(720,733,2):
 for pi,typ in enumerate(types):
  for j in (range(5) if pi==0 else [3,4]):
   os=[]
   for r in obs:
    if r['frame']!=f:continue
    dd=[d for d in r['detections'] if d['person']==typ]
    if not dd or dd[0]['scores'][j]<.4:continue
    d=dd[0];W=np.array(r['w2c']);os.append(dict(camera=r['camera'],P=np.array(r['K'])@W[:3],W=W,xy=np.array(d['keypoints'][j])))
   best=None
   for a,b in itertools.combinations(os,2):
    A=np.stack([xy*p[2]-p[k] for o in [a,b] for p in [o['P']] for k,xy in enumerate(o['xy'])]);_,_,v=np.linalg.svd(A)
    if abs(v[-1,3])<1e-12:continue
    x=v[-1,:3]/v[-1,3];e=err(x,os);ins=e<8;key=(int(ins.sum()),-float(np.median(e[ins]))) if ins.any() else (0,-1e9)
    if best is None or key>best[0]:best=(key,x,ins)
   rr=dict(frame=f,person=typ,index=j,accepted=False)
   if best is None or best[0][0]<4:rows.append(rr);continue
   _,x,ins=best
   for _ in range(3):
    x=least_squares(lambda y:np.concatenate([proj(y,o)[0]-o['xy'] for o,b in zip(os,ins) if b]),x,loss='soft_l1',f_scale=3,max_nfev=100).x;e=err(x,os);new=e<8
    if np.array_equal(ins,new) or new.sum()<4:ins=new;break
    ins=new
   dirs=[]
   for o,b in zip(os,ins):
    if b:
     center=-o['W'][:3,:3].T@o['W'][:3,3];v=x-center;dirs.append(v/np.linalg.norm(v))
   angle=max([float(np.degrees(np.arccos(np.clip(a@b,-1,1)))) for a,b in itertools.combinations(dirs,2)],default=0.)
   rr.update(accepted=bool(ins.sum()>=4 and angle>2),point=x.tolist(),max_ray_angle_deg=angle,observations=[dict(camera=o['camera'],error_px=float(ei),inlier=bool(b)) for o,ei,b in zip(os,e,ins)]);rows.append(rr)
save(O/'head-support.json',dict(records=rows,accepted=sum(r['accepted'] for r in rows),total=len(rows),goal_complete=False))
text=(S/'fit_enerf_rgb_head.py').read_text()
def once(a,b):
 global text
 assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b,1)
text=text.replace('dynamic-human-enerf-rgb-pose-correction-074/body','dynamic-human-enerf-second-interval-079/body').replace("SEQ=LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074'","SEQ=LAB/'outputs/dynamic-human-enerf-second-interval-079/pose'").replace('dynamic-human-enerf-rgb-pose-correction-074/head','dynamic-human-enerf-second-interval-079/head').replace('estimated_offsets/motion-fit.npz','fixed_clock/motion-fit.npz')
once("support={(r['person'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OLD/'triangulation.json')['records'] if r['accepted']}","support={(r['person'],r['frame'],r['index']):{o['camera'] for o in r.get('observations',[]) if o['inlier']} for r in read(OUT/'head-support.json')['records'] if r['accepted']}")
once("detection=next(d for d in r['detections'] if d['person']==person)","available=[d for d in r['detections'] if d['person']==person]\n  if not available:continue\n  detection=available[0]")
once('support.get((person,j),set())',"support.get((person,r['frame'],j),set())")
once('assert len(rows)==546 and len(contexts)==105',"assert len(rows)>=200 and len(contexts)==105;assert all(sum(r['person']==pi for r in rows)>=50 for pi in [0,1])")
text=text.replace('744,757','720,733').replace('745,756','721,732').replace('times-750','times-726').replace('ts-750','ts-726')
once("targets='Same546training camera/time/head indices as065; correctedRGB predictedcoordinates; dark5face points, light2ears only; RTMWscore>=0.4 and059fixedcamera support. No body/foot/hand or odd-time observations in fitting.'","targets='New079even-time geometric consensus, dark5face landmarks and light2ears. No old059support or reservedtime observations reused.'")
once('exactly074 RGB-corrected estimated-offset body branch','exactly079new-interval fixed-clock body branch')
once('Identical canonical reference and exact affine covariance renderer, identical15000Gaussianupdates andfixed development-time validation.','Rebase canonical surface to final head-refined physicalframe726. Both later appearance branches share this exact geometry and motion.')
once("v0=t(ref['vertices']);inv0=t(ref['inverse_linear'])","""v0=t(ref['vertices']);inv0=t(ref['inverse_linear'])
# Rebase after head refinement; all final affines use the final canonical726.
with torch.no_grad():
 pose,tr=query(t([[1.,0.,0.,0.]]));v,j,M=lbs(betas,pose[:,0],model.v_template,model.shapedirs,model.posedirs,model.J_regressor,model.parents,model.lbs_weights,return_T=True)
 j=model.vertex_joint_selector(v,j)*scales[:,None,None]+tr[:,0,None,:];v0=v*scales[:,None,None]+tr[:,0,None,:];M0=M[:,:,:3,:3]*scales[:,None,None,None];inv0=torch.linalg.inv(M0)
 assert float((M0@inv0-torch.eye(3,device='cuda')).abs().max())<1e-4
 np.savez_compressed(OUT/'reference.npz',vertices=v0.cpu().numpy(),joints=j.cpu().numpy(),linear=M0.cpu().numpy(),inverse_linear=inv0.cpu().numpy(),faces=model.faces,betas=f['betas'],scale=f['scales'],basepose=pose[:,0].cpu().numpy(),translation=tr[:,0].cpu().numpy(),reference_frame=np.array(726))
""")
(S/'fit_enerf_second_head.py').write_text(text)
(S/'run_enerf_second_head.sh').write_text('#!/bin/bash\nset -euo pipefail\nsource /opt/gs-slam-lab/surface-gs/env.sh\npython /mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/scripts/dynamic_human/fit_enerf_second_head.py\n',newline='\n')
print(json.dumps(dict(accepted=sum(r['accepted'] for r in rows),total=len(rows),status='head_support_and_script_ready_not_fitted')))
