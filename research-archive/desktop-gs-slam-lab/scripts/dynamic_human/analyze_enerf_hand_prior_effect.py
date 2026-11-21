"""Did the soft constraint actually change geometry? CPU selected-point computation."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_trajectory_model import HandTrajectory
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-prior-control-098';F=L/'outputs/dynamic-human-enerf-hand-fine-085/fine-graph.npz';torch.set_num_threads(4)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');z=dict(np.load(O/'association.npz'))
with np.load(S/'heldout/capture-affines-estimated.npz') as archive:clock={k:archive[k] for k in ['camera','frame','capture_time','linear','translation']}
ids=z['selected_parent_ids'];local=z['selected_local_ids'];weight=z['weight'].astype(float);weight/=weight.sum();lookup={(str(c),int(t)):i for i,(c,t) in enumerate(zip(clock['camera'],clock['frame']))};records=[]
paths={'parent':S/'local_bspline/resume.pt','free_motion':L/'outputs/dynamic-human-enerf-hand-trajectories-095/points/resume.pt','soft_prior':O/'soft_prior/resume.pt'}
for branch,path in paths.items():
 cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.numpy() for k,v in cp['params'].items()};scene=float(cp['scene_scale']);g=GraphMotion(S/'graph.npz',scene,device='cpu');g.load_state_dict(cp['graph']);g.requires_grad_(False);anchor=p['anchor'][ids];idx=p['node_indices'][ids];w=p['node_weights'][ids];w=w/w.sum(1,keepdims=True);node_anchor=g.node_anchor.numpy();node_points=g.node_points.numpy()
 if branch!='parent':
  d=HandTrajectory('points',F,scene,device='cpu');d.load_state_dict(cp['trajectory']);d.requires_grad_(False)
 with torch.no_grad():
  for c in sorted(set(z['camera'])-{'00','06','12'}):
   for f in range(720,733,2):
    i=lookup[c,f];A=clock['linear'][i].astype(float);b=clock['translation'][i].astype(float);body=np.einsum('nij,nj->ni',A[anchor],p['means'][ids])+b[anchor];nodes=np.einsum('nij,nj->ni',A[node_anchor],node_points)+b[node_anchor];tau=torch.tensor(np.float32(clock['capture_time'][i]),dtype=torch.float32).sub(726).div(6);R,tr=g(tau);R=R.numpy();tr=tr.numpy();mean=(np.einsum('nkij,nkj->nki',R[idx],body[:,None]-nodes[idx])+nodes[idx]+tr[idx]);mean=(mean*w[:,:,None]).sum(1)
    if branch!='parent':mean+=d(tau).numpy()[local]
    delta=mean-z['target'][i];distance=np.linalg.norm(delta,axis=-1);normal=((mean-z['surface'][i])*z['normal'][i]).sum(1);motion=np.log1p((distance/(.015*scene))**2);surface=np.log1p((np.maximum(abs(normal)-.003*scene,0)/(.006*scene))**2);loss=float(np.sum(weight*(motion+surface)*.5));records.append(dict(branch=branch,camera=c,frame=f,prior_loss=loss,weighted_tracking_RMSE_world=float(np.sqrt(np.sum(weight*distance**2))),weighted_absolute_normal_distance_world=float(np.sum(weight*abs(normal))),unweighted_median_tracking_distance_world=float(np.median(distance))))
 del cp,p,g
metrics={b:{k:float(np.mean([r[k] for r in records if r['branch']==b])) for k in ['prior_loss','weighted_tracking_RMSE_world','weighted_absolute_normal_distance_world','unweighted_median_tracking_distance_world']} for b in paths}
save(O/'prior-effect.json',dict(status='315training_context_parameter_only_comparisons_complete',records=records,metrics=metrics,scope='Distances to uncertain MANO candidate, not true hand geometry. Lower prior loss only verifies effect of the added penalty; image clarity must be judged independently.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(metrics),flush=True)
