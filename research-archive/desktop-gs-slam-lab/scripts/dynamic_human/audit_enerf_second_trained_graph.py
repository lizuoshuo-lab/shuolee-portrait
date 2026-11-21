"""CPU validation of both saved motion graphs with independent SciPy rotations."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from scipy.spatial.transform import Rotation
from scipy.interpolate import BSpline
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-second-interval-079';torch.set_num_threads(4)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
g=dict(np.load(R/'graph.npz'));m=dict(np.load(R/'head/capture-affines.npz'));bs=BSpline.basis_element([-2,-1,0,1,2],extrapolate=False);reports=[]
for branch in ['global_cubic','local_bspline']:
 cp=torch.load(R/branch/'resume.pt',map_location='cpu',weights_only=False);assert cp['step']==15000 and cp['graph_sha256']==sha(R/'graph.npz') and cp['motion_sha256']==sha(R/'head/capture-affines.npz')
 expectedsteps=[15000]*(2 if branch=='global_cubic' else 4);assert [int(x['step']) for x in cp['graph_optim']['state'].values()]==expectedsteps
 assert all(torch.isfinite(v).all() for v in cp['params'].values()) and all(torch.isfinite(v).all() for v in cp['graph'].values())
 for k in ['node_points','node_anchor','edges','edge_weights']:assert np.array_equal(g[k],cp['graph'][k].numpy())
 p={k:v.numpy() for k,v in cp['params'].items()};assert p['node_indices'].dtype==np.int64 and p['anchor'].dtype==np.int64
 assert p['anchor'].min()>=0 and p['anchor'].max()<13780 and p['node_indices'].min()>=0 and p['node_indices'].max()<384
 assert np.all(p['node_weights']>=0) and np.max(abs(p['node_weights'].sum(-1)-1))<1e-6
 assert np.all(g['node_person'][p['node_indices']]==(p['anchor']//6890)[:,None])
 ids=np.linspace(0,len(p['means'])-1,512,dtype=np.int64);idx=p['node_indices'][ids];w=p['node_weights'][ids].astype(np.float64);w/=w.sum(-1,keepdims=True)
 rot=.1*np.tanh(cp['graph']['rotation_raw'].numpy().astype(np.float64));trans=.03*cp['scene_scale']*np.tanh(cp['graph']['translation_raw'].numpy().astype(np.float64));assert np.linalg.norm(rot)>0 and np.linalg.norm(trans)>0
 if branch=='local_bspline':
  centers=cp['graph']['local_centers'].numpy().astype(np.float64);assert np.array_equal(centers,np.linspace(-1.5,1.5,7));lr=.03*np.tanh(cp['graph']['local_rotation_raw'].numpy().astype(np.float64));lt=.01*cp['scene_scale']*np.tanh(cp['graph']['local_translation_raw'].numpy().astype(np.float64));assert np.linalg.norm(lr)>0 and np.linalg.norm(lt)>0
 detmin=1.;smin=1.;maxangle=0.;maxdisp=0.
 for physical in list(m['capture_time'])+[726.]:
  tau=(float(physical)-726)/6;b=np.array([tau,tau*tau,tau*tau*tau]);rv=np.einsum('k,nkd->nd',b,rot);d=np.einsum('k,nkd->nd',b,trans)
  if branch=='local_bspline':
   v=np.nan_to_num(bs((tau-centers)/.5))-np.nan_to_num(bs(-centers/.5));rv+=np.einsum('k,nkd->nd',v,lr);d+=np.einsum('k,nkd->nd',v,lt)
  Q=Rotation.from_rotvec(rv).as_matrix();assert np.allclose(Q@Q.transpose(0,2,1),np.eye(3),atol=1e-12)
  if physical==726:assert np.array_equal(Q,np.tile(np.eye(3),(384,1,1))) and np.count_nonzero(d)==0
  A=np.einsum('ni,nijk->njk',w,Q[idx]);de=np.linalg.det(A);si=np.linalg.svd(A,compute_uv=False);assert de.min()>0 and si.min()>0
  detmin=min(detmin,float(de.min()));smin=min(smin,float(si.min()));maxangle=max(maxangle,float(np.linalg.norm(rv,axis=1).max()));maxdisp=max(maxdisp,float(np.linalg.norm(d,axis=1).max()))
 report=dict(branch=branch,checkpoint_sha256=sha(R/branch/'resume.pt'),step=15000,points=len(p['means']),graph_optimizer_steps=expectedsteps,
 contexts=195,all_nodes=384,sampled_gaussians_per_context=512,minimum_blend_determinant=detmin,minimum_blend_singular=smin,max_node_angle_rad=maxangle,max_node_displacement_world=maxdisp,physical726_identity_exact=True,goal_complete=False)
 save(R/branch/'independent-graph-audit.json',report);reports.append(report);del cp,p
save(R/'independent-trained-graph-audit.json',dict(status='both_saved_graphs_CPU_SciPy_rotations_material_identity_and_blend_maps_verified',records=reports,scope='Sampled linear map conditioning, no GT geometry or visual clarity claim.',goal_complete=False));print(json.dumps(reports),flush=True)
