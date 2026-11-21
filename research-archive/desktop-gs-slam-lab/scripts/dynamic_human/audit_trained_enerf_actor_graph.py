"""Independent CPU check of saved graph parameters using SciPy rotations."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
R=L/'outputs/dynamic-human-enerf-actor-supervision-070'
H=L/'outputs/dynamic-human-enerf-head-motion-066'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
torch.set_num_threads(4)
cp=torch.load(R/'residual_graph/resume.pt',map_location='cpu',weights_only=False)
assert cp['step']==15000 and cp['graph_sha256']==sha(R/'graph.npz')
assert cp['motion_sha256']==sha(H/'capture-affines.npz')
assert [int(v['step']) for v in cp['graph_optim']['state'].values()]==[15000,15000]
assert all(torch.isfinite(v).all() for v in cp['params'].values())
g=dict(np.load(R/'graph.npz'));m=dict(np.load(H/'capture-affines.npz'))
for k in ['node_points','node_anchor','edges','edge_weights']:
 assert np.array_equal(g[k],cp['graph'][k].numpy()),k
rawR=cp['graph']['rotation_raw'].numpy().astype(np.float64)
rawd=cp['graph']['translation_raw'].numpy().astype(np.float64)
assert np.isfinite(rawR).all() and np.isfinite(rawd).all()
rot=.1*np.tanh(rawR);trans=.03*cp['scene_scale']*np.tanh(rawd)
assert np.linalg.norm(rot)>0 and np.linalg.norm(trans)>0
p={k:v.numpy() for k,v in cp['params'].items()}
assert p['node_indices'].dtype==np.int64
assert p['node_indices'].min()>=0 and p['node_indices'].max()<384
assert np.all(p['node_weights']>=0) and np.max(abs(p['node_weights'].sum(-1)-1))<1e-6
assert np.all(g['node_person'][p['node_indices']]==g['node_person'][p['node_indices']][:,:1])
ids=np.linspace(0,len(p['means'])-1,512,dtype=np.int64)
idx=p['node_indices'][ids];weights=p['node_weights'][ids].astype(np.float64)
weights/=weights.sum(-1,keepdims=True)
angles=[];node_displacements=[];detmin=1.;smin=1.;checks=[]
for c,f,t in zip(m['camera'],m['frame'],m['capture_time']):
 tau=(float(t)-750)/6;basis=np.array([tau,tau*tau,tau*tau*tau])
 rv=np.einsum('k,nkd->nd',basis,rot);d=np.einsum('k,nkd->nd',basis,trans)
 Q=Rotation.from_rotvec(rv).as_matrix()
 assert np.allclose(Q@Q.transpose(0,2,1),np.eye(3),atol=1e-12)
 linear=np.einsum('ni,nijk->njk',weights,Q[idx])
 determinants=np.linalg.det(linear);singular=np.linalg.svd(linear,compute_uv=False)
 assert determinants.min()>0 and singular.min()>0
 detmin=min(detmin,float(determinants.min()));smin=min(smin,float(singular.min()))
 angles.extend(np.linalg.norm(rv,axis=1));node_displacements.extend(np.linalg.norm(d,axis=1))
 checks.append(dict(camera=str(c),frame=int(f),physical_frame=float(t),sampled_gaussians=512,min_determinant=float(determinants.min()),min_singular_value=float(singular.min())))
zero=np.einsum('k,nkd->nd',np.zeros(3),rot)
assert np.array_equal(Rotation.from_rotvec(zero).as_matrix(),np.tile(np.eye(3),(384,1,1)))
report=dict(status='saved15000step_graph_CPU_SciPy_rotations_bindings_and_sampled_linear_maps_verified',
 checkpoint_sha256=sha(R/'residual_graph/resume.pt'),graph_sha256=sha(R/'graph.npz'),motion_sha256=sha(H/'capture-affines.npz'),
 context_count=195,sampled_gaussians_per_context=512,total_gaussians=len(p['means']),
 node_rotation_angle_rad=dict(median=float(np.median(angles)),max=float(np.max(angles))),
 node_translation_world_units=dict(median=float(np.median(node_displacements)),max=float(np.max(node_displacements))),
 sampled_min_determinant=detmin,sampled_min_singular_value=smin,reference_physical750_identity_exact=True,
 trained_graph_optimizer_steps=[15000,15000],body_motion_unchanged=True,checks=checks,
 scope='CPU independent SciPy rotations for all384nodes/all195contexts; 512fixedGaussians/context linearblend conditioning. No GTgeometry claim. Fullpose and licensed model arrays stay local.',goal_complete=False)
(R/'trained-graph-independent-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
