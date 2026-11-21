"""Discarded GPU check: parent render equivalence, local motion isolation and real gradients."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_joint_model import HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';O=L/'outputs/dynamic-human-enerf-hand-joint-084'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(8);parent=S/'local_bspline/resume.pt';before=sha(parent);cp=torch.load(parent,map_location='cpu',weights_only=False);keys=['means','scales','quats','opacities','sh0','shN'];p={k:torch.nn.Parameter(v.cuda(),requires_grad=k in keys) for k,v in cp['params'].items()};active=torch.from_numpy(np.load(O/'support/active-gaussians.npy')).cuda();nodes=torch.from_numpy(np.load(O/'support/active-nodes.npy')).cuda()
graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['graph']);ref.requires_grad_(False);joint=HandJointRenderer(S/'head/capture-affines.npz',graph,ref,active);old=ResidualGraphRenderer(S/'head/capture-affines.npz',ref)
for k in keys:p[k].register_hook(lambda grad:grad*active.reshape((-1,)+(1,)*(grad.ndim-1)))
for par in graph.parameters():par.register_hook(lambda grad:grad*nodes[:,None,None])
checks=[]
for c,t in [('04',732),('07',730),('13',728)]:
 z=np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz');m=np.load(F/'hand-masks'/f'camera{c}-frame{t:06d}.npz');x0,y0,x1,y1=m['crop'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;W=tensor(z['w2c']);K=tensor(K)
 with torch.no_grad():
  rgb,alpha,_=old.render(p,c,t,W,K,160,160,2);new,al,_=joint.render(p,c,t,W,K,160,160,2);err=max(float((rgb-new).abs().max()),float((alpha-al).abs().max()));assert err<1e-6
  means,cov,linear,_=joint.geometry(p,c,t);assert (means-joint.positions(p,c,t)).abs().max()<1e-7
  graph.local_translation_raw[nodes]+=0.01;p['means'][active]+=0.0001
  changed,C,lin,_=joint.geometry(p,c,t);assert torch.equal(means[~active],changed[~active]) and torch.equal(cov[~active],C[~active]) and torch.equal(linear[~active],lin[~active]);assert (changed[active]-means[active]).abs().max()>0
  graph.load_state_dict(cp['graph']);p['means'].copy_(cp['params']['means'].cuda())
 new,al,info=joint.render(p,c,t,W,K,160,160,2);target=tensor(z['rgb'][y0:y1,x0:x1])/255;loss=(new[0]+.5*(1-al[0])-target).square().mean();loss.backward();gradients={k:float(p[k].grad[active].abs().sum()) for k in keys};assert all(p[k].grad is not None and torch.isfinite(p[k].grad).all() and p[k].grad[~active].abs().max()==0 for k in keys);assert gradients['means']>0 and gradients['sh0']>0
 for k in keys:p[k].grad=None
 for par in graph.parameters():assert par.grad is not None and torch.isfinite(par.grad).all() and par.grad[~nodes].abs().max()==0;par.grad=None
 checks.append(dict(camera=c,frame=t,initial_render_max_error=err,nonhand_geometry_exact_after_discarded_perturbation=True,active_parameter_gradient_sums=gradients))
r=next(r for r in read(F/'model-support/completion.json')['records'] if r['model_visible']>0);z=np.load(F/'model-support'/r['filename']);src=torch.load(F/'model-support'/f'camera{r["camera"]}-frame{r["source_frame"]:06d}-source.pt',map_location='cpu',weights_only=False);source={k:src[k].cuda() for k in ['W','K']};pair=dict(camera=r['camera'],source_frame=r['source_frame'],target_frame=r['target_frame'],xy=tensor(z['local_xy']),normal=tensor(z['normal']));normal,flow,a=joint.current_normal_flow(p,pair,source);assert (normal-tensor(z['parent_normal'])).abs().max()<1e-5;normal.square().mean().backward();assert p['means'].grad is not None and torch.isfinite(p['means'].grad).all();assert all(torch.isfinite(v.grad).all() for v in graph.parameters());assert sha(parent)==before
save(O/'runtime-smoke.json',dict(status='3native_parent_render_equivalence_nonhand_isolation_photo_and_currentflow_gradients_passed',parent_sha256=before,checks=checks,flow_parent_max_error=float((normal-tensor(z['parent_normal'])).abs().max()),active_Gaussians=int(active.sum()),active_nodes=int(nodes.sum()),no_model_checkpoint_saved=True,goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print('HAND_JOINT_SMOKE_OK',flush=True)
