from pathlib import Path
import json,shutil,numpy as np,torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate,split
from enerf_articulated_model import initial_parameters
from enerf_graph_renderer import GraphMotion,bind_graph,ResidualGraphRenderer
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-residual-graph-067';DATA=LAB/'outputs/dynamic-human-enerf-articulated-control-064';MOTION=LAB/'outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz'
torch.set_num_threads(4);torch.manual_seed(73);p=initial_parameters(DATA/'initial.npz');scale=float(torch.linalg.vector_norm(p['means'].max(0).values-p['means'].min(0).values)*.5);bind_graph(p,OUT/'graph.npz');g=GraphMotion(OUT/'graph.npz',scale);m=ResidualGraphRenderer(MOTION,g);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
with torch.no_grad():p['shN'].copy_(torch.rand_like(p['shN'])*.03);p['scales'][:,0].add_(.15)
def context(camera):
 z=dict(np.load(DATA/'native-input'/f'camera{camera}-frame000750.npz'));x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0
 return z,tensor(z['w2c']),tensor(K),int(x1-x0),int(y1-y0),tensor(z['rgb'][y0:y1,x0:x1]/255)
z,W,K,w,h,target=context('04')
with torch.no_grad():
 r0,a0,_=m.base.render(p,'04',750,W,K,w,h,2);r,a,_=m.render(p,'04',750,W,K,w,h,2);rgb_error=float((r-r0).abs().mean());alpha_error=float((a-a0).abs().mean());assert rgb_error<1e-5 and alpha_error<1e-5
g.rotation_raw.data.normal_(std=.01);g.translation_raw.data.normal_(std=.01)
r,a,info=m.render(p,'04',750,W,K,w,h,2);loss=(r[0]+.5*(1-a[0])-target).square().mean()+.01*info['graph_rigidity'];loss.backward();grad={k:float(v.grad.abs().sum()) for k,v in p.items() if v.requires_grad};graphgrad={k:float(v.grad.abs().sum()) for k,v in g.named_parameters()};assert all(np.isfinite(v) and v>0 for v in [*grad.values(),*graphgrad.values()]);actual_change=float((r-r0).abs().mean());assert actual_change>1e-9
# Even with nonzero motion coefficients, physical time750 has exact residual identity.
z,W,K,w,h,_=context('01')
with torch.no_grad():
 ref_r,ref_a,_=m.base.render(p,'01',750,W,K,w,h,2);test_r,test_a,_=m.render(p,'01',750,W,K,w,h,2);reference_error=float((ref_r-test_r).abs().mean());assert reference_error<1e-5
opt={k:torch.optim.Adam([v],lr=.001) for k,v in p.items() if v.requires_grad};strategy=DefaultStrategy();strategy.check_sanity(p,opt);state=strategy.initialize_state(scale)
fields=['anchor','node_indices','node_weights'];old={k:p[k].clone() for k in fields};chosen=torch.zeros(len(p['means']),device='cuda',dtype=torch.bool);chosen[:10]=True;duplicate(p,opt,state,chosen)
for k in fields:assert torch.equal(p[k],torch.cat([old[k],old[k][:10]])) and not p[k].requires_grad
old={k:p[k].clone() for k in fields};chosen=torch.zeros(len(p['means']),device='cuda',dtype=torch.bool);chosen[:10]=True;split(p,opt,state,chosen)
for k in fields:assert torch.equal(p[k],torch.cat([old[k][10:],old[k][:10],old[k][:10]])) and not p[k].requires_grad
result=dict(status='actual_CUDA_identity_full_covariance_motion_gradients_and_density_material_bindings_verified',identity_rgb_mae=rgb_error,identity_alpha_mae=alpha_error,nonzero_motion_rgb_change=actual_change,reference_time_identity_rgb_mae=reference_error,finite_nonzero_gaussian_gradients=grad,finite_nonzero_graph_gradients=graphgrad,all_capture_rotation_bound_radians=m.max_rotation_bound,actual_duplicate_split_material_binding_check=True,max_gpu_bytes=torch.cuda.max_memory_allocated(),qualification='Only discarded smoke forwards/backward. No graph model has been trained; no clarity claim.',goal_complete=False)
(OUT/'runtime-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
for n in ['smoke_enerf_graph_renderer.py','enerf_graph_renderer.py','enerf_residual_graph.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,OUT/n)
print(json.dumps(result))
