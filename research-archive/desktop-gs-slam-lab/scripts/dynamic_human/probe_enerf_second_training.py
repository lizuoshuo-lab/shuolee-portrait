"""Discarded real CUDA forwards/backwards; old and zero-extended models agree."""
from pathlib import Path
import json,hashlib,gc
import numpy as np,torch
from enerf_articulated_model import initial_parameters
from enerf_second_renderer import GraphMotion as Cubic,ResidualGraphRenderer as CR,bind_graph
from enerf_second_temporal_graph import GraphMotion as Local,ResidualGraphRenderer as LR
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-second-interval-079'
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();t=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
assert not (R/'runtime-smoke.json').exists();torch.set_num_threads(4);torch.manual_seed(73)
records=[]
for c,fr in [('04',720),('07',732)]:
 z=np.load(R/'native-input'/f'camera{c}-frame{fr:06d}.npz');x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;w=int(x1-x0);h=int(y1-y0);target=t(z['rgb'][y0:y1,x0:x1]/255);matte=t(z['matte'][y0:y1,x0:x1])[...,None];outputs=[];grads=[]
 for Model,Renderer in [(Cubic,CR),(Local,LR)]:
  p=initial_parameters(R/'initial.npz');scale=float(torch.linalg.vector_norm(p['means'].max(0).values-p['means'].min(0).values)*.5);g=Model(R/'graph.npz',scale);bind_graph(p,R/'graph.npz');model=Renderer(R/'head/capture-affines.npz',g)
  rgb,a,info=model.render(p,c,fr,t(z['w2c']),t(K),w,h,0);loss=((rgb[0]+.5*(1-a[0]))-(target*matte+.5*(1-matte))).abs().mean();loss.backward()
  gradient={n:float(v.grad.abs().sum()) for n,v in g.named_parameters()};assert all(v>0 and np.isfinite(v) for v in gradient.values())
  assert torch.isfinite(p['means'].grad).all() and p['means'].grad.abs().sum()>0
  outputs.append((rgb.detach().cpu().numpy(),a.detach().cpu().numpy()));grads.append(gradient)
  refR,refd=g(t(0));assert float((refR-torch.eye(3,device='cuda')).abs().max())==0 and float(refd.abs().max())==0
  del rgb,a,info,p,g,model,loss;gc.collect();torch.cuda.empty_cache()
 dr=float(abs(outputs[0][0]-outputs[1][0]).max());da=float(abs(outputs[0][1]-outputs[1][1]).max());assert dr==0 and da==0
 records.append(dict(camera=c,frame=fr,RGB_max_difference=dr,alpha_max_difference=da,graph_gradient_L1=grads))
save(R/'runtime-smoke.json',dict(status='new_interval_forward_gradients_zero_extension_verified',records=records,initial_sha256=sha(R/'initial.npz'),graph_sha256=sha(R/'graph.npz'),motion_sha256=sha(R/'head/capture-affines.npz'),discarded=True,trained_updates=0,goal_complete=False));print('NEW_INTERVAL_RUNTIME_PROBE_PASSED',flush=True)
