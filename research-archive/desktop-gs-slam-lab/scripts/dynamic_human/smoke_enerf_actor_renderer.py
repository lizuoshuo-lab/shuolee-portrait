from pathlib import Path
import json,numpy as np,torch,hashlib
from enerf_articulated_model import initial_parameters
from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer,bind_graph
from enerf_actor_renderer import ActorSupervisedRenderer,ownership_loss
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-actor-supervision-070';I=L/'outputs/dynamic-human-enerf-articulated-control-064';motion=L/'outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz'
torch.set_num_threads(8);torch.manual_seed(73);p=initial_parameters(I/'initial.npz');scale=float(torch.linalg.vector_norm(p['means'].max(0).values-p['means'].min(0).values)*.5);graph=GraphMotion(R/'graph.npz',scale);bind_graph(p,R/'graph.npz');base=ResidualGraphRenderer(motion,graph);model=ActorSupervisedRenderer(motion,graph)
z=np.load(I/'native-input/camera04-frame000750.npz');x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda');labels=torch.tensor(np.load(R/'masks/camera04-frame000750.npz')['labels'][y0:y1,x0:x1],device='cuda',dtype=torch.long)
with torch.no_grad():rgb0,a0,_=base.render(p,'04',750,tensor(z['w2c']),tensor(K),int(x1-x0),int(y1-y0),2)
rgb,alpha,info=model.render(p,'04',750,tensor(z['w2c']),tensor(K),int(x1-x0),int(y1-y0),2);weights=info['actor_contributions'];err=dict(rgb=float((rgb-rgb0).abs().max()),alpha=float((alpha-a0).abs().max()),sum_alpha=float((weights.sum(-1,keepdim=True)-alpha).abs().max()));assert max(err.values())<5e-6
loss=ownership_loss(weights,alpha,labels);loss.backward();gradient={k:float(p[k].grad.abs().sum()) for k in ['means','opacities']};gradient['graph_translation']=float(graph.translation_raw.grad.abs().sum());assert all(np.isfinite(v) and v>0 for v in gradient.values())
# Loss uses joint contributions; class-constant perfectly assigned probabilities should approach zero.
dummy=torch.tensor([[[[1.,0.],[0.,1.]]]],requires_grad=True);ones=torch.ones(1,1,2,1);lab=torch.tensor([[0,1]]);good=float(ownership_loss(dummy,ones,lab));bad=float(ownership_loss(dummy.flip(-1),ones,lab));assert good<2e-6 and bad>10
report=dict(passed=True,status='actual_cuda_RGB_alpha_identity_actor_sum_and_ownership_gradients_verified',max_absolute_errors=err,loss=float(loss),gradients=gradient,perfect_assignment_loss=good,wrong_assignment_loss=bad,goal_complete=False)
(R/'actor-runtime-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
