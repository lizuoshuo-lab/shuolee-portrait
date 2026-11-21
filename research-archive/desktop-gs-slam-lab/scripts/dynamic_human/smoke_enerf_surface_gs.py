from pathlib import Path
import json,time,hashlib,subprocess
import torch,numpy as np,gsplat
from gsplat.rendering import rasterization_2dgs
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-surface-gs-055');OUT.mkdir(exist_ok=True)
torch.manual_seed(73);torch.set_num_threads(4);started=time.time()
mean=torch.tensor([[-.1,-.1,3],[.1,-.1,3],[.1,.1,3],[-.1,.1,3]],device='cuda',requires_grad=True)
q=torch.tensor([[1.,0,0,0]]*4,device='cuda',requires_grad=True)
s=torch.full((4,3),.12,device='cuda',requires_grad=True);o=torch.full((4,),.8,device='cuda',requires_grad=True)
c=torch.rand(4,3,device='cuda',requires_grad=True);K=torch.tensor([[[100.,0,32],[0,100.,32],[0,0,1.]]],device='cuda')
r,a,n,nd,d,median,info=rasterization_2dgs(mean,q,s,o,c[None],torch.eye(4,device='cuda')[None],K,64,64,
    packed=False,render_mode='RGB+ED',distloss=True,near_plane=.1,far_plane=20)
assert all(torch.isfinite(t).all() for t in [r,a,n,nd,d,median])
info['gradient_2dgs'].retain_grad();loss=(r[...,:3]*torch.linspace(0,1,64,device='cuda')[None,None,:,None]).sum()+.01*d.mean();loss.backward()
assert mean.grad is not None and torch.isfinite(mean.grad).all() and mean.grad.abs().sum()>0
assert info['gradient_2dgs'].grad is not None and torch.isfinite(info['gradient_2dgs'].grad).all()
torch.cuda.synchronize()
report=dict(status='isolated_2dgs_native_CUDA_forward_backward_and_geometry_outputs_passed',version=gsplat.__version__,module=gsplat.__file__,
    device=torch.cuda.get_device_name(0),compute_capability=list(torch.cuda.get_device_capability()),render_shape=list(r.shape),
    alpha_max=float(a.max()),means_grad_sum=float(mean.grad.abs().sum()),gradient2dgs_sum=float(info['gradient_2dgs'].grad.abs().sum()),
    elapsed_seconds=time.time()-started,production_training_started=False,goal_complete=False)
(OUT/'runtime-smoke.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
