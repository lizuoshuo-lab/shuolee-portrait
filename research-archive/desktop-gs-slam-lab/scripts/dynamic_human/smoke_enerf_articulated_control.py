from pathlib import Path
import json,shutil
import numpy as np,torch
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate,split
from enerf_articulated_model import initial_parameters,ArticulatedRenderer
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';MOTION=LAB/'outputs/dynamic-human-enerf-continuous-body-063';torch.set_num_threads(8);torch.manual_seed(73)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
z=np.load(OUT/'native-input/camera04-frame000750.npz');x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K=tensor(K);W=tensor(z['w2c']);w=int(x1-x0);h=int(y1-y0);p=initial_parameters(OUT/'initial.npz');m=ArticulatedRenderer(MOTION/'estimated_offsets/capture-affines.npz');i=m.lookup['04',750]
origA=m.A[i].clone();origb=m.b[i].clone();origI=m.inverse[i].clone();m.A[i]=torch.eye(3,device='cuda');m.b[i]=0;m.inverse[i]=torch.eye(3,device='cuda')
with torch.no_grad():
 # Nonzero higherorder SH and anisotropic scales make the identity check meaningful.
 p['shN'].copy_(torch.rand_like(p['shN'])*.03);p['scales'][:,0].add_(.15)
 ra,aa,_=m.render(p,'04',750,W,K,w,h,2);rb,ab,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),W[None],K[None],w,h,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
 identity_rgb_mae=float(abs(ra-rb).mean());identity_alpha_mae=float(abs(aa-ab).mean());assert identity_rgb_mae<1e-5 and identity_alpha_mae<1e-5,(identity_rgb_mae,identity_alpha_mae)
 m.A[i]=origA;m.b[i]=origb;m.inverse[i]=origI
r,a,_=m.render(p,'04',750,W,K,w,h,2);target=tensor(z['rgb'][y0:y1,x0:x1]/255);loss=(r[0]+.5*(1-a[0])-target).square().mean();loss.backward();grad={k:float(v.grad.abs().sum()) for k,v in p.items() if v.requires_grad}
assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in p.values() if v.requires_grad)
# Actual strategy operations must preserve the discrete binding, without an optimizer.
opt={k:torch.optim.Adam([v],lr=.001) for k,v in p.items() if v.requires_grad};strategy=DefaultStrategy();strategy.check_sanity(p,opt);state=strategy.initialize_state(1.);old=p['anchor'].clone();mask=torch.zeros(len(old),device='cuda',dtype=torch.bool);mask[:10]=True;duplicate(p,opt,state,mask);assert torch.equal(p['anchor'],torch.cat([old,old[:10]]));old=p['anchor'].clone();mask=torch.zeros(len(old),device='cuda',dtype=torch.bool);mask[:10]=True;split(p,opt,state,mask);assert torch.equal(p['anchor'],torch.cat([old[10:],old[:10],old[:10]]))
assert p['anchor'].dtype==torch.long and not p['anchor'].requires_grad
report=dict(status='identity_affine_covariance_gradients_and_density_bindings_verified',identity_rgb_mae=identity_rgb_mae,identity_alpha_mae=identity_alpha_mae,actual_deformation_loss=float(loss),finite_nonzero_gradients=grad,binding_duplicate_split='actualopsverified',max_gpu_bytes=torch.cuda.max_memory_allocated(),note='Smokeparametersdiscarded; trainingstartsagainfromimmutableinitial.npz',goal_complete=False);save(OUT/'runtime-smoke.json',report);shutil.copy2(__file__,OUT/Path(__file__).name);print(json.dumps(report))
