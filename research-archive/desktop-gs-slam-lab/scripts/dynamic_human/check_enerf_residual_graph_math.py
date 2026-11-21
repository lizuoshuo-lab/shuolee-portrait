from pathlib import Path
import json,shutil
import torch
from enerf_residual_graph import rotation_matrix,transform_gaussians,graph_rigidity,polynomial_transform
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-residual-graph-067';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(73);dtype=torch.float64
means=torch.randn(8,3,dtype=dtype);g=torch.randn(4,3,dtype=dtype);Q=torch.randn(8,3,3,dtype=dtype);cov=Q@Q.transpose(-1,-2)+.01*torch.eye(3,dtype=dtype);indices=torch.tensor([[0,1,2,3]]*8);weights=torch.rand(8,4,dtype=dtype);weights/=weights.sum(-1,keepdim=True)
zero=torch.zeros(4,3,dtype=dtype);R=rotation_matrix(zero);v,C,A=transform_gaussians(means,cov,g,R,zero,indices,weights);identity_error=max(float((v-means).abs().max()),float((C-cov).abs().max()));assert identity_error<1e-12
w=torch.tensor([.3,-.2,.1],dtype=dtype);common=rotation_matrix(w);translation=torch.tensor([.02,.07,-.03],dtype=dtype);d=(common@g.T).T-g+translation
v,C,A=transform_gaussians(means,cov,g,common.expand(4,-1,-1),d,indices,weights);rigid_error=max(float((v-(means@common.T+translation)).abs().max()),float((C-common@cov@common.T).abs().max()));assert rigid_error<1e-12
edges=torch.tensor([[0,1],[1,2],[2,3],[3,0]]);ew=torch.ones(4,dtype=dtype);consistent=float(graph_rigidity(g,common.expand(4,-1,-1),d,edges,ew));assert consistent<1e-25
bad=d.clone();bad[1,0]+=.1;inconsistent=float(graph_rigidity(g,common.expand(4,-1,-1),bad,edges,ew));assert inconsistent>1e-4
rotcoef=(torch.randn(4,3,3,dtype=dtype)*.02).requires_grad_();transcoef=(torch.randn(4,3,3,dtype=dtype)*.01).requires_grad_();tau=torch.tensor(.37,dtype=dtype)
def forward(a,b):
 r,d=polynomial_transform(a,b,tau);v,C,_=transform_gaussians(means,cov,g,r,d,indices,weights);return torch.cat([v.flatten(),C.flatten()])
gradcheck=torch.autograd.gradcheck(forward,(rotcoef,transcoef),eps=1e-6,atol=1e-5,rtol=1e-4);assert gradcheck
r0,d0=polynomial_transform(rotcoef,transcoef,torch.tensor(0.,dtype=dtype));assert torch.equal(r0,torch.eye(3,dtype=dtype).expand(4,-1,-1)) and torch.equal(d0,zero)
rotation_zero=torch.zeros(2,3,dtype=dtype,requires_grad=True);zero_gradcheck=torch.autograd.gradcheck(rotation_matrix,(rotation_zero,),eps=1e-6,atol=1e-6);assert zero_gradcheck
result=dict(status='CPU_identity_globalrigid_equivariance_covariance_and_actual_finite_difference_gradients_verified',identity_error=identity_error,common_rigid_transform_error=rigid_error,consistent_graph_rigidity=consistent,inconsistent_graph_rigidity=inconsistent,full_motion_gradient_check=gradcheck,zero_rotation_gradient_check=zero_gradcheck,reference_time_exact_identity=True,qualification='Mathematics only. NoCUDAintegration,noimageoptimization,noRGBclarityclaim.',goal_complete=False)
(OUT/'math-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for file in [Path(__file__),Path(__file__).with_name('enerf_residual_graph.py')]:shutil.copy2(file,OUT/file.name)
print(json.dumps(result))
