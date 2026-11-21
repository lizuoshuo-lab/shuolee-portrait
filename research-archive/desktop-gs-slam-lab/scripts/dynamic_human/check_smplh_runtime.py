"""Validate shape/hand-rotation derivatives of the supplied neutral model."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,torch,smplx
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-hand-prior-096'
p=Path('/mnt/d/GS-SLAM-Data/models/smplh_300/SMPLH_NEUTRAL.npz')
torch.set_num_threads(4)
m=smplx.SMPLH(model_path=str(p),ext='npz',gender='neutral',num_betas=10,use_pca=False,flat_hand_mean=True).double()
zero=torch.zeros(1,45,dtype=torch.float64);x=zero.clone();x[0,3]=.25;x.requires_grad_(True)
base=m(right_hand_pose=zero,left_hand_pose=zero);out=m(right_hand_pose=x,left_hand_pose=zero)
assert out.vertices.shape==(1,6890,3) and torch.isfinite(out.vertices).all()
v=(out.vertices-base.vertices).square().sum();grad=torch.autograd.grad(v,x)[0];h=1e-5
def value(a):return (m(right_hand_pose=a,left_hand_pose=zero).vertices-base.vertices).square().sum()
xp=x.detach().clone();xm=x.detach().clone();xp[0,3]+=h;xm[0,3]-=h
fd=float((value(xp)-value(xm))/(2*h));error=abs(fd-float(grad[0,3]));assert error<1e-7 and abs(float(grad[0,3]))>1e-8
delta=(out.vertices-base.vertices).detach().norm(dim=-1)[0];changed=int((delta>1e-9).sum())
assert 0<changed<2000
report=dict(status='SMPLH_forward_and_right_finger_derivative_valid_no_scene_training',python=sys.executable,smplx_module=smplx.__file__,vertices=6890,skinning_joints=52,shape_coefficients=10,archive_shape_coefficients=300,hand_pose_dimensions=45,rotation_test_radians=.25,changed_vertices=changed,maximum_model_unit_displacement=float(delta.max()),autodiff_fd_error=error,model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),not_HaMeR_compatible_MANO=True,no_scene_refit=True,no_rendered_clarity_claim=True,goal_complete=False)
(O/'smplh-runtime-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
