from pathlib import Path
import hashlib,json,shutil
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_hand_trajectory_model import HandTrajectory,HandTrajectoryRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';O.mkdir(exist_ok=True);F=L/'outputs/dynamic-human-enerf-hand-fine-085/fine-graph.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(8);cp=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);base=HandJointRenderer(S/'head/capture-affines.npz',g,g,active);checks=[]
for mode in ['nodes','points']:
 d=HandTrajectory(mode,F,cp['scene_scale']);model=HandTrajectoryRenderer(S/'head/capture-affines.npz',g,active,d)
 for c,t in [('04',720),('04',732),('17',732)]:
  z=np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz');W=tensor(z['w2c']);K=tensor(z['K'])
  with torch.no_grad():
   a=base.render(p,c,t,W,K,1920,1080,2);b=model.render(p,c,t,W,K,1920,1080,2);err=max(float((a[k]-b[k]).abs().max()) for k in [0,1]);assert err==0
   mm,mc,ml,_=model.geometry(p,c,t);bm,bc,bl,_=base.geometry(p,c,t);assert torch.equal(mm,bm) and torch.equal(mc,bc) and torch.equal(ml,bl)
  checks.append(dict(mode=mode,camera=c,frame=t,initial_RGB_alpha_max_error=err))
 # Analytic temporal derivative against centered finite difference, arbitrary
 # coefficients only in this discarded smoke object, never a training parent.
 with torch.no_grad():d.coefficients.copy_(torch.randn_like(d.coefficients)*.1)
 assert d(torch.tensor(0.,device='cuda')).abs().max()==0
 d=d.double();tau=torch.tensor(.32,dtype=torch.float64,device='cuda',requires_grad=True);v=d(tau).square().mean();dv=torch.autograd.grad(v,tau)[0];h=1e-5;fd=(d(tau+h).square().mean()-d(tau-h).square().mean())/(2*h);assert abs(float(dv-fd))<1e-10
 value=d(torch.tensor(.6,dtype=torch.float64,device='cuda'));assert float(value.norm(dim=-1).max())<=.06*cp['scene_scale']*3**.5
 loss=value.square().mean()+d.regularization();loss.backward();assert torch.isfinite(d.coefficients.grad).all() and d.coefficients.grad.abs().max()>0
 checks.append(dict(mode=mode,trainable_motion_parameters=d.coefficients.numel(),derivative_error=float(abs(dv-fd)),reference_identity=True,finite_nonzero_gradient=True))
save(O/'runtime-smoke.json',dict(status='both_initial_renderers_exact_derivatives_and_gradient_checked',records=checks,no_model_checkpoint_saved=True,goal_complete=False))
save(O/'protocol.json',dict(status='prepared_for_two_matched_training_runs',parent_sha256=sha(S/'local_bspline/resume.pt'),support_sha256=sha(F),scene_scale=float(cp['scene_scale']),steps_each=3000,training='Original105images:15cameras x7evenframes720..732;00/06/12neverfitted. Exact088multi_timecamera/frame/background schedule.',surface='Same6330Gaussianparameters and LR as088;246654otherGaussians frozen;nodensitychange.',motion='Original384nodegraph/captureaffines/clocksoffsets frozen. Newtranslationfield shares time acrosscameras. Same7compactcubicB-splinebasis,referenceidentity726, coefficientbounds .03scene_scale. Two branches:32sharednodes interpolated by existinglocalbindings versus6330per-Gaussiantrajectories.',regularization='.1mean(||delta/scene_scale||componentwise squared)+10mean((second framedifference/scene_scale)^2), evaluatedat13integerframes. No guessed fingerlabels, opticalflowloss or per-camera parameters.',motion_raw_coefficient_lr=.003,lr_decay_end=1/3,qualification='Comparedwith085,LR/reg/spatialfieldandrotationdesign differ; onlynodes-versuspoints here is controlled. Translationresidualcannotrotate Gaussian covariance or localSH frame. Not a physicalhand model.',acceptance='Original134native cases,15same-timehandcrops,existing6targethands/12handobjectregions,heldouttime/camera andcontinuous sequences. Scalar gains or twofitted views alone insufficient.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);shutil.copy2(L/'scripts/dynamic_human/enerf_hand_trajectory_model.py',O/'enerf_hand_trajectory_model.py');print('PREPARED',json.dumps(checks),flush=True)
