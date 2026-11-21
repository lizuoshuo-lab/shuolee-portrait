"""Independent NumPy reconstruction of trained local trajectories, CPU only."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from enerf_hand_trajectory_model import HandTrajectory
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-prior-control-098';F=L/'outputs/dynamic-human-enerf-hand-fine-085/fine-graph.npz';torch.set_num_threads(4)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();z=np.load(S/'heldout/capture-affines-estimated.npz');times=z['capture_time'].astype(np.float32);tau=(times-726)/6;support=np.load(F);stats={}
def B(t,centers):
 x=(np.asarray(t)[...,None]-centers)/.5;a=abs(x);v=np.zeros_like(x);inside=a<1;edge=(a>=1)&(a<2);v[inside]=2/3-x[inside]**2+.5*a[inside]**3;v[edge]=(2-a[edge])**3/6;return v
def independent(t,q,scale,mode):
 centers=q['centers'].numpy().astype(np.float64);basis=B(np.asarray(t,dtype=np.float64),centers)-B(np.array(0.),centers);d=np.einsum('...k,nkd->...nd',basis,.03*scale*np.tanh(q['coefficients'].numpy().astype(np.float64)))
 if mode=='nodes':d=(d[...,q['indices'].numpy(),:]*q['weights'].numpy()[...,None]).sum(-2)
 return d
for branch in ['soft_prior']:
 path=O/branch/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);m=HandTrajectory('points',F,cp['scene_scale'],device='cpu');m.load_state_dict(cp['trajectory']);m.requires_grad_(False)
 with torch.no_grad():actual=m(torch.from_numpy(tau)).numpy();ref=m(torch.tensor(0.)).numpy();reg=float(m.regularization())
 expected=independent(tau,cp['trajectory'],cp['scene_scale'],'points');error=float(abs(actual-expected).max());assert error<1e-7 and abs(ref).max()==0
 d=independent(np.arange(-6,7)/6,cp['trajectory'],cp['scene_scale'],'points');acc=d[2:]-2*d[1:-1]+d[:-2];computed=.1*np.mean((d/cp['scene_scale'])**2)+10*np.mean((acc/cp['scene_scale'])**2);assert abs(reg-computed)<1e-8
 norm=np.linalg.norm(actual,axis=-1);assert norm.max()<=.06*cp['scene_scale']*np.sqrt(3)+1e-7;pair={c:int(np.where((z['camera']==c)&(z['frame']==732))[0][0]) for c in ['04','17']};difference=np.linalg.norm(actual[pair['04']]-actual[pair['17']],axis=-1)
 stats[branch]=dict(checkpoint_sha256=before,contexts=234,numpy_max_absolute_error=error,reference_frame726_exact_zero=True,motion_parameter_count=m.coefficients.numel(),displacement_world_units={str(q):float(np.quantile(norm,q)) for q in [.5,.95,.99,1]},frame04_17_732_residual_difference_world_units={str(q):float(np.quantile(difference,q)) for q in [.5,.95,1]},max_integer_second_difference_world_units=float(np.linalg.norm(acc,axis=-1).max()),regularizer=reg,regularizer_independent_error=abs(reg-computed),raw_coefficient_abs_quantiles={str(q):float(np.quantile(abs(cp['trajectory']['coefficients'].numpy()),q)) for q in [.5,.95,1]},fraction_tanh_abs_gt_095=float(np.mean(abs(np.tanh(cp['trajectory']['coefficients'].numpy()))>.95)))
 np.savez_compressed(O/f'{branch}-residual-positions.npz',camera=z['camera'],frame=z['frame'],capture_time_float32=times,delta=actual);assert sha(path)==before
save(O/'motion-audit.json',dict(status='one234context_field_independently_reconstructed',records=stats,scope='Residualcenters only,nottrueobservedmotion. Sameglobaltime functions forallcameras. Covariance/SHframe hasno extra residual rotation. No clarity claim.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(stats),flush=True)
