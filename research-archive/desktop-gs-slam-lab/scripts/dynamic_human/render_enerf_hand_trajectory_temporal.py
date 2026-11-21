"""Full native render then fixed hand-context crops, two continuous branches."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_trajectory_model import HandTrajectory,HandTrajectoryRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';T=O/'temporal';T.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (T/'completion.json').exists();plan=read(U/'temporal/protocol.json');save(T/'protocol.json',plan);torch.set_num_threads(8);active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();audit={(r['camera'],r['frame']):r for r in read(S/'input-audit.json')['records']};inputs={};checks=[];photo=T/'source';photo.mkdir(exist_ok=True)
for c in plan['cameras']:
 for t in plan['frames']:
  stem=f'camera{c}-frame{t:06d}';path=S/'native-input'/f'{stem}.npz';assert sha(path)==audit[c,t]['input_sha256'];z=np.load(path);inputs[c,t]=dict(W=tensor(z['w2c']),K=tensor(z['K']),bbox=plan['boxes'][c],input_sha256=audit[c,t]['input_sha256'])
  for suffix in ['npz','png']:shutil.copy2(U/'temporal/source'/f'{stem}.{suffix}',photo/f'{stem}.{suffix}')
for branch,old in [('parent','parent'),('frozen','multi_time')]:
 dest=T/branch;dest.mkdir(exist_ok=True)
 for c,t in inputs:
  stem=f'camera{c}-frame{t:06d}'
  for suffix in ['npz','png']:
   src=U/'temporal'/old/f'{stem}.{suffix}';shutil.copy2(src,dest/src.name);assert sha(src)==sha(dest/src.name)
  checks.append(dict(branch=branch,camera=c,frame=t,reused088=True))
with torch.no_grad():
 for branch in ['nodes','points']:
  root=O/branch;path=root/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);d=HandTrajectory(branch,L/'outputs/dynamic-human-enerf-hand-fine-085/fine-graph.npz',cp['scene_scale']);d.load_state_dict(cp['trajectory']);model=HandTrajectoryRenderer(S/'heldout/capture-affines-estimated.npz',g,active,d);dest=T/branch;dest.mkdir(exist_ok=True)
  for (c,t),o in inputs.items():
   rgb,alpha,_=model.render(p,c,t,o['W'],o['K'],1920,1080,2);x0,y0,x1,y1=o['bbox'];r=rgb[0,y0:y1,x0:x1].cpu().numpy();a=alpha[0,y0:y1,x0:x1,0].cpu().numpy();assert np.isfinite(r).all() and np.isfinite(a).all();pred=np.clip(r+.5*(1-a[...,None]),0,1);np.savez_compressed(dest/f'camera{c}-frame{t:06d}.npz',rgb=r,alpha=a);Image.fromarray(u8(pred)).save(dest/f'camera{c}-frame{t:06d}.png');err=None
   if t==731:
    existing=np.load(root/'native-evaluation'/f'camera{c}-frame{t:06d}-prediction.npz');err=max(float(abs(r-existing['rgb'][y0:y1,x0:x1]).max()),float(abs(a-existing['alpha'][y0:y1,x0:x1]).max()));assert err<1e-6
   checks.append(dict(branch=branch,camera=c,frame=t,input_sha256=o['input_sha256'],bbox=o['bbox'],checkpoint_sha256=before,evaluation_reproduction_max_error=err));print('HAND_TRAJECTORY_TEMPORAL',branch,c,t,flush=True)
  assert sha(path)==before;del model,p,g,d,cp;torch.cuda.empty_cache()
assert len(checks)==104;save(T/'completion.json',dict(status='52new_native_ROIs_plus52exact_reused088',records=checks,frames_per_camera=13,cameras=plan['cameras'],goal_complete=False));shutil.copy2(__file__,T/Path(__file__).name)
