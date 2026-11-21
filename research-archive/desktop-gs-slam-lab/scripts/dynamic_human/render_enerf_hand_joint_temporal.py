"""Actual full-resolution renders cropped to fixed source-defined hand windows,13consecutiveframes."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-joint-084';T=O/'temporal';T.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (T/'completion.json').exists();plan=read(T/'protocol.json');torch.set_num_threads(8);active=torch.from_numpy(np.load(O/'support/active-gaussians.npy')).cuda();inputs={};audit={(r['camera'],r['frame']):r for r in read(S/'input-audit.json')['records']};checks=[];photo=T/'source';photo.mkdir(exist_ok=True)
for c in plan['cameras']:
 box=plan['boxes'][c];x0,y0,x1,y1=box
 for t in plan['frames']:
  path=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(path)==audit[c,t]['input_sha256'];z=np.load(path);inputs[c,t]=dict(W=tensor(z['w2c']),K=tensor(z['K']),bbox=box,input_sha256=audit[c,t]['input_sha256']);rgb=z['rgb'][y0:y1,x0:x1];matte=z['matte'][y0:y1,x0:x1];valid=z['valid'][y0:y1,x0:x1];np.savez_compressed(photo/f'camera{c}-frame{t:06d}.npz',rgb=rgb,matte=matte,valid=valid);Image.fromarray(rgb).save(photo/f'camera{c}-frame{t:06d}.png')
with torch.no_grad():
 for branch in ['parent','joint_photo','joint_flow']:
  root=S/'local_bspline' if branch=='parent' else O/branch;path=root/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['graph'] if branch=='parent' else cp['reference_graph']);ref.requires_grad_(False);model=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,ref,active);dest=T/branch;dest.mkdir(exist_ok=True)
  for (c,t),o in inputs.items():
   rgb,alpha,_=model.render(p,c,t,o['W'],o['K'],1920,1080,2);x0,y0,x1,y1=o['bbox'];r=rgb[0,y0:y1,x0:x1].cpu().numpy();a=alpha[0,y0:y1,x0:x1,0].cpu().numpy();assert np.isfinite(r).all() and np.isfinite(a).all();pred=np.clip(r+.5*(1-a[...,None]),0,1);np.savez_compressed(dest/f'camera{c}-frame{t:06d}.npz',rgb=r,alpha=a);Image.fromarray(u8(pred)).save(dest/f'camera{c}-frame{t:06d}.png');err=None
   if t==731:
    existing=np.load(root/'native-evaluation'/f'camera{c}-frame{t:06d}-prediction.npz');err=max(float(abs(r-existing['rgb'][y0:y1,x0:x1]).max()),float(abs(a-existing['alpha'][y0:y1,x0:x1]).max()));assert err<1e-6
   checks.append(dict(branch=branch,camera=c,frame=t,input_sha256=o['input_sha256'],bbox=o['bbox'],checkpoint_sha256=before,evaluation_reproduction_max_error=err));print('HAND_JOINT_TEMPORAL',branch,c,t,flush=True)
  assert sha(path)==before;del model,p,g,ref,cp;torch.cuda.empty_cache()
assert len(checks)==78;save(T/'completion.json',dict(status='78actualnativehandwindows_saved_from_full_resolution_renders',records=checks,frames_per_camera=13,cameras=plan['cameras'],goal_complete=False));shutil.copy2(__file__,T/Path(__file__).name)
