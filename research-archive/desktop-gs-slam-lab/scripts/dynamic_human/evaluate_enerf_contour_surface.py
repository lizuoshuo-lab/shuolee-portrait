from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-contour-surface-103';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['prior_control','contour_prior'],required=True);args=ap.parse_args();R=O/args.branch;E=R/'native-evaluation';E.mkdir(exist_ok=True);assert not (E/'metrics.json').exists();protocol=read(O/'protocol.json');assert read(R/'completion.json')['scope_steps']==6000;path=R/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);parent=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy'));assert cp['renderer']=='mano_surface103'
for k,v in parent['params'].items():
 if k=='opacities':assert torch.equal(v[~active],cp['params'][k][~active])
 else:assert torch.equal(v,cp['params'][k]),k
assert float(cp['params']['opacities'][active].sigmoid().max())<=.020001
assert all(torch.equal(v,cp['graph'][k]) for k,v in parent['graph'].items());assert all(int(v['step'])==6000 for v in cp['hand_optim']['state'].values()) and all(int(v['step'])==6000 for v in cp['old_optim']['state'].values())
torch.set_num_threads(6);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active.cuda());hand=ManoGaussianSurface(L/protocol['priors'][args.branch]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],False);initial={k:v.clone() for k,v in hand.state_dict().items()};hand.load_state_dict(cp['hand_surface']);hand.requires_grad_(False)
trainable=['offset_raw','log_scale_delta','opacity_logits','color_logits']
for k,v in initial.items():
 if k not in trainable:assert torch.equal(v,hand.state_dict()[k]),k
save(R/'frozen-state-audit.json',dict(status='alloriginalgeometry_color_motion_and246654opacities_exact',checkpoint_sha256=before,original_opacity_max_change=float((parent['params']['opacities'][active]-cp['params']['opacities'][active]).abs().max()),new_parameter_changes={k:float((initial[k]-hand.state_dict()[k]).abs().max()) for k in trainable},all_optimizer_steps=6000,goal_complete=False))
records=read(U/'evaluation-protocol.json')['records'];rows=[];channelchecks=[]
with torch.no_grad():
 for r in records:
  c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';source=(U if r.get('input_location')=='088' else S)/'native-input'/f'{stem}.npz';assert sha(source)==r['input_sha256'];z=np.load(source);W=tensor(z['w2c']);K=tensor(z['K']);raw,alpha,_=render_hand_surface(p,old,hand,c,t,W,K,1920,1080,emit_surface_weight=True)
  if (c,t) in [('00',721),('04',732),('13',731)]:
   ordinary=render_hand_surface(p,old,hand,c,t,W,K,1920,1080);error=max(float((raw[...,:3]-ordinary[0]).abs().max()),float((alpha-ordinary[1]).abs().max()));assert error<2e-6,error;channelchecks.append(dict(camera=c,frame=t,RGB_alpha_difference=error))
  rgb=raw[0,:,:,:3].cpu().numpy();new_weight=raw[0,:,:,3].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all() and (new_weight<=alpha+1e-5).all();pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];bb=np.zeros_like(z['valid']);bb[y0:y1,x0:x1]=True;metrics={}
  for name,mask in {'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&bb}.items():
   d=pred.astype(np.float64)[mask]-target.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  np.savez_compressed(E/f'{stem}-prediction.npz',rgb=rgb,alpha=alpha,surface_weight=new_weight);Image.fromarray(u8(pred)).save(E/f'{stem}-trained.png');Image.fromarray(u8(alpha)).save(E/f'{stem}-alpha.png');rows.append(dict(camera=c,frame=t,split=r['split'],metrics=metrics,input_sha256=r['input_sha256'],bbox=r['bbox'],reused079parent_prediction=False,visible_new_weight_sum=float(new_weight.sum())));save(E/'progress.json',dict(images=len(rows)));print('MANO_SURFACE_EVAL',args.branch,c,t,flush=True)
assert len(rows)==134 and sha(path)==before;save(E/'metrics.json',dict(status='134native_contexts_complete_comparison_pending',records=rows,checkpoint_sha256=before,surface_weight_channel_checks=channelchecks,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
