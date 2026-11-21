"""Two matched native-RGB trials: articulated surface with fixed/refined hand pose."""
from pathlib import Path
import argparse,json,time,hashlib,shutil,os
import numpy as np,torch
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-hand-visibility-100'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['narrow_reset','context_reset'],required=True);args=ap.parse_args();R=O/args.branch;R.mkdir(exist_ok=True);assert not (R/'completion.json').exists();assert read(O/'smoke.json')['no_optimizer_steps'];protocol=read(O/'protocol.json');torch.set_num_threads(6);torch.manual_seed(10001)
parent=S/'local_bspline/resume.pt';parent_sha=sha(parent);assert protocol['parent_sha256']==parent_sha;cp=torch.load(parent,map_location='cpu',weights_only=False);active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();p={k:torch.nn.Parameter(v.cuda(),requires_grad=k=='opacities') for k,v in cp['params'].items()};p['opacities'].register_hook(lambda grad:grad*active);g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'head/capture-affines.npz',g,g,active);hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],protocol['initial_color'],True)
reset=torch.from_numpy(np.load(O/('reset-mask.npy' if args.branch=='narrow_reset' else 'context-reset-mask.npy'))).cuda();assert read(O/'selection/visual-review.json')['approved_for_bounded_trial']
with torch.no_grad():
 hand.opacity_logits.fill_(float(np.log(.2/.8)))
 p['opacities'][reset]=torch.minimum(p['opacities'][reset],torch.tensor(float(np.log(.02/.98)),device='cuda'))
rates=dict(offset_raw=.003,log_scale_delta=.005,opacity_logits=.05,color_logits=.01,pose_delta=.002,translation_delta=.001)
groups=[dict(params=[getattr(hand,k)],lr=v,name=k) for k,v in rates.items() if getattr(hand,k).requires_grad]
opt=torch.optim.Adam(groups,eps=1e-15);oldopt=torch.optim.Adam([p['opacities']],lr=.05,eps=1e-15)
obs={};records=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105 and not {r['camera'] for r in records}&{'00','06','12'}
for r in records:
 path=S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;obs[r['camera'],r['frame']]=dict(rgb=torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).cuda(),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],W=tensor(z['w2c']),K=tensor(K),width=int(x1-x0),height=int(y1-y0))
schedule=np.load(L/'outputs/dynamic-human-enerf-hand-time-scope-088/multi_time/schedule.npz');camera=np.tile(schedule['camera'],2);frames=np.tile(schedule['frame'],2);background=tensor(np.tile(schedule['background'],(2,1)));assert len(camera)==6000 and protocol['steps_each']==6000;np.savez(R/'schedule.npz',camera=camera,frame=frames,background=background.cpu().numpy())
protocol.update(initial_reset_count=int(reset.sum()),initial_new_opacity=.2,branch=args.branch,new_Gaussians=len(hand.faces),trainable_hand_parameters=sum(v.numel() for v in hand.parameters() if v.requires_grad),original_opacity_parameters=int(active.sum()),learning_rates={k:rates[k] for k in rates if getattr(hand,k).requires_grad},old_hand_opacity_lr=.05,lr_decay_end=1/3,schedule_sha256=sha(R/'schedule.npz'));save(R/'protocol.json',protocol)
start=0;history=[];elapsed=0
if (R/'resume.pt').exists():
 q=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);assert q['renderer']=='mano_surface100' and q['parent_sha256']==parent_sha and q['branch']==args.branch;start=q['scope_steps'];history=q['history'];elapsed=q['elapsed'];hand.load_state_dict(q['hand_surface']);opt.load_state_dict(q['hand_optim']);oldopt.load_state_dict(q['old_optim'])
 with torch.no_grad():p['opacities'].copy_(q['params']['opacities'].cuda())
 del q
began=time.monotonic()-elapsed;torch.cuda.reset_peak_memory_stats()
def checkpoint(step):
 q=dict(cp);q.update(renderer='mano_surface100',branch=args.branch,parent_sha256=parent_sha,scope_steps=step,step=15000+step,params={k:v.detach().cpu() for k,v in p.items()},hand_surface={k:v.detach().cpu() for k,v in hand.state_dict().items()},hand_optim=opt.state_dict(),old_optim=oldopt.state_dict(),history=history,elapsed=time.monotonic()-began);torch.save(q,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(start,6000):
 c,f=str(camera[i]),int(frames[i]);o=obs[c,f];rgb,alpha,_=render_hand_surface(p,old,hand,c,f,o['W'],o['K'],o['width'],o['height']);bg=background[i];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid'];photo=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean();reg=hand.regularization();loss=photo+reg;assert torch.isfinite(loss);loss.backward()
 if i==start:
  assert p['opacities'].grad[~active].abs().max()==0
  for k in rates:
   v=getattr(hand,k)
   if v.requires_grad:assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().max()>0,k
 opt.step();opt.zero_grad(set_to_none=True);oldopt.step();oldopt.zero_grad(set_to_none=True);factor=(1/3)**((i+1)/6000)
 for group in opt.param_groups:group['lr']=rates[group['name']]*factor
 oldopt.param_groups[0]['lr']=.05*factor
 if (i+1)%200==0 or i==start:
  assert torch.equal(p['opacities'][~active].detach().cpu(),cp['params']['opacities'][~active.cpu()]);row=dict(step=i+1,loss=float(loss),photo=float(photo),regularizer=float(reg),new_opacity_mean=float(hand.opacity_logits.sigmoid().mean()),new_opacity_p90=float(torch.quantile(hand.opacity_logits.sigmoid(),.9)),max_local_offset_world=float((.005*cp['scene_scale']*hand.offset_raw.tanh()).norm(dim=-1).max()),max_pose_delta_rad=float((.25*hand.pose_delta.tanh()).abs().max()),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print('MANO_SURFACE',args.branch,json.dumps(row),flush=True)
 if (i+1)%2000==0:checkpoint(i+1)
assert sha(parent)==parent_sha
save(R/'completion.json',dict(status='6000RGBupdates_complete_native_evaluation_pending',initial_reset_count=int(reset.sum()),initial_new_opacity=.2,branch=args.branch,scope_steps=6000,new_Gaussians=len(hand.faces),total_Gaussians=len(p['means'])+len(hand.faces),old_trainable_opacities=int(active.sum()),all_other_original_parameters_frozen=True,checkpoint_sha256=sha(R/'resume.pt'),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
