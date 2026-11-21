"""Native fifteen-view hand surface fit with local standard GS density control."""
from pathlib import Path
import json,time,hashlib,shutil,os
import numpy as np,torch
from gsplat.strategy.ops import split,duplicate,remove
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';U=L/'outputs/dynamic-human-enerf-hand-time-scope-088';O=L/'outputs/dynamic-human-enerf-hand-adaptive-094';R=O/'adaptive';R.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (R/'resume.pt').exists(),'Do not overwrite an existing run';torch.set_num_threads(8);torch.manual_seed(8801);np.random.seed(8801)
parent=S/'local_bspline/resume.pt';parent_sha=sha(parent);cp=torch.load(parent,map_location='cpu',weights_only=False);original_active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();active=original_active.clone()
rates=dict(means=.00016*cp['scene_scale'],scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125);p={k:torch.nn.Parameter(v.cuda(),requires_grad=k in rates) for k,v in cp['params'].items()};fixed={k:v.detach()[~active].clone() for k,v in p.items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);model=HandJointRenderer(S/'head/capture-affines.npz',g,g,active);opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()}
state=dict(origin=torch.arange(len(active),device='cuda'),grad2d=torch.zeros(len(active),device='cuda'),count=torch.zeros(len(active),device='cuda'));hooks=[]
def mask_and_hooks():
 global active,hooks
 for h in hooks:h.remove()
 active=original_active[state['origin']];model.active=active;model.ids=torch.where(active)[0];hooks=[p[k].register_hook(lambda grad:grad*active.reshape((-1,)+(1,)*(grad.ndim-1))) for k in rates]
mask_and_hooks();obs={}
records=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train' and r['frame']==732];assert len(records)==15 and not {r['camera'] for r in records}&{'00','06','12'}
for r in records:
 path=S/'native-input'/f"camera{r['camera']}-frame000732.npz";assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;obs[r['camera']]=dict(rgb=torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).cuda(),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],W=tensor(z['w2c']),K=tensor(K),width=int(x1-x0),height=int(y1-y0))
schedule=np.load(U/'single_time/schedule.npz');camera=schedule['camera'];assert len(camera)==3000 and np.all(schedule['frame']==732);background=tensor(schedule['background']);shutil.copy2(U/'single_time/schedule.npz',R/'schedule.npz')
save(R/'protocol.json',dict(parent_sha256=parent_sha,control='088single_time; identical initial state,3000camera/background draws,LR,loss,frozenmotion.',change='Density control only within descendants of6330selectedGaussians. Standard gsplat split/duplicate, no opacity reset; low-opacity local prune.',density=dict(gradient_threshold=.0002,large_scale_fraction_of_scene=.01,start=400,stop=2000,every=200,opacity_prune=.005,active_cap=25000,split_scale_divisor=1.6),gradient='signed mean2d gradients, screen-normalized halfwidth/height; average over visible observations. No absolute-gradient CUDA variant.',bindings='Children inherit bodyanchor and graphbindings; all384motionnodes frozen. Nonselected246654Gs remain exact.',limitations='Local density control is adapted to a moving scene. This is not a full official3DGS training reproduction. Frame732 nominal captures retain estimated camera time differences. No heldout training.',goal_complete=False))
began=time.monotonic();history=[];events=[]
def checkpoint(step):
 q=dict(cp);q.update(renderer='hand_adaptive_094',step=15000+step,scope_steps=step,parent_sha256=parent_sha,params={k:v.detach().cpu() for k,v in p.items()},optim={k:v.state_dict() for k,v in opt.items()},active_Gaussians=active.cpu(),origin=state['origin'].cpu(),history=history,density_events=events,elapsed=time.monotonic()-began);torch.save(q,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(3000):
 c=str(camera[i]);o=obs[c];rgb,alpha,info=model.render(p,c,732,o['W'],o['K'],o['width'],o['height'],2);info['means2d'].retain_grad();bg=background[i];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid'];loss=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean();assert torch.isfinite(loss);loss.backward()
 with torch.no_grad():
  grads=info['means2d'].grad[0].clone();grads[:,0]*=o['width']/2;grads[:,1]*=o['height']/2;vis=(info['radii'][0]>0).all(-1)&active;state['grad2d'][vis]+=grads[vis].norm(dim=-1);state['count'][vis]+=1
 for k,v in opt.items():v.step();v.zero_grad(set_to_none=True);v.param_groups[0]['lr']=rates[k]*((1/3)**((i+1)/3000))
 if 400<=i+1<=2000 and (i+1)%200==0:
  with torch.no_grad():
   score=state['grad2d']/state['count'].clamp_min(1);candidate=active&(score>.0002);budget=max(25000-int(active.sum()),0)
   if int(candidate.sum())>budget:
    allowed=torch.where(candidate)[0];chosen=allowed[torch.topk(score[allowed],budget).indices] if budget else allowed[:0];candidate.zero_();candidate[chosen]=True
   small=p['scales'].exp().max(-1).values<=.01*cp['scene_scale'];du=candidate&small;sp=candidate&~small;nd=int(du.sum());ns=int(sp.sum());ev=dict(step=i+1,before_active=int(active.sum()),duplicate=nd,split=ns,visible_selected=int((active&(state['count']>0)).sum()),gradient_quantiles=[float(x) for x in torch.quantile(score[active],tensor([.5,.9,.99,1]))])
   if nd:duplicate(p,opt,state,du)
   sp=torch.cat([sp,torch.zeros(nd,dtype=torch.bool,device='cuda')])
   if ns:split(p,opt,state,sp)
   active=original_active[state['origin']];pr=active&(p['opacities'].sigmoid()<.005);ev['pruned']=int(pr.sum())
   if pr.any():remove(p,opt,state,pr)
   state['grad2d'].zero_();state['count'].zero_();mask_and_hooks();ev['after_active']=int(active.sum());events.append(ev);print('DENSITY',json.dumps(ev),flush=True);save(R/'density-events.json',events)
 if (i+1)%100==0:
  assert all(torch.equal(v[~active],fixed[k]) for k,v in p.items());row=dict(step=i+1,loss=float(loss),points=len(active),active=int(active.sum()),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print('ADAPTIVE',json.dumps(row),flush=True)
 if (i+1)%1000==0:checkpoint(i+1)
assert sha(parent)==parent_sha;save(R/'completion.json',dict(status='3000updates_adaptive_surface_fit_complete_evaluation_pending',scope_steps=3000,points=len(active),active_Gaussians=int(active.sum()),nonhand_parameters_exactly_preserved=True,all_motion_exactly_frozen=True,checkpoint_sha256=sha(R/'resume.pt'),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
