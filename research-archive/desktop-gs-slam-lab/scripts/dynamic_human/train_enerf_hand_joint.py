"""Matched local geometry/appearance/motion continuation with optional normal flow."""
from pathlib import Path
import argparse,json,time,hashlib,random,os,shutil
import numpy as np,torch
import torch.nn.functional as F
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';F0=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=F0/'model-support';O=L/'outputs/dynamic-human-enerf-hand-joint-084'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['joint_photo','joint_flow'],required=True);args=ap.parse_args();R=O/args.branch;R.mkdir(exist_ok=True);assert not (R/'completion.json').exists();steps=3000
assert read(O/'support/visual-review.json')['training_approved'];assert read(O/'runtime-smoke.json')['no_model_checkpoint_saved'];assert read(O/'protocol.json')['steps_each']==steps
torch.set_num_threads(8);torch.manual_seed(8401);np.random.seed(8401);random.seed(8401);parent=S/'local_bspline/resume.pt';parent_sha=sha(parent);cp=torch.load(parent,map_location='cpu',weights_only=False)
rates=dict(means=.000016*cp['scene_scale'],scales=.0005,quats=.0001,opacities=.005,sh0=.00025,shN=.0000125);p={k:torch.nn.Parameter(v.cuda(),requires_grad=k in rates) for k,v in cp['params'].items()};active=torch.from_numpy(np.load(O/'support/active-gaussians.npy')).cuda();nodes=torch.from_numpy(np.load(O/'support/active-nodes.npy')).cuda();assert int(active.sum())==6330 and int(nodes.sum())==22
fixed={k:v.detach().clone() for k,v in p.items()};graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);reference=GraphMotion(S/'graph.npz',cp['scene_scale']);reference.load_state_dict(cp['graph']);reference.requires_grad_(False);initial_graph={k:v.detach().clone() for k,v in graph.named_parameters()};model=HandJointRenderer(S/'head/capture-affines.npz',graph,reference,active)
for k in rates:p[k].register_hook(lambda grad:grad*active.reshape((-1,)+(1,)*(grad.ndim-1)))
for par in graph.parameters():par.register_hook(lambda grad:grad*nodes[:,None,None])
opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()};gopt=torch.optim.Adam(graph.parameters(),lr=.0003,eps=1e-15)
observations=[];records=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105
for r in records:
 path=S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;observations.append(dict(camera=r['camera'],frame=r['frame'],rgb=torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).cuda(),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],K=tensor(K),W=tensor(z['w2c']),width=int(x1-x0),height=int(y1-y0)))
sources={};pairs=[]
for r in read(M/'completion.json')['records']:
 if not r['model_visible']:continue
 c,t=r['camera'],r['source_frame'];key=(c,t)
 if key not in sources:
  source=torch.load(M/f'camera{c}-frame{t:06d}-source.pt',map_location='cpu',weights_only=False);sources[key]={k:source[k].cuda() for k in ['W','K']}
 assert sha(M/r['filename'])==r['support_sha256'];q=np.load(M/r['filename']);pairs.append(dict(camera=c,source_frame=t,target_frame=r['target_frame'],xy=tensor(q['local_xy']),normal=tensor(q['normal']),normal_target=tensor(q['normal_target'])))
photo_rng=random.Random(8401);flow_rng=random.Random(8403);draws=[photo_rng.randrange(105) for _ in range(steps)];flowdraws=[flow_rng.randrange(len(pairs)) for _ in range(steps)];background=torch.rand(steps,3,generator=torch.Generator().manual_seed(8402)).cuda();np.savez_compressed(R/'schedule.npz',image_index=np.array(draws),flow_index=np.array(flowdraws),background=background.cpu().numpy())
save(R/'protocol.json',dict(branch=args.branch,parent_sha256=parent_sha,parent_step=15000,additional_joint_steps=steps,flow_weight=0 if args.branch=='joint_photo' else .05,normal_scale_px=2.,active_Gaussians=6330,active_nodes=22,learning_rates=rates,graph_lr=.0003,schedule_sha256=sha(R/'schedule.npz'),renderer='HandJointRenderer; nonselectedGaussiansalwaysuseparentgraph',current_visibility='Recomputed forfloweachstep,detached geometry/opacity contributionweights; all1527originalpointsretained.',goal_complete=False))
start=0;history=[];elapsed=0.
if (R/'resume.pt').exists():
 resume=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);assert resume['parent_sha256']==parent_sha and resume['renderer']=='hand_joint_084';graph.load_state_dict(resume['graph']);gopt.load_state_dict(resume['graph_optim'])
 with torch.no_grad():
  for k in p:p[k].copy_(resume['params'][k].cuda())
 for k in opt:opt[k].load_state_dict(resume['optim'][k])
 start=resume['additional_joint_steps'];history=resume['history'];elapsed=resume['elapsed'];del resume
began=time.monotonic()-elapsed
def checkpoint(step):
 out=dict(cp);out.update(renderer='hand_joint_084',parent_sha256=parent_sha,parent_step=15000,step=15000+step,additional_joint_steps=step,params={k:v.detach().cpu() for k,v in p.items()},optim={k:v.state_dict() for k,v in opt.items()},graph={k:v.detach().cpu() for k,v in graph.state_dict().items()},reference_graph={k:v.detach().cpu() for k,v in reference.state_dict().items()},graph_optim=gopt.state_dict(),history=history,elapsed=time.monotonic()-began,branch=args.branch,active_nodes=nodes.cpu(),active_Gaussians=active.cpu());torch.save(out,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(start,steps):
 o=observations[draws[i]];pair=pairs[flowdraws[i]];rgb,alpha,info=model.render(p,o['camera'],o['frame'],o['W'],o['K'],o['width'],o['height'],2);bg=background[i];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid'];photo=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean();reg=10*info['graph_rigidity']+.001*info['graph_rotation_size']+.1*info['graph_translation_size']
 normal,flow,flowalpha=model.current_normal_flow(p,pair,sources[pair['camera'],pair['source_frame']]);flowloss=F.smooth_l1_loss(normal/2,pair['normal_target']/2);loss=photo+reg+(0 if args.branch=='joint_photo' else .05)*flowloss;assert torch.isfinite(loss);loss.backward()
 if i==start:
  assert all(p[k].grad is not None and torch.isfinite(p[k].grad).all() and p[k].grad[~active].abs().max()==0 for k in rates);assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad[~nodes].abs().max()==0 for v in graph.parameters())
 gopt.step();gopt.zero_grad(set_to_none=True);gopt.param_groups[0]['lr']=.0003*((1/3)**((i+1)/steps))
 for k,v in opt.items():v.step();v.zero_grad(set_to_none=True);v.param_groups[0]['lr']=rates[k]*((1/3)**((i+1)/steps))
 if (i+1)%100==0 or i==start:
  torch.cuda.synchronize();assert all(torch.equal(v[~active],fixed[k][~active]) for k,v in p.items());assert all(torch.equal(v[~nodes],initial_graph[k][~nodes]) for k,v in graph.named_parameters());row=dict(step=i+1,photo=float(photo),regularizer=float(reg),normal_loss=float(flowloss),sample_mean_abs_normal_error_px=float((normal-pair['normal_target']).abs().mean()),sample_min_alpha=float(flowalpha.min()),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print('HAND_JOINT_TRAIN',args.branch,json.dumps(row),flush=True)
 if (i+1)%1000==0 or i+1==steps:checkpoint(i+1)
assert sha(parent)==parent_sha;save(R/'completion.json',dict(status='3000matched_local_joint_updates_complete_native_evaluation_pending',step=18000,additional_joint_steps=3000,points=len(p['means']),active_Gaussians=6330,active_nodes=22,nonhand_parameters_exactly_frozen=True,elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),checkpoint_sha256=sha(R/'resume.pt'),goal_complete=False));snap=R/'source-snapshot';snap.mkdir(exist_ok=True)
for name in ['train_enerf_hand_joint.py','enerf_hand_joint_model.py','enerf_second_temporal_graph.py','enerf_second_renderer.py','enerf_residual_graph.py']:shutil.copy2(L/'scripts/dynamic_human'/name,snap/name)
print('HAND_JOINT_TRAIN_COMPLETE',args.branch,flush=True)
