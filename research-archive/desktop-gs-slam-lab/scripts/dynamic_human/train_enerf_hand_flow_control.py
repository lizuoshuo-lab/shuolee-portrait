"""Matched graph-only continuation, with/without qualified hand normal flow."""
from pathlib import Path
import argparse,json,time,hashlib,random,os,shutil
import numpy as np,torch
import torch.nn.functional as F
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import predicted_normal
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=O/'model-support'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['photo_control','normal_flow'],required=True);args=ap.parse_args();R=O/args.branch;R.mkdir(exist_ok=True);assert not (R/'completion.json').exists();steps=1500
torch.set_num_threads(8);torch.manual_seed(8301);np.random.seed(8301);random.seed(8301)
support=read(M/'completion.json');parent=S/'local_bspline/resume.pt';assert sha(parent)==support['parent_sha256'];cp=torch.load(parent,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};assert all(not v.requires_grad for v in p.values())
graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',graph);active=torch.from_numpy(np.load(M/'active-nodes.npy')).cuda();assert int(active.sum())==27
initial_graph={k:v.detach().clone() for k,v in graph.named_parameters()}
for param in graph.parameters():param.register_hook(lambda grad:grad*active[:,None,None])
optimizer=torch.optim.Adam(graph.parameters(),lr=.0003,eps=1e-15);observations=[]
records=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105
for r in records:
 path=S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0
 observations.append(dict(camera=r['camera'],frame=r['frame'],rgb=torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).cuda(),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],K=tensor(K),W=tensor(z['w2c']),width=int(x1-x0),height=int(y1-y0)))
sources={};pairs=[]
for r in support['records']:
 if not r['model_visible']:continue
 c,t=r['camera'],r['source_frame'];key=(c,t)
 if key not in sources:
  source=torch.load(M/f'camera{c}-frame{t:06d}-source.pt',map_location='cpu',weights_only=False);sources[key]={k:v.cuda() if torch.is_tensor(v) else v for k,v in source.items()}
 assert sha(M/r['filename'])==r['support_sha256'];q=np.load(M/r['filename']);pairs.append(dict(camera=c,source_frame=t,target_frame=r['target_frame'],xy=tensor(q['local_xy']),normal=tensor(q['normal']),normal_target=tensor(q['normal_target']),alpha=tensor(q['alpha'])))
photo_rng=random.Random(8301);flow_rng=random.Random(8303);draws=[photo_rng.randrange(105) for _ in range(steps)];flowdraws=[flow_rng.randrange(len(pairs)) for _ in range(steps)];background=torch.rand(steps,3,generator=torch.Generator().manual_seed(8302)).cuda()
np.savez_compressed(R/'schedule.npz',image_index=np.array(draws),flow_index=np.array(flowdraws),background=background.cpu().numpy())
save(R/'protocol.json',dict(branch=args.branch,parent_sha256=sha(parent),parent_step=15000,additional_graph_steps=steps,flow_weight=0 if args.branch=='photo_control' else .05,normal_scale_px=2.,active_nodes=27,active_nodes_sha256=sha(M/'active-nodes.npy'),schedule_sha256=sha(R/'schedule.npz'),Gaussian_parameters_frozen=True,body_motion_sha256=sha(S/'head/capture-affines.npz'),goal_complete=False))
start=0;history=[];elapsed=0.
if (R/'resume.pt').exists():
 resume=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);assert resume['parent_sha256']==sha(parent);graph.load_state_dict(resume['graph']);optimizer.load_state_dict(resume['graph_optim']);start=resume['additional_graph_steps'];history=resume['history'];elapsed=resume['elapsed'];del resume
began=time.monotonic()-elapsed
def checkpoint(step):
 out=dict(cp);out.update(parent_sha256=sha(parent),parent_step=15000,step=15000+step,additional_graph_steps=step,graph={k:v.detach().cpu() for k,v in graph.state_dict().items()},graph_optim=optimizer.state_dict(),history=history,elapsed=time.monotonic()-began,branch=args.branch,active_nodes=active.cpu());torch.save(out,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(start,steps):
 o=observations[draws[i]];pair=pairs[flowdraws[i]];rgb,alpha,info=model.render(p,o['camera'],o['frame'],o['W'],o['K'],o['width'],o['height'],2);bg=background[i];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid']
 photo=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean();reg=10*info['graph_rigidity']+.001*info['graph_rotation_size']+.1*info['graph_translation_size']
 normal,_=predicted_normal(p,model,pair,sources[pair['camera'],pair['source_frame']]);flowloss=F.smooth_l1_loss(normal/2,pair['normal_target']/2);loss=photo+reg+(0 if args.branch=='photo_control' else .05)*flowloss;assert torch.isfinite(loss)
 loss.backward()
 if i==start:
  assert all(v.grad is None for v in p.values());assert all(v.grad is not None and torch.isfinite(v.grad).all() and v.grad[~active].abs().max()==0 for v in graph.parameters());assert sum(float(v.grad.abs().sum()) for v in graph.parameters())>0
 optimizer.step();optimizer.zero_grad(set_to_none=True);optimizer.param_groups[0]['lr']=.0003*((1/3)**((i+1)/steps))
 if (i+1)%100==0 or i==start:
  torch.cuda.synchronize();assert all(torch.equal(v[~active],initial_graph[k][~active]) for k,v in graph.named_parameters());row=dict(step=i+1,photo=float(photo),regularizer=float(reg),normal_loss=float(flowloss),sample_mean_abs_normal_error_px=float((normal-pair['normal_target']).abs().mean()),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print('HAND_FLOW_TRAIN',args.branch,json.dumps(row),flush=True)
 if (i+1)%500==0 or i+1==steps:checkpoint(i+1)
assert sha(parent)==support['parent_sha256'];save(R/'completion.json',dict(status='1500matched_graph_updates_complete_native_evaluation_pending',step=16500,additional_graph_steps=1500,points=len(p['means']),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),Gaussian_parameters_frozen=True,inactive_graph_nodes_exactly_frozen=True,checkpoint_sha256=sha(R/'resume.pt'),goal_complete=False))
snap=R/'source-snapshot';snap.mkdir(exist_ok=True)
for name in ['train_enerf_hand_flow_control.py','enerf_hand_flow_model.py','enerf_second_temporal_graph.py','enerf_second_renderer.py','enerf_residual_graph.py']:shutil.copy2(L/'scripts/dynamic_human'/name,snap/name)
print('HAND_FLOW_TRAIN_COMPLETE',args.branch,flush=True)
