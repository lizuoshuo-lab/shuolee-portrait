"""Matched local node versus per-Gaussian continuous translation fields."""
from pathlib import Path
import argparse,json,time,hashlib,random,shutil,os
import numpy as np,torch
from train_enerf_camera10_time import ss
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_trajectory_model import HandTrajectory,HandTrajectoryRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';J=L/'outputs/dynamic-human-enerf-hand-joint-084';O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['nodes','points'],required=True);args=ap.parse_args();R=O/args.branch;R.mkdir(exist_ok=True);assert not (R/'completion.json').exists();assert read(O/'runtime-smoke.json')['no_model_checkpoint_saved'];assert read(O/'protocol.json')['steps_each']==3000
torch.set_num_threads(8);torch.manual_seed(8801);np.random.seed(8801);random.seed(8801);parent=S/'local_bspline/resume.pt';parent_sha=sha(parent);cp=torch.load(parent,map_location='cpu',weights_only=False);active=torch.from_numpy(np.load(J/'support/active-gaussians.npy')).cuda();assert int(active.sum())==6330
rates=dict(means=.00016*cp['scene_scale'],scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125);p={k:torch.nn.Parameter(v.cuda(),requires_grad=k in rates) for k,v in cp['params'].items()};fixed={k:v.detach().clone() for k,v in p.items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['graph']);ref.requires_grad_(False);trajectory=HandTrajectory(args.branch,L/'outputs/dynamic-human-enerf-hand-fine-085/fine-graph.npz',cp['scene_scale']);model=HandTrajectoryRenderer(S/'head/capture-affines.npz',g,active,trajectory);motion_opt=torch.optim.Adam(trajectory.parameters(),lr=.003,eps=1e-15);opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()}
for k in rates:p[k].register_hook(lambda grad:grad*active.reshape((-1,)+(1,)*(grad.ndim-1)))
records=[r for r in read(S/'input-audit.json')['records'] if r['split']=='train'];cams=sorted({r['camera'] for r in records});assert len(records)==105 and len(cams)==15 and not set(cams)&{'00','06','12'};obs={}
for r in records:
    path=S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;obs[r['camera'],r['frame']]=dict(camera=r['camera'],frame=r['frame'],rgb=torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).cuda(),matte=tensor(z['matte'][y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],W=tensor(z['w2c']),K=tensor(K),width=int(x1-x0),height=int(y1-y0))
steps=3000;schedule=L/'outputs/dynamic-human-enerf-hand-time-scope-088/multi_time/schedule.npz';z=np.load(schedule);camera=z['camera'];times=z['frame'];background=tensor(z['background']);assert len(camera)==steps and set(times)==set(range(720,733,2));shutil.copy2(schedule,R/'schedule.npz');protocol=read(O/'protocol.json');protocol.update(branch=args.branch,trainable_motion_parameters=trajectory.coefficients.numel(),learning_rates=rates,schedule_sha256=sha(schedule));save(R/'protocol.json',protocol)
start=0;history=[];elapsed=0
if (R/'resume.pt').exists():
    q=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);assert q['parent_sha256']==parent_sha and q['renderer']=='hand_trajectory_095';start=q['scope_steps'];history=q['history'];elapsed=q['elapsed'];trajectory.load_state_dict(q['trajectory']);motion_opt.load_state_dict(q['motion_optim'])
    with torch.no_grad():
        for k in p:p[k].copy_(q['params'][k].cuda())
    for k in opt:opt[k].load_state_dict(q['optim'][k])
    del q
began=time.monotonic()-elapsed
def checkpoint(step):
    q=dict(cp);q.update(renderer='hand_trajectory_095',branch=args.branch,parent_sha256=parent_sha,scope_steps=step,step=15000+step,params={k:v.detach().cpu() for k,v in p.items()},optim={k:v.state_dict() for k,v in opt.items()},active_Gaussians=active.cpu(),trajectory={k:v.detach().cpu() for k,v in trajectory.state_dict().items()},motion_optim=motion_opt.state_dict(),trajectory_mode=args.branch,history=history,elapsed=time.monotonic()-began);torch.save(q,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(start,steps):
    o=obs[camera[i],times[i]];rgb,alpha,_=model.render(p,o['camera'],o['frame'],o['W'],o['K'],o['width'],o['height'],2);bg=background[i];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid'];loss=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean();photo=loss;regularizer=trajectory.regularization();loss=photo+regularizer;assert torch.isfinite(loss);loss.backward();assert trajectory.coefficients.grad is not None and torch.isfinite(trajectory.coefficients.grad).all()
    if i==start:assert all(p[k].grad is not None and torch.isfinite(p[k].grad).all() and p[k].grad[~active].abs().max()==0 for k in rates)
    motion_opt.step();motion_opt.zero_grad(set_to_none=True);motion_opt.param_groups[0]['lr']=.003*((1/3)**((i+1)/steps))
    for k,v in opt.items():v.step();v.zero_grad(set_to_none=True);v.param_groups[0]['lr']=rates[k]*((1/3)**((i+1)/steps))
    if (i+1)%100==0 or i==start:
        assert all(torch.equal(v[~active],fixed[k][~active]) for k,v in p.items());row=dict(step=i+1,loss=float(loss),photo=float(photo),regularizer=float(regularizer),max_sample_displacement=float(model.residual(o['camera'],o['frame']).detach().norm(dim=-1).max()),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print('HAND_SCOPE',args.branch,json.dumps(row),flush=True)
    if (i+1)%1000==0 or i+1==steps:checkpoint(i+1)
assert sha(parent)==parent_sha;save(R/'completion.json',dict(status='3000updates_local_trajectory_surface_fit_complete_evaluation_pending',branch=args.branch,scope_steps=3000,points=len(p['means']),active_Gaussians=6330,nonhand_parameters_exactly_preserved=True,original_motion_exactly_frozen=True,new_trajectory_parameters=trajectory.coefficients.numel(),checkpoint_sha256=sha(R/'resume.pt'),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated(),goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
