"""Matched static, nominal-time dynamic and clock-corrected dynamic baselines."""
from pathlib import Path
import argparse,json,hashlib,random,time,os,shutil
import numpy as np,torch
from gsplat.strategy import DefaultStrategy
from train_enerf_camera10_time import ss
from enerf_graph_renderer import bind_graph
from actor23_graph_model import ActorGraph,render,geometry

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-actor23-independent-106';M=O/'model'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','train'],required=True);ap.add_argument('--branch',choices=['static','dynamic_nominal','dynamic_clock'],default='dynamic_clock');a=ap.parse_args();torch.set_num_threads(6);torch.manual_seed(10601);np.random.seed(10601);random.seed(10601)
init=read(M/'initialization-and-graph.json');assert sha(M/'initial.npz')==init['initial_sha256'] and sha(M/'graph.npz')==init['graph_sha256'];z=np.load(M/'initial.npz');N=len(z['points']);scale=init['scene_scale'];C0=.28209479177387814
p=torch.nn.ParameterDict({k:torch.nn.Parameter(tt(v)) for k,v in dict(means=z['points'],scales=np.full((N,3),np.log(.009)),quats=np.tile([1,0,0,0],(N,1)),opacities=np.full(N,-2.1972246),sh0=((z['colors']-.5)/C0)[:,None,:],shN=np.zeros((N,8,3))).items()})
rates=dict(means=.00016*scale,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20);opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()};strategy=DefaultStrategy(refine_start_iter=400,refine_stop_iter=10000,reset_every=3000,refine_every=100,verbose=True);strategy.check_sanity(p,opt);state=strategy.initialize_state(scale);bind_graph(p,M/'graph.npz');g=ActorGraph(M/'graph.npz',scale);clocks=read(O/'geometry-time/candidate-clock-offsets.json')['offsets'];audit=read(O/'training-input-audit.json')['records'];plan=read(O/'validation-reservation.json');obs=[]
for r in (audit if a.mode=='train' else [r for r in audit if r['camera']=='04' and r['frame']==624]):
    assert r['camera'] not in plan['heldout_cameras'] and r['frame']%2==0;path=O/r['input'].replace('\\','/');assert sha(path)==r['input_sha256'];z=np.load(path);x0,y0,x1,y1=z['bbox']
    obs.append(dict(camera=r['camera'],frame=r['frame'],W=tt(z['w2c']),K=tt(z['K']),rgb=torch.as_tensor(z['rgb'][y0:y1,x0:x1],device='cuda'),matte=tt(z['matte'][y0:y1,x0:x1])[...,None],valid=tt(z['valid'][y0:y1,x0:x1])[...,None],bbox=[int(x0),int(y0),int(x1),int(y1)]))
def loss(o,index,bg,degree):
    raw,alpha,info=render(p,g,index,o['W'],o['K'],degree=degree);x0,y0,x1,y1=o['bbox'];rgb=raw[:,y0:y1,x0:x1];alpha=alpha[:,y0:y1,x0:x1];pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid']
    photometric=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean()
    return photometric,info
if a.mode=='smoke':
    assert not (M/'runtime-smoke.json').exists();o=obs[0]
    with torch.no_grad():
        mu,cov,A,_,_=geometry(p,g,624);identity_error=float((mu-p['means']).abs().max());assert identity_error<2e-6
        r0=render(p,g,612,o['W'],o['K']);r1=render(p,g,624,o['W'],o['K']);image_error=max(float((r0[0]-r1[0]).abs().max()),float((r0[1]-r1[1]).abs().max()));assert image_error<2e-6
    photo,info=loss(o,624,tt([.5,.5,.5]),0);photo.backward();gradients={k:float(v.grad.norm()) for k,v in g.named_parameters()};assert all(np.isfinite(x) and x>0 for x in gradients.values());assert torch.isfinite(p['means'].grad).all()
    save(M/'runtime-smoke.json',dict(status='identity_native_render_and_actual_graph_gradients_verified',native_canvas=[1920,1080],loss_slice=o['bbox'],identity_mean_max_error=identity_error,zero_motion_render_max_error=image_error,graph_gradient_norms=gradients,initial_photometric_loss=float(photo),no_optimizer_steps=True,goal_complete=False));print('ACTOR23_SMOKE_PASSED',flush=True);exit()
assert len(obs)==195 and read(M/'runtime-smoke.json')['no_optimizer_steps'];R=O/a.branch;R.mkdir(exist_ok=True);assert not (R/'completion.json').exists();steps=15000
spec=dict(branch=a.branch,steps=steps,initial_sha256=sha(M/'initial.npz'),graph_sha256=sha(M/'graph.npz'),input_manifest_sha256=sha(O/'training-input-audit.json'),clock_file_sha256=sha(O/'geometry-time/candidate-clock-offsets.json'),reference_index=612,time_normalizer_frames=12,train_images=195,seed=10601,native_canvas=[1920,1080],loss='Full native render then fixed foreground-bbox slice:0.8L1+0.2DSSIM+0.1matteL1; random uniform RGB background. Masks include objects and uncertainty.',graph_regularizers=dict(rigidity=10,rotation_size=.001,translation_size=.1),graph_parameters='384node local rigid graph:3cubic+7B-spline coefficients per rotation/translation. Cubic coefficients bounded .15rad and .08scene_scale; local .05rad and .02scene_scale. Reference exactly identity.',learning_rates=rates,graph_learning_rate=.003,point_limit=800000,densification='400..10000every100;reset every3000;same gsplatDefaultStrategy',comparison='Static freezes all motion coefficients at zero. Dynamic nominal uses image index; dynamic clock uses image index + training-only fitted camera offset. All three share geometry/appearance initialization and195native photos, same15000draws and backgrounds. No heldout model selection.',scope='New actors, local research baseline, not official4K4D/OmniRe or complete large scene. Same recording site as other ENeRF actors; not a new environment.',goal_complete=False)
if (R/'protocol.json').exists():assert read(R/'protocol.json')==spec
else:save(R/'protocol.json',spec)
rng=random.Random(10601);draws=[rng.randrange(len(obs)) for _ in range(steps)];bg=torch.rand(steps,3,generator=torch.Generator().manual_seed(90601));np.savez_compressed(R/'sampling-schedule.npz',camera=np.array([obs[i]['camera'] for i in draws]),frame=np.array([obs[i]['frame'] for i in draws]),background=bg.numpy());save(R/'initial-and-sampling-audit.json',dict(initial={k:sha256 for k,v in p.items() if (sha256:=hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest())},sampling_sha256=sha(R/'sampling-schedule.npz'),draw_sha256=hashlib.sha256(np.array(draws,np.int32).tobytes()).hexdigest(),initial_graph={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in g.state_dict().items()}))
g.requires_grad_(a.branch!='static');go=torch.optim.Adam(g.parameters(),lr=.003,eps=1e-15);bg=bg.cuda();begin=0;history=[];elapsed=0
if (R/'resume.pt').exists():
    cp=torch.load(R/'resume.pt',weights_only=False,map_location='cpu');assert cp['protocol']==spec;p=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda(),requires_grad=k not in ['node_indices','node_weights']) for k,v in cp['params'].items()});opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()}
    for k in opt:opt[k].load_state_dict(cp['optim'][k])
    g.load_state_dict(cp['graph']);go.load_state_dict(cp['graph_optim']);state={k:v.cuda() if torch.is_tensor(v) else v for k,v in cp['strategy'].items()};begin=cp['step'];history=cp['history'];elapsed=cp['elapsed'];torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda']);np.random.set_state(cp['numpy']);random.setstate(cp['random'])
started=time.monotonic()-elapsed
def checkpoint(step):
    cp=dict(protocol=spec,renderer='actor23_component_graph',step=step,scene_scale=scale,params={k:v.detach().cpu() for k,v in p.items()},optim={k:v.state_dict() for k,v in opt.items()},graph={k:v.detach().cpu() for k,v in g.state_dict().items()},graph_optim=go.state_dict(),strategy=state,history=history,elapsed=time.monotonic()-started,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),random=random.getstate());torch.save(cp,R/'resume.partial');os.replace(R/'resume.partial',R/'resume.pt')
for i in range(begin,steps):
    o=obs[draws[i]];index=612 if a.branch=='static' else o['frame']+(clocks[o['camera']] if a.branch=='dynamic_clock' else 0);photo,info=loss(o,index,bg[i],min(2,i//1000));reg=10*info['rigidity']+.001*info['rotation_size']+.1*info['translation_size'];total=photo+reg;assert torch.isfinite(total);strategy.step_pre_backward(p,opt,state,i,info);total.backward()
    if a.branch!='static':go.step();go.zero_grad(set_to_none=True);go.param_groups[0]['lr']=.003*(.1**((i+1)/steps))
    for op in opt.values():op.step();op.zero_grad(set_to_none=True)
    strategy.step_post_backward(p,opt,state,i,info,packed=False);opt['means'].param_groups[0]['lr']=rates['means']*(.01**((i+1)/steps))
    if len(p['means'])>800000:checkpoint(i+1);raise RuntimeError('Point budget exceeded; checkpoint preserved')
    if (i+1)%100==0 or i==0:
        torch.cuda.synchronize();row=dict(step=i+1,loss=float(total),photometric_loss=float(photo),regularizer=float(reg),points=len(p['means']),elapsed_seconds=time.monotonic()-started,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(R/'progress.json',row);print(a.branch,json.dumps(row),flush=True)
    if (i+1)%1000==0:checkpoint(i+1)
assert all(torch.isfinite(v).all() for v in p.values())
if a.branch=='static':assert all(torch.count_nonzero(v)==0 for v in g.parameters())
save(R/'completion.json',dict(status='15000_updates_complete_native_evaluation_required',branch=a.branch,steps=steps,points=len(p['means']),checkpoint_sha256=sha(R/'resume.pt'),elapsed_seconds=time.monotonic()-started,all_parameters_finite=True,static_motion_exactly_zero=a.branch=='static',goal_complete=False))
for name in ['train_actor23_control.py','actor23_graph_model.py']:shutil.copy2(L/'scripts/dynamic_human'/name,R/name)
print('ACTOR23_TRAINING_COMPLETE',a.branch,flush=True)
