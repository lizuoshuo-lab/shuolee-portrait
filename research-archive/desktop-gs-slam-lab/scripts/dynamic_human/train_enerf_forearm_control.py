from pathlib import Path
import argparse,json,time,hashlib,random,os,shutil
import numpy as np,torch
from gsplat.strategy import DefaultStrategy
from train_enerf_camera10_time import ss
from enerf_articulated_model import initial_parameters
from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer,bind_graph
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';MOTION=LAB/'outputs/dynamic-human-enerf-continuous-body-063'
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=['residual_graph']);ap.add_argument('--steps',type=int,default=15000);ap.add_argument('--result-root',type=Path,default=OUT);ap.add_argument('--motion-file',type=Path);a=ap.parse_args();root=a.result_root/a.branch;root.mkdir(parents=True,exist_ok=True);assert not (root/'completion.json').exists();motion_file=a.motion_file or MOTION/a.branch/'capture-affines.npz'
 assert read(OUT/'runtime-smoke.json')['status']=='identity_affine_covariance_gradients_and_density_bindings_verified'
 torch.set_num_threads(8);torch.manual_seed(73);np.random.seed(73);random.seed(73);tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
 repair_audit=read(a.result_root/'mask-independent-audit.json'); repairs={(x['camera'],x['frame']):x for x in repair_audit['records'] if x['accepted']}; assert set(repairs)=={('01',750),('01',752),('01',754)}
 records=[r for r in read(OUT/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105 and all(r['frame']%2==0 for r in records);obs=[]
 for r in records:
  f=OUT/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(f)==r['input_sha256'];z=np.load(f);x0,y0,x1,y1=z['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0
  train_matte=z['matte']
  if (r['camera'],r['frame']) in repairs:
   row=repairs[r['camera'],r['frame']];candidate=a.result_root/'candidates'/f"camera{r['camera']}-frame{r['frame']:06d}.npz"
   assert sha(f)==row['source_sha256'] and sha(candidate)==row['candidate_sha256']
   train_matte=np.load(candidate)['matte'];assert train_matte.shape==z['matte'].shape
  obs.append(dict(camera=r['camera'],frame=r['frame'],rgb=torch.tensor(z['rgb'][y0:y1,x0:x1],device='cuda'),matte=tensor(train_matte[y0:y1,x0:x1])[...,None],valid=tensor(z['valid'][y0:y1,x0:x1])[...,None],K=tensor(K),W=tensor(z['w2c']),width=int(x1-x0),height=int(y1-y0)))
 p=initial_parameters(OUT/'initial.npz');scene_scale=float(torch.linalg.vector_norm(p['means'].max(0).values-p['means'].min(0).values)*.5)
 rates=dict(means=.00016*scene_scale,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.0025/20);opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()};strategy=DefaultStrategy(refine_start_iter=400,refine_stop_iter=10000,reset_every=3000,refine_every=100,verbose=True);strategy.check_sanity(p,opt);state=strategy.initialize_state(scene_scale)
 rng=random.Random(73);draws=[rng.randrange(len(obs)) for _ in range(a.steps)];background=torch.rand(a.steps,3,generator=torch.Generator().manual_seed(9073));np.savez_compressed(root/'sampling-schedule.npz',camera=np.array([obs[i]['camera'] for i in draws]),frame=np.array([obs[i]['frame'] for i in draws]),background=background.numpy())
 save(root/'initial-and-sampling-audit.json',dict(initial_parameters={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in p.items()},draw_sha256=hashlib.sha256(np.array(draws,dtype=np.int32).tobytes()).hexdigest(),background_sha256=hashlib.sha256(background.numpy().tobytes()).hexdigest(),steps=a.steps,initial_points=len(p['means'])))
 assert read(root/'initial-and-sampling-audit.json')==read(LAB/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph/initial-and-sampling-audit.json')
 graph_file=a.result_root/'graph.npz';graph=GraphMotion(graph_file,scene_scale);bind_graph(p,graph_file);model=ResidualGraphRenderer(motion_file,graph);gopt=torch.optim.Adam(graph.parameters(),lr=.003,eps=1e-15)
 graph_protocol=dict(graph_sha256=sha(graph_file),nodes=384,coefficients=6912,learning_rate=.003,regularizers=dict(rigidity=10.0,rotation_size=.001,translation_size=.1),reference_physical_frame=750,reference_identity=True,body_motion_frozen=True,topology='traininginitialvisualhullonly,4frozenmaterialweights,8edgespernode',selection='Fixed15000updates; exploratorydevelopmentcandidate; noheldoutcheckpointselection',goal_complete=False)
 if (root/'graph-protocol.json').exists():assert read(root/'graph-protocol.json')==graph_protocol
 save(root/'graph-protocol.json',graph_protocol)
 save(root/'protocol.json',dict(branch=a.branch,updates=a.steps,learning_rates=rates,input_count=105,input_manifest_sha256=sha(OUT/'input-audit.json'),motion_sha256=sha(motion_file),renderer='body-affine plus local residualgraph; covariance and canonicalSH2 transformed',selection='fixedfinaliteration, no validationcheckpointselection',goal_complete=False))
 background=background.cuda();start=0;history=[];elapsed=0
 if (root/'resume.pt').exists():
  cp=torch.load(root/'resume.pt',map_location='cpu',weights_only=False);p=torch.nn.ParameterDict({k:torch.nn.Parameter(v.cuda(),requires_grad=k not in ['anchor','node_indices','node_weights']) for k,v in cp['params'].items()});opt={k:torch.optim.Adam([p[k]],lr=v,eps=1e-15) for k,v in rates.items()}
  for k in opt:opt[k].load_state_dict(cp['optim'][k])
  assert cp['graph_sha256']==sha(graph_file) and cp['motion_sha256']==sha(motion_file);graph.load_state_dict(cp['graph']);gopt.load_state_dict(cp['graph_optim'])
  state={k:v.cuda() if torch.is_tensor(v) else v for k,v in cp['strategy'].items()};start=cp['step'];history=cp['history'];elapsed=cp['elapsed'];random.setstate(cp['random']);np.random.set_state(cp['numpy']);torch.set_rng_state(cp['torch']);torch.cuda.set_rng_state(cp['cuda'])
 began=time.monotonic()-elapsed
 def checkpoint(step):
  torch.save(dict(graph_sha256=sha(graph_file),motion_sha256=sha(motion_file),scene_scale=scene_scale,graph={k:v.detach().cpu() for k,v in graph.state_dict().items()},graph_optim=gopt.state_dict(),step=step,params={k:v.detach().cpu() for k,v in p.items()},optim={k:v.state_dict() for k,v in opt.items()},strategy=state,history=history,elapsed=time.monotonic()-began,random=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state()),root/'resume.partial');os.replace(root/'resume.partial',root/'resume.pt')
 for i in range(start,a.steps):
  o=obs[draws[i]];bg=background[i];degree=min(2,i//1000);rgb,alpha,info=model.render(p,o['camera'],o['frame'],o['W'],o['K'],o['width'],o['height'],degree)
  pred=(rgb[0]+bg*(1-alpha[0]))*o['valid'];target=(o['rgb'].float()/255*o['matte']+bg*(1-o['matte']))*o['valid'];loss=.8*(pred-target).abs().mean()+.2*(1-ss(pred.permute(2,0,1)[None],target.permute(2,0,1)[None]))+.1*((alpha[0]-o['matte']).abs()*o['valid']).mean()
  photometric=loss;regularization=10*info['graph_rigidity']+.001*info['graph_rotation_size']+.1*info['graph_translation_size'];loss=photometric+regularization
  assert torch.isfinite(loss);strategy.step_pre_backward(p,opt,state,i,info);loss.backward()
  if i==start:assert p['means'].grad is not None and torch.isfinite(p['means'].grad).all() and p['means'].grad.abs().sum()>0
  if i==start:
   assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in graph.parameters())
  gopt.step();gopt.zero_grad(set_to_none=True);gopt.param_groups[0]['lr']=.003*(.1**((i+1)/a.steps))
  for v in opt.values():v.step();v.zero_grad(set_to_none=True)
  strategy.step_post_backward(p,opt,state,i,info,packed=False);opt['means'].param_groups[0]['lr']=rates['means']*(.01**((i+1)/a.steps))
  assert p['anchor'].dtype==torch.long and not p['anchor'].requires_grad
  assert p['node_indices'].dtype==torch.long and not p['node_weights'].requires_grad and not p['node_indices'].requires_grad
  if len(p['means'])>800000:checkpoint(i+1);raise RuntimeError('800000 point budget exceeded; checkpoint preserved')
  if (i+1)%100==0 or i==0:
   torch.cuda.synchronize();row=dict(step=i+1,loss=float(loss),photometric_loss=float(photometric),rigidity=float(info['graph_rigidity']),rotation_size=float(info['graph_rotation_size']),translation_size=float(info['graph_translation_size']),points=len(p['means']),elapsed_seconds=time.monotonic()-began,max_gpu_bytes=torch.cuda.max_memory_allocated());history.append(row);save(root/'progress.json',row);print(a.branch,json.dumps(row),flush=True)
  if (i+1)%1000==0 or i+1==a.steps:checkpoint(i+1)
 save(root/'completion.json',dict(status='15000update_residual_graph_training_complete_native_evaluation_pending',step=a.steps,points=len(p['means']),max_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-began,goal_complete=False));snap=root/'source-snapshot';snap.mkdir(exist_ok=True)
 for n in ['train_enerf_forearm_control.py','enerf_articulated_model.py','enerf_graph_renderer.py','enerf_residual_graph.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
if __name__=='__main__':main()
