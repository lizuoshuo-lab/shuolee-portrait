"""Verify exact frozen state, then render all original120native evaluation cases."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import predicted_normal
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=O/'model-support'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['photo_control','normal_flow'],required=True);args=ap.parse_args();R=O/args.branch;E=R/'native-evaluation';E.mkdir(exist_ok=True);assert not (E/'metrics.json').exists();assert read(R/'completion.json')['additional_graph_steps']==1500
torch.set_num_threads(8);cp=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);parent=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);assert cp['step']==16500 and cp['parent_step']==15000 and cp['additional_graph_steps']==1500 and cp['parent_sha256']==sha(S/'local_bspline/resume.pt')
assert cp['params'].keys()==parent['params'].keys();assert all(torch.equal(cp['params'][k],v) for k,v in parent['params'].items());active=cp['active_nodes'];assert torch.equal(active,torch.from_numpy(np.load(M/'active-nodes.npy')))
dynamic=['rotation_raw','translation_raw','local_rotation_raw','local_translation_raw'];changed={}
for k,v in parent['graph'].items():
 if k in dynamic:
  assert torch.equal(cp['graph'][k][~active],v[~active]);changed[k]=float((cp['graph'][k][active]-v[active]).abs().max())
 else:assert torch.equal(cp['graph'][k],v)
assert any(v>0 for v in changed.values());assert all(int(v['step'])==1500 for v in cp['graph_optim']['state'].values());assert all(int(v['step'])==15000 for o in cp['optim'].values() for v in o['state'].values())
save(R/'frozen-state-audit.json',dict(status='allGaussianParameters_and357inactive_graphnodes_exactly_preserved',parent_sha256=cp['parent_sha256'],checkpoint_sha256=sha(R/'resume.pt'),points=len(cp['params']['means']),parameter_tensors_verified=list(cp['params']),inactive_nodes=int((~active).sum()),active_nodes=int(active.sum()),max_active_changes=changed,graph_optimizer_steps=1500,Gaussian_optimizer_steps_unchanged=15000,goal_complete=False))
p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',graph)
normalrows=[];normals=[];flowdir=R/'flow-evaluation';flowdir.mkdir(exist_ok=True);sources={}
with torch.no_grad():
 for r in read(M/'completion.json')['records']:
  if not r['model_visible']:continue
  c,t=r['camera'],r['source_frame'];key=(c,t)
  if key not in sources:
   z=torch.load(M/f'camera{c}-frame{t:06d}-source.pt',map_location='cpu',weights_only=False);sources[key]={k:v.cuda() if torch.is_tensor(v) else v for k,v in z.items()}
  z=np.load(M/r['filename']);pair=dict(camera=c,source_frame=t,target_frame=r['target_frame'],xy=tensor(z['local_xy']),normal=tensor(z['normal']),normal_target=tensor(z['normal_target']),alpha=tensor(z['alpha']));normal,flow=predicted_normal(p,model,pair,sources[key]);error=(normal-pair['normal_target']).abs().cpu().numpy();normals.extend(error.tolist());np.savez_compressed(flowdir/r['filename'],normal=normal.cpu().numpy(),flow=flow.cpu().numpy(),absolute_error=error);normalrows.append(dict(camera=c,source_frame=t,target_frame=r['target_frame'],count=len(error),mean_abs_normal_error_px=float(error.mean()),filename=r['filename']))
save(R/'normal-flow-evaluation.json',dict(records=normalrows,count=len(normals),mean_absolute_error_px=float(np.mean(normals)),median_px=float(np.median(normals)),p90_px=float(np.quantile(normals,.9)),scope='Trainingnormal-componentconsistency,notunseenflowGT orimageclarity.'));sources.clear();del model,parent,cp;torch.cuda.empty_cache()
motion=S/'heldout/capture-affines-estimated.npz';assert sha(motion)==read(S/'heldout/motion-extension-audit.json')['branches']['estimated']['sha256'];model=ResidualGraphRenderer(motion,graph)
records=[r for r in read(S/'input-audit.json')['records'] if r['frame']%2==1 or (r['camera'] in ['01','04','08','17'] and r['frame'] in [720,726,732])];assert len(records)==120
save(E/'protocol.json',dict(status='fixed_before_native_rendering',images=120,resolution=[1920,1080],image_inputs='Original079sameRGBmatteandcameras,90knownodd18otherodd12traindiagnostic.',appearance='ActualGaussianrender; noimageblend,sharpen,upscale,RAFTimagewarp or temporalinterpolation.',motion_sha256=sha(motion),goal_complete=False))
rows=[]
with torch.no_grad():
 for r in records:
  c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';path=S/'native-input'/f'{stem}.npz';assert sha(path)==r['input_sha256'];z=np.load(path);rgb,alpha,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=rgb[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all();pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(z['valid']);box[y0:y1,x0:x1]=True;masks={'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&box};metrics={}
  for name,mask in masks.items():
   d=pred.astype(np.float64)[mask]-ref.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  np.savez_compressed(E/f'{stem}-prediction.npz',rgb=rgb,alpha=alpha);Image.fromarray(u8(pred)).save(E/f'{stem}-trained.png');Image.fromarray(u8(alpha)).save(E/f'{stem}-alpha.png');rows.append(dict(camera=c,frame=t,split=r['split'],metrics=metrics,input_sha256=r['input_sha256'],bbox=r['bbox']));save(E/'progress.json',dict(images=len(rows)));print('HAND_FLOW_NATIVE_EVAL',args.branch,c,t,flush=True)
save(E/'metrics.json',dict(status='120native_predictions_saved_independent_comparison_pending',records=rows,goal_complete=False));shutil.copy2(__file__,R/'source-snapshot'/Path(__file__).name)
