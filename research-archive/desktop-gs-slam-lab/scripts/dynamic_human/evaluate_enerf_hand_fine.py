"""Frozen-boundary audit and original120native render contexts for local joint controls."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_fine_model import HandFineGraph,HandFineRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=F/'model-support';O=L/'outputs/dynamic-human-enerf-hand-fine-085'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',choices=['fine_photo','fine_flow'],required=True);args=ap.parse_args();R=O/args.branch;E=R/'native-evaluation';E.mkdir(exist_ok=True);assert not (E/'metrics.json').exists();assert read(R/'completion.json')['additional_joint_steps']==3000
torch.set_num_threads(8);cp=torch.load(R/'resume.pt',map_location='cpu',weights_only=False);parent=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);assert cp['renderer']=='hand_fine_085' and cp['step']==18000 and cp['parent_sha256']==sha(S/'local_bspline/resume.pt');active=cp['active_Gaussians'];nodes=cp['active_nodes'];assert torch.equal(active,torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy'))) and torch.equal(nodes,torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-nodes.npy')))
changes={}
for k,v in parent['params'].items():
 assert torch.equal(v[~active],cp['params'][k][~active])
 if k in ['anchor','node_indices','node_weights']:assert torch.equal(v,cp['params'][k])
 else:changes[k]=float((v[active]-cp['params'][k][active]).abs().max())
assert all(v>0 for v in changes.values());dynamic=['rotation_raw','translation_raw','local_rotation_raw','local_translation_raw']
for k,v in parent['graph'].items():
 assert torch.equal(cp['reference_graph'][k],v)
 if k in dynamic:assert torch.equal(cp['graph'][k][~nodes],v[~nodes])
 else:assert torch.equal(cp['graph'][k],v)
assert all(int(v['step'])==3000 for v in cp['graph_optim']['state'].values());assert all(int(v['step'])==3000 for o in cp['optim'].values() for v in o['state'].values())
save(R/'frozen-state-audit.json',dict(status='246654nonhandGaussian_parameters_parentmotion_and362inactive_nodes_exactly_preserved',parent_sha256=cp['parent_sha256'],checkpoint_sha256=sha(R/'resume.pt'),points=len(active),active_Gaussians=int(active.sum()),frozen_Gaussians=int((~active).sum()),active_nodes=int(nodes.sum()),additional_local_nodes=32,frozen_nodes=int((~nodes).sum()),max_active_parameter_changes=changes,all_optimizer_steps=3000,goal_complete=False))
fine=HandFineGraph(O/'fine-graph.npz',cp['scene_scale'])
assert cp['fine_graph_source_sha256']==sha(O/'fine-graph.npz')
base=fine.state_dict()
for k,v in base.items():
 if k in dynamic:assert torch.isfinite(cp['fine_graph'][k]).all() and cp['fine_graph'][k].abs().max()>0
 else:assert torch.equal(v.cpu(),cp['fine_graph'][k])
fine.load_state_dict(cp['fine_graph'])

p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['reference_graph']);ref.requires_grad_(False);model=HandFineRenderer(S/'head/capture-affines.npz',graph,ref,active.cuda(),fine);sources={};rows=[];errors=[];alphas=[];flowdir=R/'flow-evaluation';flowdir.mkdir(exist_ok=True)
with torch.no_grad():
 for r in read(M/'completion.json')['records']:
  if not r['model_visible']:continue
  c,t=r['camera'],r['source_frame'];key=(c,t)
  if key not in sources:
   s=torch.load(M/f'camera{c}-frame{t:06d}-source.pt',map_location='cpu',weights_only=False);sources[key]={k:s[k].cuda() for k in ['W','K']}
  z=np.load(M/r['filename']);pair=dict(camera=c,source_frame=t,target_frame=r['target_frame'],xy=tensor(z['local_xy']),normal=tensor(z['normal']),normal_target=tensor(z['normal_target']));normal,flow,alpha=model.current_normal_flow(p,pair,sources[key]);err=(normal-pair['normal_target']).abs().cpu().numpy();errors.extend(err.tolist());alphas.extend(alpha[:,0].cpu().tolist());np.savez_compressed(flowdir/r['filename'],normal=normal.cpu().numpy(),flow=flow.cpu().numpy(),alpha=alpha[:,0].cpu().numpy(),absolute_error=err);rows.append(dict(camera=c,source_frame=t,target_frame=r['target_frame'],count=len(err),mean_abs_normal_error_px=float(err.mean()),filename=r['filename']))
save(R/'normal-flow-evaluation.json',dict(records=rows,count=len(errors),mean_absolute_error_px=float(np.mean(errors)),median_px=float(np.median(errors)),p90_px=float(np.quantile(errors,.9)),alpha_min=float(min(alphas)),alpha_below095=int((np.array(alphas)<.95).sum()),scope='Currentvisibility normal-component fit to same1527estimatedtrainingpoints,notGT orheldoutmotion accuracy.'));del model,parent,cp;torch.cuda.empty_cache()
motion=S/'heldout/capture-affines-estimated.npz';assert sha(motion)==read(S/'heldout/motion-extension-audit.json')['branches']['estimated']['sha256'];model=HandFineRenderer(motion,graph,ref,active.cuda(),fine)
records=[r for r in read(S/'input-audit.json')['records'] if r['frame']%2==1 or (r['camera'] in ['01','04','08','17'] and r['frame'] in [720,726,732])];assert len(records)==120;save(E/'protocol.json',dict(status='fixed_before_native_rendering',images=120,resolution=[1920,1080],targets='Original079 RGBmatteandcameras,90knownodd18otherodd12train.',appearance='ActualGaussianrender;noenhancement orimagewarp.',motion_sha256=sha(motion),goal_complete=False));rows=[]
with torch.no_grad():
 for r in records:
  c,t=r['camera'],r['frame'];stem=f'camera{c}-frame{t:06d}';path=S/'native-input'/f'{stem}.npz';assert sha(path)==r['input_sha256'];z=np.load(path);rgb,alpha,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=rgb[0].cpu().numpy();alpha=alpha[0,:,:,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all();pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);target=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];box=np.zeros_like(z['valid']);box[y0:y1,x0:x1]=True;masks={'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5),'foreground_bbox':z['valid']&box};metrics={}
  for name,mask in masks.items():
   d=pred.astype(np.float64)[mask]-target.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  np.savez_compressed(E/f'{stem}-prediction.npz',rgb=rgb,alpha=alpha);Image.fromarray(u8(pred)).save(E/f'{stem}-trained.png');Image.fromarray(u8(alpha)).save(E/f'{stem}-alpha.png');rows.append(dict(camera=c,frame=t,split=r['split'],metrics=metrics,input_sha256=r['input_sha256'],bbox=r['bbox']));save(E/'progress.json',dict(images=len(rows)));print('HAND_FINE_NATIVE_EVAL',args.branch,c,t,flush=True)
save(E/'metrics.json',dict(status='120native_predictions_saved_independent_comparison_pending',records=rows,goal_complete=False));shutil.copy2(__file__,R/'source-snapshot'/Path(__file__).name)
