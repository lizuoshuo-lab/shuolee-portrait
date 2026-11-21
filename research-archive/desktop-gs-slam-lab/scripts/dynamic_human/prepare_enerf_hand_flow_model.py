from pathlib import Path
import json,hashlib,shutil
import numpy as np,cv2,torch
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import geometry_at,project,splat,sample,predicted_normal
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';N=O/'normal-flow';M=O/'model-support';M.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (M/'completion.json').exists();assert read(N/'visual-review.json')['training_approved'];torch.set_num_threads(8)
cp=torch.load(S/'local_bspline/resume.pt',map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',graph)
records=read(N/'preparation.json')['records'];active=torch.zeros(384,device='cuda',dtype=torch.bool);out=[];sources={};checks=[]
with torch.no_grad():
 for r in records:
  if not r['accepted']:continue
  c,t=r['camera'],r['source_frame'];q=dict(np.load(N/r['filename']));key=(c,t)
  if key not in sources:
   z=np.load(S/'native-input'/f'camera{c}-frame{t:06d}.npz');box=q['source_crop'];K=z['K'].copy();K[0,2]-=box[0];K[1,2]-=box[1];W=tensor(z['w2c']);K=tensor(K);means,C=geometry_at(p,model,c,t);uv,depth=project(means,W,K)
   source=dict(means=means,cov=C,W=W,K=K,crop=box);light=(p['anchor']>=6890).float();features=torch.stack([light,depth,torch.ones_like(depth)*2,torch.ones_like(depth)*-3],-1);value,alpha=splat(features,source,p);expected=value[:, :,1:2]/alpha.clamp_min(1e-8)
   error=float((value[:,:,2:]-alpha*torch.tensor([2.,-3.],device='cuda')).abs().max());assert error<1e-4
   maskdata=np.load(O/'hand-masks'/f'camera{c}-frame{t:06d}.npz');mask=cv2.erode(maskdata['mask'].astype(np.uint8),np.ones((3,3),np.uint8));assert np.array_equal(box,maskdata['crop'])
   inside=(uv[:,0]>=0)&(uv[:,0]<159)&(uv[:,1]>=0)&(uv[:,1]<159)&(depth>.1)&(light>.5);ids=torch.where(inside)[0]
   near=(sample(tensor(mask)[...,None],uv[ids])[:,0]>.999)&(abs(depth[ids]-sample(expected,uv[ids])[:,0])<.03*cp['scene_scale'])
   gidx=p['node_indices'][ids[near]].reshape(-1);gidx=gidx[graph.node_anchor[gidx]>=6890];active[gidx]=True
   source.update(alpha=alpha,light=value[:,:,0:1]);sources[key]=source
   torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in source.items()},M/f'camera{c}-frame{t:06d}-source.pt');checks.append(dict(camera=c,frame=t,constant_feature_error=error,source_means_check=True,selected_Gaussians=int(near.sum())))
  source=sources[key];xy=tensor(q['xy']-q['source_crop'][:2]);al=sample(source['alpha'],xy);light=sample(source['light'],xy)/al.clamp_min(1e-8);gate=((al[:,0]>=.95)&(light[:,0]>=.9)).cpu().numpy()
  if not gate.any():out.append(dict(**r,model_visible=0));continue
  pair=dict(camera=c,source_frame=t,target_frame=r['target_frame'],xy=xy[gate],normal=tensor(q['normal'][gate]),normal_target=tensor(q['normal_target'][gate]),alpha=al[gate]);pred,flow=predicted_normal(p,model,pair,source)
  np.savez_compressed(M/r['filename'],xy=q['xy'][gate],local_xy=pair['xy'].cpu().numpy(),normal=q['normal'][gate],normal_target=q['normal_target'][gate],alpha=al[gate].cpu().numpy(),parent_normal=pred.cpu().numpy(),parent_flow=flow.cpu().numpy(),source_crop=q['source_crop'])
  out.append(dict(**r,model_visible=int(gate.sum()),support_sha256=sha(M/r['filename']),parent_mean_abs_normal_error=float((pred-pair['normal_target']).abs().mean())));print('HAND_MODEL_SUPPORT',c,t,r['target_frame'],int(gate.sum()),flush=True)
np.save(M/'active-nodes.npy',active.cpu().numpy());assert 0<int(active.sum())<96
save(M/'completion.json',dict(status='frozen_source_visibility_support_complete',parent_sha256=sha(S/'local_bspline/resume.pt'),records=out,source_checks=checks,active_nodes=int(active.sum()),active_node_indices=torch.where(active)[0].cpu().tolist(),active_node_sha256=sha(M/'active-nodes.npy'),visible_normal_constraints=sum(r['model_visible'] for r in out),goal_complete=False));shutil.copy2(__file__,M/Path(__file__).name);print('HAND_MODEL_SUPPORT_COMPLETE',int(active.sum()),sum(r['model_visible'] for r in out),flush=True)
