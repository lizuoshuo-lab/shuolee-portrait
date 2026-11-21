"""Read-only check of frozen versus current source visibility in center-flow rendering."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer
from enerf_hand_flow_model import means_at,geometry_at,project,sample,splat
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';M=O/'model-support';D=O/'visibility-diagnostic';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
assert not (D/'summary.json').exists();torch.set_num_threads(8);allrows={}
with torch.no_grad():
 for branch in ['parent','photo_control','normal_flow']:
  path=S/'local_bspline/resume.pt' if branch=='parent' else O/branch/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(S/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ResidualGraphRenderer(S/'head/capture-affines.npz',graph);cache={};rows=[];arrays=[]
  for r in read(M/'completion.json')['records']:
   if not r['model_visible']:continue
   c,t=r['camera'],r['source_frame'];key=(c,t)
   if key not in cache:
    old=torch.load(M/f'camera{c}-frame{t:06d}-source.pt',map_location='cpu',weights_only=False);old={k:v.cuda() if torch.is_tensor(v) else v for k,v in old.items()};means,cov=geometry_at(p,model,c,t);current=dict(old,means=means,cov=cov);uv,_=project(means,old['W'],old['K']);old_uv,_=project(old['means'],old['W'],old['K']);displacement=torch.linalg.norm(uv-old_uv,dim=-1);shiftmap,_=splat(displacement[:,None],old,p);cache[key]=(old,current,uv,shiftmap)
   old,current,uv,shiftmap=cache[key];z=np.load(M/r['filename']);xy=tensor(z['local_xy']);n=tensor(z['normal']);target=tensor(z['normal_target']);uv1,_=project(means_at(p,model,c,r['target_frame']),old['W'],old['K']);flow=uv1-uv
   currentmap,al=splat(flow,current,p);a=sample(al,xy)[:,0];f=sample(currentmap,xy)/a.clamp_min(1e-8)[:,None];frozenmap,_=splat(flow,old,p);ff=sample(frozenmap,xy)/tensor(z['alpha']);normal=(f*n).sum(-1);frozennormal=(ff*n).sum(-1);shift=sample(shiftmap,xy)[:,0]/tensor(z['alpha'])[:,0]
   values=dict(current_normal_error=(normal-target).abs().cpu().numpy(),frozen_normal_error=(frozennormal-target).abs().cpu().numpy(),normal_component_change=(normal-frozennormal).abs().cpu().numpy(),parent_weighted_source_center_shift=shift.cpu().numpy(),current_alpha=a.cpu().numpy())
   assert all(np.isfinite(x).all() for x in values.values());np.savez_compressed(D/f'{branch}-{r["filename"]}',**values);arrays.append(values);rows.append(dict(camera=c,source_frame=t,target_frame=r['target_frame'],count=len(a),current_normal_mae=float(values['current_normal_error'].mean())))
  allv={k:np.concatenate([v[k] for v in arrays]) for k in arrays[0]};assert len(allv['current_alpha'])==1527
  stats={k:dict(mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=float(v.max())) for k,v in allv.items()}
  if branch=='parent':assert stats['normal_component_change']['maximum']<1e-5 and stats['parent_weighted_source_center_shift']['maximum']<1e-5
  assert sha(path)==before;allrows[branch]=dict(count=1527,checkpoint_sha256=before,statistics=stats,alpha_below095=int((allv['current_alpha']<.95).sum()),records=rows);print('VISIBILITY_DIAGNOSTIC',branch,json.dumps(stats),flush=True);del cache,model,graph,p,cp;torch.cuda.empty_cache()
save(D/'summary.json',dict(status='read_only_current_vs_frozen_source_visibility_complete',branches=allrows,scope='Same1527training constraints. Both flow variants approximate Gaussian center displacement; neither is ground-truth surface flow. No parameter optimization, no RGB modification.',goal_complete=False))
