"""Approximate rendered depth ordering; not a source geometry ground truth."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from gsplat.rendering import rasterization
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-hand-visibility-100';D=O/'depth-order';D.mkdir(exist_ok=True);read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
torch.set_num_threads(6);path=O/'context_reset/resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(O/'context-reset-mask.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active);hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],True);hand.load_state_dict(cp['hand_surface']);hand.requires_grad_(False);cases=read(O/'attribution/summary.json')['records'];rows=[]
with torch.no_grad():
 for r in cases:
  c,f=r['camera'],r['frame'];stem=f'camera{c}-frame{f:06d}';z=np.load(S/'native-input'/f'{stem}.npz');x0,y0,x1,y1=r['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K=tensor(K);W=tensor(z['w2c']);valid=z['valid'][y0:y1,x0:x1];mu,cv,_,_=old.geometry(p,c,f);hm,hc,ha,hcolor,_=hand.geometry(c,f)
  values={}
  for name,means,cov,alpha in [('old',mu,cv,p['opacities'].sigmoid()),('new',hm,hc,ha)]:
   centerdepth=(means@W[:3,:3].T+W[:3,3])[:,2];moment,al,_=rasterization(means,None,None,alpha,centerdepth[:,None],W[None],K[None],160,160,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3);depth=(moment[0,:,:,0]/al[0,:,:,0].clamp_min(1e-8)).cpu().numpy();values[name]=dict(depth=depth,alpha=al[0,:,:,0].cpu().numpy())
  full=np.load(O/'context_reset/native-evaluation'/f'{stem}-prediction.npz');weight=full['surface_weight'][y0:y1,x0:x1];na=values['new']['alpha'];oa=values['old']['alpha'];gap=values['new']['depth']-values['old']['depth'];mask=valid&(na>.2)&(oa>.95)&(weight<.1*na);assert mask.sum()>20
  row=dict(camera=c,frame=f,pixels=int(mask.sum()),new_minus_old_expected_depth_world_quantiles=np.quantile(gap[mask],[0,.1,.5,.9,1]).tolist(),fraction_new_depth_behind_old=float((gap[mask]>0).mean()),scene_scale=float(cp['scene_scale']),scope='Pixels with model-defined new alpha>0.2, old alpha>0.95 and visible new weight<10percent of newonlyalpha. Positive depthgap means new component expected centerdepth lies behind old component expected centerdepth. Neither is ground-truth first surface depth.')
  rows.append(row);np.savez_compressed(D/f'{stem}.npz',old_depth=values['old']['depth'],new_depth=values['new']['depth'],old_alpha=oa,new_alpha=na,visible_new_weight=weight,mask=mask,bbox=r['bbox']);print('HAND_DEPTH',c,f,row,flush=True)
assert sha(path)==before;save(D/'summary.json',dict(status='four_contexts_actual_expected_depth_order_audited',checkpoint_sha256=before,records=rows,formula='D=sum(alpha_i*T_i*z_i)/sum(alpha_i*T_i), with z_i theGaussian center camera depth. This doesnot recover truevisible hand geometry.',goal_complete=False));shutil.copy2(__file__,D/Path(__file__).name)
