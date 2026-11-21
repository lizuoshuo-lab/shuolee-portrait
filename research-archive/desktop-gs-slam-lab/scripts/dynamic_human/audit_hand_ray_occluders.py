"""Trace the actual sorted GS tile list at fixed training-image pixels.

CPU compositing is checked against CUDA output; anchor ownership is heuristic.
"""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,smplx
from gsplat.rendering import rasterization
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';P=L/'outputs/dynamic-human-enerf-hand-visibility-100';O=L/'outputs/dynamic-human-hand-body-alignment-101/ray-occluders';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tt=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
assert not (O/'summary.json').exists();torch.set_num_threads(4)
path=P/'context_reset/resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};n=len(p['means']);act=np.load(P/'context-reset-mask.npy');assoc=np.load(P/'reset-mask.npy')
g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,torch.from_numpy(act).cuda())
hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(P/'protocol.json')['initial_color'],True);hand.load_state_dict(cp['hand_surface']);hand.requires_grad_(False)
smpl=smplx.SMPL('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl',num_betas=10)
dom=smpl.lbs_weights.argmax(-1).numpy();anchor=p['anchor'].cpu().numpy();person=anchor//6890;joint=dom[anchor%6890]
names=['associated1937','other_context4393','frozen_light_wrist_or_hand_anchor','frozen_other_light_anchor','frozen_dark_anchor','new_MANO']
group=np.full(n+1538,5,np.int64);group[:n]=np.where(person==1,np.where(np.isin(joint,[21,23]),2,3),4);group[:n][act]=1;group[:n][assoc]=0
tags=torch.nn.functional.one_hot(torch.tensor(group,device='cuda'),6).float()
plan=[r for r in read(P/'contribution/summary.json')['records'] if r['camera'] in ['03','04','13'] and r['frame']==732];assert len(plan)==3
save(O/'protocol.json',dict(status='fixed_before_ray_trace',source_checkpoint_sha256=before,cases=[dict(camera=r['camera'],frame=r['frame']) for r in plan],selection='16evenlyspaced native pixel indices per context where new-onlyalpha>0.2 and jointnewweight<10percentnewalpha, valid source pixel. Model-defined occlusion diagnostics, not hand labels.',scope='Use actual gsplat tile order, alpha clamp0.999, alpha threshold1/255 and exclusive transmittance stopping at1e-4. Compare six group weights against CUDA per pixel before attributing contributors.',no_training=True,heldout_inputs_used=False,goal_complete=False))
records=[]
with torch.no_grad():
 for r in plan:
  c,f=r['camera'],r['frame'];z=np.load(S/'native-input'/f'camera{c}-frame{f:06d}.npz');x0,y0,x1,y1=r['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K=tt(K);W=tt(z['w2c']);mu,cv,_,_=old.geometry(p,c,f);hm,hc,ha,_,_=hand.geometry(c,f);means=torch.cat([mu,hm]);cov=torch.cat([cv,hc]);alpha=torch.cat([p['opacities'].sigmoid(),ha])
  rr,al,meta=rasterization(means,None,None,alpha,tags,W[None],K[None],160,160,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
  _,nal,_=rasterization(hm,None,None,ha,torch.ones((1538,1),device='cuda'),W[None],K[None],160,160,covars=hc,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
  weights=rr[0].cpu().numpy();na=nal[0,:,:,0].cpu().numpy();valid=z['valid'][y0:y1,x0:x1];mask=valid&(na>.2)&(weights[:,:,5]<.1*na);yx=np.argwhere(mask);assert len(yx)>=16;yx=yx[np.linspace(0,len(yx)-1,16).round().astype(int)]
  m=meta['means2d'][0].cpu().numpy();conic=meta['conics'][0].cpu().numpy();op=meta['opacities'][0].cpu().numpy();depth=meta['depths'][0].cpu().numpy();flat=meta['flatten_ids'].cpu().numpy();offset=meta['isect_offsets'].cpu().numpy().ravel();tw=meta['tile_width'];tile=meta['tile_size'];traces=[];errs=[];tally=np.zeros(6);jtally={};topids=[]
  for y,x in yx:
   ti=(y//tile)*tw+x//tile;start=int(offset[ti]);end=int(offset[ti+1]) if ti+1<len(offset) else len(flat);ids=flat[start:end];d=m[ids]-[x+.5,y+.5];co=conic[ids];sig=.5*(co[:,0]*d[:,0]**2+co[:,2]*d[:,1]**2)+co[:,1]*d[:,0]*d[:,1];a=np.minimum(.999,op[ids]*np.exp(-np.maximum(sig,0)));a[(sig<0)|(a<1/255)]=0
   T=1.;contributions=[];calc=np.zeros(6)
   for j,aa in zip(ids,a):
    if aa==0:continue
    nextT=T*(1-aa)
    if nextT<=1e-4:break
    w=float(T*aa);calc[group[j]]+=w;contributions.append((int(j),w,float(depth[j])));T=nextT
   err=float(np.max(abs(calc-weights[y,x])));errs.append(err);assert err<2e-5,(c,f,int(x),int(y),err)
   tally+=calc
   for j,w,dep in contributions:
    if j<n and not act[j]:
     key=f'person{person[j]}_dominant_joint{joint[j]}';jtally[key]=jtally.get(key,0)+w
   top=sorted(contributions,key=lambda a:a[1],reverse=True)[:8];topids.extend(j for j,w,dep in top if j<n and not act[j])
   traces.append(dict(pixel_source_xy=[int(x+x0),int(y+y0)],weights={name:float(calc[k]) for k,name in enumerate(names)},CUDA_max_difference=err,top_contributors=[dict(id=j,visible_weight=w,camera_depth=dep,group=names[group[j]],anchor_person=int(person[j]) if j<n else None,anchor_dominant_joint=int(joint[j]) if j<n else None) for j,w,dep in top]))
  rec=dict(camera=c,frame=f,selected_pixels=16,group_mean_weights={name:float(tally[k]/16) for k,name in enumerate(names)},frozen_dominant_anchor_joint_weights_per_pixel={k:float(v/16) for k,v in sorted(jtally.items(),key=lambda a:-a[1])},max_CUDA_error=max(errs),unique_top_frozen_Gaussians=len(set(topids)),traces=traces)
  records.append(rec);print('RAY_OCCLUDERS',json.dumps({k:v for k,v in rec.items() if k!='traces'}),flush=True)
assert sha(path)==before
save(O/'summary.json',dict(status='48native_training_pixels_exact_tile_compositing_verified',records=records,source_checkpoint_unchanged=True,group_labels_are_heuristic_anchors_not_semantics=True,scope='Selected model-occluded pixels only. These rays identify existing rasterized contributions, not which geometry is anatomically correct. No source labels or checkpoint changed.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name)
