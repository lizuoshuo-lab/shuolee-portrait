"""Measure native composite contributions and group-removal counterfactuals, read-only."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import spherical_harmonics
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';A=L/'outputs/dynamic-human-enerf-hand-prior-control-098';P=L/'outputs/dynamic-human-enerf-mano-surface-099';O=L/'outputs/dynamic-human-enerf-hand-visibility-100';O.mkdir(exist_ok=True);D=O/'attribution';D.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda');u8=lambda a:np.rint(np.clip(a,0,1)*255).astype(np.uint8)
assert not (D/'summary.json').exists();torch.set_num_threads(6);path=P/'photo_pose/resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();z=np.load(A/'association.npz');assoc=torch.zeros_like(active);assoc[z['selected_parent_ids']]=True;assert int(assoc.sum())==1937;g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active);hand=ManoGaussianSurface(H/'all_views/parameters.npz',S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(P/'protocol.json')['initial_color'],True);hand.load_state_dict(cp['hand_surface']);hand.requires_grad_(False);plan=[r for r in read(P/'contribution/summary.json')['records'] if (r['camera'],r['frame']) in [('04',731),('04',732),('13',731),('03',732)]];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);rows=[]
with torch.no_grad():
 for r in plan:
  c,f=r['camera'],r['frame'];stem=f'camera{c}-frame{f:06d}';z=np.load(S/'native-input'/f'{stem}.npz');x0,y0,x1,y1=r['bbox'];K=z['K'].copy();K[0,2]-=x0;K[1,2]-=y0;K=tensor(K);W=tensor(z['w2c']);raw=z['rgb'][y0:y1,x0:x1];valid=z['valid'][y0:y1,x0:x1];means,cov,linear,meta=old.geometry(p,c,f);center=-W[:3,:3].T@W[:3,3];dirs=(old.base.inverse[meta[0],p['anchor']]@torch.linalg.solve(linear,(means-center)[...,None]))[...,0];dirs=torch.nn.functional.normalize(dirs,dim=-1);colors=(spherical_harmonics(2,dirs,torch.cat([p['sh0'],p['shN']],1))+.5).clamp_min(0);hm,hc,ha,hcolor,_=hand.geometry(c,f);n=len(means);means=torch.cat([means,hm]);cov=torch.cat([cov,hc]);alpha=torch.cat([p['opacities'].sigmoid(),ha]);color=torch.cat([colors,hcolor]);tags=torch.zeros((n+len(hm),4),device='cuda');tags[:n,0]=assoc;tags[:n,1]=active&~assoc;tags[:n,2]=~active;tags[n:,3]=1
  newonly=rasterization(hm,None,None,ha,hcolor,W[None],K[None],160,160,covars=hc,packed=False,near_plane=.1,far_plane=20,eps2d=.3);na=newonly[1][0,:,:,0].cpu().numpy();support=valid&(na>.05);results={};arrays=[raw]
  for mode in ['joint','mute_associated','mute_context']:
   aa=alpha.clone()
   if mode!='joint':aa[:n][assoc if mode=='mute_associated' else active]=0
   rr,al,_=rasterization(means,None,None,aa,torch.cat([color,tags],-1),W[None],K[None],160,160,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3);rgb=rr[0,:,:,:3].cpu().numpy();ww=rr[0,:,:,3:].cpu().numpy();alphaout=al[0,:,:,0].cpu().numpy();assert np.max(abs(ww.sum(-1)-alphaout))<2e-5
   results[mode]=dict(new_visible_fraction=float(ww[:,:,3][valid].sum()/max(float(na[valid].sum()),1e-9)),group_mean_weights_on_model_support={name:float(ww[:,:,i][support].mean()) for i,name in enumerate(['associated','other_context','frozen','new'])});arrays.append(u8(rgb+.5*(1-alphaout[...,None])));np.savez_compressed(D/f'{stem}-{mode}.npz',rgb=rgb,alpha=alphaout,group_weight=ww,new_only_alpha=na,bbox=r['bbox'])
   if mode=='joint':jointw=ww
  arrays.extend([np.repeat(u8(jointw[:,:,i])[...,None],3,-1) for i in range(4)]);im=Image.new('RGB',(640,404),'white');d=ImageDraw.Draw(im)
  for i,(arr,label) in enumerate(zip(arrays,['Original','Joint RGB','Mute associated','Mute all context','Associated weight','Other context weight','Frozen weight','New hand weight'])):
   y=(i//4)*202;d.text(((i%4)*160+3,y+4),f'{label}',font=font,fill='black');d.text(((i%4)*160+3,y+22),f'{c}/{f}',font=font,fill='black');im.paste(Image.fromarray(arr),((i%4)*160,y+42))
  out=D/f'{stem}.png';im.save(out);rows.append(dict(camera=c,frame=f,bbox=r['bbox'],metrics=results,file=out.name,sha256=sha(out)));print('OCCLUDER_GROUPS',c,f,results,flush=True)
assert sha(path)==before
save(D/'summary.json',dict(status='four_native_contexts_readonly_group_attribution_complete',checkpoint_sha256=before,group_sizes=dict(associated=1937,other_context=4393,frozen=246654,new=1538),records=rows,scope='Temporary zero-opacity counterfactuals only; no source checkpoint changes. Groups are geometric/support associations, not semantic GT. Muting all context is an attribution bound and removes toy/arm content; not a proposed reconstruction.',goal_complete=False));shutil.copy2(__file__,D/Path(__file__).name)
