"""Read-only counterfactuals: separate intrinsic opacity from scene occlusion."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from gsplat.rendering import rasterization
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
from enerf_mano_gaussian_surface import ManoGaussianSurface,render_hand_surface
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';H=L/'outputs/dynamic-human-hand-consensus-097';O=L/'outputs/dynamic-human-enerf-contour-surface-103';A=O/'visibility';A.mkdir(exist_ok=True);assert not (A/'summary.json').exists()
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
plan=read(O/'contribution/summary.json')['records'];cases=[r for r in plan if (r['camera'],r['frame']) in [('04',731),('04',732),('13',731),('03',732)]];assert len(cases)==4;torch.set_num_threads(6);active=torch.from_numpy(np.load(L/'outputs/dynamic-human-enerf-hand-joint-084/support/active-gaussians.npy')).cuda();rows=[];font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',13)
with torch.no_grad():
 for branch in ['prior_control','contour_prior']:
  path=O/branch/'resume.pt';before=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);g.requires_grad_(False);old=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,g,active);hand=ManoGaussianSurface(L/read(O/'protocol.json')['priors'][branch]['path'],S/'heldout/capture-affines-estimated.npz',cp['scene_scale'],read(O/'protocol.json')['initial_color'],False);hand.load_state_dict(cp['hand_surface'])
  for r in cases:
   c,f=r['camera'],r['frame'];stem=f'camera{c}-frame{f:06d}';z=np.load(S/'native-input'/f'{stem}.npz');W=tensor(z['w2c']);K=tensor(z['K']);x0,y0,x1,y1=r['bbox'];sl=np.s_[y0:y1,x0:x1];hm,hc,ha,hcolor,_=hand.geometry(c,f)
   newrgb,newalpha,_=rasterization(hm,None,None,ha,hcolor,W[None],K[None],1920,1080,covars=hc,packed=False,near_plane=.1,far_plane=20,eps2d=.3);oldrgb,oldalpha,_=render_hand_surface(p,old,hand,c,f,W,K,1920,1080,disable_new=True);nr=newrgb[0].cpu().numpy()[sl];na=newalpha[0,:,:,0].cpu().numpy()[sl];orr=oldrgb[0].cpu().numpy()[sl];oa=oldalpha[0,:,:,0].cpu().numpy()[sl]
   full=np.load(O/branch/'native-evaluation'/f'{stem}-prediction.npz');fr=full['rgb'][sl];fa=full['alpha'][sl];fw=full['surface_weight'][sl];assert (fw<=na+2e-5).all();valid=z['valid'][sl];m=valid&(na>.05);fullpred=np.clip(fr+.5*(1-fa[...,None]),0,1);oldpred=np.clip(orr+.5*(1-oa[...,None]),0,1);newpred=np.clip(nr+.5*(1-na[...,None]),0,1);target=z['rgb'][sl].astype(np.float32)/255*z['matte'][sl][...,None]+.5*(1-z['matte'][sl][...,None]);fg=valid&(z['matte'][sl]>.5)
   metrics=dict(support_pixels_new_alpha_gt_005=int(m.sum()),new_only_alpha_mean_on_support=float(na[m].mean()),visible_new_weight_mean_on_support=float(fw[m].mean()),visible_fraction_of_new_only_weight=float(fw[valid].sum()/max(float(na[valid].sum()),1e-9)),context_RGB_MAE_full_vs_trained_old_only=float(abs(fullpred-oldpred)[valid].mean()),context_foreground_psnr_full=float(-10*np.log10(np.mean((fullpred.astype(float)[fg]-target.astype(float)[fg])**2))),context_foreground_psnr_trained_old_only=float(-10*np.log10(np.mean((oldpred.astype(float)[fg]-target.astype(float)[fg])**2))))
   arrays=[z['rgb'][sl],u8(fullpred),u8(oldpred),u8(newpred),np.repeat(u8(na)[...,None],3,-1),np.repeat(u8(fw)[...,None],3,-1)];h,w=na.shape;im=Image.new('RGB',(w*6,h+50),'white');d=ImageDraw.Draw(im);d.text((4,3),f'{branch} | camera{c} frame{f}',font=font,fill='black')
   for i,(a,label) in enumerate(zip(arrays,['Original','Joint RGB','Trained old only','New only RGB','New only alpha','Visible new weight'])):im.paste(Image.fromarray(a),(i*w,50));d.text((i*w+4,27),label,font=font,fill='black')
   out=A/f'{branch}-{stem}.png';im.save(out);np.savez_compressed(A/f'{branch}-{stem}.npz',new_only_rgb=nr,new_only_alpha=na,trained_old_rgb=orr,trained_old_alpha=oa,visible_new_weight=fw);rows.append(dict(branch=branch,camera=c,frame=f,metrics=metrics,file=out.name,sha256=sha(out)))
  assert sha(path)==before;del p,g,old,hand,cp;torch.cuda.empty_cache()
save(A/'summary.json',dict(status='8read_only_visibility_counterfactuals_complete',records=rows,scope='Same trained parameters; remove one component only for attribution. New-only image is not a reconstruction accepted for the full scene. New-only alpha support is model-defined, not source hand GT.',goal_complete=False));shutil.copy2(__file__,A/Path(__file__).name);print(json.dumps(rows,indent=2),flush=True)
