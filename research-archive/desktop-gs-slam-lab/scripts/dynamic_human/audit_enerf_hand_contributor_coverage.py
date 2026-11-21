"""Actual compositing support, not center projection, in unchanged source hand masks."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from gsplat.rendering import rasterization
from enerf_second_temporal_graph import GraphMotion
from enerf_hand_joint_model import HandJointRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';J=L/'outputs/dynamic-human-enerf-hand-joint-084';F=L/'outputs/dynamic-human-enerf-open-hand-flow-083';O=L/'outputs/dynamic-human-enerf-hand-contributions-086';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
assert not (O/'summary.json').exists();torch.set_num_threads(8);path=S/'local_bspline/resume.pt';parent=sha(path);cp=torch.load(path,map_location='cpu',weights_only=False);p={k:v.cuda() for k,v in cp['params'].items()};active=torch.from_numpy(np.load(J/'support/active-gaussians.npy')).cuda();g=GraphMotion(S/'graph.npz',cp['scene_scale']);g.load_state_dict(cp['graph']);ref=GraphMotion(S/'graph.npz',cp['scene_scale']);ref.load_state_dict(cp['graph']);ref.requires_grad_(False);model=HandJointRenderer(S/'head/capture-affines.npz',g,ref,active);features=torch.stack([active.float(),(~active).float()],-1)
rows=[r for r in read(F/'hand-masks/completion.json')['records'] if r['camera'] not in ['14','16']];assert len(rows)==39 and all(r['frame'] in [728,730,732] for r in rows)
save(O/'protocol.json',dict(status='fixed_before_diagnostic',parent_sha256=parent,active_mask_sha256=sha(J/'support/active-gaussians.npy'),scope='39originaltrainingmasks,eroded1px; nooptimization ornewlabels. Selected6330andfrozen246654features rasterizedtogetherwithoriginalalphasorting. Selectedshare=selectedcontribution/totalalpha,includingocclusion.',goal_complete=False))
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);records=[];panels={};(O/'arrays').mkdir(exist_ok=True)
with torch.no_grad():
 for r in rows:
    c,t=r['camera'],r['frame'];ip=S/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(ip)==r['input_sha256'];z=np.load(ip);m=np.load(F/'hand-masks'/r['filename']);x0,y0,x1,y1=m['crop'];W=tensor(z['w2c']);K=tensor(z['K']);means,cov,_,_=model.geometry(p,c,t)
    value,alpha,_=rasterization(means,None,None,p['opacities'].sigmoid(),features,W[None],K[None],1920,1080,covars=cov,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
    rgb,alpha_rgb,_=model.render(p,c,t,W,K,1920,1080,2);identity=float((value.sum(-1,keepdim=True)-alpha).abs().max());alpha_match=float((alpha_rgb-alpha).abs().max());assert identity<2e-6 and alpha_match<2e-6
    pred=(rgb[0]+.5*(1-alpha_rgb[0]))[y0:y1,x0:x1].cpu().numpy();val=value[0,y0:y1,x0:x1].cpu().numpy();a=alpha[0,y0:y1,x0:x1,0].cpu().numpy();share=val[...,0]/np.maximum(a,1e-8);mask=cv2.erode(m['mask'].astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&z['valid'][y0:y1,x0:x1];raw=z['rgb'][y0:y1,x0:x1];matte=z['matte'][y0:y1,x0:x1];target=raw/255*matte[...,None]+.5*(1-matte[...,None]);err=abs(pred-target).mean(-1);assert mask.sum()>0
    masks=dict(original_hand=mask,alpha_supported=mask&(a>.5));stats={}
    for name,mm in masks.items():
        assert mm.sum()>0
        stats[name]=dict(pixels=int(mm.sum()),mean_active_share=float(share[mm].mean()),median_active_share=float(np.median(share[mm])),fraction_active_below_half=float((share[mm]<.5).mean()),fraction_active_below_quarter=float((share[mm]<.25).mean()),RGB_MAE=float(err[mm].mean()),error_mass_under_inactive_majority=float(err[mm&(share<.5)].sum()/max(err[mm].sum(),1e-12)))
    stem=f'camera{c}-frame{t:06d}';np.savez_compressed(O/'arrays'/f'{stem}.npz',source_rgb=raw,prediction=pred,contribution=val,alpha=a,active_share=share,hand_mask=mask,error=err,crop=m['crop']);records.append(dict(camera=c,frame=t,input_sha256=r['input_sha256'],mask_sha256=sha(F/'hand-masks'/r['filename']),contribution_identity_max_error=identity,RGB_alpha_max_error=alpha_match,statistics=stats))
    arrays=[raw,u8(pred),np.repeat(u8(share)[...,None],3,-1),np.repeat(mask[...,None].astype(np.uint8)*255,3,-1)];im=Image.new('RGB',(640,184),'white');d=ImageDraw.Draw(im)
    for k,(arr,label) in enumerate(zip(arrays,['Original','Parent render','Selected share 0-1','Estimated hand mask'])):im.paste(Image.fromarray(arr),(160*k,24));d.text((160*k+3,4),label,font=font,fill='black')
    panels[c,t]=im;print('HAND_CONTRIBUTION',c,t,stats['alpha_supported']['mean_active_share'],flush=True)
 for c in sorted({r['camera'] for r in rows}):
    sheet=Image.new('RGB',(640,616),'white');d=ImageDraw.Draw(sheet)
    for i,t in enumerate([728,730,732]):d.text((3,i*204+3),f'Camera{c} frame{t}',font=font,fill='black');sheet.paste(panels[c,t],(0,i*204+20))
    sheet.save(O/f'camera{c}-contribution.png')
assert sha(path)==parent
summary={name:{key:float(np.mean([r['statistics'][name][key] for r in records])) for key in ['mean_active_share','median_active_share','fraction_active_below_half','fraction_active_below_quarter','RGB_MAE','error_mass_under_inactive_majority']} for name in ['original_hand','alpha_supported']}
save(O/'summary.json',dict(status='39trainingcontexts_compositing_support_measured_visual_review_pending',parent_sha256=parent,active_Gaussians=6330,frozen_Gaussians=246654,records=records,unweighted_context_means=summary,qualification='Handmasks are estimated and mayinclude/exclude wrongsurface. Active share measuresoptimizationcoverage,notskinownership. Grayblack=excludedmaterial,white=selectedmaterial. No causalityclaim ornewmodel.',goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name);print(json.dumps(summary),flush=True)
