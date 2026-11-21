from pathlib import Path
import json,hashlib,shutil
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from gsplat import rasterization
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
OUT=LAB/'outputs/dynamic-human-enerf-surface-tracks-054';RENDER=OUT/'rendered';RENDER.mkdir(exist_ok=True)
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert not (OUT/'depth-comparison.json').exists()
z=np.load(OUT/'tracks.npz');checkpoint=BASE/'first-frame/resume.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert cp['step']==6000;p={k:v.cuda() for k,v in cp['params'].items()};del cp
torch.set_num_threads(8);tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
rows=[];images={};cams={};rendered={}
with torch.no_grad():
 for c in sorted(set(z['camera'])):
  o=np.load(BASE/f'train-camera-{c}.npz');cams[c]={k:o[k] for k in ['K','w2c']};images[c]=o['rgb']
  r,a,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),
    tensor(o['w2c'])[None],tensor(o['K'])[None],1920,1080,sh_degree=2,render_mode='RGB+ED',packed=False,near_plane=.1,far_plane=20,eps2d=.3)
  rgb=r[0,:,:,:3].cpu().numpy();depth=r[0,:,:,3].cpu().numpy();alpha=a[0,:,:,0].cpu().numpy()
  assert np.isfinite(depth).all();np.savez_compressed(RENDER/f'camera{c}-depth.npz',depth=depth,alpha=alpha)
  rendered[c]=np.rint(np.clip(rgb+.5*(1-alpha[...,None]),0,1)*255).astype(np.uint8)
  Image.fromarray(rendered[c]).save(RENDER/f'camera{c}-rgb.png')
  ids=np.flatnonzero(z['camera']==c);xy=z['xy'][ids].astype(np.float32);mx=xy[:,0,None];my=xy[:,1,None]
  samples=cv2.remap(depth,mx,my,cv2.INTER_LINEAR).ravel();al=cv2.remap(alpha,mx,my,cv2.INTER_LINEAR).ravel()
  gt=(z['points'][z['track_id'][ids]]@o['w2c'][:3,:3].T+o['w2c'][:3,3])[:,2]
  rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(o['K']).T
  X=(rays*samples[:,None]-o['w2c'][:3,3])@o['w2c'][:3,:3]
  for j,ind in enumerate(ids):
   rows.append(dict(observation=int(ind),track=int(z['track_id'][ind]),camera=str(c),xy=z['xy'][ind].tolist(),alpha=float(al[j]),
     expected_depth=float(samples[j]),triangulated_depth=float(gt[j]),abs_relative_depth_error=float(abs(samples[j]-gt[j])/gt[j]),
     world_from_render=X[j].tolist(),reference_reprojection_px=float(z['reprojection'][ind])))
  print('DEPTH',c,len(ids),flush=True)
for row in rows:
 other=np.flatnonzero((z['track_id']==row['track'])&(z['camera']!=row['camera']));errs=[]
 for ind in other:
  c=str(z['camera'][ind]);cam=cams[c];q=cam['K']@(cam['w2c'][:3,:3]@row['world_from_render']+cam['w2c'][:3,3])
  errs.append(float(np.linalg.norm(q[:2]/q[2]-z['xy'][ind])))
 row['cross_view_median_reprojection_px']=float(np.median(errs))
valid=[r for r in rows if r['alpha']>.9]
trackrows=[]
for i in range(len(z['points'])):
 rr=[r for r in valid if r['track']==i]
 if len(rr)>=3:trackrows.append(dict(track=i,observations=len(rr),median_relative_depth_error=float(np.median([r['abs_relative_depth_error'] for r in rr])),
   median_model_cross_view_px=float(np.median([r['cross_view_median_reprojection_px'] for r in rr]))))
ranked=sorted(trackrows,key=lambda r:r['median_model_cross_view_px']);selected=ranked[:6]+ranked[-6:];patchdir=OUT/'identity-review';patchdir.mkdir(exist_ok=True)
def patch(im,xy):
 x,y=np.rint(xy).astype(int);a,b=max(0,x-60),max(0,y-60);c,d=min(1920,x+60),min(1080,y+60)
 pic=np.full((120,120,3),235,np.uint8);pic[b-y+60:d-y+60,a-x+60:c-x+60]=im[b:d,a:c];return pic
for row in selected:
 rr=sorted([r for r in valid if r['track']==row['track']],key=lambda r:r['camera']);idx=np.rint(np.linspace(0,len(rr)-1,3)).astype(int);chosen=[rr[j] for j in idx]
 panel=Image.new('RGB',(750,157),'white');d=ImageDraw.Draw(panel)
 for j,r in enumerate(chosen):
  c=r['camera'];d.text((j*250+3,3),f"cam{c} depth {100*r['abs_relative_depth_error']:.2f}%",fill='black')
  for k,(label,im) in enumerate([('Photo',images[c]),('Render',rendered[c])]):
   d.text((j*250+k*124+3,18),label,fill='black');panel.paste(Image.fromarray(patch(im,r['xy'])),(j*250+k*124,37))
   x=j*250+k*124+60;y=97;d.line((x-3,y,x+3,y),fill='red');d.line((x,y-3,x,y+3),fill='red')
 fn=f"track{row['track']:03d}.png";panel.save(patchdir/fn);row['panel']=fn;row['shown_cameras']=[r['camera'] for r in chosen]
summary=dict(status='train_view_expected_depth_compared_to_multiview_texture_tracks_review_pending',
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),observations=len(rows),alpha_above09=len(valid),tracks_with3opaque_observations=len(trackrows),
 median_abs_relative_depth_error=float(np.median([r['abs_relative_depth_error'] for r in valid])),
 p90_abs_relative_depth_error=float(np.quantile([r['abs_relative_depth_error'] for r in valid],.9)),
 median_model_cross_view_reprojection_px=float(np.median([r['cross_view_median_reprojection_px'] for r in valid])),
 records=rows,track_records=trackrows,selected=selected,goal_complete=False)
save(OUT/'depth-comparison.json',summary);shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k not in ['records','track_records','selected']}))
