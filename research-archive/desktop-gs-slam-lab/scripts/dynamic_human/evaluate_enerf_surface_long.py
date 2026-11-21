"""Six native views of056, scored with exactly the frozen044 references."""
from pathlib import Path
import json,hashlib,torch,numpy as np,os
from PIL import Image,ImageDraw
from gsplat.rendering import rasterization_2dgs
def rasterization(*args,**kwargs):
 r,a,n,nd,d,m,info=rasterization_2dgs(*args,**kwargs,render_mode='RGB+ED')
 return r[...,:3],a,info
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
OUT=Path(os.environ['ENERF_RUN_DIR']);RUN=OUT/'first-frame'
BASE=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044/source_support_blending/native-evaluation'
EVAL=OUT/'native-evaluation';EVAL.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert read(RUN/'completion.json')['step']==30000 and not (EVAL/'metrics.json').exists()
torch.set_num_threads(8);checkpoint=RUN/'resume.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
assert cp['step']==30000 and all(torch.isfinite(v).all() for v in cp['params'].values())
p={k:v.cuda() for k,v in cp['params'].items()};del cp
save(OUT/'independent-state-audit.json',dict(status='fixed30000checkpoint_finite_parameters_and_input_membership_verified',
 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),step=30000,gaussian_count=len(p['means']),
 training_cameras=[r['camera'] for r in read(OUT/'input-audit.json')['records']],goal_complete=False))
save(EVAL/'protocol.json',dict(method='2DGaussian surface control, SH2,30000updates',resolution=[1920,1080],
 appearance_sources='No source image sampled at inference; rendered from optimized Gaussian and SH parameters.',
 comparison='Same six development cameras and unchanged044native reference/matte/region definitions; 056extended optimization schedule, compared to0556000update regularized endpoint; development cameras.',goal_complete=False))
records=[];tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda');u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
with torch.no_grad():
 for row in read(BASE/'metrics.json')['records']:
  name=row['camera'];prefix=f'camera{name}-frame000750';z=np.load(BASE/f'{prefix}-raw.npz')
  rgb,al,_=rasterization(p['means'],p['quats'],p['scales'].exp(),p['opacities'].sigmoid(),torch.cat([p['sh0'],p['shN']],1),
    tensor(z['w2c'])[None],tensor(z['K'])[None],1920,1080,sh_degree=2,packed=False,near_plane=.1,far_plane=20,eps2d=.3)
  rgb=rgb[0].cpu().numpy();alpha=al[0,...,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
  pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1)
  ref=z['source'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
  x0,y0,x1,y1=row['bbox'];ym=y0+round(.6*(y1-y0));masks={'full_valid':z['valid'],'foreground_matte_gt0_5':z['valid']&(z['matte']>.5)}
  for label,box in [('foreground_union_bbox',[x0,y0,x1,y1]),('upper_union',[x0,y0,x1,ym]),('lower_union',[x0,ym,x1,y1])]:
   m=np.zeros_like(z['valid']);m[box[1]:box[3],box[0]:box[2]]=True;masks[label]=m&z['valid']
  metrics={}
  for label,m in masks.items():
   diff=pred.astype(np.float64)[m]-ref.astype(np.float64)[m];metrics[label]=dict(pixels=int(m.sum()),psnr_db=float(-10*np.log10(np.mean(diff**2))),mae=float(abs(diff).mean()))
  for suffix,arr in [('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(alpha)),('source',z['source'])]:Image.fromarray(arr).save(EVAL/f'{prefix}-{suffix}.png')
  np.savez_compressed(EVAL/f'{prefix}-raw.npz',rgb=rgb,alpha=alpha,source=z['source'],matte=z['matte'],valid=z['valid'],K=z['K'],w2c=z['w2c'])
  for label,box in [('full-person',[x0,y0,x1,y1]),('upper',[x0,y0,x1,ym])]:
   a,b,c,d=box;w=c-a;h=d-b;panel=Image.new('RGB',(2*w,h+38),'white');draw=ImageDraw.Draw(panel)
   for i,(title,im) in enumerate([('Reference (released matte)',u8(ref)),('2DGS30000',u8(pred))]):
    panel.paste(Image.fromarray(im[b:d,a:c]),(i*w,38));draw.text((i*w+8,10),title,fill='black')
   panel.save(EVAL/f'{prefix}-{label}-comparison.png')
  # Match the established independent native-pixel auditor's schema.
  records.append(dict(camera=name,split=row['split'],source_cameras=[],source_rgb_sha256=row['source_rgb_sha256'],source_matte_sha256=row['source_matte_sha256'],bbox=row['bbox'],metrics=metrics))
  print('NATIVE_2DGS',json.dumps(records[-1]),flush=True)
save(EVAL/'metrics.json',dict(status='six_native_2dgs_renders_complete_review_pending',records=records,goal_complete=False))
