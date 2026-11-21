from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,torch
from PIL import Image,ImageDraw
from enerf_articulated_model import ArticulatedRenderer
LAB=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064';MOTION=LAB/'outputs/dynamic-human-enerf-continuous-body-063'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8)
ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=['fixed_offsets','estimated_offsets','direct_reprojection','head_only']);ap.add_argument('--result-root',type=Path,default=OUT);ap.add_argument('--motion-file',type=Path);a=ap.parse_args();ROOT=a.result_root/a.branch;motion_file=a.motion_file or MOTION/a.branch/'capture-affines.npz';E=ROOT/'native-evaluation';E.mkdir(exist_ok=True);assert not (E/'metrics.json').exists();assert read(ROOT/'completion.json')['step']==15000
torch.set_num_threads(8);cp=torch.load(ROOT/'resume.pt',map_location='cpu',weights_only=False);assert cp['step']==15000 and all(torch.isfinite(v).all() for v in cp['params'].values());p={k:v.cuda() for k,v in cp['params'].items()};steps={k:[int(st['step']) for st in o['state'].values() if 'step' in st] for k,o in cp['optim'].items()};assert all(v==[15000] for v in steps.values());del cp
save(ROOT/'training-state-audit.json',dict(status='15000updates_finite_checkpoint_optimizer_steps_and_discrete_bindings_verified',step=15000,points=len(p['means']),checkpoint_sha256=sha(ROOT/'resume.pt'),optimizer_steps=steps,anchor_min=int(p['anchor'].min()),anchor_max=int(p['anchor'].max()),goal_complete=False));assert p['anchor'].dtype==torch.long and p['anchor'].min()>=0 and p['anchor'].max()<13780
assert sha(motion_file)==read(ROOT/'protocol.json')['motion_sha256'];model=ArticulatedRenderer(motion_file);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda');records=[r for r in read(OUT/'input-audit.json')['records'] if r['split']=='validation_time' or (r['camera'] in ['01','04','08','17'] and r['frame'] in [744,750,756])];assert len(records)==102
save(E/'protocol.json',dict(status='fixed_before_evaluation',selection='All90odd-timevalidationimages plus12even-time diagnosticimages; 15knowntrainingcameras, no00/06/12.',images=102,resolution=[1920,1080],motion_sha256=sha(motion_file),appearance='RenderedonlyfromtrainedGaussiansandselectedfrozenmotion; noimage-basedblend,upscale,sharpening,retimedsourcepixels or imageinterpolation.',foreground_reference='Original062RGB withoriginalreleasedmatte, samefloat32undistort convention. Matte isnotsemanticGT.',goal_complete=False))
rows=[]
with torch.no_grad():
 for row in records:
  c,t=row['camera'],row['frame'];f=OUT/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(f)==row['input_sha256'];z=np.load(f);r,al,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=r[0].cpu().numpy();alpha=al[0,...,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all();pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None])
  masks={'full_valid':z['valid'],'foreground':z['valid']&(z['matte']>.5)};x0,y0,x1,y1=z['bbox'];mask=np.zeros_like(z['valid']);mask[y0:y1,x0:x1]=True;masks['foreground_bbox']=mask&z['valid'];metrics={}
  for name,mask in masks.items():
   d=pred.astype(np.float64)[mask]-ref.astype(np.float64)[mask];metrics[name]=dict(pixels=int(mask.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
  name=f'camera{c}-frame{t:06d}';np.savez_compressed(E/f'{name}-prediction.npz',rgb=rgb,alpha=alpha)
  for label,im in [('source',z['rgb']),('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(alpha))]:Image.fromarray(im).save(E/f'{name}-{label}.png')
  w=int(x1-x0);h=int(y1-y0);panel=Image.new('RGB',(2*w,h+32),'white');d=ImageDraw.Draw(panel)
  for i,(label,im) in enumerate([('Original native photograph',z['rgb']),('Articulated Gaussian reconstruction',u8(pred))]):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));d.text((i*w+5,10),label,fill='black')
  panel.save(E/f'{name}-person-comparison.png');rows.append(dict(camera=c,frame=t,split=row['split'],input_sha256=row['input_sha256'],bbox=row['bbox'],metrics=metrics));print('ARTICULATED_EVAL',a.branch,c,t,flush=True)
save(E/'metrics.json',dict(status='102nativeimages_complete_independentpixel_and_visualreview_pending',records=rows,goal_complete=False));shutil.copy2(__file__,ROOT/'source-snapshot'/Path(__file__).name)
