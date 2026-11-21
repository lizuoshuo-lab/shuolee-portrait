"""Read-only54native renders: two frozen models and a clock-zero diagnostic."""
from pathlib import Path
import json,hashlib,shutil,gc
import numpy as np,torch
from PIL import Image,ImageDraw
from enerf_articulated_model import ArticulatedRenderer
from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-heldout-cameras-068';H=L/'outputs/dynamic-human-enerf-head-motion-066';G=L/'outputs/dynamic-human-enerf-residual-graph-067'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
torch.set_num_threads(8);protocol=read(R/'protocol.json');extension=read(R/'motion-extension-audit.json');records=[r for r in read(R/'input-audit.json')['records'] if r['frame']%2];assert len(records)==18
branches=[('head_estimated',H/'head_only','head','estimated'),('graph_estimated',G/'residual_graph','graph','estimated'),('graph_zero',G/'residual_graph','graph','zero')]
for name,source,kind,clock in branches:
 out=R/name;E=out/'native-evaluation';E.mkdir(parents=True,exist_ok=True);assert not (E/'metrics.json').exists();checkpoint=source/'resume.pt';assert sha(checkpoint)==protocol['checkpoints'][kind]
 cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==15000 and all(torch.isfinite(v).all() for v in cp['params'].values());p={k:v.cuda() for k,v in cp['params'].items()};motion=R/f'capture-affines-{clock}.npz';assert sha(motion)==extension['branches'][clock]['sha256']
 if kind=='graph':
  assert cp['motion_sha256']==extension['source_affines_sha256'];assert cp['graph_sha256']==sha(G/'graph.npz');graph=GraphMotion(G/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);graph.eval();model=ResidualGraphRenderer(motion,graph)
 else:model=ArticulatedRenderer(motion)
 del cp;rows=[];torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for r in records:
   c,t=r['camera'],r['frame'];prefix=f'camera{c}-frame{t:06d}';f=R/'native-input'/f'{prefix}.npz';assert sha(f)==r['input_sha256'];z=np.load(f);rgb,alpha,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=rgb[0].cpu().numpy();alpha=alpha[0,...,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all()
   pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);x0,y0,x1,y1=z['bbox'];bboxmask=np.zeros_like(z['valid']);bboxmask[y0:y1,x0:x1]=True;masks=dict(full_valid=z['valid'],foreground=z['valid']&(z['matte']>.5),foreground_bbox=z['valid']&bboxmask);metrics={}
   for label,m in masks.items():
    d=pred.astype(np.float64)[m]-ref.astype(np.float64)[m];metrics[label]=dict(pixels=int(m.sum()),psnr_db=float(-10*np.log10(np.mean(d*d))),mae=float(abs(d).mean()))
   np.savez_compressed(E/f'{prefix}-prediction.npz',rgb=rgb,alpha=alpha)
   for label,image in [('source',z['rgb']),('reference',u8(ref)),('trained',u8(pred)),('alpha',u8(alpha))]:Image.fromarray(image).save(E/f'{prefix}-{label}.png')
   w=int(x1-x0);h=int(y1-y0);panel=Image.new('RGB',(2*w,h+32),'white');d=ImageDraw.Draw(panel)
   for i,(label,im) in enumerate([('Original native photograph',z['rgb']),(name,u8(pred))]):panel.paste(Image.fromarray(im[y0:y1,x0:x1]),(i*w,32));d.text((i*w+5,10),label,fill='black')
   panel.save(E/f'{prefix}-person-comparison.png');rows.append({**r,'metrics':metrics});print('HELDOUT_RENDER',name,c,t,flush=True)
 save(E/'metrics.json',dict(status='18native_frozen_model_heldout_camera_renders_complete',records=rows,goal_complete=False));assert sha(checkpoint)==protocol['checkpoints'][kind]
 save(out/'frozen-model-audit.json',dict(status='readonly_frozen15000step_checkpoint_before_after_hash_identical',checkpoint_sha256=sha(checkpoint),motion_extension_sha256=sha(motion),old_body_contexts_preserved=195,appearance_updates=0,graph_updates=0,body_pose_updates=0,clock_hypothesis=clock,max_gpu_bytes=torch.cuda.max_memory_allocated(),goal_complete=False))
 del model,p
 if kind=='graph':del graph
 gc.collect();torch.cuda.empty_cache()
shutil.copy2(__file__,R/Path(__file__).name)
