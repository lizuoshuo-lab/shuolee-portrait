"""Evaluate072on the same18new-camera odd frames using frozen068clocks."""
from pathlib import Path
import json,hashlib,shutil,numpy as np,torch
from PIL import Image
from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';I=L/'outputs/dynamic-human-enerf-heldout-cameras-068';E=R/'heldout/native-evaluation';E.mkdir(exist_ok=True,parents=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();u8=lambda x:np.rint(np.clip(x,0,1)*255).astype(np.uint8);tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
assert not (E/'metrics.json').exists();checkpoint=R/'residual_graph/resume.pt';before=sha(checkpoint);cp=torch.load(checkpoint,map_location='cpu',weights_only=False);assert cp['step']==15000
p={k:v.cuda() for k,v in cp['params'].items()};graph=GraphMotion(R/'graph.npz',cp['scene_scale']);graph.load_state_dict(cp['graph']);model=ResidualGraphRenderer(I/'capture-affines-estimated.npz',graph);torch.set_num_threads(8)
assert sha(I/'capture-affines-estimated.npz')==read(I/'motion-extension-audit.json')['branches']['estimated']['sha256']
assert cp['motion_sha256']==read(I/'motion-extension-audit.json')['source_affines_sha256'];records=[r for r in read(I/'input-audit.json')['records'] if r['frame']%2];assert len(records)==18
save(E/'protocol.json',dict(status='fixed_before_rendering',checkpoint_sha256=before,images=18,clock_estimates='Frozen068same21even-image estimates, no refitting for072',model_updates=0,goal_complete=False));rows=[]
with torch.no_grad():
 for r in records:
  c,t=r['camera'],r['frame'];name=f'camera{c}-frame{t:06d}';file=I/'native-input'/f'{name}.npz';assert sha(file)==r['input_sha256'];z=np.load(file);rgb,alpha,_=model.render(p,c,t,tensor(z['w2c']),tensor(z['K']),1920,1080,2);rgb=rgb[0].cpu().numpy();alpha=alpha[0,...,0].cpu().numpy();assert np.isfinite(rgb).all() and np.isfinite(alpha).all();np.savez_compressed(E/f'{name}-prediction.npz',rgb=rgb,alpha=alpha)
  pred=np.clip(rgb+.5*(1-alpha[...,None]),0,1);ref=z['rgb'].astype(np.float32)/255*z['matte'][...,None]+.5*(1-z['matte'][...,None]);m=z['valid']&(z['matte']>.5);delta=pred[m].astype(np.float64)-ref[m].astype(np.float64);psnr=float(-10*np.log10(np.mean(delta*delta)))
  Image.fromarray(u8(pred)).save(E/f'{name}-trained.png');rows.append(dict(camera=c,frame=t,foreground_psnr_db=psnr,input_sha256=r['input_sha256']));print('ACTOR_HELDOUT',c,t,psnr,flush=True)
assert sha(checkpoint)==before
save(E/'metrics.json',dict(status='18native_newcamera_images_frozen_model_and_sameclock_complete',records=rows,checkpoint_sha256=before,model_updates=0,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name)
