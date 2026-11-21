"""Official torchvision RAFT-Large, unchanged native training photo pairs."""
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,torch
from torch.nn.functional import pad
from torchvision.models.optical_flow import raft_large,Raft_Large_Weights
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-open-hand-flow-083/raft-fields';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'completion.json').exists();weights=Raft_Large_Weights.DEFAULT
torch.set_num_threads(4);model=raft_large(weights=weights,progress=True).cuda().eval();transform=weights.transforms()
modelpath=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[-1];assert modelpath.exists()
rows=read(S/'input-audit.json')['records'];by={(r['camera'],r['frame']):r for r in rows};cams=read(S/'protocol.json')['training_cameras']
save(O/'protocol.json',dict(status='fixed_before_RAFT_inference',model=weights.name,url=weights.url,model_sha256=sha(modelpath),precision='float32',iterations=20,
    input='Native RGB, union of the two whole-person boxes plus64px context. Replicate padding only to multiples8; no resize, no enhancement.',normalization='Official torchvision weights.transforms(): uint8 tofloat32 and[-1,1].',training_only=True,cameras=cams,frames=[728,730,732],not_flow_ground_truth=True))
out=[];start=time.monotonic()
with torch.inference_mode():
 for c in cams:
  for a,b in [(728,730),(730,732)]:
   dest=O/f'camera{c}-frames{a}-{b}.npz'
   if dest.exists():raise RuntimeError('Existing partial output: inspect before restart')
   paths=[S/'native-input'/f'camera{c}-frame{t:06d}.npz' for t in [a,b]];zs=[]
   for t,p in zip([a,b],paths):assert sha(p)==by[c,t]['input_sha256'];zs.append(dict(np.load(p)))
   boxes=np.array([z['bbox'] for z in zs]);lo=np.maximum(0,boxes[:,:2].min(0)-64);hi=np.minimum([1920,1080],boxes[:,2:].max(0)+64);x0,y0=map(int,lo);x1,y1=map(int,hi);h,w=y1-y0,x1-x0
   images=[torch.from_numpy(z['rgb'][y0:y1,x0:x1].copy()).permute(2,0,1).unsqueeze(0) for z in zs]
   aa,bb=transform(*images);aa=pad(aa.cuda(),(0,(-w)%8,0,(-h)%8),mode='replicate');bb=pad(bb.cuda(),(0,(-w)%8,0,(-h)%8),mode='replicate')
   assert aa.dtype==torch.float32 and aa.min()>=-1 and aa.max()<=1
   f=model(aa,bb,num_flow_updates=20)[-1][0,:,:h,:w].permute(1,2,0).cpu().numpy();back=model(bb,aa,num_flow_updates=20)[-1][0,:,:h,:w].permute(1,2,0).cpu().numpy();assert np.isfinite(f).all() and np.isfinite(back).all()
   np.savez_compressed(dest,forward=f,backward=back,context_crop=[x0,y0,x1,y1]);out.append(dict(camera=c,source_frame=a,target_frame=b,source_sha256=by[c,a]['input_sha256'],target_sha256=by[c,b]['input_sha256'],sha256=sha(dest),filename=dest.name,context_crop=[x0,y0,x1,y1]));print('RAFT_NATIVE',c,a,b,flush=True)
save(O/'completion.json',dict(status='30nativepairs60directions_inferred',records=out,elapsed_seconds=time.monotonic()-start,max_gpu_bytes=torch.cuda.max_memory_allocated(),goal_complete=False));shutil.copy2(__file__,O/Path(__file__).name)
