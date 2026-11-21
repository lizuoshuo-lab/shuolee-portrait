"""Frozen 118 alpha-supervision audit. No optimization and no parameter changes."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json
import numpy as np
from PIL import Image
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118/all-persons-20261027')
OUT=REC/'indoor118-negative-alpha';OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.cli import gpu_lock
rt.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets',ROOT)
rt.save(OUT/'plan.json',dict(created=rt.now(),hypothesis='Visible-person masks encode occlusion as zero alpha; the current all-image alpha loss may erase body surfaces behind furniture.',
 variable='Read-only fitted mesh zbuffer and mask votes; synthetic gradient check. All K/E, fits, Gaussians, masks unchanged.',
 limits='Outside-mask mesh pixels can also indicate pose or camera error. Region counting alone does not prove external occlusion or isolate the face ghost cause.',
 budget='89 fixed raw cameras at450px height, six saved images, no training.',fit_sha256=rt.sha(ROOT/'fit/fitted.npz'),training_source_sha256=rt.sha(PROJECT/'src/shuolee_portrait/training.py')))
with gpu_lock():
 import torch,smplx,nvdiffrast.torch as dr
 model=smplx.SMPLX(str(rt.c.MODEL),num_betas=10,num_expression_coeffs=10,use_pca=False,batch_size=1)
 fit=dict(np.load(ROOT/'fit/fitted.npz'));faces=fit['faces'];v=torch.as_tensor(fit['vertices_world'][0],dtype=torch.float32,device='cuda')
 skin=model.lbs_weights.numpy();legw=skin[:,[4,5,7,8,10,11]].sum(1)
 lower=(legw[faces].mean(1)>.5);faceids=torch.as_tensor(faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext()
 rows=rt.read(ROOT/'data/training-inputs.json')['records'];records=[];positive=np.zeros(len(faces),int);negative=np.zeros(len(faces),int)
 selected=set(np.linspace(0,len(rows)-1,6).astype(int))
 for i,r in enumerate(rows):
  H=450;W=round(r['wh'][0]*H/r['wh'][1]);K=torch.as_tensor(np.diag([W/r['wh'][0],H/r['wh'][1],1])@np.asarray(r['K']),dtype=torch.float32,device='cuda');E=torch.as_tensor(r['w2c'],dtype=torch.float32,device='cuda')
  cam=v@E[:3,:3].T+E[:3,3];z=cam[:,2]
  clip=torch.stack([2*K[0,0]/W*cam[:,0]+(2*K[0,2]/W-1)*z,-2*K[1,1]/H*cam[:,1]+(1-2*K[1,2]/H)*z,1.00002*z-.0200002,z],-1)
  rast,_=dr.rasterize(ctx,clip[None],faceids,(H,W));tid=rast[0,:,:,3].flip(0).cpu().numpy().astype(int)-1
  body=tid>=0;mask=np.asarray(Image.open(r['mask']).convert('L').resize((W,H),Image.Resampling.NEAREST))>127
  region=body & lower[np.maximum(tid,0)];outside=body & ~mask
  positive[np.unique(tid[body&mask])]+=1;negative[np.unique(tid[outside])]+=1
  records.append(dict(frame=r['frame'],visible_mesh_pixels=int(body.sum()),outside_mask_pixels=int(outside.sum()),
    lower_leg_pixels=int(region.sum()),lower_leg_zero_alpha_pixels=int((region&~mask).sum()),lower_leg_zero_alpha_fraction=float((region&~mask).sum()/max(region.sum(),1))))
  if i in selected:
   rgb=np.asarray(Image.open(r['image']).convert('RGB').resize((W,H),Image.Resampling.LANCZOS));overlay=rgb.copy().astype(float)
   overlay[region&mask]=.5*overlay[region&mask]+.5*np.array([20,210,60]);overlay[region&~mask]=.5*overlay[region&~mask]+.5*np.array([250,30,50])
   Image.fromarray(rgb).save(OUT/f'{r["frame"]}_原始画面.png')
   Image.fromarray(overlay.astype('uint8')).save(OUT/f'{r["frame"]}_腿部投影_红色为零mask监督区域.png')
   Image.fromarray(mask.astype('uint8')*255).save(OUT/f'{r["frame"]}_人物mask.png')
 # Isolated mechanism check; not a trained improvement.
 a=torch.tensor([.6,.6],requires_grad=True);target=torch.tensor([0.,1.]);((a-target).abs()*.1).sum().backward()
 oldgrad=a.grad.tolist();a.grad=None
 weight=torch.tensor([0.,1.]);((a-target).abs()*weight*.1).sum().backward();ignoregrad=a.grad.tolist()
 assert oldgrad[0]>0 and ignoregrad[0]==0
np.savez_compressed(OUT/'surface-mask-votes.npz',positive_views=positive,zero_mask_views=negative,lower_leg_faces=lower)
result=dict(created=rt.now(),state='measured_requires_raw_image_review',frames=len(rows),records=records,
 lower_leg_faces=int(lower.sum()),lower_leg_faces_positive_and_negative_at_least3=int((lower&(positive>=3)&(negative>=3)).sum()),
 lower_leg_faces_never_mask_supported=int((lower&(positive==0)&(negative>0)).sum()),
 alpha_gradient_control=dict(old_gradient=oldgrad,ignored_first_pixel_gradient=ignoregrad,meaning='Zero target drives predicted opacity down; ignore weight stops that contribution only.'),
 conclusion='Mechanism measured, but visible mesh outside mask is not automatically furniture occlusion. Actual images must separate wrong projection from physical occlusion.',
 camera_cause_proven=False,face_ghost_cause_proven=False,training_updates=0,quality_accepted=False)
rt.save(OUT/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
