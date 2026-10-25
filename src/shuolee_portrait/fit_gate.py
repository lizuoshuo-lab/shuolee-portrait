import numpy as np
from PIL import Image
from . import runtime as rt

def evaluate(root,cfg):
 import torch,nvdiffrast.torch as dr
 fit=dict(np.load(root/'fit/fitted.npz'));obs=dict(np.load(root/'data/observations.npz'));weights=dict(np.load(root/'fit/observation-weights.npz'))
 h=obs['boxes'][:,3]-obs['boxes'][:,1];error=np.linalg.norm(fit['projected_coco17']-weights['body_target'],axis=-1)/h[:,None]
 reliable=weights['body']*weights['visibility']>.25
 if reliable.sum()<len(h)*4:raise RuntimeError('Insufficient reliable body landmarks for the fit gate.')
 median=float(np.median(error[reliable]));p90=float(np.quantile(error[reliable],.9))
 passed=median<=cfg['max_fit_median_body_fraction'] and p90<=cfg['max_fit_p90_body_fraction']
 result=dict(median_body_fraction=median,p90_body_fraction=p90,coarse_initialization_gate_passed=passed,quality_accepted=False,from_scratch=True,physical_accuracy_proven=False,thresholds='Diagnostic limits, not a calibrated quality guarantee',fit_sha256=rt.sha(root/'fit/fitted.npz'))
 rt.save(root/'fit/geometry-gate.json',result)
 rows=rt.read(root/'data/training-inputs.json')['records'];out=root/'reviews/fit';out.mkdir(parents=True)
 ctx=dr.RasterizeCudaContext();faces=torch.tensor(fit['faces'],device='cuda',dtype=torch.int32);v=torch.tensor(fit['vertices_world'][0],device='cuda');g=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
 with torch.no_grad():
  for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
   r=rows[i];H=768;W=round(r['wh'][0]*H/r['wh'][1]);K=g(np.diag([W/r['wh'][0],H/r['wh'][1],1])@np.array(r['K']));E=g(r['w2c']);cam=v@E[:3,:3].T+E[:3,3];z=cam[:,2]
   clip=torch.stack([2*K[0,0]/W*cam[:,0]+(2*K[0,2]/W-1)*z,-2*K[1,1]/H*cam[:,1]+(1-2*K[1,2]/H)*z,1.00002*z-.0200002,z],-1)
   rast,_=dr.rasterize(ctx,clip[None],faces,(H,W));mask=(rast[0,:,:,3]>0).flip(0).cpu().numpy();im=Image.open(r['image']).convert('RGB').resize((W,H),Image.Resampling.LANCZOS);rgb=np.array(im)
   overlay=rgb.astype(float);overlay[mask]=overlay[mask]*.6+np.array([15,190,210])*.4
   im.save(out/f'{r["frame"]}_input.png');Image.fromarray(overlay.astype('uint8')).save(out/f'{r["frame"]}_mesh_overlay.png')
 if not passed:raise RuntimeError('Fresh body fit failed coarse geometric gate. See fit/geometry-gate.json and reviews/fit. Gaussian training was not started.')
