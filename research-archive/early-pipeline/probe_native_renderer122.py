"""One bounded known-geometry CUDA probe. No person, optimizer or old model loaded."""
from pathlib import Path
import os, sys, json, hashlib, time, traceback, math, argparse
from datetime import datetime, timezone
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
parser=argparse.ArgumentParser();parser.add_argument('--attempt',choices=['cuda-path-fixed','projection-observability-fix']);args=parser.parse_args()
if args.attempt:
 O=O/args.attempt;O.mkdir(exist_ok=True)
repo=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
sys.path.insert(0,str(repo))
os.environ.setdefault('MAX_JOBS','2');os.environ.setdefault('TORCH_CUDA_ARCH_LIST','12.0')
def save(name,data):(O/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if (O/'native-renderer-result.json').exists():raise RuntimeError('This bounded probe already finished; inspect its evidence, do not repeat')
if (O/'native-renderer-progress.json').exists():raise RuntimeError('Probe already registered; verify PID before considering recovery')
t0=time.monotonic()
save('native-renderer-plan.json',{'scope':'Fresh synthetic points; asymmetric full K; two known poses; pinhole and explicit OpenCV/KB four-D fisheye; forward/backward and parameter roundtrip','pixels':[192,144],'maxProjectionErrorPx':.05,'variants':3,'no_person_loaded':True,'no_training':True,'no_optimizer':True,'no_camera_estimation':True,'no_parameter_sweep':True,'no_new_weights':True,'repair':'Prior fixture read undefined means2d for opacity-culled subpixel Gaussians. Test the geometric UT with opacity culling disabled; render using the already registered anisotropic gradient fixture. Keep error threshold and backend unchanged.','note':'Synthetic implementation accuracy is not evidence of <1px physical face accuracy'})
save('native-renderer-progress.json',{'pid':os.getpid(),'phase':'import','started':datetime.now(timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
result={'passed':False,'person_training_started':False,'fixtures':[]}
try:
 import torch,gsplat,numpy as np
 from PIL import Image
 from native_portrait_renderer122 import NativeCameraContract,render_native
 from gsplat.cuda._wrapper import fully_fused_projection_with_ut
 torch.manual_seed(122);device='cuda'
 result.update(torch=torch.__version__,gsplat=gsplat.__version__,gsplat_path=gsplat.__file__,gpu=torch.cuda.get_device_name(),source_sha256=hashlib.sha256((repo/'gsplat/rendering.py').read_bytes()).hexdigest())
 T=lambda a:torch.tensor(a,dtype=torch.float32,device=device)
 W,H=192,144
 K=T([[[103.1,0,91.25],[0,98.7,77.75],[0,0,1]]]*2)
 V=torch.eye(4,device=device).repeat(2,1,1);a=.12;V[1,:3,:3]=T([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]]);V[1,:3,3]=T([.06,-.02,.1])
 m=T([[0,0,3],[.7,.4,3],[-1.35,.55,2.8],[1.5,-.7,2.7]])
 q=T([[1,0,0,0]]*len(m));small=T([[.001,.001,.001]]*len(m));opacity=T([.75]*len(m));colors=T([[1,.2,.1],[.1,1,.2],[.2,.1,1],[1,1,.1]])
 D=T([[.015,-.001,0,0]]*2)
 def analytical(model):
  x=torch.einsum('cij,nj->cni',V[:,:3,:3],m)+V[:,:3,3,None].transpose(1,2)
  xy=x[...,:2];z=x[...,2:];n=xy/z
  if model=='opencv_fisheye4':
   radius=torch.linalg.vector_norm(xy,dim=-1,keepdim=True);theta=torch.atan2(radius,z)
   theta_d=theta*(1+D[:,None,0:1]*theta**2+D[:,None,1:2]*theta**4+D[:,None,2:3]*theta**6+D[:,None,3:4]*theta**8)
   n=torch.where(radius>1e-9,xy*theta_d/torch.clamp(radius,min=1e-9),torch.zeros_like(xy))
  return n*torch.stack([K[:,0,0],K[:,1,1]],dim=-1)[:,None]+torch.stack([K[:,0,2],K[:,1,2]],dim=-1)[:,None]
 for model in ['pinhole','opencv_fisheye4']:
  save('native-renderer-progress.json',{'pid':os.getpid(),'phase':'forward_'+model,'elapsed':time.monotonic()-t0})
  contract=NativeCameraContract(model,'synthetic_known_camera')
  projection=fully_fused_projection_with_ut(m,q,small,None,V,K,W,H,camera_model='pinhole' if model=='pinhole' else 'fisheye',radial_coeffs=D if model!='pinhole' else None)
  visible=bool((projection[0]>0).all());expected=analytical(model)
  err=torch.linalg.vector_norm(projection[1]-expected,dim=-1) if visible else None
  rgb,alpha,meta=render_native(m,q,T([[.03,.018,.01]]*len(m)),opacity,colors,V,K,W,H,contract=contract,radial_coeffs=D if model!='pinhole' else None)
  torch.cuda.synchronize()
  entry={'model':model,'maxProjectionErrorPx':float(err.max()) if visible else None,'points_visible':visible,'projectionRadii':projection[0].cpu().tolist(),'expectedCenters':expected.cpu().tolist(),'actualCenters':projection[1].cpu().tolist() if visible else None,'finite':bool(torch.isfinite(rgb).all() and torch.isfinite(alpha).all()),'nonzeroAlpha':bool(alpha.max()>0),'renderPointsVisible':bool((meta['radii']>0).all()),'contract':contract.manifest()}
  entry['passed']=visible and entry['maxProjectionErrorPx']<.05 and entry['finite'] and entry['nonzeroAlpha'] and entry['renderPointsVisible'];result['fixtures'].append(entry)
  for ci in range(2):Image.fromarray((rgb[ci].detach().clamp(0,1).cpu().numpy()*255).astype('uint8')).save(O/(model+'_view'+str(ci)+'.png'))
 # Same native renderer after parameter serialization; no cross-backend equivalence assumption.
 save('native-renderer-progress.json',{'pid':os.getpid(),'phase':'gradients_and_roundtrip','elapsed':time.monotonic()-t0})
 means=m.clone().requires_grad_();scales=T([[.03,.018,.01]]*len(m)).requires_grad_();quats=q.clone().requires_grad_()
 contract=NativeCameraContract('pinhole','synthetic_known_camera')
 out=render_native(means,quats,scales,opacity,colors,V,K,W,H,contract=contract)
 yy,xx=torch.meshgrid(torch.arange(H,device=device),torch.arange(W,device=device),indexing='ij')
 loss=(out[0]*(1+xx[None,:,:,None]/W+yy[None,:,:,None]/H)).sum();loss.backward()
 grads={'means':means.grad,'scales':scales.grad,'quats':quats.grad}
 finite=all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values());nonzero=all(float(grads[k].abs().max())>0 for k in ['means','scales'])
 snapshot={'means':m.cpu().numpy(),'quats':q.cpu().numpy(),'scales':scales.detach().cpu().numpy(),'opacities':opacity.cpu().numpy(),'colors':colors.cpu().numpy(),'viewmats':V.cpu().numpy(),'Ks':K.cpu().numpy()}
 np.savez(O/'native-fixture.npz',**snapshot);back=np.load(O/'native-fixture.npz');reloaded={k:T(back[k]) for k in snapshot}
 readback=render_native(**reloaded,width=W,height=H,contract=contract)
 diff=float((out[0].detach()-readback[0]).abs().max());result['fixtures'].append({'model':'pinhole-gradient-roundtrip','finite_gradients':finite,'nonzero_geometry_gradients':nonzero,'nativeImageMaxAbsDiff':diff,'passed':finite and nonzero and diff<1e-6})
 for ci in range(2):Image.fromarray((out[0][ci].detach().clamp(0,1).cpu().numpy()*255).astype('uint8')).save(O/('anisotropic_pinhole_view'+str(ci)+'.png'))
 # Unsupported domains must fail before rendering, never default to a plausible image.
 rejected=0
 for bad in [NativeCameraContract('FISHEYE_8','native_fisheye'),NativeCameraContract('opencv_fisheye4','rectified'),NativeCameraContract('pinhole','rectified','integer_centers')]:
  try:bad.validate()
  except ValueError:rejected+=1
 result['unsupported_contracts_rejected']=rejected;result['passed']=all(f['passed'] for f in result['fixtures']) and rejected==3
except Exception as e:
 result.update(error=type(e).__name__,message=str(e),traceback=traceback.format_exc())
finally:
 result.update(seconds=time.monotonic()-t0,finished=datetime.now(timezone.utc).isoformat(),adapter_sha256=hashlib.sha256(Path(__file__).with_name('native_portrait_renderer122.py').read_bytes()).hexdigest())
 save('native-renderer-result.json',result);save('native-renderer-progress.json',{'pid':os.getpid(),'phase':'completed' if result['passed'] else 'failed','result':'native-renderer-result.json','passed':result['passed']})
 print(json.dumps(result,ensure_ascii=False,indent=2))
