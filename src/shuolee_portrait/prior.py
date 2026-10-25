"""GVHMR predicts this video's pose prior using this run's measured SfM matrices."""
import gc
import numpy as np
from PIL import Image
from . import runtime as rt

def infer(root,cfg):
 import torch,cv2
 import hmr4d
 hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
 from hmr4d.network.hmr2.configs import get_config
 from hmr4d.network.hmr2 import HMR2
 from hmr4d.network.hmr2.utils.preproc import IMAGE_MEAN,IMAGE_STD
 rows=rt.read(root/'data/inputs.json')['records'];out=root/'inference';out.mkdir()
 crops=[];boxes=[]
 for r in rows:
  rgb=np.array(Image.open(r['image']).convert('RGB'));b=np.array(r['initial_box_xyxy']);center=(b[:2]+b[2:])/2;size=max(b[2:]-b[:2])*1.2
  # Identical isotropic raster map for the square HMR crop and reverse landmark map.
  M=np.array([[256/size,0,127.5-center[0]*256/size],[0,256/size,127.5-center[1]*256/size]],np.float32)
  crops.append(cv2.warpAffine(rgb,M,(256,256),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT));boxes.append([*center,size])
 imgs=((torch.from_numpy(np.stack(crops)).float()/255-IMAGE_MEAN)/IMAGE_STD).permute(0,3,1,2);boxes=torch.tensor(boxes,dtype=torch.float32)
 def load(path):
  unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path)
  if unsafe:raise RuntimeError(f'Unreviewed checkpoint globals: {unsafe}')
  return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 config=get_config(str(rt.PROJECT/'third_party/gvhmr/hmr4d/network/hmr2/configs/model_config.yaml'))
 config.defrost();config.MODEL.BBOX_SHAPE=[192,256];config.SMPL.MEAN_PARAMS=str(rt.ASSETS/'gvhmr/hmr4d/network/hmr2/configs/smpl_mean_params.npz');config.freeze()
 with torch.inference_mode():
  model=HMR2(config);state=load(rt.ASSETS/'gvhmr/hmr2/epoch=10-step=25000.ckpt')['state_dict'];model.load_state_dict({k:v for k,v in state.items() if k.split('.')[0] in ['backbone','smpl_head']},strict=True);del state
  model.cuda().eval();features=[]
  for i in range(0,len(imgs),4):features.append(model({'img':imgs[i:i+4].cuda()}).cpu())
  features=torch.cat(features);del model;gc.collect();torch.cuda.empty_cache()
  from hmr4d.utils.preproc.vitpose_pytorch import build_model
  from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
  from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
  model=build_model('ViTPose_huge_coco_256x192',checkpoint=None);model.load_state_dict(load(rt.ASSETS/'gvhmr/vitpose/vitpose-h-multi-coco.pth')['state_dict'],strict=True);model.cuda().eval();points=[]
  for i in range(0,len(imgs),4):
   x=imgs[i:i+4,:,:,32:224].cuda();h,f=model(torch.cat([x,x.flip(3)])).chunk(2);heat=((h+flip_heatmap_coco17(f))*.5).cpu().numpy();b=boxes[i:i+4]
   p,s=keypoints_from_heatmaps(heatmaps=heat,center=b[:,:2].numpy(),scale=(torch.cat([b[:,[2]]*.75,b[:,[2]]],1)/200).numpy(),use_udp=True);points.append(torch.from_numpy(np.concatenate([p,s],-1)))
  kp=torch.cat(points);torch.save(kp,out/'vitpose-keypoints.pt');del model;gc.collect();torch.cuda.empty_cache()
  # PROJ_ROOT redirects only generic model resources. Module imports remain vendored code.
  import hmr4d
  hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
  import hmr4d.model.gvhmr.gvhmr_pl_demo,hmr4d.model.gvhmr.utils.endecoder,hmr4d.network.gvhmr.relative_transformer
  from hydra import initialize_config_module,compose
  from hydra.utils import instantiate
  from hmr4d.configs import register_store_gvhmr
  from hmr4d.utils.geo.hmr_cam import normalize_kp2d
  from hmr4d.utils.geo_transform import compute_cam_angvel
  from hmr4d.utils.smplx_utils import make_smplx
  register_store_gvhmr()
  with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):settings=compose(config_name='demo',overrides=['video_name=shuolee_new_video'])
  model=instantiate(settings.model,_recursive_=False);model.load_state_dict(load(rt.ASSETS/'gvhmr/gvhmr/gvhmr_siga24_release.ckpt')['state_dict'],strict=True);model.cuda().eval()
  Ks=torch.tensor(np.array([r['K'] for r in rows]),dtype=torch.float32);Rs=torch.tensor(np.array([r['w2c'] for r in rows])[:,:3,:3],dtype=torch.float32)
  batch={'length':torch.tensor([len(rows)]),'obs':normalize_kp2d(kp,boxes)[None],'bbx_xys':boxes[None],'K_fullimg':Ks[None],'cam_angvel':compute_cam_angvel(Rs)[None],'f_imgseq':features[None]}
  result=model.pipeline.forward({k:v.cuda() for k,v in batch.items()},train=False,postproc=True,static_cam=False)
  body=make_smplx('supermotion').cuda();params={k:v[0] for k,v in result['pred_smpl_params_incam'].items()};predbody=body(**params)
  arrays={'incam_'+k:v.cpu().numpy() for k,v in params.items()};arrays['incam_joints']=predbody.joints.cpu().numpy()
  if not all(np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Nonfinite GVHMR prediction.')
  np.savez_compressed(out/'prediction-postprocessed.npz',**arrays)
 rt.save(out/'receipt.json',dict(frames=len(rows),from_this_video=True,previous_person_parameters_loaded=False,camera_origin='Current run full-image SfM',camera_count=len(Ks),geometry_is_prior_not_ground_truth=True))
