"""GVHMR predicts this video's pose prior using this run's measured SfM matrices."""
import gc
import numpy as np
from PIL import Image
from . import runtime as rt


def camera_intrinsics_for_prior(rows, cfg):
 """Convert renderer corner coordinates once, without mutating camera records."""
 K=np.asarray([r['K'] for r in rows],dtype=np.float32).copy()
 if cfg.get('pixel_center_convention')=='renderer_corner_to_detector_integer':
  K[:,0,2]-=np.float32(.5);K[:,1,2]-=np.float32(.5)
 return K

def inference_settings():
 """Compose the actual upstream inference graph so doctor checks its imports."""
 import hmr4d
 hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
 import hmr4d.model.gvhmr.gvhmr_pl_demo,hmr4d.model.gvhmr.utils.endecoder,hmr4d.network.gvhmr.relative_transformer
 import hmr4d.model.gvhmr.pipeline.gvhmr_pipeline
 from hydra import initialize_config_module,compose
 from hydra.core.global_hydra import GlobalHydra
 from hmr4d.configs import register_store_gvhmr
 register_store_gvhmr()
 # SAM2 initializes Hydra when imported. Compose GVHMR in its own temporary
 # context and restore the caller's context, including when composition fails.
 previous=GlobalHydra.instance().hydra
 GlobalHydra.instance().clear()
 try:
  with initialize_config_module(version_base='1.3',config_module='hmr4d.configs'):
   return compose(config_name='demo',overrides=['video_name=shuolee_new_video'])
 finally:
  GlobalHydra.instance().clear()
  if previous is not None:GlobalHydra.instance().initialize(previous)

def infer(root,cfg):
 import torch
 import hmr4d
 hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
 from hmr4d.network.hmr2.configs import get_config
 from hmr4d.network.hmr2 import HMR2
 from hmr4d.network.hmr2.utils.preproc import IMAGE_MEAN,IMAGE_STD
 from .crops import prepare_prior_crop
 rows=rt.read(root/'data/inputs.json')['records'];out=root/'inference';out.mkdir()
 def progress(name,done=0):
  rt.save(root/'progress.json',dict(stage='prior',component=name,done=done,total=len(rows),updated=rt.now()))
  print(f'prior {name} {done}/{len(rows)}',flush=True)
 progress('prepare_crops')
 crops=[];boxes=[];crop_records=[]
 for r in rows:
  rgb=np.array(Image.open(r['image']).convert('RGB'))
  crop,box,mapping=prepare_prior_crop(rgb,r['initial_box_xyxy'])
  crops.append(crop);boxes.append(box)
  crop_records.append(dict(frame=r['frame'],source_sha256=r['source_sha256'],**mapping))
 rt.save(out/'crop-mapping.json',dict(records=crop_records,training_image_warped=False,physical_camera_accuracy_proven=False))
 imgs=((torch.from_numpy(np.stack(crops)).float()/255-IMAGE_MEAN)/IMAGE_STD).permute(0,3,1,2);boxes=torch.tensor(np.asarray(boxes),dtype=torch.float32)
 def load(path):
  unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(path)
  if unsafe:raise RuntimeError(f'Unreviewed checkpoint globals: {unsafe}')
  return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
 config=get_config(str(rt.PROJECT/'third_party/gvhmr/hmr4d/network/hmr2/configs/model_config.yaml'))
 config.defrost();config.MODEL.BBOX_SHAPE=[192,256];config.SMPL.MEAN_PARAMS=str(rt.ASSETS/'gvhmr/hmr4d/network/hmr2/configs/smpl_mean_params.npz');config.freeze()
 with torch.inference_mode():
  progress('load_HMR2')
  model=HMR2(config);state=load(rt.ASSETS/'gvhmr/hmr2/epoch=10-step=25000.ckpt')['state_dict'];model.load_state_dict({k:v for k,v in state.items() if k.split('.')[0] in ['backbone','smpl_head']},strict=True);del state
  model.cuda().eval();features=[]
  for i in range(0,len(imgs),4):
   features.append(model({'img':imgs[i:i+4].cuda()}).cpu());progress('HMR2_features',min(i+4,len(rows)))
  features=torch.cat(features);del model;gc.collect();torch.cuda.empty_cache()
  from hmr4d.utils.preproc.vitpose_pytorch import build_model
  from hmr4d.utils.geo.flip_utils import flip_heatmap_coco17
  from hmr4d.utils.kpts.kp2d_utils import keypoints_from_heatmaps
  progress('load_ViTPose')
  model=build_model('ViTPose_huge_coco_256x192',checkpoint=None);model.load_state_dict(load(rt.ASSETS/'gvhmr/vitpose/vitpose-h-multi-coco.pth')['state_dict'],strict=True);model.cuda().eval();points=[]
  for i in range(0,len(imgs),4):
   x=imgs[i:i+4,:,:,32:224].cuda();h,f=model(torch.cat([x,x.flip(3)])).chunk(2);heat=((h+flip_heatmap_coco17(f))*.5).cpu().numpy();b=boxes[i:i+4]
   p,s=keypoints_from_heatmaps(heatmaps=heat,center=b[:,:2].numpy(),scale=(torch.cat([b[:,[2]]*.75,b[:,[2]]],1)/200).numpy(),use_udp=True);points.append(torch.from_numpy(np.concatenate([p,s],-1)))
   progress('ViTPose',min(i+4,len(rows)))
  kp=torch.cat(points);torch.save(kp,out/'vitpose-keypoints.pt');del model;gc.collect();torch.cuda.empty_cache()
  from hydra.utils import instantiate
  from hmr4d.utils.geo.hmr_cam import normalize_kp2d
  from hmr4d.utils.geo_transform import compute_cam_angvel
  from hmr4d.utils.smplx_utils import make_smplx
  progress('load_GVHMR')
  settings=inference_settings()
  model=instantiate(settings.model,_recursive_=False);model.load_state_dict(load(rt.ASSETS/'gvhmr/gvhmr/gvhmr_siga24_release.ckpt')['state_dict'],strict=True);model.cuda().eval()
  Ks=torch.tensor(camera_intrinsics_for_prior(rows,cfg),dtype=torch.float32);Rs=torch.tensor(np.array([r['w2c'] for r in rows])[:,:3,:3],dtype=torch.float32)
  # This native-pixel input explicitly stores renderer corner-centre K.
  # HMR's keypoints/boxes use integer pixel centres, so its camera uses K-0.5.
  # Older runs retain their original prior behavior for reproducibility.
  if cfg.get('pixel_center_convention')=='renderer_corner_to_detector_integer':
   rt.save(out/'intrinsic-contract.json',dict(renderer_K=rows[0]['K'],prior_K_integer=Ks[0].tolist(),keypoints='integer_centres',calibration_accuracy_proven=False))
  batch={'length':torch.tensor([len(rows)]),'obs':normalize_kp2d(kp,boxes)[None],'bbx_xys':boxes[None],'K_fullimg':Ks[None],'cam_angvel':compute_cam_angvel(Rs)[None],'f_imgseq':features[None]}
  result=model.pipeline.forward({k:v.cuda() for k,v in batch.items()},train=False,postproc=True,static_cam=False)
  progress('GVHMR_inference',len(rows))
  body=make_smplx('supermotion').cuda();params={k:v[0] for k,v in result['pred_smpl_params_incam'].items()};predbody=body(**params)
  arrays={'incam_'+k:v.cpu().numpy() for k,v in params.items()};arrays['incam_joints']=predbody.joints.cpu().numpy()
  if not all(np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Nonfinite GVHMR prediction.')
  np.savez_compressed(out/'prediction-postprocessed.npz',**arrays)
 dt=np.diff([r['time_seconds'] for r in rows])
 rt.save(out/'receipt.json',dict(frames=len(rows),from_this_video=True,previous_person_parameters_loaded=False,camera_origin='Current run full-image SfM',camera_count=len(Ks),geometry_is_prior_not_ground_truth=True,median_frame_interval_seconds=float(np.median(dt)),frame_stride=cfg['frame_stride'],temporal_prior_cadence_requires_review=cfg['frame_stride']!=1))
