"""GVHMR predicts this video's pose prior using this run's measured SfM matrices."""
import gc
from pathlib import Path
import numpy as np
from PIL import Image
from . import runtime as rt


def camera_intrinsics_for_prior(rows, cfg):
 """Convert renderer corner coordinates once, without mutating camera records."""
 K=np.asarray([r['K'] for r in rows],dtype=np.float32).copy()
 if cfg.get('pixel_center_convention')=='renderer_corner_to_detector_integer':
  K[:,0,2]-=np.float32(.5);K[:,1,2]-=np.float32(.5)
 return K


def _fresh_source_binding(root, rows, cfg):
    """Bind an original prediction to the actual current video and frame files."""
    root = Path(root)
    inputs_path = root / 'data/inputs.json'
    inputs = rt.read(inputs_path)
    run = rt.read(root / 'run.json')
    if inputs.get('records') != rows or not rows:
        raise ValueError('Fresh prior records changed or are empty.')
    video = Path(run['video'])
    digest = run.get('video_sha256')
    if not video.is_file() or rt.sha(video) != digest:
        raise ValueError('The current source video is missing or its SHA-256 changed.')
    if inputs.get('video_sha256', digest) != digest:
        raise ValueError('Fresh prior inputs belong to a different video.')
    ids, sources, hashes, times = [], [], [], []
    for slot, row in enumerate(rows):
        frame = row.get('frame')
        source = row.get('source_frame_index')
        if not isinstance(frame, str) or not frame or frame in ids:
            raise ValueError('Fresh prior frame identities must be unique nonempty strings.')
        if type(source) is not int or source < 0 or (sources and source <= sources[-1]):
            raise ValueError('Fresh prior source frame indices must be increasing integers.')
        if row.get('official_prediction_slot', slot) != slot:
            raise ValueError('Fresh prior prediction slots disagree with the frame order.')
        if row.get('video_sha256', digest) != digest:
            raise ValueError('A fresh prior frame belongs to a different video.')
        stamp = row.get('time_seconds')
        if (type(stamp) not in (int, float) or not np.isfinite(stamp) or stamp < 0 or
                (times and stamp <= times[-1])):
            raise ValueError('Fresh prior timestamps must be finite and strictly increasing.')
        image = Path(row['image'])
        if not image.is_file() or rt.sha(image) != row.get('source_sha256'):
            raise ValueError(f'{frame}: source image is missing or changed.')
        with Image.open(image) as opened:
            wh = list(opened.size)
        if row.get('wh') != wh:
            raise ValueError(f'{frame}: source image size disagrees with its camera.')
        box = np.asarray(row.get('initial_box_xyxy'), dtype=np.float64)
        if (box.shape != (4,) or not np.isfinite(box).all() or
                np.any(box[2:] <= box[:2]) or np.any(box < 0) or
                np.any(box > [wh[0], wh[1], wh[0], wh[1]])):
            raise ValueError(f'{frame}: invalid fresh prior image crop.')
        K = np.asarray(row.get('K'), dtype=np.float64)
        E = np.asarray(row.get('w2c'), dtype=np.float64)
        if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-9 or abs(K[1, 0]) > 1e-9):
            raise ValueError('Fresh prior requires finite zero-skew pinhole intrinsics.')
        if (E.shape != (4, 4) or not np.isfinite(E).all() or
                not np.allclose(E[3], [0, 0, 0, 1]) or
                not np.allclose(E[:3, :3] @ E[:3, :3].T, np.eye(3), atol=1e-5) or
                abs(np.linalg.det(E[:3, :3]) - 1) > 1e-5):
            raise ValueError('Fresh prior requires rigid world-to-camera matrices.')
        ids.append(frame); sources.append(source); hashes.append(row['source_sha256']); times.append(stamp)
    return dict(video_sha256=digest, frame_ids=ids, source_frames=sources, image_sha256=hashes,
                inputs_sha256=rt.sha(inputs_path), renderer_K=[r['K'] for r in rows],
                prior_K=camera_intrinsics_for_prior(rows, cfg).tolist(), w2c=[r['w2c'] for r in rows],
                time_seconds=times, pixel_center_convention=cfg.get('pixel_center_convention', 'as_provided'))


def _forward_prediction(pipeline, batch, *, postprocess=True):
    """One fresh forward; upstream postprocessing replaces the incam body pose."""
    if type(postprocess) is not bool:
        raise TypeError('postprocess must be a boolean.')
    return pipeline.forward(batch, train=False, postproc=postprocess, static_cam=False)


def _prediction_arrays(result, body, frames):
    names = dict(body_pose=63, betas=10, global_orient=3, transl=3)
    predicted = result['pred_smpl_params_incam']
    params = {}
    for name, width in names.items():
        value = predicted[name]
        if tuple(value.shape) != (1, frames, width):
            raise ValueError(f'Unexpected fresh incam {name} shape: {tuple(value.shape)}.')
        params[name] = value[0]
    posed = body(**params)
    arrays = {'incam_' + key: value.detach().cpu().numpy().copy() for key, value in params.items()}
    arrays['incam_joints'] = posed.joints.detach().cpu().numpy().copy()
    joints = arrays['incam_joints']
    if joints.ndim != 3 or joints.shape[0] != frames or joints.shape[1] == 0 or joints.shape[2] != 3:
        raise ValueError('Unexpected fresh incam joint shape.')
    if not all(np.issubdtype(value.dtype, np.floating) and np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite or nonfloating GVHMR prediction.')
    return arrays


def _save_prediction(root, rows, cfg, arrays, *, postprocess=True, source_binding=None):
    """Publish one variant and its provenance; never relabel a postprocessed result."""
    if type(postprocess) is not bool:
        raise TypeError('postprocess must be a boolean.')
    out = Path(root) / 'inference'
    if not postprocess:
        if source_binding is None or _fresh_source_binding(root, rows, cfg) != source_binding:
            raise ValueError('Fresh prior source binding changed during inference.')
    prediction = out / ('prediction-postprocessed.npz' if postprocess else 'prediction-original.npz')
    receipt = out / ('receipt.json' if postprocess else 'original-receipt.json')
    if prediction.exists() or receipt.exists():
        raise FileExistsError('A prior prediction or receipt already exists; use a fresh run.')
    np.savez_compressed(prediction, **arrays)
    intervals = np.diff([r['time_seconds'] for r in rows])
    report = dict(frames=len(rows), from_this_video=True, previous_person_parameters_loaded=False,
                  camera_origin='Current run full-image SfM', camera_count=len(rows),
                  geometry_is_prior_not_ground_truth=True,
                  median_frame_interval_seconds=float(np.median(intervals)) if len(intervals) else None,
                  frame_stride=cfg['frame_stride'], temporal_prior_cadence_requires_review=cfg['frame_stride'] != 1,
                  prediction_variant='postprocessed_incam' if postprocess else 'original_incam',
                  postprocessing_applied=postprocess, fresh_inference=True, cached_prediction_loaded=False,
                  prediction_sha256=rt.sha(prediction), quality_accepted=False,
                  physical_calibration_accepted=False, training_started=False)
    if not postprocess:
        report.update(source_binding)
    rt.save(receipt, report)
    return report

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

def infer(root,cfg,*,postprocess=True):
 """Predict fresh incam parameters; False skips upstream IK and writes the original variant."""
 if type(postprocess) is not bool:raise TypeError('postprocess must be a boolean.')
 root=Path(root);rows=rt.read(root/'data/inputs.json')['records']
 source_binding=None if postprocess else _fresh_source_binding(root,rows,cfg)
 out=root/'inference';out.mkdir()
 import torch
 import hmr4d
 hmr4d.PROJ_ROOT=rt.ASSETS/'gvhmr'
 from hmr4d.network.hmr2.configs import get_config
 from hmr4d.network.hmr2 import HMR2
 from hmr4d.network.hmr2.utils.preproc import IMAGE_MEAN,IMAGE_STD
 from .crops import prepare_prior_crop
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
  result=_forward_prediction(model.pipeline,{k:v.cuda() for k,v in batch.items()},postprocess=postprocess)
  progress('GVHMR_inference',len(rows))
  body=make_smplx('supermotion').cuda()
  arrays=_prediction_arrays(result,body,len(rows))
 _save_prediction(root,rows,cfg,arrays,postprocess=postprocess,source_binding=source_binding)
