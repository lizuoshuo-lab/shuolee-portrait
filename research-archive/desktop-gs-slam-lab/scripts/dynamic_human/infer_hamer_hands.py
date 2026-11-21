"""Official HaMeR on105training contexts, three crop scales; no Gaussian updates."""
from pathlib import Path
import json,hashlib,time,contextlib,io,gc
import numpy as np,torch,cv2
from PIL import Image,ImageDraw,ImageFont
from torch.utils.data import default_collate
from hamer.configs import get_config
from hamer.models import HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.datasets.utils import generate_image_patch_cv2

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');O=L/'outputs/dynamic-human-hand-prior-096';R=Path('/mnt/d/GS-SLAM-Data/models/hamer/_DATA');M=Path('/mnt/d/GS-SLAM-Data/models/mano_v1_2/numpy');P=O/'predictions';P.mkdir(exist_ok=True);V=O/'prior-panels';V.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 return h.hexdigest()
assert not (O/'inference-summary.json').exists()
weights=R/'hamer_ckpts/checkpoints/hamer.ckpt';receipt=read(O/'weights-extraction.json');digest=sha(weights);assert digest==next(r['sha256'] for r in receipt['extracted'] if r['file'].endswith('hamer.ckpt'))
cfg=get_config(str(R/'hamer_ckpts/model_config.yaml'),update_cachedir=False);cfg.defrost();cfg.MANO.MODEL_PATH=str(M);cfg.MANO.MEAN_PARAMS=str(R/'data/mano_mean_params.npz');cfg.MANO.DATA_DIR=str(R/'data');cfg.MODEL.BBOX_SHAPE=[192,256];cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None);cfg.freeze()
plan=read(O/'input-manifest.json');expected=plan['expected_model_cfg']
for k,v in expected.items():assert cfg.MODEL[k]==v,(k,cfg.MODEL[k],v)
torch.set_num_threads(6);torch.manual_seed(42);model=HAMER(cfg,init_renderer=False)
checkpoint=torch.load(weights,map_location='cpu',weights_only=False);state=checkpoint['state_dict'];before={k:v.detach().clone() for k,v in model.state_dict().items() if k.startswith('mano.')};loaded=model.load_state_dict(state,strict=False)
assert not loaded.missing_keys and not loaded.unexpected_keys,(loaded.missing_keys,loaded.unexpected_keys)
changes=[]
for k,v in before.items():
 b=model.state_dict()[k];changes.append(dict(name=k,exact=torch.equal(v,b),max_abs_difference=float((v-b).abs().max()) if v.numel() else 0))
save(O/'network-load.json',dict(status='official_checkpoint_loaded_all_keys_match',weights_sha256=digest,missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys,MANO_buffer_comparison=changes,model_cfg_verified_against_preprocessing=True,init_renderer=False,no_GS_updates=True,goal_complete=False))
del checkpoint,state,before;gc.collect();model=model.eval().cuda();torch.cuda.reset_peak_memory_stats();records=[];started=time.monotonic();font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12)
rows=sorted(plan['records'],key=lambda r:(not (r['camera'] in ['03','04','13','17'] and r['frame'] in [720,726,732]),r['camera'],r['frame']))
colors=[(235,70,20),(210,40,160),(30,150,210),(65,180,55),(165,75,210)]
for n,r in enumerate(rows):
 c,t=r['camera'],r['frame'];path=L/r['source'];assert sha(path)==r['source_sha256'];z=np.load(path);rgb=z['rgb'];box=np.array(r['context_bbox'],np.float32);center=(box[:2]+box[2:])/2;factors=np.array([.85,1.,1.15],np.float32);boxes=np.stack([np.r_[center-(box[2:]-box[:2])/2*f,center+(box[2:]-box[:2])/2*f] for f in factors]);bgr=np.ascontiguousarray(rgb[:,:,::-1]);ds=ViTDetDataset(cfg,bgr,boxes,np.ones(3),rescale_factor=1.)
 with contextlib.redirect_stdout(io.StringIO()):items=[ds[k] for k in range(3)]
 batch={k:v.cuda() for k,v in default_collate(items).items()}
 stem=f'camera{c}-frame{t:06d}';dest=P/f'{stem}.npz'
 if dest.exists():
  with np.load(dest) as previous:a={k:previous[k] for k in previous.files if k.startswith(('pred_','mano_')) or k=='focal_length'}
 else:
  with torch.no_grad():pred=model(batch)
  a={k:v.detach().cpu().numpy() for k,v in pred.items() if torch.is_tensor(v)}
  a.update({'mano_'+k:v.detach().cpu().numpy() for k,v in pred['pred_mano_params'].items()})
 sizes=np.array([i['box_size'] for i in items]);centers=np.array([i['box_center'] for i in items]);source_xy=a['pred_keypoints_2d']*sizes[:,None,None]+centers[:,None,:]
 assert source_xy.shape==(3,21,2) and all(np.isfinite(v).all() for v in a.values())
 projected=[];maxerror=0.
 for k,item in enumerate(items):
  _,A=generate_image_patch_cv2(bgr,*item['box_center'],item['box_size'],item['box_size'],256,256,False,1.,0,border_mode=cv2.BORDER_CONSTANT);inv=cv2.invertAffineTransform(A);xy=(a['pred_keypoints_2d'][k]+.5)*256;back=xy@inv[:,:2].T+inv[:,2];maxerror=max(maxerror,float(abs(back-source_xy[k]).max()))
  pts=a['pred_keypoints_3d'][k]+a['pred_cam_t'][k];independent=pts[:,:2]/pts[:,2:]*cfg.EXTRA.FOCAL_LENGTH/256*sizes[k]+centers[k];assert np.max(abs(independent-source_xy[k]))<1e-3
  projected.append(A)
 assert maxerror<1e-3
 spread=np.max(np.linalg.norm(source_xy-source_xy[1:2],axis=-1),axis=0)
 if not dest.exists():np.savez(dest,**a,source_keypoints_2d=source_xy,crop_factors=factors,box_sizes=sizes,box_centers=centers,affine_native_to_patch=np.array(projected),K=z['K'],w2c=z['w2c'])
 else:
  with np.load(dest) as previous:assert np.array_equal(previous['source_keypoints_2d'],source_xy)
 rec=dict(camera=c,frame=t,file=str(dest.relative_to(O)),sha256=sha(dest),source_sha256=r['source_sha256'],crop_keypoint_spread_median_px=float(np.median(spread)),crop_keypoint_spread_p95_px=float(np.quantile(spread,.95)),max_affine_backprojection_error_px=maxerror,scope='Trainingphoto prior qualification only. Model joints are predictions, not GT. Camera-space scale/depth from monocular network is not calibrated scene geometry.')
 if c in ['03','04','13','17'] and t in [720,726,732]:
  x0,y0,x1,y1=map(int,box);raw=Image.fromarray(rgb[y0:y1,x0:x1]);overlay=raw.copy();d=ImageDraw.Draw(overlay);pts=source_xy[1]-[x0,y0]
  for f in range(5):
   chain=[0]+list(range(1+4*f,5+4*f))
   for i,j in zip(chain[:-1],chain[1:]):d.line([tuple(pts[i]),tuple(pts[j])],fill=colors[f],width=1)
  im=Image.new('RGB',(320,206),'white');draw=ImageDraw.Draw(im);draw.text((4,4),f'{c}/{t} | HaMeR prior, NOT GS result',font=font,fill='black');draw.text((4,24),'Native source              Predicted joints',font=font,fill='black');im.paste(raw,(0,46));im.paste(overlay,(160,46));p=V/f'{stem}.png';im.save(p);rec['visual']=str(p.relative_to(O))
 records.append(rec);save(O/'inference-progress.json',dict(status='running',images=len(records),forwards=len(records)*3,elapsed_seconds=time.monotonic()-started,last_camera=c,last_frame=t));print('HAMER',n+1,c,t,rec['crop_keypoint_spread_median_px'],flush=True)
save(O/'observations.json',dict(status='105training_images315crop_predictions_complete',records=records,accepted_supervision=False,goal_complete=False))
save(O/'inference-summary.json',dict(status='network_predictions_complete_visual_and_geometry_qualification_pending',images=len(records),crops=len(records)*3,elapsed_seconds=time.monotonic()-started,max_gpu_GB=torch.cuda.max_memory_allocated()/1024**3,checkpoint_sha256=digest,crop_spread_median_over_images_px=float(np.median([r['crop_keypoint_spread_median_px'] for r in records])),crop_spread_p95_over_images_px=float(np.quantile([r['crop_keypoint_spread_median_px'] for r in records],.95)),projection_max_error_px=max(r['max_affine_backprojection_error_px'] for r in records),Gaussian_parameters_changed=False,heldout_photos_used=False,accepted_as_training_labels=False,goal_complete=False));print('HAMER_INFERENCE_COMPLETE',len(records),flush=True)
