"""All cameras are estimated from this run's full RGB frames, never interpolated."""
import json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import read,save,sha

def extract(video,root,cfg):
 out=root/'raw';out.mkdir()
 probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_streams','-show_entries','frame=best_effort_timestamp_time:stream=width,height,r_frame_rate,pix_fmt,color_transfer,color_space,color_primaries:stream_tags=rotate','-of','json',str(video)],capture_output=True,text=True,check=True)
 info=json.loads(probe.stdout);frames=info['frames']
 if info['streams'][0].get('color_transfer') in ['smpte2084','arib-std-b67']:
  raise RuntimeError('HDR video requires an explicitly reviewed SDR conversion before this pipeline; do not silently reinterpret PQ/HLG as sRGB.')
 times=[float(f['best_effort_timestamp_time']) for f in frames]
 if not times or np.any(np.diff(times)<=0):raise RuntimeError('Missing or non-increasing video timestamps. No synthetic timestamps are substituted.')
 stride=int(cfg['frame_stride']);ids=list(range(0,len(times),stride))
 if not cfg['min_frames']<=len(ids)<=cfg['max_frames']:raise RuntimeError(f'{len(ids)} selected frames outside declared {cfg["min_frames"]}..{cfg["max_frames"]} budget. Select an explicit stride/clip in a new config; frames are never silently discarded.')
 M=int(cfg['max_image_size'])
 vf=f"select=not(mod(n\\,{stride})),scale=w='min(iw,{M})':h='min(ih,{M})':force_original_aspect_ratio=decrease"
 with (root/'logs/decode.log').open('w') as log:
  subprocess.run(['ffmpeg','-nostdin','-hide_banner','-v','warning','-n','-i',str(video),'-vf',vf,'-pix_fmt','rgb24','-vsync','0',str(out/'frame_%06d.png')],stdout=log,stderr=subprocess.STDOUT,check=True)
 files=sorted(out.glob('frame_*.png'))
 if len(files)!=len(ids):raise RuntimeError('Decoded frame identity/count does not match ffprobe.')
 records=[];wh=None
 for p,idx in zip(files,ids):
  with Image.open(p) as im:shape=list(im.size)
  if wh is None:wh=shape
  if shape!=wh:raise RuntimeError('Video changes image dimensions; shared-intrinsics contract is invalid.')
  records.append(dict(frame=p.stem,name=p.name,source_frame_index=idx,time_seconds=times[idx],wh=shape,source_sha256=sha(p)))
 save(root/'decoded.json',dict(video_sha256=sha(video),decoded_total=len(times),frame_stride=stride,records=records,decoder='FFmpeg autorotation followed by isotropic fit within max_image_size',camera_parameters='estimated later in this exact decoded pixel domain'))

def sfm(root,cfg):
 import pycolmap as pc
 if pc.__version__!='3.10.0':raise RuntimeError(f'Validated API contract requires pycolmap 3.10.0, found {pc.__version__}.')
 raw=root/'raw';sfm=root/'sfm';sfm.mkdir();db=sfm/'database.db';maps=sfm/'models';maps.mkdir()
 device=pc.Device.cuda if pc.has_cuda else pc.Device.cpu
 opt=pc.SiftExtractionOptions();opt.max_image_size=cfg['max_image_size'];opt.max_num_features=8192;opt.num_threads=8
 pc.extract_features(str(db),str(raw),camera_mode=pc.CameraMode.SINGLE,camera_model=cfg['camera_model'],sift_options=opt,device=device)
 import sqlite3
 with sqlite3.connect(db) as connection:detected=connection.execute('SELECT rows FROM keypoints').fetchall()
 if len(detected)!=len(read(root/'decoded.json')['records']) or not detected or min(x[0] for x in detected)<cfg['min_track_observations']:
  raise RuntimeError('COLMAP could not read every RGB frame or extract sufficient features. Stop before native matching; inspect sfm.log.')
 mo=pc.SiftMatchingOptions();mo.guided_matching=True;mo.num_threads=8
 pc.match_exhaustive(str(db),sift_options=mo,device=device)
 io=pc.IncrementalPipelineOptions();io.num_threads=8;io.ba_refine_focal_length=True;io.ba_refine_principal_point=False;io.ba_refine_extra_params=True
 models=pc.incremental_mapping(str(db),str(raw),str(maps),options=io)
 if not models:raise RuntimeError('SfM could not initialize a connected scene. Preserve logs and change the capture, not the pose interpolation.')
 index=max(models,key=lambda k:models[k].num_reg_images());rec=models[index]
 decoded=read(root/'decoded.json')['records'];expected={r['name'] for r in decoded};actual={i.name for i in rec.images.values()}
 report=dict(frames=len(decoded),registered=len(actual),missing=sorted(expected-actual),points=rec.num_points3D(),mean_reprojection_error_px=rec.compute_mean_reprojection_error(),camera_origin='Estimated by full-frame COLMAP; not device calibration',interpolation_used=False,physical_accuracy_proven=False)
 save(root/'camera-audit.json',report)
 if actual!=expected:raise RuntimeError(f'Only {len(actual)}/{len(expected)} frames registered. No dropped-camera or interpolation fallback.')
 if report['mean_reprojection_error_px']>cfg['max_sfm_mean_error_px']:raise RuntimeError('SfM training residual exceeds configured diagnostic limit.')
 counts=[int(i.num_points3D) for i in rec.images.values()]
 if min(counts)<cfg['min_track_observations']:raise RuntimeError('One or more views lack sufficient registered image observations.')
 rect=root/'rectified';pc.undistort_images(str(rect),str(maps/str(index)),str(raw),output_type='COLMAP')
 ud=pc.Reconstruction(str(rect/'sparse'));by_name={i.name:i for i in ud.images.values()};records=[]
 for row in decoded:
  im=by_name[row['name']];cam=ud.cameras[im.camera_id]
  if cam.model.name not in ['PINHOLE','SIMPLE_PINHOLE']:raise RuntimeError('Undistorted camera is not pinhole.')
  K=np.array(cam.calibration_matrix());E=np.eye(4);E[:3]=im.cam_from_world.matrix();image=rect/'images'/row['name']
  if abs(np.linalg.det(E[:3,:3])-1)>1e-6 or not np.isfinite(E).all():raise RuntimeError('Invalid rigid camera transform.')
  with Image.open(image) as p:wh=list(p.size)
  if wh!=[cam.width,cam.height]:raise RuntimeError('Image and camera dimensions differ after undistortion.')
  records.append(dict(**{k:v for k,v in row.items() if k not in ['wh','source_sha256']},image=str(image),source_sha256=sha(image),wh=wh,K=K.tolist(),w2c=E.tolist(),mask=None,mask_sha256=None,official_prediction_slot=len(records),camera_kind='fresh_full_scene_sfm',camera_registration={'interpolation_used':False},observation_weight=1.0))
 save(root/'data/inputs.json',dict(records=records,pixel_domain='undistorted_integer_centers',intrinsic_origin='COLMAP_pixel_corners',coordinate_contract='K and w2c refer to the full undistorted image. COLMAP pixel corners; integer-centre landmarks project with -0.5.',from_scratch=True))
