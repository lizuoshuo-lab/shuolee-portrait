"""T5: preregister full native clip03 frames, decode once and draft main-person masks.

No camera estimation, feature rematching, old subject geometry, fitting or GS.
Mask outputs remain proposals until actual full-view/occlusion review.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,sys,time,hashlib,subprocess,signal,traceback
import numpy as np
from PIL import Image

H=Path(__file__).parent
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
I=R/'video-update-20261027';O=R/'native-supervision';DATA=Path('/mnt/d/GS-SLAM-Data/robust122-whiteT-native-supervision')
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118');ASSETS=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')

def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
 temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def progress(state,**kw):save(O/'progress.json',dict(stage='T5',state=state,pid=os.getpid(),updated=now(),**kw))
def verify(plan):
 for p,h in plan['frozen'].items():
  if sha(p)!=h:raise ValueError('Frozen T5 source changed: '+p)

def register():
 if O.exists() or DATA.exists():raise RuntimeError('T5 already exists; inspect current state')
 q=read(R/'queue-state.json')
 if q.get('running'):raise RuntimeError('Existing running task; inspect first')
 source=next(x for x in read(I/'content-review.json')['videos'] if x['id']=='clip03')
 if sha(source['path'])!=source['sha256']:raise ValueError('Source video changed')
 command=['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries',
  'frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration,width,height,pix_fmt,pict_type,key_frame','-of','json',source['path']]
 cp=subprocess.run(command,capture_output=True,text=True,timeout=120,check=True)
 frames=json.loads(cp.stdout)['frames']
 if len(frames)!=464 or any((r['width'],r['height'])!=(1520,2704) for r in frames):raise ValueError('Decoded frame/raster count differs from source')
 times=np.array([float(r['best_effort_timestamp_time']) for r in frames]);pts=[int(r['best_effort_timestamp']) for r in frames]
 if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Monotonic native timestamps required')
 files=[Path(__file__),Path(source['path']),I/'content-review.json',I/'clip03-scene-support-plan.json',
  PROJECT/'configs/server-assets.json',PROJECT/'src/shuolee_portrait/observations.py',PROJECT/'src/shuolee_portrait/runtime.py',
  PROJECT/'third_party/sam2/sam2/sam2_image_predictor.py',ASSETS/'detection/yolov8n.pt',ASSETS/'sam2/sam2.1_hiera_tiny.pt',
  R/'native-fresh-preparation/review.json']
 frozen={str(p):sha(p) for p in files}
 O.mkdir();DATA.mkdir()
 save(O/'frame-clock.json',dict(source_sha256=source['sha256'],probe_command=command,time_base='1/1000000',
  frames=[dict(frame_id=i,pts=pts[i],time_seconds=float(times[i])) for i in range(len(frames))],
  interval_min_seconds=float(np.diff(times).min()),interval_median_seconds=float(np.median(np.diff(times))),interval_max_seconds=float(np.diff(times).max()),
  inferred_or_interpolated_timestamps=False))
 save(O/'frame-probe.json',json.loads(cp.stdout))
 save(O/'frame-plan.json',dict(kind='preregistered_training_frames',source_sha256=source['sha256'],frame_ids=list(range(464)),
  registered=now(),selection='Explicitly select all464 source frames before full decode/mask inference; no selection by residual, quality or view score.',
  reason='Preserve full captured temporal cadence for the existing temporal prior and retain all observed view evidence.',
  frame_clock=ref(O/'frame-clock.json'),required_camera_count=464,cameras_registered=False,physical_admission=False))
 frozen.update({str(O/'frame-plan.json'):sha(O/'frame-plan.json'),str(O/'frame-clock.json'):sha(O/'frame-clock.json')})
 plan=dict(stage='T5',registered=now(),source=source,frame_plan=ref(O/'frame-plan.json'),data_root=str(DATA),frozen=frozen,
  scope='Native supervision preparation across entire source; not a repeated camera/feature experiment and not a human fit.',
  operation='Exact sequential FFmpeg PNG RGB48 decode, noautorotate/noresize/no second undistortion. Pinned localYOLOv8n personidentity +SAM2.1tiny fullimage mask; model internally resizes, result returnsnativepixels. No mask interpreted as physicaltruth.',
  parameters=dict(person_confidence=.45,max_persons=8,ambiguous_area_ratio=.65,tracking_min_iou=.1,tracking_iou_margin=.05,prompt_box_padding_fraction=.04,
   mask_min_score=.5,mask_min_area_fraction=.005,mask_max_area_fraction=.95,mask_hole_fill=0,mask_sprinkle_remove=0),
  review='Save every alpha proposal, all detected boxes and selectedpersonidentity. Review every silhouette and first/middle/last/nativehead-hands-feet before deriving validforeground or externalocclusion support. Do not write allvalid observation masks without review.',
  no_changes=['K','externalposes','materialtrack identities/splits','oldgeometry','oldGaussian','oldoptimizer','faceacceptance'],
  budget=dict(runs=1,decode_seconds=300,mask_seconds=1200,total_seconds=1800,GPU_tasks=1,person_fits=0,GS_steps=0),
  stop='Source/frames/hash change,detectoridentityambiguous,coarsemaskinvalid orbudget: preserve completed outputs and error; no threshold relaxation, no automatic rerun.',
  downstream='All464 cameras must be actually registered and independently admitted before fit/train. The old7diagnostic cameras are not filled or promoted. Draftalpha is not supervisedmask acceptance.')
 save(O/'plan.json',plan)
 q.update(updated=now(),phase='T5_full_native_supervision_registered',running=None,next='Run registered T5 once; all464frames explicit, no camera/fitting/training.');save(R/'queue-state.json',q)
 print(json.dumps(dict(stage='T5_registered',frames=464,clock_min=plan['source']['duration'],data_root=str(DATA))),flush=True)

def run():
 plan=read(O/'plan.json');verify(plan)
 if (O/'progress.json').exists():raise RuntimeError('Existing T5 run; inspect instead of restarting')
 def timeout(*_):raise TimeoutError('Registered T5 budget reached')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(plan['budget']['total_seconds'])
 start=time.monotonic();progress('decode_native')
 (DATA/'rgb').mkdir();(DATA/'alpha-proposal').mkdir();(DATA/'reviews').mkdir()
 command=['/usr/bin/ffmpeg','-nostdin','-hide_banner','-loglevel','error','-noautorotate','-threads','4','-i',plan['source']['path'],
  '-map','0:v:0','-vsync','0','-pix_fmt','rgb48be','-threads','4','-start_number','0',str(DATA/'rgb/frame_%06d.png')]
 with (O/'decode.log').open('x') as log:
  cp=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 save(O/'decode-command.json',dict(argv=command,exit_code=cp.returncode))
 if cp.returncode:raise RuntimeError('Native FFmpeg decode failed')
 paths=sorted((DATA/'rgb').glob('frame_*.png'));frame_plan=read(O/'frame-plan.json');clock=read(O/'frame-clock.json')['frames']
 if len(paths)!=464:raise ValueError('Full native frame count mismatch')
 frames=[];identity=[]
 for i,p in enumerate(paths):
  with Image.open(p) as im:
   if im.size!=(1520,2704) or im.mode!='RGB':raise ValueError('Exact native RGB raster required')
  rgbhash=sha(p)
  frames.append(dict(frame_id=i,rgb=ref(p),width=1520,height=2704,time_seconds=clock[i]['time_seconds']))
  old=I/'clip03-scene-support'/p.name
  if old.exists():
   ok=rgbhash==sha(old);identity.append(dict(frame_id=i,byte_identical=ok,old_sha256=sha(old),new_sha256=rgbhash))
   if not ok:raise ValueError('Existing7 diagnostic native image bytes changed')
 if len(identity)!=7:raise ValueError('Seven frozen native anchors required')
 save(O/'decode-receipt.json',dict(source_sha256=plan['source']['sha256'],frame_plan_sha256=sha(O/'frame-plan.json'),frame_clock_sha256=sha(O/'frame-clock.json'),
  geometric_resampling=False,rotation_applied=False,pixel_convention='integer_centers',native_storage='PNG RGB48 from existing10-bitHEVC; PIL/model reader yields8bitRGB, same as frozen7diagnostic inputs',
  frames=[dict(frame_id=f['frame_id'],rgb_sha256=f['rgb']['sha256'],width=f['width'],height=f['height'],time_seconds=f['time_seconds']) for f in frames],
  frozen7_byte_comparison=identity,command=command))
 save(O/'native-frames.json',dict(frame_plan=ref(O/'frame-plan.json'),source=plan['source'],frames=frames))
 print('T5_FULL_NATIVE_DECODE',len(frames),flush=True)
 # Only one local GPU task. This lock is also used by native fit/train.
 import fcntl
 with open('/tmp/shuolee-native122-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU process active; inspect before running another')
  os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
  sys.path.insert(0,str(PROJECT/'src'))
  from shuolee_portrait import runtime as rt
  rt.configure(ASSETS)
  from shuolee_portrait.observations import load_detector
  import torch,cv2
  from sam2.build_sam import build_sam2
  from sam2.sam2_image_predictor import SAM2ImagePredictor
  torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;cv2.setNumThreads(2)
  manifest=read(PROJECT/'configs/server-assets.json')['files']
  for name in ['sam2/sam2.1_hiera_tiny.pt','detection/yolov8n.pt']:
   if sha(ASSETS/name)!=next(r['sha256'] for r in manifest if r['path']==name):raise ValueError('Official existing model hash mismatch')
  detector=load_detector().to('cuda')
  net=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ASSETS/'sam2/sam2.1_hiera_tiny.pt'),device='cuda',apply_postprocessing=False)
  segmenter=SAM2ImagePredictor(net,max_hole_area=0,max_sprinkle_area=0)
  pars=plan['parameters'];previous=None;rows=[];maskstart=time.monotonic()
  with torch.inference_mode():
   for row in frames:
    f=row['frame_id']
    if time.monotonic()-maskstart>1200:raise TimeoutError('Mask inference budget')
    rgb=np.array(Image.open(row['rgb']['path']).convert('RGB'));height,width=rgb.shape[:2]
    boxes=detector.predict(rgb[:,:,::-1].copy(),classes=[0],conf=pars['person_confidence'],verbose=False,device=0)[0].boxes.xyxy.cpu().numpy()
    if not 1<=len(boxes)<=pars['max_persons']:raise ValueError(f'Person count invalid at frame{f}')
    areas=np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    if previous is None:
     order=np.argsort(-areas)
     if len(order)>1 and areas[order[1]]>=pars['ambiguous_area_ratio']*areas[order[0]]:raise ValueError(f'Initial person identity ambiguous at{f}')
     identity_scores=None
    else:
     lo=np.maximum(boxes[:,:2],previous[:2]);hi=np.minimum(boxes[:,2:],previous[2:]);inter=np.maximum(hi-lo,0).prod(1)
     iou=inter/(areas+np.prod(previous[2:]-previous[:2])-inter+1e-8);order=np.argsort(-iou)
     if iou[order[0]]<pars['tracking_min_iou'] or (len(order)>1 and iou[order[0]]-iou[order[1]]<pars['tracking_iou_margin']):raise ValueError(f'Person tracking ambiguous at{f}')
     identity_scores=iou.tolist()
    selected=int(order[0]);box=boxes[selected];previous=box.copy();pad=pars['prompt_box_padding_fraction']*max(box[2:]-box[:2])
    prompt=np.clip(box+[-pad,-pad,pad,pad],[0,0,0,0],[width,height,width,height])
    segmenter.set_image(rgb)
    masks,scores,_=segmenter.predict(box=prompt,point_coords=((prompt[:2]+prompt[2:])/2)[None],point_labels=np.ones(1,np.int32),multimask_output=False)
    mask=masks[0].astype(np.uint8);fraction=float(mask.mean());score=float(scores[0])
    if not pars['mask_min_area_fraction']<fraction<pars['mask_max_area_fraction'] or score<pars['mask_min_score']:raise ValueError(f'Coarse mask rejected at{f},score{score},area{fraction}')
    path=DATA/'alpha-proposal'/f'frame_{f:06d}.png';Image.fromarray(mask*255).save(path)
    yy,xx=np.nonzero(mask);bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
    result=dict(frame_id=f,time_seconds=row['time_seconds'],rgb=row['rgb'],alpha_proposal=ref(path),selected_person_index=selected,detected_boxes_xyxy=boxes.tolist(),
     tracking_ious=identity_scores,prompted_box_xyxy=prompt.tolist(),mask_bbox_xyxy=bbox,mask_pixels=int(mask.sum()),mask_fraction=fraction,SAM_score=score,
     touches_frame_border=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),physical_quality_accepted=False)
    rows.append(result);save(O/'mask-rows.json',rows)
    # Plain masked image: no text, point annotations or inferred geometry.
    if f%16==0 or f==463:
     white=rgb.copy();white[mask==0]=240
     Image.fromarray(white).resize((304,541),Image.Resampling.LANCZOS).save(DATA/'reviews'/f'mask-preview_{f:06d}.png')
    progress('draft_person_masks',done=len(rows),total=len(frames),elapsed_seconds=time.monotonic()-start,mask_seconds=time.monotonic()-maskstart)
    if f%16==0 or f==463:print('T5_MASK',f,round(score,3),round(fraction,3),flush=True)
  del segmenter,net,detector;torch.cuda.empty_cache()
 verify(plan)
 save(O/'result.json',dict(stage='T5',state='draft_supervision_complete_pending_actual_review',updated=now(),seconds=time.monotonic()-start,
  source_sha256=plan['source']['sha256'],training_frame_count=464,frame_selection='all_source_frames_explicit_preregistration',native_frames=464,alpha_proposals=len(rows),
  frozen7_image_bytes_unchanged=True,mean_mask_fraction=float(np.mean([r['mask_fraction'] for r in rows])),minimum_SAM_score=min(r['SAM_score'] for r in rows),
  border_frames=[r['frame_id'] for r in rows if r['touches_frame_border']],multi_detection_frames=[r['frame_id'] for r in rows if len(r['detected_boxes_xyxy'])>1],
  RGB_valid_and_external_occlusion_masks_created=False,full_camera_registration=False,physical_admission=False,person_fit=False,GS_training=False,
  limits=['SAM scores do not prove edges, hair, fingers or feet are correct.','Main-person masking does not establish occluder visibility or physicalstaticity.','No newK/poses/materialcorrespondences or geometricerror improvement.']))
 save(O/'progress.json',dict(stage='T5',state='draft_supervision_complete_pending_actual_review',pid=None,updated=now()))
 signal.alarm(0);print('T5_DRAFT_NATIVE_SUPERVISION_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['register','run']);a=p.parse_args()
 try:globals()[a.stage]()
 except BaseException as e:
  if O.exists() and a.stage=='run':save(O/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()));save(O/'progress.json',dict(stage='T5',state='failed',pid=None,updated=now()))
  raise
