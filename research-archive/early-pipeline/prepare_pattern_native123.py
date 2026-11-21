"""One native-frame preflight for an unstarted subject; no camera, detection or fit."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, time, shutil, subprocess
H=Path(__file__).resolve().parent
TOP=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/local-clothed-surface-123'
OUT=REC/'pattern-preflight'
RUN=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/pattern')
PROJECT=RUN.parents[1]/'runtime'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
subject=next(x for x in read(REC/'protocol.json')['subjects'] if x['id']=='pattern')
video=Path('/mnt/d/00/工作/数据集')/subject['video']
assert not OUT.exists() and not RUN.exists() and not Path(subject['run_root']).exists()
OUT.mkdir()
plan=dict(registered=now(),operation='Decode all99 predeclared stride5 native frames of the never-started pattern subject. Verify source hash, timestamps and pixel dimensions. Inspect five evenly indexed full images. No feature detection, camera estimation, subject fitting or training.',
    source=subject,run_root=str(RUN),budget=dict(wall_seconds=180,selected_frames=99,source_frames=491,GPU_tasks=0,camera_candidates=0),
    input_domain='Already rectified/cropped MP4. No second distortion, no synthetic original fish-eye pixels, no geometry from other people.',
    stop='Any source hash, count, dimensions, rotation or timestamp discrepancy stops preparation. No repeated decoding or threshold changes.',
    permission='Existing all-subject123 authorization. Standalone input preparation does not admit camera, fit or training.',script_sha256=sha(Path(__file__)))
save(OUT/'plan.json',plan)
start=time.monotonic()
assert sha(video)==subject['video_sha256']
info=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]))
stream=info['streams'][0]
assert [stream['width'],stream['height']]==subject['source_wh']==[2704,1520]
assert int(stream['nb_frames'])==491
assert not stream.get('tags',{}).get('rotate')
assert all(float(x.get('rotation',0))==0 for x in stream.get('side_data_list',[]))
assert shutil.disk_usage('/mnt/c').free>8*1024**3
RUN.mkdir(parents=True)
(RUN/'logs').mkdir()
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.video_sfm import extract
cfg=dict(frame_stride=5,min_frames=99,max_frames=99,max_image_size=2704)
save(OUT/'progress.json',dict(updated=now(),state='decoding',pid=__import__('os').getpid(),training_started=False))
extract(video,RUN,cfg)
decoded=read(RUN/'decoded.json')
assert decoded['decoded_total']==491 and len(decoded['records'])==99
assert [x['source_frame_index'] for x in decoded['records']]==list(range(0,491,5))
assert all(x['wh']==[2704,1520] for x in decoded['records'])
assert time.monotonic()-start<180
shutil.copy2(RUN/'decoded.json',OUT/'decoded.json')
images=[]
for slot in [0,24,49,74,98]:
    r=decoded['records'][slot];dest=OUT/f'slot{slot:03d}_source{r["source_frame_index"]:04d}.png'
    shutil.copy2(RUN/'raw'/r['name'],dest)
    images.append(dict(slot=slot,source_frame=r['source_frame_index'],file=dest.name,sha256=sha(dest),wh=r['wh']))
result=dict(updated=now(),state='decoded_actual_image_review_pending',seconds=time.monotonic()-start,source_sha256=subject['video_sha256'],selected_frames=99,source_frames=491,wh=[2704,1520],
    source_indices_preserved=True,all_native_pixels=True,second_undistortion=False,camera_estimated=False,detector_run=False,SMPLX_fitted=False,training_started=False,quality_accepted=False,run_root=str(RUN),images=images)
save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='completed',pid=None,training_started=False))
print(json.dumps(result,ensure_ascii=False))
