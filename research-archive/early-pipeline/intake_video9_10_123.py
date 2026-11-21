"""Once-only native sample intake. No feature, camera, model, or training work."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, time
from PIL import Image

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'video9-10-native-intake'
SOURCE=Path('D:/Projects/Shuolee-analysis-20261028/arrived-videos-validation.json')
DATA=Path('D:/00/工作/数据集')
FF=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
def now(): return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

if O.exists(): raise RuntimeError('Intake exists: review saved evidence instead of repeating.')
e=json.loads(SOURCE.read_text(encoding='utf-8-sig')); specs=[]
for v in e['videos']:
    assert v['name'] in ['video9.mp4','video10.mp4'] and v['decode_exit_code']==0
    p=DATA/v['name']; h=sha(p); assert h==v['sha256']
    s=next(x for x in v['probe']['streams'] if x['codec_type']=='video')
    n=int(s['nb_frames']); indices=[round(i*(n-1)/8) for i in range(9)]
    specs.append(dict(name=p.name,path=str(p),sha256=h,bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,
        width=s['width'],height=s['height'],frames=n,duration=s['duration'],indices_zero_based=indices,
        probe=v['probe'],full_decode_reused=True,full_decode_evidence=str(SOURCE),
        timestamp_check=v.get('input_timestamp_check'),null_muxer_warning_is_input_nonmonotonicity=v.get('null_muxer_warning_is_input_nonmonotonicity')))
assert len(specs)==2
O.mkdir()
save(O/'plan.json',dict(registered=now(),source_script_sha256=sha(Path(__file__)),validation_sha256=sha(SOURCE),
    purpose='Check whether new sources supply expression-stable subject coverage and nearby fixed material unavailable in prior grayT controls.',
    scope='Nine predetermined uniformly spaced native frames per clip. Native pixels retained; only separate overview pages resized. Manual content review is not physical correspondence or subpixel evidence.',
    budget=dict(clips=2,frames_per_clip=9,GPU_tasks=0,seconds_per_clip=90),
    stop='No repeated extraction, new descriptors/poses/fitting/training, or automatic admission. Stop on changed source hash or decode failure. Do not infer EIS/calibration from unnamed metadata.',
    videos=specs))
t=time.time(); out=[]
save(O/'progress.json',dict(state='extracting_registered_frames',updated=now(),GPU_tasks=0))
try:
    for s in specs:
        q=O/Path(s['name']).stem; q.mkdir(); p=Path(s['path'])
        expr="select='"+'+'.join(f'eq(n,{i})' for i in s['indices_zero_based'])+"'"
        cmd=[str(FF),'-nostdin','-hide_banner','-loglevel','error','-threads','2','-noautorotate','-i',str(p),
             '-map','0:v:0','-vf',expr,'-fps_mode','passthrough','-frames:v','9',str(q/'sample_%02d.png')]
        c=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
        (q/'decode.log').write_text(c.stderr,encoding='utf-8'); assert c.returncode==0,c.stderr
        imgs=sorted(q.glob('sample_*.png')); assert len(imgs)==9
        samples=[]
        for f,i in zip(imgs,s['indices_zero_based']):
            dest=q/f'native_frame_{i:06d}.png'; f.rename(dest)
            with Image.open(dest) as im: assert im.size==(s['width'],s['height'])
            samples.append(dict(index_zero_based=i,file=str(dest.relative_to(O)),sha256=sha(dest)))
        for j in range(3):
            overview=Image.new('RGB',(1352,760*3),'#fff')
            for k in range(3):
                with Image.open(O/samples[j*3+k]['file']) as im:
                    assert im.size==(2704,1520)
                    overview.paste(im.convert('RGB').resize((1352,760),Image.Resampling.LANCZOS),(0,760*k))
            overview.save(q/f'overview_{j:02d}.jpg',quality=94)
        assert (p.stat().st_size,p.stat().st_mtime_ns)==(s['bytes'],s['mtime_ns']) and sha(p)==s['sha256']
        out.append(dict(name=s['name'],sha256=s['sha256'],samples=samples,source_unchanged=True,native_geometric_resampling=False,rotation_applied=False))
        print(json.dumps(dict(video=s['name'],samples=s['indices_zero_based'],seconds=time.time()-t)),flush=True)
    save(O/'result.json',dict(updated=now(),state='native_frames_ready_visual_review_pending',wall_seconds=time.time()-t,
        videos=out,GPU_tasks=0,new_camera=False,new_fit=False,new_training=False,calibration_validated=False,quality_accepted=False))
    save(O/'progress.json',dict(updated=now(),state='ready_for_manual_review',wall_seconds=time.time()-t,GPU_tasks=0))
except Exception as ex:
    save(O/'failure.json',dict(updated=now(),error=repr(ex),wall_seconds=time.time()-t)); raise
