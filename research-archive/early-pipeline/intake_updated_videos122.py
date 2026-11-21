"""Inspect user-supplied new videos. No feature extraction, camera solve, or training."""
from pathlib import Path
import hashlib,json,subprocess,time
from datetime import datetime,timezone

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
DATA=Path('/mnt/d/00/工作/数据集');O=R/'video-update-20261027'
names=['cb3f4555-8b01.mp4','视频-777d211114ab4f39a1b5612466dda6d5.mp4','视频-dc1571c16c894d9989695137a386fb1e.mp4','视频-f5420380646f495f872c9ce9bcce5448.mp4','视频-fc2a55a8ce9d4263b1ebcb7915a6b18c.mp4']
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

if O.exists():raise RuntimeError('Intake already exists; inspect it instead of repeating')
O.mkdir();before=[]
for name in names:
    p=DATA/name;st=p.stat();before.append(dict(filename=name,path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(p)))
save(O/'intake-plan.json',dict(registered=now(),source_sha256=sha(Path(__file__)),user_update='User identified updated winter-hat portrait and confirmed dataset synchronization.',
    files=before,scope='Read container metadata and decode exact first/middle/last encoded frames. No geometric resampling or autorotation, no inherited old K/pose/feature/geometry. Confirm input content before planning any diagnostic.',
    uncertainty='New clip identity and processing relation must be established from content/metadata. Metadata alone does not prove calibration or geometric processing.',
    budget=dict(videos=5,decoded_samples_per_video=3,subprocess_timeout_seconds=120,GPU_tasks=0)))
result=[]
for i,spec in enumerate(before,1):
    p=Path(spec['path']);out=O/f'clip{i:02d}';out.mkdir()
    prob=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],timeout=30))
    save(out/'ffprobe.json',prob)
    v=next(s for s in prob['streams'] if s['codec_type']=='video');n=int(v['nb_frames']);picks=[0,n//2,n-1]
    expr='select='+"'"+'+'.join(f'eq(n,{f})' for f in picks)+"'"
    cmd=['/usr/bin/ffmpeg','-nostdin','-hide_banner','-loglevel','error','-noautorotate','-i',str(p),'-vf',expr,'-vsync','0','-frames:v','3',str(out/'sample_%02d.png')]
    cp=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (out/'decode.log').write_text(cp.stderr)
    if cp.returncode:raise RuntimeError(f'Decode failed:{spec["filename"]}')
    images=sorted(out.glob('sample_*.png'));assert len(images)==3
    after=p.stat();assert (after.st_size,after.st_mtime_ns)==(spec['bytes'],spec['mtime_ns']) and sha(p)==spec['sha256']
    entry=dict(**spec,id=f'clip{i:02d}',width=v['width'],height=v['height'],frames=n,r_frame_rate=v.get('r_frame_rate'),avg_frame_rate=v.get('avg_frame_rate'),
        duration=v.get('duration',prob['format'].get('duration')),codec=v.get('codec_name'),pix_fmt=v.get('pix_fmt'),sample_aspect_ratio=v.get('sample_aspect_ratio'),
        side_data=v.get('side_data_list',[]),tags=v.get('tags',{}),samples=[dict(frame=f,file=str(a.relative_to(O)),sha256=sha(a)) for f,a in zip(picks,images)],
        source_changed_during_read=False,rotation_applied=False,geometric_resampling=False,content_review='pending',intrinsic_or_pose_reuse=False)
    result.append(entry);print(json.dumps({k:entry[k] for k in ['id','filename','width','height','frames','duration','codec','side_data']},ensure_ascii=False),flush=True)
save(O/'result.json',dict(updated=now(),state='metadata_and_native_samples_ready',videos=result,old_online4_sha256='2f6c85b516da974a39986e7567cffedec0ed9a226b10c6be7ef07dbc39bf1b32',
    new_training_started=False,new_camera_estimated=False,original_files_modified=False,new_input_domain_validated=False))
