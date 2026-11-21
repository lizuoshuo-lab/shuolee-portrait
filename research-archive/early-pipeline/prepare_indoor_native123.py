"""Once-only existing-video177 input preparation. No model or geometric fit."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time,os,sys,subprocess,signal,traceback
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';A=REC/'remaining-native-input-audit';OUT=REC/'indoor-preflight'
ROOT=Path('/opt/gs-slam-lab/shuolee-local-surface-123');RUN=ROOT/'preflight/indoor';PROJECT=ROOT/'runtime'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    plan=read(A/'indoor-intake-plan.json');start=time.monotonic()
    assert not OUT.exists() and not RUN.exists() and not (ROOT/'indoor').exists()
    video=Path('/mnt/d/00/工作/数据集/video8.mp4');assert sha(video)==plan['video_sha256']
    assert shutil.disk_usage('/mnt/c').free>8*1024**3
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]))
    s=probe['streams'][0]
    assert [s['width'],s['height']]==plan['native_wh'] and int(s['nb_frames'])==884
    assert not s.get('tags',{}).get('rotate') and all(float(x.get('rotation',0))==0 for x in s.get('side_data_list',[]))
    OUT.mkdir();RUN.mkdir();(RUN/'logs').mkdir()
    save(OUT/'plan.json',dict(plan,execution_registered=now(),script_sha256=sha(Path(__file__)),extractor_sha256=sha(PROJECT/'src/shuolee_portrait/video_sfm.py')))
    save(OUT/'progress.json',dict(updated=now(),state='decoding',pid=os.getpid(),command=sys.argv,training=False))
    def alarm(*_):raise TimeoutError('Registered240second budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(240)
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait.video_sfm import extract
    extract(video,RUN,dict(frame_stride=5,min_frames=177,max_frames=177,max_image_size=2704))
    decoded=read(RUN/'decoded.json');rows=decoded['records']
    assert decoded['decoded_total']==884 and len(rows)==177
    assert [r['source_frame_index'] for r in rows]==plan['selected_indices']
    assert all(r['wh']==plan['native_wh'] for r in rows)
    shutil.copy2(RUN/'decoded.json',OUT/'decoded.json');save(OUT/'ffprobe.json',probe)
    old=read(Path('/opt/gs-slam-lab/shuolee-raw-domain-121/indoor/observations.json'))['records']
    oldby={r['raw_source_frame_index']:r for r in old};shared=[]
    import numpy as np
    from PIL import Image
    for r in rows:
        if r['source_frame_index'] not in oldby:continue
        o=oldby[r['source_frame_index']];src=RUN/'raw'/r['name'];a=np.array(Image.open(src).convert('RGB'));b=np.array(Image.open(o['image']).convert('RGB'))
        assert sha(Path(o['image']))==o['source_sha256'] and a.shape==b.shape
        d=np.abs(a.astype(np.int16)-b.astype(np.int16))
        shared.append(dict(source_frame=r['source_frame_index'],old_file_sha256=o['source_sha256'],new_file_sha256=r['source_sha256'],
            pixel_exact=bool(np.array_equal(a,b)),max_channel_difference=int(d.max()),mean_channel_difference=float(d.mean()),
            changed_pixels=int(np.any(d,axis=-1).sum()),timestamp_difference_seconds=float(r['time_seconds']-o['pts_seconds'])))
        assert abs(r['time_seconds']-o['pts_seconds'])<.00001
    assert len(shared)==4
    images=[]
    for slot in plan['review_slots']:
        r=rows[slot];p=OUT/f'slot{slot:03d}_source{r["source_frame_index"]:04d}.png';shutil.copy2(RUN/'raw'/r['name'],p)
        assert sha(p)==r['source_sha256']
        images.append(dict(slot=slot,source_frame=r['source_frame_index'],file=p.name,sha256=r['source_sha256'],wh=r['wh']))
    signal.alarm(0)
    result=dict(updated=now(),state='native177_prepared_actual_review_pending',source_sha256=plan['video_sha256'],source_frames=884,selected_frames=177,
        predeclared_indices_preserved=True,wh=plan['native_wh'],shared_historical_frame_checks=shared,
        old_feature_observations_reused=False,pixel_domain='Native rectified/resized/cropped video raster; FFmpeg RGBdecode; no further geometric transform',
        images=images,seconds=time.monotonic()-start,run_root=str(RUN),features=False,cameras=False,SMPLX=False,training=False,quality_accepted=False)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None,training=False))
    print(json.dumps(dict(state=result['state'],frames=177,seconds=result['seconds'],historical_pixel_parity=shared,training=False),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),training=False))
            save(OUT/'progress.json',dict(updated=now(),state='failed_preserved',pid=None,training=False))
        raise
