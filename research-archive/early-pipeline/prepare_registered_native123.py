"""Manifest-driven, once-only native intake. No camera, detector or fit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,traceback

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

def main(planpath):
    plan=read(planpath);out=Path(plan['output']);run=Path(plan['run_root']);project=Path(plan['project']);start=time.monotonic()
    assert not out.exists() and not run.exists(),'Existing input; inspect rather than repeat'
    video=Path(plan['video']);assert sha(video)==plan['video_sha256']
    assert sha(project/'src/shuolee_portrait/video_sfm.py')==plan['extractor_sha256']
    assert shutil.disk_usage('/mnt/c').free>8*1024**3
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]));stream=probe['streams'][0]
    assert [stream['width'],stream['height']]==plan['native_wh'] and int(stream['nb_frames'])==plan['source_frames']
    assert not stream.get('tags',{}).get('rotate') and all(float(x.get('rotation',0))==0 for x in stream.get('side_data_list',[]))
    out.mkdir();run.mkdir();(run/'logs').mkdir()
    save(out/'plan.json',dict(plan,execution_registered=now(),script_sha256=sha(__file__)))
    save(out/'progress.json',dict(state='decoding',pid=os.getpid(),command=sys.argv,training=False))
    def alarm(*_):raise TimeoutError('Native intake budget exhausted')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(plan['budget']['seconds'])
    sys.path.insert(0,str(project/'src'))
    from shuolee_portrait.video_sfm import extract
    extract(video,run,dict(frame_stride=plan['stride'],min_frames=plan['selected_frames'],max_frames=plan['selected_frames'],max_image_size=max(plan['native_wh'])))
    decoded=read(run/'decoded.json');rows=decoded['records']
    assert decoded['decoded_total']==plan['source_frames'] and len(rows)==plan['selected_frames']
    assert [r['source_frame_index'] for r in rows]==plan['selected_indices']
    assert all(r['wh']==plan['native_wh'] for r in rows)
    shutil.copy2(run/'decoded.json',out/'decoded.json');save(out/'ffprobe.json',probe)
    old=read(plan['historical_observations'])['records'];by={r['raw_source_frame_index']:r for r in old}
    import numpy as np
    from PIL import Image
    shared=[]
    for row in rows:
        i=row['source_frame_index']
        if i not in by:continue
        oldrow=by[i];a=np.array(Image.open(run/'raw'/row['name']).convert('RGB'));b=np.array(Image.open(oldrow['image']).convert('RGB'))
        assert sha(oldrow['image'])==oldrow['source_sha256'] and a.shape==b.shape
        delta=np.abs(a.astype(np.int16)-b.astype(np.int16));dt=row['time_seconds']-oldrow['pts_seconds'];assert abs(dt)<.00001
        shared.append(dict(source_frame=i,pixel_exact=bool(np.array_equal(a,b)),max_channel_difference=int(delta.max()),mean_channel_difference=float(delta.mean()),
            changed_pixels=int(np.any(delta,axis=-1).sum()),timestamp_difference_seconds=float(dt),old_sha256=oldrow['source_sha256'],new_sha256=row['source_sha256']))
    assert [r['source_frame'] for r in shared]==plan['shared_historical_indices']
    images=[]
    for slot in plan['review_slots']:
        row=rows[slot];p=out/f'slot{slot:03d}_source{row["source_frame_index"]:04d}.png';shutil.copy2(run/'raw'/row['name'],p)
        images.append(dict(slot=slot,source_frame=row['source_frame_index'],file=p.name,sha256=sha(p),wh=row['wh']))
    signal.alarm(0)
    result=dict(updated=now(),state='native_frames_prepared_review_pending',subject=plan['subject'],source_sha256=plan['video_sha256'],source_frames=plan['source_frames'],selected_frames=len(rows),
        predeclared_indices_preserved=True,wh=plan['native_wh'],shared_historical_frame_checks=shared,old_feature_observations_reused=False,
        pixel_domain='Native rectified/resized/cropped video raster; consistent FFmpeg RGBdecode; no geometric transform',images=images,
        seconds=time.monotonic()-start,run_root=str(run),features=False,cameras=False,SMPLX=False,training=False,quality_accepted=False)
    save(out/'result.json',result);save(out/'progress.json',dict(state='complete',pid=None,training=False));print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    try:main(args.plan)
    except Exception as e:
        out=Path(read(args.plan)['output'])
        if out.exists():save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),training=False))
        raise
