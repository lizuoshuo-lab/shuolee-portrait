"""One registered native decode of the existing unused field subject.

No feature, camera, prior, fit or training. Reuses the existing decoder contract.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess,sys,time,signal,traceback
H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'; REC=R.parent/'local-clothed-surface-123'
OUT=REC/'field-preflight'; RUN=Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/field')
PROJECT=RUN.parents[1]/'runtime'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    subject=next(s for s in read(REC/'protocol.json')['subjects'] if s['id']=='field')
    mapping=next(s for s in read(REC/'source-name-reconciliation.json')['subjects'] if s['subject']=='field')
    assert len(mapping['exact_matches'])==1
    match=mapping['exact_matches'][0]; video=Path('/mnt/d')/match['path'].replace('\\','/')[3:]
    assert subject['video_sha256']==mapping['expected_sha256']==match['sha256']
    assert not OUT.exists() and not RUN.exists() and not Path(subject['run_root']).exists()
    OUT.mkdir()
    plan=dict(registered=now(),source=subject,verified_source_path=str(video),source_identity='Full SHA binding to existing all-person123 protocol; only renamed video7.mp4.',
        stage='native_input_preflight_only',previous_turn_classification='progress:212 material strips reviewed;3 contradictoryFIT segments traced;report/viewer/PPT updated',
        current_turn_completed='Registered pattern consumer audit stopped at missing consumer/role-semantics gap; no pattern solver or label coercion.',
        hypothesis='Before any new per-person candidate, check exact raster/PTS and whether head, feet, and static non-person structures are actually visible. This is an intake audit, not a camera-model comparison.',
        selected_frames=list(range(0,882,5)),review_slots=[0,44,88,132,176],
        operation='Decode existing177 predeclared stride5 native frames once; copy5 predefined fullframe PNGs unchanged for actual review.',
        budget=dict(wall_seconds=180,decode_passes=1,GPU_tasks=0,features=0,cameras=0,fit=0,training=0),
        input_domain='Already rectified/resized/cropped MP4; no second undistortion, rotation or geometry from another subject.',
        stop='Any hash, count, timestamp, raster or rotation mismatch stops preparation. No decoder rerun or new feature plan until actual review.',
        permission='Existing all-person123 authorization. No new subject or training candidate added.',script_sha256=sha(Path(__file__)))
    save(OUT/'plan.json',plan); start=time.monotonic()
    assert sha(video)==subject['video_sha256']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(video)]))
    stream=probe['streams'][0]
    assert [stream['width'],stream['height']]==subject['source_wh']==[2704,1520]
    assert int(stream['nb_frames'])==subject['source_frames']==882
    assert not stream.get('tags',{}).get('rotate') and all(float(x.get('rotation',0))==0 for x in stream.get('side_data_list',[]))
    assert shutil.disk_usage('/mnt/c').free>8*1024**3
    RUN.mkdir(parents=True);(RUN/'logs').mkdir()
    sys.path.insert(0,str(PROJECT/'src'))
    from shuolee_portrait.video_sfm import extract
    def alarm(*_):raise TimeoutError('Registered180second intake budget exceeded')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(180)
    save(OUT/'progress.json',dict(updated=now(),stage='decoding',pid=os.getpid(),training=False))
    extract(video,RUN,dict(frame_stride=5,min_frames=177,max_frames=177,max_image_size=2704))
    decoded=read(RUN/'decoded.json');rows=decoded['records']
    assert decoded['decoded_total']==882 and len(rows)==177
    assert [r['source_frame_index'] for r in rows]==plan['selected_frames']
    assert all(r['wh']==[2704,1520] for r in rows)
    assert time.monotonic()-start<180
    shutil.copy2(RUN/'decoded.json',OUT/'decoded.json');save(OUT/'ffprobe.json',probe)
    images=[]
    for slot in plan['review_slots']:
        r=rows[slot];p=OUT/f'slot{slot:03d}_source{r["source_frame_index"]:04d}.png';shutil.copy2(RUN/'raw'/r['name'],p)
        assert sha(p)==r['source_sha256']
        images.append(dict(slot=slot,source_frame=r['source_frame_index'],file=p.name,sha256=sha(p),wh=r['wh']))
    signal.alarm(0)
    result=dict(updated=now(),state='decoded_actual_image_review_pending',source_sha256=subject['video_sha256'],selected_frames=177,
        source_frames=882,wh=[2704,1520],images=images,run_root=str(RUN),seconds=time.monotonic()-start,
        no_second_undistortion=True,all_registered_source_indices_preserved=True,features=False,cameras=False,SMPLX=False,training=False,quality_accepted=False)
    save(OUT/'result.json',result);save(OUT/'progress.json',dict(updated=now(),stage='complete',pid=None,training=False))
    for root in [R,REC]:
        q=read(root/'queue-state.json');q.update(updated=now(),current_stage='field_native_input_actual_review',current_subject='field',running=None,worker_pid=None,
            next='Actually inspect the five registered field source frames before any feature or geometry work.',latest_evidence=str(OUT/'result.json'),training_running=False)
        if root==REC:
            for s in q['subjects']:
                if s['id']=='field':s.update(stage='native_input_review_pending',preflight_root=str(RUN))
        save(root/'queue-state.json',q)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),training=False))
        raise
