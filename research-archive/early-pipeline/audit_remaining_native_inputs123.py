"""Bounded read-only audit of existing original videos and historical coverage.

No decoder, feature extractor, geometry solver or neural inference is called.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,time
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';HIST=REC.parent/'raw-domain-121'
OUT=REC/'remaining-native-input-audit'
RUN=Path('/opt/gs-slam-lab/shuolee-local-surface-123')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def now():return datetime.now(timezone.utc).isoformat()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not OUT.exists(),'One bounded inventory, never overwrite.'
    protocol=read(REC/'protocol.json');subjects=[s for s in protocol['subjects'] if s['id'] in ('grass','indoor')]
    assert len(subjects)==2
    OUT.mkdir();start=time.monotonic()
    plan=dict(registered=now(),purpose='Compare already authorized full123 frame coverage with failed121 diagnostic subsets before deciding whether any new native intake has additional evidence.',
        subjects=['grass','indoor'],review_slots=[0,6,12,18,23],budget=dict(seconds=60,video_decodes=0,features=0,cameras=0,SMPLX=0,training=0),
        protections='No historical pose, optimized point, descriptor or held observation promoted to fit. No changes to121. Hash failures stop the audit.',
        actual_review_required=True,new_source_material_expected=False,script_sha256=sha(Path(__file__)))
    save(OUT/'plan.json',plan);results=[]
    for s in subjects:
        sid=s['id'];video=Path('/mnt/d/00/工作/数据集')/('video5.mp4' if sid=='grass' else 'video8.mp4')
        assert sha(video)==s['video_sha256']
        oldroot=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')/sid
        op=oldroot/'observations.json';obs=read(op);records=obs['records'];assert len(records)==24
        assert all(r['original_video_sha256']==s['video_sha256'] and r['wh']==s['source_wh'] for r in records)
        old=[int(r['raw_source_frame_index']) for r in records];new=list(range(0,s['source_frames'],5))
        assert len(new)==s['selected_frames']
        oldresult=read(HIST/f'{sid}-global-fixedk.json')
        assert not oldresult['training_started'] and oldresult['registered_count']==0
        assert not (RUN/sid).exists() and not (RUN/'preflight'/sid).exists()
        samples=[]
        for slot in plan['review_slots']:
            r=records[slot];src=Path(r['image']);assert sha(src)==r['source_sha256']
            dest=OUT/f'{sid}_slot{slot:02d}_source{r["raw_source_frame_index"]:04d}.png';shutil.copy2(src,dest)
            assert sha(dest)==r['source_sha256']
            samples.append(dict(slot=slot,source_frame=r['raw_source_frame_index'],file=dest.name,sha256=r['source_sha256'],wh=r['wh']))
        results.append(dict(subject=sid,video=str(video),video_sha256=s['video_sha256'],same_historical_video=True,
            historical_indices=old,predeclared123_indices=new,shared_indices=sorted(set(old)&set(new)),
            previously_unexamined123_indices=sorted(set(new)-set(old)),historical_max_frame_gap=max(b-a for a,b in zip(old,old[1:])),
            proposed_frame_gap=5,historical_observations_sha256=sha(op),historical_result_sha256=sha(HIST/f'{sid}-global-fixedk.json'),
            historical_reason=oldresult['reason'],historical_registered_cameras=0,samples=samples,full123_preflight_exists=False))
    assert time.monotonic()-start<60
    result=dict(updated=now(),state='source_and_coverage_verified_actual_image_review_pending',subjects=results,
        genuine_new_video=False,newly_selected_frames_are_not_new_capture=True,
        limitation='Higher temporal density is a potential information change, not evidence that new rigid materials exist or that calibration is correct.',
        decoder_calls=0,feature_calls=0,geometry_calls=0,training_started=False,quality_accepted=False,seconds=time.monotonic()-start)
    save(OUT/'result.json',result)
    print(json.dumps(dict(state=result['state'],subjects=[dict(subject=r['subject'],old=len(r['historical_indices']),predeclared=len(r['predeclared123_indices']),shared=len(r['shared_indices']),new_indices=len(r['previously_unexamined123_indices']),old_max_gap=r['historical_max_frame_gap'],new_gap=5) for r in results],seconds=result['seconds']),ensure_ascii=False))
if __name__=='__main__':main()
