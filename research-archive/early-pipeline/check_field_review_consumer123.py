"""Apply the completed field review to the existing frozen observation entry."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,signal,sys,traceback
import local_surface123_frozen_observations as mod
import local_surface123_camera_reviewed as camera
H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';F=REC/'field-preflight'
O=F/'native-material';B=O/'topology-review';V=B/'background-full-review';OUT=V/'consumer-replay'
read=mod.read;sha=mod.sha
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ref(p):return dict(path=str(p),sha256=sha(p))
def now():return datetime.now(timezone.utc).isoformat()

def main():
    assert not OUT.exists(),'Completed replay must not be repeated.'
    plan=read(V/'consumer-plan.json')
    for n,h in plan['sources'].items():assert sha(H/n)==h,n
    assert sha(V/'review.json')==plan['review_sha256'] and sha(V/'index.json')==plan['index_sha256']
    root=Path('/opt/gs-slam-lab/shuolee-local-surface-123/field');assert not root.exists()
    OUT.mkdir();started=time.monotonic()
    def timeout(*_):raise TimeoutError('120-second real field consumer replay budget')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
    source=Path(plan['source_video']);subject=plan['subject'].copy()
    assert sha(source)==subject['video_sha256']
    doc=dict(schema=mod.SCHEMA,subject_id='field',purpose='engineering_replay',pixel_domain='native_exported_MP4_integer_centers',
        source_video=ref(source),decoded=ref(F/'decoded.json'),feature_manifest=ref(O/'feature-manifest.json'),
        feature_root=str(O/'features'),segments=ref(B/'segments.json'),physical_review=ref(V/'review.json'))
    manifest=OUT/'real-field-contract.json';save(manifest,doc);subject['frozen_observations']=ref(manifest)
    bundle=mod.load_subject_input(subject)
    review=read(V/'review.json');byid={t.segment_id:t for t in bundle.tracks}
    verified=[]
    for r in review['rows']:
        t=byid[r['segment_id']]
        assert t.partition==r['split'] and t.family_id==r['family_id'] and t.development_only==r['development_only']
        assert len(t.observations)==len(r['actual_saved_views_reviewed'])
        original={(o.slot,o.keypoint_id):o for o in t.observations}
        for v in r['actual_saved_views_reviewed']:
            o=original[(v['slot'],v['keypoint'])]
            assert o.source_frame==v['source_frame'] and o.native_xy==tuple(v['uv'])
        assert 'incomplete_node_review' not in t.fit_reasons
        assert 'measurement_uncertainty_unknown' in t.fit_reasons and 'staticity_unknown' in t.fit_reasons
        assert t.material_role=='unknown' and not t.family_conflict
        verified.append(dict(segment_id=t.segment_id,family_id=t.family_id,original_partition=t.partition,
            native_observations_preserved=len(t.observations),all_saved_views_consumed=True,fit_reasons=t.fit_reasons))
    assert len(bundle.frames)==177 and sum(v['native_observations_preserved'] for v in verified)==102
    assert not bundle.fit_tracks
    try:camera.camera(root,bundle)
    except mod.ObservationContractError as e:
        denial=str(e);assert denial=='no_reviewed_static_background_FIT_tracks'
    else:raise AssertionError('Unsupported real field input reached camera solver')
    assert not root.exists() and 'cv2' not in sys.modules and 'scipy.optimize' not in sys.modules
    for n,h in plan['sources'].items():assert sha(H/n)==h,n
    assert sha(V/'review.json')==plan['review_sha256']
    save(OUT/'source-audit.json',bundle.audit());save(OUT/'reviewed-node-consumption.json',dict(records=verified))
    result=dict(updated=now(),state='real_field_review_consumed_camera_not_admitted',
        frames=len(bundle.frames),all_original_tracks=len(bundle.tracks),all_original_observations=sum(len(t.observations) for t in bundle.tracks),
        reviewed_families=10,reviewed_observations=102,admissible_fit_tracks=len(bundle.fit_tracks),
        held_tracks_preserved=len(bundle.held_tracks),actual_camera_entry='local_surface123_camera_reviewed.camera',
        actual_camera_denial=denial,geometric_solver_imported=False,runroot_created=False,
        previously_implemented_entry_reused=True,new_runtime_source_changes=False,
        geometry_calls=0,SMPLX_calls=0,training_calls=0,quality_accepted=False,
        bound_native_data_preserved=True,seconds=time.monotonic()-started,
        next='No new candidate from this102-view sample. A solve needs exact static material observations with independent localization support and enough full177-frame coverage; current semantic bindings alone do not provide this.')
    save(OUT/'result.json',result);signal.alarm(0)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc()))
        raise
