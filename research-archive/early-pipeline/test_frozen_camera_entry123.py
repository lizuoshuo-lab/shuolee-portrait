"""Engineering regressions on real frozen failures and synthetic transport.

No feature detector, matching, camera solve, human fitting or GPU is permitted.
Synthetic declarations below exercise schemas; they are never physical evidence.
"""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import tempfile,json,hashlib,sys,copy,time,traceback,ast
import numpy as np
from PIL import Image
import local_surface123_frozen_observations as mod

H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';OUT=REC/'camera-observation-entry'
O=REC/'pattern-preflight/native-material';V=O/'bridge-control/physical-review'
sys.path.insert(0,'/opt/gs-slam-lab/shuolee-local-surface-123/runtime/src')
import local_surface123 as runner
import local_surface123_camera_reviewed as camera
read=mod.read;sha=mod.sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def ref(p):return dict(path=str(p),sha256=sha(p))
def now():return datetime.now(timezone.utc).isoformat()

def make_fixture(base):
    """Three artificial rasters/feature lists, no reconstructed geometry."""
    base.mkdir();(base/'features').mkdir();(base/'raw').mkdir()
    video=base/'SYNTHETIC_NOT_A_VIDEO.bin';video.write_bytes(b'synthetic transport fixture; not real footage')
    subject=dict(id='synthetic_transport',video_sha256=sha(video),source_wh=[64,48],source_frames=11,selected_frames=3,video='unused_original_name.mp4')
    rows=[];frames=[]
    for i in range(3):
        p=base/'raw'/f'frame_{i+1:06d}.png';Image.new('RGB',(64,48),(i*20,40,80)).save(p)
        feature=base/'features'/f'{i:03d}.npz'
        np.savez(feature,xy=np.array([[10.+i,10.],[20.+i,20.],[30.+i,30.],[40.+i,20.]],np.float64))
        rows.append(dict(frame=p.stem,name=p.name,source_frame_index=i*5,time_seconds=i*.2,wh=[64,48],source_sha256=sha(p)))
        frames.append(dict(slot=i,source_frame=i*5,image=str(p),source_sha256=sha(p),feature_file=feature.name,sha256=sha(feature)))
    decoded=base/'decoded.json';save(decoded,dict(video_sha256=sha(video),decoded_total=11,frame_stride=5,records=rows))
    features=base/'feature-manifest.json';save(features,dict(frames=frames))
    records=[];review=[]
    support=base/'SYNTHETIC_NO_PHYSICAL_EVIDENCE.json';save(support,dict(synthetic=True,only_for_transport_tests=True,physical_evidence=False))
    for j in range(4):
        # Two original FIT and two HELD. The last two share a HELD family.
        family='family_fit'+str(j) if j<2 else 'family_held'
        split='FIT' if j<2 else 'HELD';sid='segment'+str(j)
        records.append(dict(id=sid,family_id=family,split=split,nodes=[[i,j] for i in range(3)],development_only=False,topologically_eligible=True))
        review.append(dict(row=j,segment_id=sid,family_id=family,split=split,development_only=False,
            classification='confirmed_static_background_material',all_nodes_actually_reviewed=True,
            exact_material_accepted=True,staticity_verified=True,localization_uncertainty_px=.2,
            actual_saved_views_reviewed=[dict(slot=i,source_frame=5*i,keypoint=j,uv=[float([10,20,30,40][j]+i),float([10,20,30,20][j])]) for i in range(3)],
            supporting_evidence=[ref(support)],note='Synthetic role declaration, not an actual physical claim.'))
    segments=base/'segments.json';save(segments,dict(records=records))
    reviewfile=base/'review.json';save(reviewfile,dict(synthetic_fixture=True,actual_saved_pixels_reviewed=True,rows=review))
    doc=dict(schema=mod.SCHEMA,subject_id=subject['id'],purpose='camera_candidate',synthetic_control=True,
        pixel_domain='native_exported_MP4_integer_centers',source_video=ref(video),decoded=ref(decoded),
        feature_manifest=ref(features),feature_root=str(base/'features'),segments=ref(segments),physical_review=ref(reviewfile))
    manifest=base/'contract.json';save(manifest,doc);subject['frozen_observations']=ref(manifest)
    return subject,doc,manifest

def main():
    started=time.monotonic();results=[]
    assert not (OUT/'result.json').exists(),'Completed engineering audit is not rerun.'
    assert sha(H/'local_surface123_camera.py')==read(OUT/'implementation.json')['old_camera_sha256']
    plan=read(REC/'camera-observation-entry-plan.json')
    assert sha(V/'review.json')==plan['review_sha256']
    assert sha(V/'consumer-audit/family-evidence.json')==plan['family_evidence_sha256']
    before={str(p):sha(p) for p in [V/'review.json',V/'consumer-audit/family-evidence.json',O/'bridge-control/segments.json',O/'feature-manifest.json']}
    def check(name,fn):
        fn();results.append(dict(name=name,passed=True));print('PASS '+name,flush=True)
    def rejected(call,reason):
        try:call()
        except mod.ObservationContractError as e:assert reason in str(e),(reason,str(e));return str(e)
        raise AssertionError('Unexpected admission: '+reason)
    subject=copy.deepcopy(next(s for s in read(REC/'protocol.json')['subjects'] if s['id']=='pattern'))
    source=Path('/mnt/d/00/工作/数据集/video4.mp4')
    doc=dict(schema=mod.SCHEMA,subject_id='pattern',purpose='engineering_replay',
        pixel_domain='native_exported_MP4_integer_centers',source_video=ref(source),
        decoded=ref(Path('/opt/gs-slam-lab/shuolee-local-surface-123/preflight/pattern/decoded.json')),
        feature_manifest=ref(O/'feature-manifest.json'),feature_root=str(O/'features'),
        segments=ref(O/'bridge-control/segments.json'),physical_review=ref(V/'review.json'))
    manifest=OUT/'real-pattern-contract.json';save(manifest,doc);subject['frozen_observations']=ref(manifest)
    bundle=mod.load_subject_input(subject)
    save(OUT/'real-source-audit.json',bundle.audit())
    audit=read(V/'consumer-audit/family-evidence.json')
    contradictions={f['family_id'] for f in audit['conflicting_families']}
    def verify_real():
        assert len(bundle.frames)==99 and len(bundle.tracks)==42609
        linked=[t for t in bundle.tracks if t.family_id in contradictions]
        assert len(linked)==10 and sum(len(t.observations) for t in linked)==35
        assert all(t.family_conflict and 'family_physical_identity_conflict' in t.fit_reasons for t in linked)
        table={(t.segment_id,o.slot,o.keypoint_id):(t,o) for t in linked for o in t.observations}
        for r in audit['linked_original_observations']:
            t,o=table[r['segment_id'],r['slot'],r['keypoint_id']]
            assert o.source_frame==r['source_frame'] and o.native_xy==tuple(r['native_xy'])
            assert o.source_image_sha256==r['source_image_sha256'] and o.feature_file_sha256==r['feature_file_sha256']
            assert t.partition==r['raw_partition'] and t.development_only==r['development_only']
    check('real_3_conflicting_families_10_segments_35_original_observations',verify_real)
    check('real_unknown_roles_are_not_background',lambda: (_ for _ in ()).throw(AssertionError()) if bundle.fit_tracks or any(t.material_role!='unknown' for t in bundle.tracks) else None)
    check('real_all_8527_HELD_tracks_preserved_outside_FIT',lambda: (_ for _ in ()).throw(AssertionError()) if len(bundle.held_tracks)!=8527 or any('heldout_cannot_fit' not in t.fit_reasons for t in bundle.held_tracks) else None)
    def real_camera_boundary():
        before_root=runner.ROOT/'pattern';assert not before_root.exists()
        error=rejected(lambda:camera.camera(before_root,bundle),'no_reviewed_static_background_FIT_tracks')
        assert not before_root.exists()
        save(OUT/'actual-camera-denial.json',dict(error=error,before_imports_and_solver=True,camera_called=False))
    check('actual_new_camera_rejects_real_unknown_and_conflicting_input_before_solve',real_camera_boundary)
    check('actual_orchestrator_missing_contract_rejected_before_new_root',lambda: rejected(lambda:runner.prepare(dict(subject, frozen_observations=None)),'frozen_observations_required'))
    check('actual_orchestrator_real_contract_rejected_before_new_root',lambda: rejected(lambda:runner.prepare(subject),'no_reviewed_static_background_FIT_tracks'))
    assert not (runner.ROOT/'pattern').exists()
    with tempfile.TemporaryDirectory(prefix='shuolee-frozen-entry-') as temp:
        root=Path(temp);s,d,m=make_fixture(root/'SYNTHETIC_ONLY')
        b=mod.load_subject_input(s)
        def transport_check():
            fit,held=mod.solver_tracks(b)
            assert [t['id'] for t in fit]==['segment0','segment1'] and [t['id'] for t in held]==['segment2','segment3']
            assert all(t['original_partition']=='FIT' for t in fit) and all(t['original_partition']=='HELD' for t in held)
            assert b.tracks[0].observations[1].native_xy==(11.,10.)
            try:b.xy[0][0,0]=99
            except ValueError:pass
            else:raise AssertionError('Native coordinates mutable')
        check('synthetic_transport_preserves_nodes_pixels_partition_and_readonly_arrays',transport_check)
        calls=[]
        def camera_spy(newroot,input_bundle):
            fit,held=mod.solver_tracks(input_bundle)
            assert sha(newroot/'decoded.json')==d['decoded']['sha256']
            assert [t['id'] for t in fit]==['segment0','segment1']
            assert len(held)==2
            for frame,row in zip(input_bundle.frames,input_bundle.decoded['records']):assert sha(newroot/'raw'/row['name'])==frame['image_sha256']
            calls.append(dict(root=str(newroot),fit_ids=[t['id'] for t in fit],held_ids=[t['id'] for t in held],actual_geometric_solver=False))
        def caller_transport():
            roots=root/'runs';roots.mkdir()
            with patch.object(runner,'ROOT',roots),patch.object(runner,'status',lambda *a,**k:None),patch.object(camera,'camera',camera_spy):
                returned=runner.prepare(s)
            assert len(calls)==1 and returned.name=='synthetic_transport'
            input_audit=read(returned/'frozen-observation-input.json')
            assert input_audit['fit_tracks']==2 and not input_audit['training_admitted']
        check('actual_prepare_calls_new_camera_with_same_frozen_input_no_decoder',caller_transport)
        # Each negative fixture mutates one source of truth, then rebinds it only
        # where the test deliberately reaches deeper structural validation.
        def edit_review(mutator,reason):
            p=Path(d['physical_review']['path']);original=p.read_bytes();r=read(p)
            try:
                mutator(r);save(p,r);dd=copy.deepcopy(d);dd['physical_review']=ref(p);save(m,dd)
                ss=dict(s,frozen_observations=ref(m));return rejected(lambda:mod.load_subject_input(ss),reason)
            finally:p.write_bytes(original);save(m,d)
        check('review_changed_native_pixel_is_rejected',lambda:edit_review(lambda r:r['rows'][0]['actual_saved_views_reviewed'][0]['uv'].__setitem__(0,10.125),'review_native_pixel_changed'))
        check('review_changed_source_frame_is_rejected',lambda:edit_review(lambda r:r['rows'][0]['actual_saved_views_reviewed'][0].__setitem__('source_frame',1),'review_source_frame_mismatch'))
        check('review_changed_partition_is_rejected',lambda:edit_review(lambda r:r['rows'][0].__setitem__('split','HELD'),'review_original_identity_mismatch'))
        def file_tamper():
            p=Path(d['feature_root'])/'000.npz';original=p.read_bytes()
            try:p.write_bytes(original+b'changed');rejected(lambda:mod.load_subject_input(s),'file_hash_mismatch')
            finally:p.write_bytes(original)
        check('feature_bytes_tampered_is_rejected',file_tamper)
        def image_tamper():
            p=Path(read(Path(d['feature_manifest']['path']))['frames'][0]['image']);original=p.read_bytes()
            try:p.write_bytes(original+b'changed');rejected(lambda:mod.load_subject_input(s),'file_hash_mismatch')
            finally:p.write_bytes(original)
        check('image_bytes_tampered_is_rejected',image_tamper)
        def mixed_family():
            p=Path(d['segments']['path']);original=p.read_bytes();r=read(p)
            try:
                r['records'][3]['split']='FIT';save(p,r);dd=copy.deepcopy(d);dd['segments']=ref(p);save(m,dd)
                rejected(lambda:mod.load_subject_input(dict(s,frozen_observations=ref(m))),'family_split_leak')
            finally:p.write_bytes(original);save(m,d)
        check('mixed_family_split_cannot_enter_loader',mixed_family)
        def role_unknown():
            p=Path(d['physical_review']['path']);original=p.read_bytes();r=read(p)
            try:
                r['rows'][0]['classification']='unresolved_physical_identity';r['rows'][1]['supporting_evidence']=[]
                save(p,r);dd=copy.deepcopy(d);dd['physical_review']=ref(p);save(m,dd)
                bb=mod.load_subject_input(dict(s,frozen_observations=ref(m)))
                assert not bb.fit_tracks and bb.tracks[0].material_role=='unknown'
                rejected(lambda:mod.solver_tracks(bb),'no_reviewed_static_background_FIT_tracks')
            finally:p.write_bytes(original);save(m,d)
        check('unknown_role_and_unbacked_claim_never_implicitly_fit',role_unknown)
        def uncertainty_missing():
            p=Path(d['physical_review']['path']);original=p.read_bytes();r=read(p)
            try:
                for row in r['rows']:row['localization_uncertainty_px']=None
                save(p,r);dd=copy.deepcopy(d);dd['physical_review']=ref(p);save(m,dd)
                bb=mod.load_subject_input(dict(s,frozen_observations=ref(m)))
                assert not bb.fit_tracks and all('measurement_uncertainty_unknown' in t.fit_reasons for t in bb.tracks)
            finally:p.write_bytes(original);save(m,d)
        check('unknown_measurement_uncertainty_stays_unknown',uncertainty_missing)
        check('manifest_tamper_rejected',lambda:rejected(lambda:mod.load_frozen_input(m,'0'*64,s),'manifest_hash_mismatch'))
        wrong=copy.deepcopy(s);wrong['video_sha256']='0'*64
        check('cross_subject_video_rejected',lambda:rejected(lambda:mod.load_subject_input(wrong),'video_identity_mismatch'))
        save(OUT/'synthetic-transport.json',dict(scope='Engineering transport with artificial images/claims and a solver spy only',camera_calls=calls,geometric_solver_calls=0,physical_calibration=False,human_training=False))
    for p,h in before.items():assert sha(Path(p))==h
    assert sha(H/'local_surface123_camera.py')==read(OUT/'implementation.json')['old_camera_sha256']
    body=(H/'local_surface123_camera_reviewed.py').read_text(encoding='utf-8');old=(H/'local_surface123_camera.py').read_text(encoding='utf-8')
    assert body[body.index('    seed_pairs=[]'):]==old[old.index('    seed_pairs=[]'):]
    calls={ast.unparse(n.func) for n in ast.walk(ast.parse(body)) if isinstance(n,ast.Call)}
    assert not any('SIFT' in n or 'knnMatch' in n or 'detectAndCompute' in n for n in calls)
    save(OUT/'code-manifest.json',dict(created=now(),purpose='Sources used by the actual repaired caller, independent of the historical solver manifest',files=[dict(name=n,sha256=sha(H/n)) for n in ['local_surface123.py','local_surface123_camera_reviewed.py','local_surface123_frozen_observations.py']]))
    result=dict(updated=now(),state='engineering_entry_checks_passed_no_camera_admitted',passed=len(results),checks=results,
        real_source_audit=bundle.audit(),original_sources_unchanged=before,
        actual_camera_entry_rejection_checked=True,actual_prepare_callsite_transport_checked=True,
        geometry_body_identical=True,geometric_solver_executed=False,feature_extraction=False,matching=False,
        positive_cases='synthetic transport only; no physical observations promoted',
        native_N3_schema_modified=False,SMPLX_fitted=False,training_started=False,quality_accepted=False,
        remaining='No real reviewed static background FIT point currently admitted; field material evidence and full camera/face admission still needed.',seconds=time.monotonic()-started)
    save(OUT/'result.json',result);print(json.dumps({k:result[k] for k in ['state','passed','seconds','quality_accepted']}),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'test-failure.json',dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),training=False))
        raise
