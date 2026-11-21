"""Once-only read-only audit of the registered pattern consumer gap.

No camera, renderer, detector or trainer is called. Existing APIs are invoked
only at their CPU validation boundary, preserving the original FIT/HELD labels.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import ast,hashlib,json,sys,time
import numpy as np
from material_observations122 import Observation,validate_observation,SPLITS
from local_surface123_alignment_gate import evaluate,require_camera_admission

H=Path(__file__).resolve().parent
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';O=REC/'pattern-preflight/native-material'
B=O/'bridge-control';V=B/'physical-review';D=V/'consumer-audit'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def now():return datetime.now(timezone.utc).isoformat()

def main():
    assert not D.exists(),'Existing audit must be reviewed, not rerun.'
    plan=read(V/'consumer-evidence-plan.json')
    paths={'review.json':V/'review.json','contradiction-trace.json':V/'contradiction-trace.json',
           'segments.json':B/'segments.json','feature-manifest.json':O/'feature-manifest.json'}
    for name,p in paths.items():assert sha(p)==plan['inputs'][name],name
    D.mkdir();start=time.monotonic()
    sources={n:sha(H/n) for n in ['material_observations122.py','material_geometry122.py',
        'native_training_admission122_v2.py','local_surface123.py','local_surface123_camera.py','local_surface123_alignment_gate.py']}
    save(D/'execution.json',dict(started=now(),source_sha256=sha(Path(__file__)),plan_sha256=sha(V/'consumer-evidence-plan.json'),
        inspected_source_hashes=sources,cpu_only=True,geometry_calls=0,training_calls=0))
    review=read(V/'review.json');index=read(V/'index.json');segments=read(B/'segments.json')['records']
    bad={r['family_id']:r for r in review['rows'] if r['classification']=='confirmed_physical_identity_contradiction'}
    assert len(bad)==3 and all(r['split']=='FIT' for r in bad.values())
    linked=[s for s in segments if s['family_id'] in bad]
    manifest=read(O/'feature-manifest.json')['frames']
    files={};obs=[]
    for s in linked:
        for slot,kp in s['nodes']:
            if slot not in files:
                f=O/'features'/manifest[slot]['feature_file'];assert sha(f)==manifest[slot]['sha256']
                files[slot]=np.load(f,allow_pickle=False)['xy']
            obs.append(dict(source_frame=manifest[slot]['source_frame'],slot=slot,keypoint_id=kp,
                source_image_sha256=manifest[slot]['source_sha256'],feature_file_sha256=manifest[slot]['sha256'],
                native_xy=files[slot][kp].tolist(),family_id=s['family_id'],segment_id=s['id'],
                raw_partition=s['split'],development_only=s['development_only'],
                family_contains_identity_contradiction=True,point_identity='unknown',
                point_level_contradiction_asserted=False))
    # This sidecar records family-level conflict. It does not relabel every
    # observation as wrong, rewrite raw partitions, or claim consumer support.
    sidecar=dict(schema='shuolee-pattern-family-evidence-audit-v1',purpose='audit_only_not_training_input',
        source_video_sha256=read(O/'decision.json')['source_sha256'],
        source_segments_sha256=sha(B/'segments.json'),source_review_sha256=sha(V/'review.json'),
        unchanged_raw_partitions=['FIT','HELD'],default_unreviewed_identity='unknown',
        conflicting_families=[dict(family_id=f,review_row=r['row'],raw_partition=r['split'],
            evidence='physical-review/review.json',identity_state='family_conflict_not_all_points_wrong') for f,r in bad.items()],
        linked_original_observations=obs,downstream_consumer_connected=False)
    save(D/'family-evidence.json',sidecar)
    probes=[]
    for rowid in [9,12,40]:
        r=index['rows'][rowid];v=r['views'][0]
        o=Observation(source_sha256=sidecar['source_video_sha256'],frame_id=v['source_frame'],track_id=rowid,
            frozen_family_id=r['family_id'],frozen_split=r['split'],group='unassigned_native_ROI_candidate',
            raw_uv=tuple(v['uv']),width=2704,height=1520,tracker_numeric_validity=True)
        try:validate_observation(o)
        except ValueError as e:probes.append(dict(row=rowid,raw_partition=r['split'],accepted=False,error=str(e)))
        else:raise AssertionError('Expected current API to reject unconverted raw partition')
    camera_source=(H/'local_surface123_camera.py').read_text(encoding='utf-8')
    tree=ast.parse(camera_source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='camera')
    calls=sorted({ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n,ast.Call)})
    literal_strings={n.value for n in ast.walk(fn) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    hooks=[n for n in ['material_observations122','physical-review','segments.json','camera_fit_permission'] if n in camera_source]
    assert not hooks and 'sift.detectAndCompute' in calls and 'matcher.knnMatch' in calls
    p=Path('/opt/gs-slam-lab/shuolee-local-surface-123/pattern')
    existed=p.exists();guard=evaluate(p,phase='camera')
    assert not guard['fitting_allowed'] and not guard['training_allowed']
    try:require_camera_admission(p)
    except RuntimeError as e:denied=str(e)
    else:raise AssertionError('Actual fit gate unexpectedly allowed pattern')
    assert p.exists()==existed and 'torch' not in sys.modules
    # The schema rejects raw FIT for missing role semantics; this is explicitly
    # not evidence that it consumed the three physical contradictions.
    result=dict(updated=now(),state='stopped_at_registered_consumer_gap',
        family_conflicts=len(bad),linked_segments=len(linked),linked_observations=len(obs),
        raw_partitions_preserved=True,raw_xy_preserved=True,raw_source_files_changed=False,
        current_N3_allowed_roles=sorted(SPLITS),original_partition_schema_probes=probes,
        physical_contradiction_checked_by_N3=False,
        schema_gap='Raw FIT/HELD contains both person and scene candidates. Assigning fit_background or heldout_human requires separately supported role semantics; an ROI is not sufficient.',
        camera_source_audit=dict(path=str(H/'local_surface123_camera.py'),sha256=sources['local_surface123_camera.py'],
            entry_line=fn.lineno,fresh_extraction=True,fresh_matching=True,reads_reviewed_segments=False,evidence_hooks_found=hooks,
            scope='Inspected this actual source function and its calls. It was not executed.'),
        candidate_run_root=str(p),candidate_run_root_exists=existed,
        existing_camera_admission_guard=guard,actual_require_camera_admission_error=denied,
        direct_consumer_integration_completed=False,new_exported_training_input=False,
        no_label_coercion=True,no_camera_solver_called=True,no_SMPLX_or_prior_called=True,no_training_called=True,
        physical_accuracy_or_alignment_improved=False,
        reason='The registered stop condition applies: no consumer of these saved pattern segments exists in the selected camera route, and direct N3 conversion changes role semantics.',
        next='Retain original evidence and stop this adapter without claiming the loop filter reached fitting. Audit remaining goal requirements and still-actionable unused subject budgets before any new experiment.',
        seconds=time.monotonic()-start)
    for name,pth in paths.items():assert sha(pth)==plan['inputs'][name]
    for name,h in sources.items():assert sha(H/name)==h
    save(D/'result.json',result)
    print(json.dumps({k:result[k] for k in ['state','family_conflicts','linked_segments','linked_observations','candidate_run_root_exists','actual_require_camera_admission_error','direct_consumer_integration_completed','seconds']},ensure_ascii=False))

if __name__=='__main__':main()
