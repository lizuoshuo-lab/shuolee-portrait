"""One frozen cross-person intake. It cannot run a detector, optimizer or trainer."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import hashlib, json, subprocess, sys, time, traceback
import numpy as np

P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'generic-2d3d-research/adapter-a0b'
sys.path.insert(0,str(P/'src'))
from shuolee_portrait.alignment_inputs import load_alignment_inputs, read, sha

def save(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(p,**kw): return dict(path=str(p),sha256=sha(Path(p)),**kw)
def frozen(plan):
    assert all(sha(P/'src/shuolee_portrait'/name)==value for name,value in plan['baseline_source'].items()), 'Production source changed.'

plan=read(O/'plan.json'); frozen(plan)
assert not (O/'progress.json').exists(), 'Intake has already started; inspect retained evidence.'
start=time.monotonic(); results=[]
save(O/'progress.json',dict(state='running',pid=__import__('os').getpid(),command=sys.argv,done=0,total=len(plan['subjects'])))
for spec in plan['subjects']:
    tick=time.monotonic(); ident=spec['id']; out=O/ident; out.mkdir()
    base=dict(subject=ident,label=spec['label'],expected_frames=spec['expected_frames'],
              real_fitting_started=False,training_started=False,quality_accepted=False)
    if spec.get('state')=='not_available_for_this_intake':
        result=dict(base,state='missing_registered_observation_stream',reason=spec['reason'],frames_read=0)
        save(out/'result.json',result);results.append(result)
        continue
    source_files=[Path(spec[k]) for k in ('frames','wholebody','face','metadata')]
    if spec.get('reviews'):source_files.append(Path(spec['reviews']))
    source_hashes={str(p):sha(p) for p in source_files}
    try:
        rows=read(Path(spec['frames']))[spec['rows_key']]
        assert len(rows)==spec['expected_frames'], 'Full registered frame count differs.'
        with np.load(spec['face'],allow_pickle=False) as archive:
            face_values=archive[spec['coordinates_key']]
            face_uv=face_values[:,17:68] if spec['face_convention']=='ibug68' else face_values[:,40:91]
        frames=link(spec['frames'],rows_key=spec['rows_key'])
        if spec.get('reviews'):
            reviews=link(spec['reviews'],format='native_face_bindings_v1')
            generated=False
        else:
            # Explicitly missing evidence, not a synthetic annotation or a physical match.
            inventory=dict(observations_sha256=sha(Path(spec['face'])),inputs_sha256=frames['sha256'],
                coordinate_domain='native_integer_pixel_centres',synthetic_control=False,
                annotation_status='unreviewed_missing_evidence_inventory',
                generated_by='A0b intake; all states inactive; no human or model visibility review inferred',records=[])
            for i,row in enumerate(rows):
                for j in range(51):
                    ibug=j+17
                    inventory['records'].append(dict(frame=row['frame'],anchor_id=f'ibug68:{ibug}',ibug68_index=ibug,
                        uv_native=face_uv[i,j].tolist(),visibility='unknown',split='UNASSIGNED',
                        physical_family_id=f'unverified_semantic_id:ibug68:{ibug}',semantic_identity_verified=False,
                        localization_sigma_px=None,reliability=None,review_evidence=None,visibility_source='not_reviewed'))
            save(out/'unreviewed-inventory.json',inventory)
            reviews=link(out/'unreviewed-inventory.json',format='reviewed_anatomical_v1')
            generated=True
        manifest=dict(schema='shuolee-alignment-input/1',purpose='real',camera_model='rectified_pinhole',
            frames=frames,
            wholebody=link(spec['wholebody'],convention='coco_wholebody_133',names_key=spec['names_key'],coordinate_domain='native_integer_pixel_centres'),
            face=link(spec['face'],convention=spec['face_convention'],coordinates_key=spec['coordinates_key'],
                names_key=spec['names_key'],coordinate_domain='native_integer_pixel_centres',
                metadata=link(spec['metadata'],rows_key=spec['metadata_rows_key'])),reviews=reviews)
        save(out/'alignment-inputs.json',manifest);save(out/'config.json',plan['config'])
        bundle=load_alignment_inputs(out/'alignment-inputs.json')
        assert np.array_equal(bundle.face_uv,face_uv), 'Face coordinates changed.'
        with np.load(spec['wholebody'],allow_pickle=False) as archive:
            assert np.array_equal(bundle.body['keypoints'],archive['keypoints']), 'Whole-body coordinates changed.'
            assert np.array_equal(bundle.body['frame_ids'],archive['frame_ids']), 'Frame identity changed.'
        run=subprocess.run([sys.executable,str(P/'run.py'),'fit-inputs','--output',str(out)],capture_output=True,text=True,timeout=90)
        (out/'CLI.stdout.txt').write_text(run.stdout,encoding='utf-8');(out/'CLI.stderr.txt').write_text(run.stderr,encoding='utf-8')
        assert run.returncode==0, run.stderr
        report=json.loads(run.stdout)
        assert report['frames']==spec['expected_frames'] and not report['fitting_allowed']
        assert report['active_fit']==0 and report['active_held']==0
        assert report['selected_implementation']=='shuolee_portrait.fitting_direct._fit_prepared'
        save(out/'CLI-result.json',report)
        missing_body=[k for k in ('scores','boxes','flip_disagreement') if k not in bundle.body]
        result=dict(base,state='frozen_input_consumer_passed_fit_not_admitted',frames_read=len(rows),
                    face_provider=spec['face_convention'],fixed_face_observations=len(rows)*51,
                    cameras_present=report['camera_records'],camera_usage=spec['camera_usage'],
                    source_native_pixels_unchanged=True,source_frame_identity_preserved=True,
                    metadata_image_identity_consistent=True,new_source_image_bytes_rechecked=False,
                    unreviewed_inventory_generated=generated,visibility_counts=report['visibility_counts'],
                    active_fit=report['active_fit'],active_held=report['active_held'],
                    missing_body_arrays=missing_body,gaps=report['gaps'],camera_admission=report['camera_admission'],
                    config_sha256=sha(out/'config.json'),seconds=time.monotonic()-tick)
        assert result['seconds']<plan['budget']['cpu_seconds_per_subject']
        bundle.assert_unchanged()
    except Exception as error:
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        result=dict(base,state='intake_rejected',error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-tick)
    assert all(sha(Path(p))==value for p,value in source_hashes.items()), 'Input mutation.'
    assert not any((out/name).exists() for name in ('fit','inference','gaussian-run','training')), 'Unexpected compute artifact.'
    result['source_files_unchanged']=True
    save(out/'result.json',result);results.append(result)
    save(O/'progress.json',dict(state='running',pid=__import__('os').getpid(),done=len(results),total=len(plan['subjects']),last_subject=ident))
    print(json.dumps({k:v for k,v in result.items() if k not in ('camera_admission','gaps')},ensure_ascii=False),flush=True)
frozen(plan)
accepted=[r for r in results if r['state']=='frozen_input_consumer_passed_fit_not_admitted']
configs={r['config_sha256'] for r in accepted}
assert len(configs)<=1, 'Per-person configuration differs.'
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='cross_person_intake_completed',
    subjects=len(results),valid_existing_streams=len(accepted),frames_read=sum(r.get('frames_read',0) for r in results),
    fixed_face_observations=sum(r.get('fixed_face_observations',0) for r in results),
    historical_camera_records_inspected=sum(r.get('cameras_present',0) for r in results),
    one_source_hashes=plan['baseline_source'],one_config_sha256=list(configs),production_changed=False,
    missing_observation_streams=[r['subject'] for r in results if r['state']=='missing_registered_observation_stream'],
    rejected=[r['subject'] for r in results if r['state']=='intake_rejected'],
    real_fitting_runs=0,training_runs=0,camera_solves=0,detector_runs=0,GPU_used=False,quality_accepted=False,
    real_fitting_allowed=False,seconds=time.monotonic()-start,results=results)
save(O/'result.json',result)
save(O/'progress.json',dict(state='completed',pid=None,done=len(results),total=len(results),seconds=result['seconds']))
print(json.dumps({k:v for k,v in result.items() if k not in ('results','one_source_hashes')},ensure_ascii=False),flush=True)
