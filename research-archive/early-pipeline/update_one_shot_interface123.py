from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

H=Path(__file__).parent
C=H.parent/'shuolee-coordination'
P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
D=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,j): p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
paths=[P/'src/shuolee_portrait'/n for n in ['sequence_input_adapter.py','sequence_fitting.py','sequence_optimizer.py','frame_pose_contract.py','camera_world_units.py','framewise_clothed_surface.py','framewise_native_training.py']]
paths += [H/'fit_full_sequence123.py',H/'assemble_sequence_bundle123.py']
interface=dict(updated=stamp,files=[dict(path=str(p),sha256=sha(p)) for p in paths],
    fitting='load_sequence_inputs(manifest) -> SharedIdentitySequence and SequenceObservations -> optimize_sequence(evaluation_mode=streamed)',
    assembly='assemble_final_bundle(run_dir, out_new, source_run, sequence_id, project_root, path_maps) -> load_assembled_bundle',
    execution='Real94 fitting is currently running, not a complete new-video pipeline.',
    prior_scope='Current run explicitly reuses the same-video original cached GVHMR incam prediction. A newly supplied video needs fresh prediction; never substitute an old individual prior.',
    helper_scope='fit_full_sequence123.py is the research orchestration example with explicit local model paths, not a portable server command as-is.',
    unresolved=['Fresh-video observation/prior producer must connect the existing upstream inference in the single engineering entry.',
                'Final real94 fit and visual review pending; final Bundle must not be assembled from initial.npz.',
                'Real free-clothing Gaussian training and model-quality acceptance not yet completed.'],
    working_fit_runtime=dict(torch='2.7.1+cu128',smplx='0.1.28',numpy='1.26.4',scipy='1.14.1',
        note='Existing standard smplx package works; the local drivestudio fork failed before fitting due incompatible SMPLXOutput.v_shaped.'),
    image_ownership='No source video, model weights, fit arrays or user images in public Git.',
    company_delivery='Integrate into the existing smplx/reconstruction project and single video command; no separate contract projects or repeated fragment release campaigns.')
j=read(C/'one-shot-reconstruction-handoff.json');j.update(updated=stamp,integrated_input_interface=interface,
    latest_research='Real94 shared SMPL-X fitting running; native trainer preserved. Same synthetic boundary correction reduces mask violations31505 to275, not a real quality claim.')
save(C/'one-shot-reconstruction-handoff.json',j)
j=read(C/'local-research-status.json');j.update(updated_at=stamp,integrated_input_interface=interface,
    boundary_correction=dict(source_sha256=sha(P/'src/shuolee_portrait/framewise_clothed_surface.py'),
        result=str(D/'native-training-entry/boundary-correction/result.json'),CPU_only=True,real_quality_accepted=False),
    company00_latest_snapshot=dict(status='systemError',message='Selected model is at capacity. Please try a different model.',
        branch='smplx/reconstruction',retry_with_substantive_interface_message_pending=True))
save(C/'local-research-status.json',j)
print(json.dumps(dict(files=len(paths),real_fit_running=True,portable_video_pipeline_complete=False)))
