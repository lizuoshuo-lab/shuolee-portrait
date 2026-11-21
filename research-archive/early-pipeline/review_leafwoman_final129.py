"""CPU-only final18000 source review; frozen1000 review code stays unchanged.

Reuse the same fit24 -> GS12 identity validation and native five-panel review.
Original terminal save precedes its last optimizer step:18000/17999 updates.
The saved snapshot capture is not claimed to be an exact-resume checkpoint.
"""
import argparse
import ast
import difflib
import hashlib
import os
from pathlib import Path

import review_leafwoman_half12_gaussian129 as previous
from run_official_gaussians123 import read,save,sha,now

OUT,FIT=previous.OUT,previous.FIT
BASE_SHA='3750943adae25a08d57030546e4bfff17f2209b3dc814080a1b7103a25c56e32'
FINAL_STEP,FINAL_UPDATES=18000,17999


def compiled_review():
    assert sha(previous.__file__)==BASE_SHA
    main,_,before=previous.compiled_review();code=before
    assert code.count('assert step == 1000')==1
    code=code.replace('assert step == 1000','assert step == FINAL_STEP')
    ast.parse(code)
    namespace=dict(main.__globals__,FINAL_STEP=FINAL_STEP)
    exec(compile(code,str(__file__)+'::same_CPU_final_source_review','exec'),namespace)
    return namespace['main'],before,code


def final_completion(step,process_exit):
    """Require terminal accounting and a new exit receipt, never the1000 exit."""
    assert step==FINAL_STEP
    process_exit=Path(process_exit)
    assert process_exit==OUT/'final-process-exit.json'
    assert process_exit!=OUT/'process-exit.json'
    process=read(process_exit)
    assert process['exit_code']==0 and process['step']==FINAL_STEP
    run=OUT/'gaussian-run';terminal=read(run/'result.json')
    assert terminal['state']=='bounded_control_completed'
    assert terminal['snapshot_step']==FINAL_STEP and terminal['optimizer_updates']==FINAL_UPDATES
    assert terminal['configured_schedule_steps']==FINAL_STEP
    snapshot=run/f'step-{step:06d}';result=read(snapshot/'result.json')
    assert result['step']==FINAL_STEP and result['frames_rendered']==previous.N
    assert result['active_sh_degree']==0 and result['points']==terminal['points']
    assert result['world_ply_sha256']==sha(snapshot/'world.ply')
    sampling=read(snapshot/'sampling.json')
    assert sampling['total_samples']==FINAL_STEP and sum(sampling['counts'])==FINAL_STEP
    assert len(sampling['counts'])==previous.N
    assert (snapshot/'checkpoint.pt').is_file()
    return snapshot,result,terminal


def review(step,process_exit):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and step==FINAL_STEP
    assert sha(previous.__file__)==BASE_SHA
    context=previous.input_context()
    snapshot,result,terminal=final_completion(step,process_exit)
    assert result['reference_frame']==context['reference']
    expected=previous.validate_prepared_subset(context,snapshot)
    paths=[previous.SOURCE_PLAN,previous.SELECTION,FIT/'selected-fit.npz',FIT/'plan.json',FIT/'result.json',
        OUT/'source-native-inputs.json',OUT/'pixel-mapping.json',OUT/'fit/fitted.npz',
        snapshot/'world.ply',snapshot/'result.json',snapshot/'current-meshes.npz',snapshot/'checkpoint.pt',
        snapshot/'sampling.json',OUT/'gaussian-run/result.json',Path(process_exit)]
    frozen={str(path):sha(path) for path in paths}
    compiled_review()[0](step)
    path=snapshot/'review/result.json';data=read(path)
    assert {row['frame'] for row in data['records']}==set(expected)
    data.update(subject='leafwoman',reference_frame=context['reference'],training_views=previous.N,
        original_fit_candidate_count=previous.FIT_N,fit24_frame_indices=context['fit24_indices'],
        fit_reference_frame=context['selection']['fit_reference_frame'],
        same_frozen_fit24_subset_no_refit=True,
        source_plan_sha256=previous.SOURCE_PLAN_SHA,selection_sha256=previous.SELECTION_SHA,
        original_terminal_save=True,global_iteration=FINAL_STEP,
        optimizer_updates_in_reviewed_snapshot=FINAL_UPDATES,
        snapshot_phase='Original terminal Scene.save before optimizer update18000;17999 optimizer updates completed.',
        snapshot_checkpoint_sha256=sha(snapshot/'checkpoint.pt'),
        exact_resume_checkpoint_claimed=False,actual_final_process_exit_code=0,
        final_process_exit_path=str(process_exit),final_process_exit_sha256=sha(process_exit),
        snapshot_point_count=result['points'],
        diagnostic_only=True,quality_accepted=False,opacity_candidate_implemented=False,
        source_flags_are_not_physical_GT=True,finger_GT_or_MAE_fabricated=False,
        unknown_knee_feet_not_rendered_as_negative_GT=True,actual_mesh_and_gaussian_limbs_not_hidden=True,
        derived_ROI_scope='Saved GS/alpha covers only pixel-mapping crop_xyxy; gray48 outside is not-rendered/unknown and metric-excluded. No alpha0 or absence-of-floaters claim outsideROI. Full actual mesh remains shown.',
        alpha_scope='Native rasterizer1-T over all retained sorted layers with truncation/early-stop; not front clothing coverage. Float32 unchanged; display clip only.',
        source_coordinates='Original integer source raster; existing ROI2x RGB/alpha BOX-downsample and paste, no geometric warp.',
        reference_warning=context['selection']['reference_supported_observation_warning'],
        continuous_novel_views=dict(state='not_rendered_not_reviewed_by_this_helper',GPU_started_by_this_helper=False,
            existing_reusable_entries=['export_cross_subject_sh0124.py','export_demo_body_snapshot123.py::replay'],
            reference_frame=context['reference'],ground_truth_available=False,old_subject_orbit_reused=False),
        frozen1000_review_helper_sha256=BASE_SHA,source_hashes=frozen,review_helper_sha256=sha(__file__))
    save(path,data)
    assert all(sha(path)==digest for path,digest in frozen.items())
    assert sha(previous.__file__)==BASE_SHA
    print('LEAFWOMAN129_FINAL_SOURCE12_GAUSSIAN_ALPHA_REVIEW_READY',step,len(data['records']),flush=True)


def audit():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and sha(previous.__file__)==BASE_SHA
    context=previous.input_context();_,before,code=compiled_review()
    assert 'assert step == FINAL_STEP' in code and 'assert step == 1000' not in code
    assert "mesh['vertices_camera'][i:i+1]" in code and "faces=mesh['faces']" in code
    # Schema-only witness from this same live experiment. Never use10000 as
    # final output and never invoke any renderer during this static audit.
    sample=OUT/'gaussian-run/step-010000/result.json';observed=None
    if sample.exists():
        observed=read(sample)
        assert observed['step']==10000 and observed['frames_rendered']==previous.N
        assert observed['reference_frame']==context['reference'] and observed['active_sh_degree']==0
        observed=dict(path=str(sample),sha256=sha(sample),keys=sorted(observed),
            use='Read-only regular snapshot schema witness, never final quality evidence.')
    save(Path(__file__).with_suffix('.audit.json'),dict(updated=now(),state='CPU_static_final129_review_ready',
        helper_sha256=sha(__file__),frozen1000_helper_sha256=BASE_SHA,
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),
        source_plan_sha256=previous.SOURCE_PLAN_SHA,selection_sha256=previous.SELECTION_SHA,
        selected_frames=context['frames'],reference_frame=context['reference'],
        fit24_frame_indices=context['fit24_indices'],training_views=previous.N,original_fit_candidate_count=previous.FIT_N,
        native_wh=list(previous.NATIVE_WH),record_id='022',only_step=FINAL_STEP,terminal_optimizer_updates=FINAL_UPDATES,
        final_exit_required=str(OUT/'final-process-exit.json'),final_snapshot_read=False,
        regular10000_schema_witness=observed,review_render_executed=False,GPU_started=False,
        training_fit_source_selection_or_frozen1000_helper_modified=False,
        current_mesh_ID_indexing_fullK_ROI_unknown_feet_rules_unchanged=True,
        original_snapshot_capture_not_claimed_exact_resume=True,quality_accepted=False,
        interface_diff=''.join(difflib.unified_diff(before.splitlines(True),code.splitlines(True),fromfile='frozen129_CPU1000_review',tofile='same129_CPUfinal18000_review'))))
    print('LEAFWOMAN129_FINAL_REVIEW_CPU_STATIC_READY',sha(__file__),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','review'])
    parser.add_argument('--step',type=int,default=FINAL_STEP)
    parser.add_argument('--process-exit',type=Path,default=OUT/'final-process-exit.json')
    args=parser.parse_args()
    audit() if args.mode=='audit' else review(args.step,args.process_exit)
