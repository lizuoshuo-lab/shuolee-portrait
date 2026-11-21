"""Reuse 001's protected native replay on the freshly fitted cold stock and tail."""

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import numpy as np

import launch_head_appearance_cold_001 as cold
import review_head_appearance_transfer_001_holdouts as common


need, read, sha, pin = cold.need, cold.read, cold.sha, cold.pin
RUN, ROOT = cold.RUN, cold.ROOT
STAGE = ROOT/'preregistration.json'
STAGE_SHA = 'a37f995242927b9486c8e9ae9ccdfd735a6e81d2552cd34ac59b2ed8499decbf'
HELPER_SHA = '40396314a3e8a62fb3693aad0d2ef903f83dc33329336a01bf78b28a39ced05b'
ORIGINAL_WRAPPER_SHA = '98fc2e20224aa75e909b0c9fa57d0659bdd7b48fc3a41cede9d18ca402e5395a'
ORIGINAL_TIMING_ERROR = 'Command-spawn, 1k, static export, or terminal panel timing is inconsistent.'


def registration():
    pin(STAGE, STAGE_SHA)
    pin(Path(common.__file__), HELPER_SHA)
    stage = read(STAGE)
    launch = read(ROOT/'launch.json')
    need(stage['schema'] == 'shuolee-head-appearance-cold-video-v1' and stage['subject'] == '001'
         and stage['candidate_commit'] == cold.COMMIT and stage['output'] == str(RUN)
         and stage['no_preprocessing_fit_or_gaussian_reuse'] is True
         and stage['expected_holdouts'] == cold.HOLDOUTS
         and stage['expected_held_source_indices'] == cold.HELD_SOURCE
         and stage['steps'] == 10000 and stage['head_steps'] == 600 and stage['sh_degree'] == 0
         and launch['preregistration_sha256'] == STAGE_SHA
         and stage['entry_command'] == launch['entry_command'] == cold.command(),
         'The declared cold launch changed.')
    pin(Path(cold.__file__), stage['wrapper_sha256'])
    for path,digest in cold.SOURCE_PINS.items():
        pin(path,digest)
    return stage


def inputs():
    stage = registration()
    spec = common.inputs()
    from shuolee_portrait import official_sequence as seq

    archive = Path(spec['pose_archive']).parent
    archived_inputs = read(archive/'inputs.json')
    archived_fit = read(archive/'fit-result.json')
    fit = read(RUN/'fit/official-fit-result.json')
    selection = read(RUN/'fit/selection.json')
    q90 = archived_inputs['frozen_q90']
    need(fit['execution_kind'] == 'fresh_official_sequence' and fit['new_fitting_started'] is True
         and fit['fitted_candidate_frames'] == stage['fit_candidates']
         and fit['selection'] == selection and selection['retained_holdouts'] == cold.HOLDOUTS
         and set(selection['selected_frames']).isdisjoint(cold.HOLDOUTS)
         and fit['source']['confidence']['per_joint_q90'] == q90
         and {r['shared_body_target_sha256'] for r in fit['frame_results']} == {archived_fit['shared_target_sha256']},
         'Cold fit target, source normalization or heldout exclusions differ.')
    source, observed = RUN/'data/inputs.json', RUN/'data/observations.npz'
    old = cold.PRIOR/'run'
    old_obs = old/'data/observations.npz'
    pin(old_obs, archived_inputs['input_sha256'][str(old_obs)])
    for path in (source, observed):
        pin(path, fit['source']['input_files'][str(path)])
    pin(RUN/'fit/fitted.npz', fit['fit_sha256'])
    pin(RUN/'fit/shared-body.json', fit['shared_body_sha256'])
    shared = read(RUN/'fit/shared-body.json')
    old_shared_path = old/'fit/shared-body.json'
    pin(old_shared_path, read(old/'fit/official-fit-result.json')['shared_body_sha256'])
    old_shared = read(old_shared_path)
    need(all(np.array_equal(shared[k], old_shared[k]) for k in ('betas', 'pose_latent'))
         and not set(cold.HOLDOUTS) & {frame for group in shared['anchor_groups'].values() for frame in group}
         and all((RUN/'fit/frames'/frame).exists() is False for frame in cold.HOLDOUTS),
         'The cold shared shape, shared pose latent or heldout fitting exclusion changed.')
    cfg = read(RUN/'config.json')
    data = seq.load_inputs(source, observed, camera_length_contract=cfg['camera_length_contract'],
        reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    need(data['confidence'] == fit['source']['confidence'] and data['scope']['retained_holdouts'] == cold.HOLDOUTS
         and all(not data['targets'][data['ids'].index(frame)].any() for frame in cold.HOLDOUTS),
         'The recomputed cold target normalization or exclusion differs.')
    pin(common.frozen.five.EVALUATOR, common.frozen.five.EVALUATOR_SHA)
    adapt = runpy.run_path(str(common.frozen.five.EVALUATOR))['with_frozen_q90']
    pin(Path(spec['pose_archive']), spec['pose_sha256'])
    old_rows = {r['frame']:r for r in read(old/'data/inputs.json')['records']}
    rows = {r['frame']:r for r in data['rows']}
    with np.load(old_obs, allow_pickle=False) as baseline, np.load(observed, allow_pickle=False) as decoded, \
            np.load(spec['pose_archive'], allow_pickle=False) as held, np.load(RUN/'fit/fitted.npz', allow_pickle=False) as fitted:
        keys = ('frame_ids','keypoints','scores','flip_disagreement','boxes','joint_names')
        need(set(baseline.files) == set(decoded.files) == set(keys)
             and all(np.array_equal(baseline[key],decoded[key]) for key in keys)
             and np.array_equal(held['shared_betas'], fitted['shared_betas'])
             and np.array_equal(held['faces'], fitted['faces'])
             and set(held['frame_ids']).isdisjoint(fitted['frame_ids']),
             'The raw observation arrays, shared beta or topology differ.')
        slots = [decoded['frame_ids'].tolist().index(frame) for frame in cold.HOLDOUTS]
        targets = adapt(seq.official, *(decoded[key][slots] for key in ('keypoints','scores','boxes','flip_disagreement')), q90)
        for i,frame in enumerate(cold.HOLDOUTS):
            row = rows[frame]
            w,h = row['wh']
            x,y = targets[i,:,:2].T
            targets[i,(x<-.5)|(x>=w-.5)|(y<-.5)|(y>=h-.5),2] = 0.
            need(all(row[key] == old_rows[frame][key] for key in common.frozen.capture.SOURCE_FIELDS)
                 and np.array_equal(held['K_integer_centers'][i],row['K_integer_centers'])
                 and np.array_equal(held['w2c'][i],row['w2c'])
                 and held['image_sha256'][i] == row['source_sha256'],
                 f'Cold native camera or source differs: {frame}.')
            pin(cold.located(row['image']),row['source_sha256'])
            pin(cold.located(row['mask']),row['mask_sha256'])
        need(np.array_equal(targets, held['observed_native_keypoints']),
             'The reconstructed 118-point holdout targets differ.')
        spec['fit_provenance'] = dict(frozen_training_fit_sha256=spec['fit_sha256'],
            new_training_fit_sha256=fit['fit_sha256'], shared_target_sha256=archived_fit['shared_target_sha256'],
            cold_observations_sha256=sha(observed), frozen_observations_sha256=sha(old_obs),
            all_70_raw_observation_arrays_equal=True, exact_shared_beta_and_pose_latent=True,
            exact_q90=True, exact_118_native_targets=True, heldouts_excluded_from_fit_and_training=True)
        for item in spec['frames']:
            frame = item['frame']
            row = rows[frame]
            item.update(image=str(cold.located(row['image'])), cold_source_mask_sha256=row['mask_sha256'],
                historical_fit_mask_sha256=str(held['mask_sha256'][item['camera_index']]),
                archived_pose_index=item['camera']['pose_index'])
            item['camera']['pose_index'] = len(fitted['frame_ids'])+item['camera_index']
        for _ in common.native_views(spec, held):
            pass
    return dict(spec, script_sha256=sha(Path(__file__)), reused_evaluator_sha256=HELPER_SHA,
                cold_preregistration_sha256=STAGE_SHA, fit_sha256=fit['fit_sha256'],
                fit_reused=False, cold_video_clock=True, selected_frames=selection['selected_frames'],
                reference_frame=selection['reference_frame'])


def candidate(spec, timing_sha):
    stage = registration()
    marker = ROOT/'wrapper-terminal.json'
    if not marker.is_file():
        need(timing_sha is None, 'The timing receipt requires the original cold wrapper.')
        return None
    wrapper = read(marker)
    if timing_sha is not None:
        pin(marker, ORIGINAL_WRAPPER_SHA)
        need(wrapper['state'] == 'incomplete' and wrapper['validation_error'] == ORIGINAL_TIMING_ERROR
             and wrapper['returncode'] == 0 and wrapper['process_status']['state'] == 'completed'
             and wrapper['process_status']['static_export_passed'] is True,
             'The preserved original has a different failure.')
        timing_path = ROOT/'timing-revalidated.json'
        pin(timing_path, timing_sha)
        timing = read(timing_path)
        immutable = ('subject', 'candidate_commit', 'returncode', 'command', 'command_spawned_at',
            'command_to_cli_end_seconds', 'prerequisite_008', 'process_status', 'preregistration_sha256',
            'application_reported_static_export_seconds', 'quality_accepted')
        need(timing['original_wrapper_sha256'] == ORIGINAL_WRAPPER_SHA
             and all(timing[key] == wrapper[key] for key in immutable),
             'The independent timing receipt does not bind the original run.')
        wrapper = timing
        spec.update(original_wrapper_sha256=ORIGINAL_WRAPPER_SHA, timing_receipt_sha256=timing_sha)
    elif wrapper['state'] == 'incomplete':
        pin(marker, ORIGINAL_WRAPPER_SHA)
        return None
    pin(RUN/'result.json', wrapper['result_sha256'])
    result = read(RUN/'result.json')
    gaussian = read(RUN/'gaussian-run/result.json')
    need(wrapper['state'] == result['state'] == 'completed' and wrapper['candidate_commit'] == cold.COMMIT
         and wrapper['preregistration_sha256'] == STAGE_SHA and wrapper['returncode'] == 0
         and wrapper['from_scratch'] is result['from_scratch'] is True
         and wrapper['fixed_fit_reused'] is wrapper['checkpoint_reused'] is False
         and result['format_passed'] is True and wrapper['selected_frames'] == spec['selected_frames']
         and result['reference_frame_id'] == wrapper['reference_frame'] == spec['reference_frame']
         and result['steps'] == gaussian['steps'] == 10000 and result['sh_degree'] == 0
         and result['final_snapshot'] == gaussian['final_snapshot'] == common.TAIL
         and result['head_appearance'] == gaussian['head_appearance'] == wrapper['head_appearance'],
         'The cold terminal does not bind the new training and completed tail.')
    pin(RUN/'fit/fitted.npz', spec['fit_sha256'])
    replay = cold.located(result['bound_replay'])
    pin(replay,result['bound_replay_sha256'])
    camera_path = cold.located(result['camera'])
    artifact = wrapper['artifacts']['camera']
    need(str(camera_path) == artifact['path'], 'The cold portable camera path differs.')
    native = RUN/'gaussian-run/training-cameras.json'
    camera_pins = {native.relative_to(RUN).as_posix():read(replay)['training_camera_sha256'],
                   camera_path.relative_to(RUN).as_posix():artifact['sha256']}
    for path,digest in camera_pins.items():
        pin(RUN/path,digest)
    return dict(stage,camera_pins=camera_pins),wrapper,result,gaussian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'terminal-check', 'render', '_inputs'))
    parser.add_argument('--timing-receipt-sha256', metavar='SHA256')
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.action == '_inputs':
        print(json.dumps(inputs()), flush=True)
        return
    payload = subprocess.check_output([sys.executable,'-B',__file__,'_inputs'],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True)
    spec = json.loads(payload.splitlines()[-1])
    finished = candidate(spec,args.timing_receipt_sha256)
    state = None
    if finished is not None:
        sys.path.insert(0, str(common.PROJECT/'src'))
        snapshots = common.launch.rebound(common.snapshots,ROOT=ROOT,RUN=RUN)
        with np.load(spec['pose_archive'], allow_pickle=False) as poses:
            state = snapshots(finished, dict(poses))
    if args.action != 'check':
        need(state is not None, 'Wait for the verified cold terminal.')
    if args.action == 'render':
        output = ROOT/'protected-holdout-transfer'
        need(not output.exists(), 'Preserve the existing cold protected receipt.')
        result = common.launch.rebound(common.render,RUN=RUN)(spec,state,output)
        print(json.dumps(dict(result=str(result),sha256=sha(result))),flush=True)
    else:
        print(json.dumps(dict(spec,terminal_ready=state is not None,
                              terminal_identity=state[-1] if state is not None else None)),flush=True)


if __name__ == '__main__':
    main()
