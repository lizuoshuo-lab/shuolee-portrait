"""Run the existing protected stock/tail evaluator on two registered 005 views."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

import review_head_appearance_transfer_001_holdouts as common


need, read, sha, pin = common.need, common.read, common.sha, common.pin
ROOT = common.launch.root_for('005')
RUN = ROOT/'run'
PLAN = common.PLAN.with_name('smpl-head-appearance-transfer-extension-preregistration.json')
PLAN_SHA = 'c9b0d89bb6d5b9c7b6693c3cc0abbcf76e5aaf232e6b3530ce396c5662f7c913'
HELPER_SHA = '40396314a3e8a62fb3693aad0d2ef903f83dc33329336a01bf78b28a39ced05b'
FRAMES = common.frozen.PRIMARY['005']
OCCLUDED_EYES = {'frame_000042': ['eye_42_47']}
REGION_PINS = {
    'frame_000027': dict(face='1c5c2d8aebf16bc2f7f912117ffa3df657ad552f589db0b98930717701534ea4',
        eye_36_41='aaf354db86127b51d424c7a871df490da363f2a0a1ffb30378f6058d5f5cb6aa',
        eye_42_47='ee043ec7536b8d829883b601c40cfd0009d8ab78a37b78d91972da76225e0c82'),
    'frame_000042': dict(face='1b48719faed84ddfba3b2a2176335e83c766c56368f579d034af7da743331b13',
        eye_36_41='ad2d130623f077c3af6bddf20bc8fb76f78bb81279bcfc2228d1f220d811467b',
        eye_42_47='381d83e9fbb6066429fec716da45256fabb933d699347c62cfeeba7636fd9ec8'),
}


def registration():
    pin(PLAN, PLAN_SHA)
    pin(Path(common.__file__), HELPER_SHA)
    plan = read(PLAN)
    subject = plan['subjects']['005']
    need(plan['state'] == 'registered_before_005_008_common_candidate_training'
         and plan['baseline_commit'] == common.frozen.queue.COMMIT
         and plan['prior_transfer_plan_sha256'] == common.PLAN_SHA
         and plan['one_shared_profile'] is True and plan['no_refit'] is True
         and plan['body_steps'] == 10000 and plan['head_appearance_steps'] == 600
         and plan['face_weight'] == .05 and plan['sh_degree'] == 0
         and plan['holdout_gates'] == dict(rgb_mae_max_regression=common.LIMITS['rgb'],
             face_mae_max_regression=common.LIMITS['face_rgb'],
             exterior_alpha_half_relative_increase_max=common.LIMITS['exterior'],
             far_distance_strictly_over_px=24, far_added_max=common.LIMITS['far_added'])
         and subject['formal_holdouts'] == FRAMES, 'The registered 005 protected scope changed.')
    for name in ('baseline_terminal', 'baseline_wrapper', 'fit', 'training_inputs', 'selection',
                 'prior_formal_receipt', 'training_cameras', 'portable_camera'):
        item = subject[name]
        pin(Path(item['path']), item['sha256'])
    return dict(subject, baseline_holdout_receipt=subject['prior_formal_receipt'],
                candidate_commit=plan['candidate_commit'])


def native_views(spec, poses):
    for item in spec['frames']:
        unknown = OCCLUDED_EYES.get(item['frame'], [])

        def visible_regions(keypoints, score, wh):
            regions, support = common.regions(keypoints, score, wh)
            for name in unknown:
                regions[name] = np.zeros_like(regions[name])
            return regions, support

        views = common.launch.rebound(common.native_views, regions=visible_regions, REGION_PINS=REGION_PINS)
        yield from views(dict(spec, frames=[item]), poses)


def inputs():
    def preflight(_subject, require_terminal):
        return common.frozen.preflight('005', require_terminal)

    # Reuse the immutable evaluator and substitute only its registered subject
    # inputs. The 001 evidence and helper bytes stay intact.
    source = SimpleNamespace(preflight=preflight, five=common.frozen.five)
    inspect = common.launch.rebound(common.inputs, registration=registration, frozen=source,
                                   PLAN_SHA=PLAN_SHA, FRAMES=FRAMES, native_views=native_views)
    spec = inspect()
    subject = registration()
    archive = read(Path(subject['prior_formal_receipt']['path']))
    excluded = {row['frame']:dict(role=row['role'], pose_screen_passed=row['pose_screen_passed'],
                                 flags=row['pose_screen_flags'])
                for row in archive['records'] if row['frame'] in subject['diagnostic_only']}
    need(set(excluded) == set(subject['diagnostic_only'])
         and excluded['frame_000011']['role'] == 'in_arc_pose_screen_failed'
         and excluded['frame_000055']['role'] == 'outside_arc_diagnostic',
         'Prior 005 pose-screen exclusions changed.')
    return dict(spec, script_sha256=sha(Path(__file__)), subject='005', prior_exclusions=excluded,
                occluded_eyes=OCCLUDED_EYES,
                eye_alpha_min_pixels=32, reused_evaluator_sha256=HELPER_SHA)


def candidate(stage_sha):
    stage_path = ROOT/'preregistration.json'
    if not stage_path.is_file():
        return None
    if stage_sha is None:
        stage_sha = sha(stage_path)
    pin(stage_path, stage_sha)
    stage = read(stage_path)
    subject = registration()
    need(stage['schema'] == 'shuolee-head-appearance-frozen-fit-transfer-v1'
         and stage['acceptance_plan_sha256'] == PLAN_SHA and stage['subject'] == '005'
         and stage['candidate_commit'] == subject['candidate_commit']
         and stage['baseline_result_sha256'] == subject['baseline_terminal']['sha256']
         and stage['baseline_commit'] == common.frozen.queue.COMMIT
         and stage['fit_reused'] is True and stage['fit_calls'] == 0
         and stage['head_appearance_steps'] == 600 and stage['head_appearance_weight'] == .05
         and stage['selected_frames'] == subject['selected_frames']
         and stage['training_slots'] == len(subject['selected_frames'])
         and stage['reference_frame'] == subject['reference_frame']
         and stage['formal_holdouts'] == FRAMES
         and stage['retained_holdouts'] == subject['retained_holdouts']
         and stage['coverage'] == subject['coverage']
         and stage['camera_pins'] == {common.launch.CAMERA_FILES[0]:subject['training_cameras']['sha256'],
                                       common.launch.CAMERA_FILES[1]:subject['portable_camera']['sha256']},
         'Candidate changed frozen 005 camera, coverage or retained holdouts.')
    for name,key in (('fit/fitted.npz', 'fit'), ('data/training-inputs.json', 'training_inputs'),
                     ('fit/selection.json', 'selection')):
        pin(RUN/name, subject[key]['sha256'])
    wrapper = ROOT/'wrapper-terminal.json'
    if not wrapper.is_file():
        return None
    terminal = read(wrapper)
    pin(RUN/'result.json', terminal['result_sha256'])
    result = read(RUN/'result.json')
    gaussian = read(RUN/'gaussian-run/result.json')
    need(terminal['state'] == result['state'] == 'completed' and result['format_passed'] is True
         and terminal['candidate_commit'] == stage['candidate_commit']
         and terminal['selected_frames'] == stage['selected_frames']
         and terminal['fit_calls'] == result['fit_calls'] == 0
         and result['fit_reused'] is True and result['gaussian_from_scratch'] is True
         and terminal['cold_video_clock'] is False
         and result['reference_frame_id'] == stage['reference_frame']
         and result['reused_fit_preregistration_sha256'] == stage_sha
         and result['steps'] == gaussian['steps'] == 10000 and result['sh_degree'] == 0
         and result['final_snapshot'] == gaussian['final_snapshot'] == terminal['final_snapshot']
         and result['head_appearance'] == gaussian['head_appearance'] == terminal['head_appearance'],
         '005 terminal does not bind the registered stock and tail.')
    for name,digest in stage['camera_pins'].items():
        pin(RUN/name, digest)
    return stage, terminal, result, gaussian


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'terminal-check', 'render', '_inputs'))
    parser.add_argument('--stage-sha', help='Expected hash of the staged candidate registration')
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.action == '_inputs':
        print(json.dumps(inputs()), flush=True)
        return
    payload = subprocess.check_output([sys.executable, '-B', __file__, '_inputs'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True)
    spec = json.loads(payload.splitlines()[-1])
    staged = candidate(args.stage_sha)
    state = None
    if staged is not None:
        sys.path.insert(0, str(common.PROJECT/'src'))
        snapshots = common.launch.rebound(common.snapshots, ROOT=ROOT, RUN=RUN)
        with np.load(spec['pose_archive'], allow_pickle=False) as poses:
            state = snapshots(staged, dict(poses))
    if args.action != 'check':
        need(state is not None and args.stage_sha is not None, 'Wait for the completed 005 tail and pin its stage SHA.')
    if args.action == 'render':
        output = ROOT/'protected-holdout-transfer'
        need(not output.exists(), 'Preserve the existing transfer receipt.')
        render = common.launch.rebound(common.render, RUN=RUN, native_views=native_views)
        result = render(spec, state, output)
        print(json.dumps(dict(result=str(result), sha256=sha(result))), flush=True)
    else:
        print(json.dumps(dict(spec, terminal_ready=state is not None,
                              terminal_identity=state[-1] if state is not None else None)), flush=True)


if __name__ == '__main__':
    main()
