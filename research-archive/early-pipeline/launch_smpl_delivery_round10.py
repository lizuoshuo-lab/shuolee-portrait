"""Run one protected cold reconstruction for each round-ten subject."""

import argparse
import hashlib
import sys
from pathlib import Path
from types import FunctionType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_unified_cold_016_006 as cold


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-joint-fit-20261111')
BASE_PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-replacement-20261111')
BASE = '7602f31d16dc13ba68c27ca63895d72a07d3a1dd'
DELIVERY = Path(__file__).resolve().parents[1]/'outputs/smpl-delivery-round9/receipt.json'
DELIVERY_SHA = '0a489c425b8ed9ea6f32f390e953330325c3123cf72130a5d4994ac1a90d805c'
DIRECTOR = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs/round9_director_review_20261111.json')
DIRECTOR_SHA = '5e3f800f4bee334037fa3bde9d05f90489186de41a1dc7efac88c831a3866ad2'
FIT_MODULE = 'src/shuolee_portrait/joint_body_fit.py'
SOURCE_FILES = (*cold.SOURCE_FILES, FIT_MODULE, 'src/shuolee_portrait/sh0_main.py',
    'src/shuolee_portrait/shape_refinement.py', 'src/shuolee_portrait/world_scale.py',
    'src/shuolee_portrait/render_support.py', 'src/shuolee_portrait/replacement_selection.py',
    'src/shuolee_portrait/coverage_refinement.py')
CHANGED = {FIT_MODULE, 'src/shuolee_portrait/official_sequence.py', 'src/shuolee_portrait/observations.py'}
need, read, pin, sha, write = cold.need, cold.read, cold.pin, cold.sha, cold.write


def rebound(fn, **kwargs):
    return FunctionType(fn.__code__, dict(fn.__globals__, **kwargs), fn.__name__, fn.__defaults__, fn.__closure__)


def root_for(subject):
    return cold.LAB/f'shuolee-smpl-delivery-round10-{subject}'


git_state = rebound(cold.git_state, PROJECT=PROJECT, BASE=BASE)
official_pins = rebound(cold.official_pins, PROJECT=PROJECT)
command = rebound(cold.command, PROJECT=PROJECT, root_for=root_for)
check_inputs = rebound(cold.check, git_state=git_state, root_for=root_for, command=command,
                       official_pins=official_pins, PROJECT=PROJECT, SOURCE_FILES=SOURCE_FILES, __file__=__file__)


def check(subject, commit):
    import subprocess

    plan = check_inputs(subject, commit)
    pointer = (PROJECT/'.git').read_text().strip().removeprefix('gitdir: ')
    git_dir = subprocess.check_output(['/usr/bin/wslpath', '-u', pointer], text=True).strip()
    changed = set(subprocess.check_output(['/usr/bin/git', '--git-dir', git_dir, '--work-tree', str(PROJECT),
        'diff', '--name-only', BASE, commit], text=True).splitlines())
    need(CHANGED <= changed and changed <= CHANGED | {'tests/test_joint_body_fit.py'},
         f'Unexpected round-ten code paths: {sorted(changed)}.')
    for name in set(SOURCE_FILES)-CHANGED:
        pin(PROJECT/name, sha(BASE_PROJECT/name))
    pin(DELIVERY, DELIVERY_SHA)
    pin(DIRECTOR, DIRECTOR_SHA)
    director = read(DIRECTOR)
    delivery = read(DELIVERY)
    need(director['receipt_sha256_verified'] == DELIVERY_SHA and director['round'] == 9
         and director['commit'] == delivery['commit'] == BASE, 'Unverified round-nine baseline.')
    reference = delivery['subjects'][subject]
    baseline_run = Path(reference['result']).parent
    pin(baseline_run/'result.json', reference['result_sha256'])
    selection = read(baseline_run/'fit/selection.json')
    need(selection['retained_holdouts'] == plan['expected_holdouts'] and len(selection['selected_frames']) == 24,
         'Baseline source layout differs.')
    return dict(plan, schema='shuolee-round10-cold-joint-fit-v1', baseline_commit=BASE,
        baseline_run=str(baseline_run), baseline_result_sha256=reference['result_sha256'],
        director_review_sha256=DIRECTOR_SHA, baseline_delivery_sha256=DELIVERY_SHA,
        baseline_selection_sha256=sha(baseline_run/'fit/selection.json'),
        baseline_reference=selection['reference_frame'], baseline_selected_frames=selection['selected_frames'],
        baseline_coverage=selection['coverage'], gaussian_changes=False)


def fit_proof(root, plan):
    import numpy as np

    run = root/'run'
    fit = read(run/'fit/official-fit-result.json')
    joint = fit['joint_body_refinement']
    for path, digest in (('file', 'sha256'), ('candidate_file', 'candidate_sha256'),
                         ('before_selection_file', 'before_selection_sha256')):
        target = (run/joint[path]).resolve()
        need(target.is_relative_to(run/'fit'), 'Joint proof path outside its fit root.')
        pin(target, joint[digest])
    metrics = read(run/joint['file'])
    before = read(run/joint['before_selection_file'])
    scope = metrics['fit_frame_ids']
    holdouts = plan['expected_holdouts']
    selection = fit['selection']
    inputs = read(run/'data/inputs.json')['records']
    training = read(run/'data/training-inputs.json')['records']
    names = [row['frame'] for row in training]
    need(metrics['state'] == 'complete' and metrics['retained_holdouts'] == holdouts
         and metrics['excluded_holdouts'] == selection['retained_holdouts'] == holdouts
         and len(scope) == len(set(scope)) == 52 and set(scope).isdisjoint(holdouts)
         and len(inputs) == 56 and {r['frame'] for r in inputs} == set(scope) | set(holdouts)
         and len(names) == len(set(names)) == 24 and set(names) == set(selection['selected_frames']) <= set(scope)
         and fit['selection'] == read(run/'fit/selection.json') and before['retained_holdouts'] == holdouts
         and fit['fitted_candidate_frames'] == 52 and fit['new_fitting_started'] is True,
         'Joint fitting, source roster or protected holdouts differ.')
    by_name = {row['frame']:row for row in inputs}
    baseline_rows = read(Path(plan['baseline_run'])/'data/training-inputs.json')['records']
    baseline = baseline_rows[0]
    camera_fields = ('wh', 'K', 'K_integer_centers', 'w2c', 'camera_kind')
    need(all(all(row[k] == baseline[k] for k in camera_fields) for row in inputs + training)
         and [by_name[name]['source_frame_index'] for name in holdouts] == plan['expected_held_source_indices']
         and all(all(by_name[row['frame']][k] == row[k] for k in ('source_frame_index', 'source_sha256'))
                 for row in baseline_rows),
         'Native pixels, fixed K, pixel centers or registered camera convention differ.')
    anchor = read(run/'fit/shared-body.json')
    anchors = [frame for frames in anchor['anchor_groups'].values() for frame in frames]
    need(len(anchors) == 24 and set(anchors) <= set(scope), 'Shared shape used a protected source.')
    pin(run/'fit/shared-body.json', fit['shared_body_sha256'])
    pin(run/'fit/fitted.npz', fit['fit_sha256'])
    pin(run/'data/training-inputs.json', fit['training_inputs_sha256'])
    with np.load(run/joint['candidate_file'], allow_pickle=False) as candidates, \
            np.load(run/'fit/fitted.npz', allow_pickle=False) as selected:
        slots = [scope.index(name) for name in names]
        need(candidates['frame_ids'].tolist() == scope and selected['frame_ids'].tolist() == names
             and np.array_equal(candidates['betas'], selected['betas'])
             and np.array_equal(candidates['betas'], np.asarray(anchor['betas'], np.float32))
             and float(selected['world_scale']) == 1.
             and all(np.array_equal(candidates[k][slots], selected[k]) for k in
                     ('vertices_camera', 'joints_camera', 'body_pose_axis_angle', 'global_orient',
                      'camera_translation', 'K_integer_centers', 'observed_native_keypoints',
                      'source_frames', 'image_sha256', 'mask_sha256')),
             'The trained fit is not the selected joint pose geometry in fixed units.')
    pin(PROJECT/FIT_MODULE, metrics['joint_module_sha256'])
    return fit, joint, metrics, before


def finish(subject, root, plan, stamp, elapsed, rc):
    import datetime
    import zipfile
    from launch_smpl_delivery_round9 import replacement_trial_proof

    run = root/'run'
    status_file = run/'status.json'
    status = read(status_file) if status_file.is_file() else {}
    receipt = dict(state='failed' if rc else 'incomplete', subject=subject, returncode=rc,
        candidate_commit=plan['candidate_commit'], baseline_commit=BASE,
        preregistration_sha256=sha(root/'preregistration.json'), command_started_at=stamp,
        command_to_end_seconds=elapsed, process_status=status, quality_accepted=False)
    if rc == 0:
        try:
            result = read(run/'result.json')
            fit, joint, metrics, before = fit_proof(root, plan)
            selection = fit['selection']
            integration = read(run/'gaussian-run/integration.json')
            cfg = read(run/'gaussian-run/config.json')
            trained = read(run/'gaussian-run/result.json')
            old = Path(plan['baseline_run'])
            old_cfg = read(old/'gaussian-run/config.json')
            old_initial = old/'gaussian-run/initialization.json'
            pin(run/'gaussian-run/initialization.json', sha(old_initial))
            keys = ('optimization', 'unified_main', 'alpha_loss_weight', 'straight_rgb_alpha_floor',
                    'sampling_policy', 'snapshot_steps', 'opacity_reset_steps', 'sh_degree')
            need(all(cfg[k] == old_cfg[k] for k in keys)
                 and integration['optimizer_updates'] == result['steps'] == plan['gaussian_step_budget'] == 10000
                 and integration['optimizer_initial_state_entries'] == 0 and integration['no_prior_gaussian_loaded']
                 and integration['initial_points'] == 20908
                 and integration['raw_native_rgb_target'] and not integration['prior_render_used_as_target']
                 and integration['frozen_surface_sha256'] is None
                 and len(trained['per_frame_updates']) == 24 and min(trained['per_frame_updates']) > 0
                 and sum(trained['per_frame_updates']) == 10000,
                 'R9 Gaussian implementation, fresh initialization or fixed schedule differs.')
            need(status.get('state') == 'completed' and status['static_export_passed']
                 and result['state'] == 'completed' and result['format_passed'] and result['from_scratch']
                 and result['frames'] == 24 and result['reference_frame_id'] == selection['reference_frame']
                 and result['final_snapshot'] == 'step-010000',
                 'Missing completed public cold-path artifact.')
            for key,digest in (('ply','sha256'), ('scene','scene_sha256'), ('bound_replay','bound_replay_sha256'),
                               ('bound_panels','bound_panels_sha256')):
                path = (run/result[key]).resolve()
                need(path.is_relative_to(run), 'Cold artifact path escapes its run.')
                pin(path, result[digest])
            with zipfile.ZipFile(run/result['scene']) as bundle:
                with bundle.open('cameras.json') as camera, bundle.open('point_cloud/iteration_10000/point_cloud.ply') as ply:
                    need(cold.stream_sha(camera) == sha(run/result['camera']) and cold.stream_sha(ply) == result['sha256'],
                         'Bundled camera or world PLY differs.')
            replay = read(run/result['bound_replay'])
            panels = read(run/result['bound_panels'])
            names = [row['frame'] for row in read(run/'data/training-inputs.json')['records']]
            need([row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']] == names
                 and replay['retained_holdouts'] == plan['expected_holdouts'],
                 'Bound playback did not use the exact newly selected real poses.')
            scale = run/'gaussian-run/world-scale.json'
            pin(scale, integration['world_scale']['sha256'])
            world = read(scale)
            need(world['terminal'] == integration['world_scale']['terminal']
                 and world['terminal']['all_poses_max']['all']['over_limit'] == 0,
                 'World-scale limit did not hold after the new fit.')
            for event in world['splits']:
                support = event['render_support']
                options = world['config']['render_support']
                replacement_trial_proof(support['coverage_trial'], event['needle']['support_replaced'], names, options)
                replacement_trial_proof(support['near_limit_trial'], event['split_parents'], names, options)
            for name,digest in plan['code_sha256'].items():
                pin(PROJECT/name, digest)
            pin(Path(plan['video']), plan['source_sha256'])
            video_input = read(run/'run.json')
            need(video_input['video_sha256'] == plan['source_sha256']
                 and video_input['camera_config_sha256'] == plan['camera_config_sha256'], 'Cold video or camera input changed.')
            time_s = (datetime.datetime.fromisoformat(status['static_export_completed_at'])-
                      datetime.datetime.fromisoformat(stamp)).total_seconds()
            receipt.update(state='completed', reference_frame=result['reference_frame_id'], points=result['points'],
                result_sha256=sha(run/'result.json'), selected_frames=selection['selected_frames'],
                coverage=selection['coverage'], before_joint_coverage=before.get('coverage'),
                full_bin_coverage=selection['full_bin_coverage'], joint_refinement_sha256=joint['sha256'],
                joint_candidate_sha256=joint['candidate_sha256'], frozen_native_cameras_verified=True,
                static_export_seconds=time_s, panels_complete_seconds=elapsed, original_video_to_export=True,
                optimizer_updates=10000, native_masks_fresh=True, reference_mesh_frozen_from_prior=False,
                world_scale_sha256=sha(scale))
        except Exception as error:
            receipt['validation_error'] = str(error)
    write(root/'wrapper-terminal.json', receipt)
    print(dict(state=receipt['state'], subject=subject, path=str(root/'wrapper-terminal.json'),
               validation_error=receipt.get('validation_error')), flush=True)
    return 0 if receipt['state'] == 'completed' else rc or 1


register = rebound(cold.register, check=check, root_for=root_for)
verify_registered = rebound(cold.verify_registered, check=check, root_for=root_for)
launch = rebound(cold.launch, verify_registered=verify_registered, root_for=root_for, command=command,
                 PROJECT=PROJECT, finish=finish)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'register', 'launch'))
    parser.add_argument('subject', choices=tuple(cold.PREVIOUS))
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.action == 'launch':
        raise SystemExit(launch(args.subject, args.commit))
    if args.action == 'register':
        register(args.subject, args.commit)
    else:
        plan = check(args.subject, args.commit)
        print(dict(state='preflight_passed', subject=args.subject, plan=plan['schema'],
                   protected_holdouts=plan['expected_holdouts']), flush=True)


if __name__ == '__main__':
    main()
