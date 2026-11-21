"""Replay the fixed 006 formal poses against the unified terminal, without fitting."""

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_unified_reused_fit_016_006 as launch
import review_006_cold_candidate_holdouts as cold


common = launch.common
require, pin, read, sha = cold.require, cold.pin, cold.read, cold.sha
PRIMARY = ('frame_000022', 'frame_000037')
DIAGNOSTIC = ('frame_000008', 'frame_000050')
REPLAY_SHA = 'fba497752658df234da5049daaf129ae812d5aec2df0d8c1755358d18688b5b9'


def preflight():
    project = common.PROJECT
    root = common.root_for('006')
    run = root/'run'
    plan_path = root/'preregistration.json'
    plan = read(plan_path)
    plan_sha = sha(plan_path)
    common.git_state(launch.COMMIT)
    pin(Path(cold.__file__), REPLAY_SHA)
    held = cold.prior.HELD
    require(plan['schema'] == 'shuolee-unified-reused-fit-v1' and plan['subject'] == '006'
            and plan['candidate_commit'] == launch.COMMIT and plan['copied_fitting_input_from'] == str(cold.RUN)
            and plan['baseline_result_sha256'] == launch.RESULT_PINS['006']
            and plan['source_video_sha256'] == cold.VIDEO_SHA
            and plan['fit_reused'] is True and plan['fit_calls'] == 0
            and plan['fresh_gaussian_optimizer'] is True and plan['cold_video_clock'] is False
            and plan['retained_holdouts'] == list(held), 'Wrong unified 006 registration.')
    pins = {plan_path: plan_sha, Path(cold.__file__): REPLAY_SHA}
    for name in (*launch.COPY, 'run.json'):
        path = run/name
        pins[path] = pin(path, plan['staged_sha256'][name])
    for name in launch.COPY:
        pin(cold.RUN/name, pins[run/name])
    for name, digest in plan['code_sha256'].items():
        pins[project/name] = pin(project/name, digest)
    metadata = read(run/'run.json')
    pins[Path(metadata['video'])] = pin(Path(metadata['video']), plan['source_video_sha256'])
    require(metadata['fit_reused'] is True and metadata['fit_calls'] == 0
            and metadata['camera_config_sha256'] == plan['camera_config_sha256']
            and metadata['assets'] == str(common.ASSETS), 'Wrong staged fit or calibration.')
    if (root/'launch.json').exists():
        launch_receipt = read(root/'launch.json')
        require(launch_receipt['preregistration_sha256'] == plan_sha
                and launch_receipt['candidate_commit'] == launch.COMMIT, 'Launch did not bind this plan.')
        pins[root/'launch.json'] = sha(root/'launch.json')

    archive, area = cold.frozen_reference()
    pins.update({Path(path): digest for path, digest in archive.pins.items()})
    source_rows = cold.source_rows(cold.RUN, 'data/inputs.json')
    official = read(run/'fit/official-fit-result.json')
    source = official['source']['input_files']
    for name in ('data/inputs.json', 'data/observations.npz'):
        path = cold.RUN/name
        pins[path] = pin(path, source[str(path)])
    require(not cold.compare_observations(cold.RUN/'data/observations.npz', cold.prior.BASE/'data/observations.npz'),
            'The archived evaluation used different detector observations.')
    selection = read(run/'fit/selection.json')
    shared = read(run/'fit/shared-body.json')
    pin(run/'fit/shared-body.json', official['shared_body_sha256'])
    selected = selection['selected_frames']
    require(official['selection'] == selection and selected == plan['selected_frames']
            and selection['retained_holdouts'] == list(held) and set(selected).isdisjoint(held)
            and set(held).isdisjoint(frame for group in shared['anchor_groups'].values() for frame in group)
            and set(held).isdisjoint(p.parent.name for p in (cold.RUN/'fit/frames').glob('*/result.json'))
            and selection['reference_frame'] == plan['reference_frame']
            and np.array_equal(np.asarray(shared['betas'], dtype=np.float32), archive.beta)
            and {report['shared_body_target_sha256'] for report in official['frame_results']} == {archive.target_hash}
            and official['source']['confidence']['per_joint_q90'] == read(archive.pose_archive/'inputs.json')['frozen_q90'],
            'Protected roster, shared beta, pose target, or confidence normalization changed.')
    pose_path = archive.pose_archive/'evaluation-fitted.npz'
    with np.load(pose_path, allow_pickle=False) as data:
        poses = dict(data)
    passing = tuple(frame for frame in held if archive.baseline[frame]['pose_screen_passed'])
    require(passing == PRIMARY and tuple(frame for frame in held if frame not in passing) == DIAGNOSTIC
            and poses['frame_ids'].tolist() == list(held) and poses['source_frames'].tolist() == list(held.values())
            and np.array_equal(poses['shared_betas'], archive.beta), 'Frozen pose arrays or original screens changed.')
    for row in archive.subset['rows']:
        current = source_rows[row['frame']]
        require(all(current[key] == row[key] for key in cold.NATIVE_FIELDS),
                f'Frozen native RGB, K or camera differs: {row["frame"]}')
        image = Path(current['image'])
        pins[image] = pin(image, current['source_sha256'])

    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs
    from shuolee_portrait.training import camera, verify_official_fit

    require(rt.PROJECT == project and verify_official_fit(run)['fit_sha256'] == pins[run/'fit/fitted.npz'],
            'The staged official fit could not be verified.')
    receipt = dict(preregistration_sha256=plan_sha, candidate_commit=launch.COMMIT,
                   protected_frames=held, primary_frames=list(PRIMARY), diagnostic_frames=list(DIAGNOSTIC),
                   evaluation_fit_sha256=pins[pose_path], shared_body_target_sha256=archive.target_hash,
                   archived_scoring_masks_fixed=True, fitting_calls=0, optimizer_updates=0, quality_accepted=False)
    status = read(run/'status.json') if (run/'status.json').is_file() else dict(state='staged')
    if status['state'] != 'completed' or not (root/'wrapper-terminal.json').is_file():
        return dict(receipt, state='terminal_not_ready', staged_frozen_inputs_exact=True,
                    reconstruction_state=status['state'], reconstruction_stage=status.get('stage')), None

    wrapper = read(root/'wrapper-terminal.json')
    pins[root/'wrapper-terminal.json'] = sha(root/'wrapper-terminal.json')
    pins[run/'status.json'] = sha(run/'status.json')
    terminal_path = run/'result.json'
    pins[terminal_path] = pin(terminal_path, wrapper['result_sha256'])
    terminal = read(terminal_path)
    snapshot = run/'gaussian-run/step-010000'
    profile = read(snapshot.parent/'config.json')
    require(wrapper['state'] == terminal['state'] == 'completed'
            and wrapper['candidate_commit'] == launch.COMMIT
            and wrapper['fit_calls'] == terminal['fit_calls'] == 0
            and wrapper['fit_reused'] is terminal['fit_reused'] is True
            and terminal['from_scratch'] is False and terminal['gaussian_from_scratch'] is True
            and terminal['format_passed'] is True and terminal['steps'] == 10000 and terminal['sh_degree'] == 0
            and wrapper['selected_frames'] == selected and wrapper['reference_frame'] == selection['reference_frame']
            and terminal['reused_fit_preregistration_sha256'] == plan_sha
            and status['sha256'] == terminal['sha256']
            and profile['optimizer_initial_state_entries'] == 0
            and profile['opacity_reset_steps'] == cold.RESET_STEPS
            and profile['sampling_policy'] == 'seeded_equal_occupied_source_bin_cycles'
            and profile['alpha_loss_normalization'] == 'foreground_area',
            'The terminal is not the registered reused-fit unified result.')
    for name, digest in ((terminal['ply'], terminal['sha256']), (terminal['scene'], terminal['scene_sha256']),
                         (terminal['bound_replay'], terminal['bound_replay_sha256']),
                         (terminal['bound_panels'], terminal['bound_panels_sha256'])):
        pins[run/name] = pin(run/name, digest)
    replay = read(run/terminal['bound_replay'])
    bound_path = snapshot/'bound-replay.json'
    pins[bound_path] = pin(bound_path, replay['bound_replay_sha256'])
    bound = read(bound_path)
    asset = common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    pins[asset] = pin(asset, plan['asset_sha256'][str(asset)])
    pins[snapshot/'bound-local.ply'] = pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    require(bound['schema'] == 'shuolee-smplx-bound-v1' and bound['bound_ply'] == 'bound-local.ply'
            and bound['points'] == terminal['points'] == wrapper['points']
            and bound['sh_degree'] == bound['optimizer_updates'] == 0,
            'The terminal bound checkpoint is invalid.')
    fit, mesh, facts = _inputs(run, snapshot, asset, bound)
    require(np.array_equal(fit['shared_betas'], poses['shared_betas'])
            and set(selected) == {frame['frame'] for frame in facts['frames']}
            and replay['retained_holdouts'] == list(held), 'Terminal beta or protected roster differs.')

    records = []
    for i, (row, scoring, benchmark) in enumerate(zip(archive.subset['rows'], archive.old_rows,
                                                      archive.old_records, strict=True)):
        frame = row['frame']
        base, area_record = archive.baseline[frame], area[frame]
        keys = ('frame', 'source_frame_index', 'native_source_sha256', 'scoring_mask_sha256',
                'fit_mask_sha256', 'pose_screen_passed', 'pose_screen_flags', 'posed_camera')
        require(all(base[key] == area_record[key] for key in keys), f'Archived baselines differ: {frame}')
        pose = len(facts['frames'])+i
        expected = dict(base['posed_camera'], pose_index=pose)
        require(camera(row, i, pose_index=pose)[1] == expected, f'Native camera projection differs: {frame}')
        score_mask = archive.helper['relocate'](scoring['mask'])
        with Image.open(row['image']) as source, Image.open(score_mask) as silhouette:
            image = source.convert('RGB')
            mask = np.asarray(silhouette.convert('L')) >= 128
        crop = tuple(benchmark['crop_xyxy'])
        original = image.crop(crop)
        records.append(SimpleNamespace(row=row, pose=pose, i=i, expected=expected, image=image, mask=mask,
            crop=crop, original=original, baseline=base, area=area_record,
            cold_picture=cold.archived_posed(archive.pose_archive/'renders', base, original, crop),
            area_picture=cold.archived_posed(cold.AREA_REVIEW, area_record, original, crop)))
    state = SimpleNamespace(rt=rt, asset=asset, snapshot=snapshot, fit=fit, mesh=mesh, facts=facts,
                            bound=bound, poses=poses, records=records, helper=archive.helper, archive_pins=pins)
    return dict(receipt, state='frozen_formal_inputs_exact', staged_frozen_inputs_exact=True,
                terminal_result_sha256=pins[terminal_path], terminal_bound_manifest_sha256=pins[bound_path],
                terminal_points=terminal['points']), state


def relabel(output, record):
    frame = record['frame']
    role = 'protected_holdout' if frame in PRIMARY else 'pose_screen_failed_diagnostic'
    path = output/record['image']
    pin(path, record['image_sha256'])
    with Image.open(path) as old:
        panel = old.convert('RGB')
    left = 3*(panel.width//4)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((left, 0, panel.width-1, 22), fill='#171717')
    draw.text((left+4, 5), 'Unified 006 / ' + ('holdout' if frame in PRIMARY else 'pose diagnostic'), fill='white')
    panel.save(path)
    return dict(record, evaluation_role=role, image_sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt, state = preflight()
    if args.action == 'check':
        print(json.dumps(dict(receipt, gpu_calls=0)))
        return
    require(state is not None, json.dumps(receipt))
    require(args.output is not None, 'Choose a fresh separate evaluation output.')
    output = args.output.resolve()
    require(not output.exists() and output.parent == common.LAB,
            'Use a fresh direct child of the local lab directory.')
    records, parity = cold.replay(state, output)
    primary, diagnostics = [], []
    for record in records:
        result = relabel(output, record)
        (primary if record['frame'] in PRIMARY else diagnostics).append(result)
    target = output/'result.json'
    common.write(target, dict(receipt, state='unified_006_frozen_holdout_replay_complete', primary_records=primary,
                              diagnostic_records=diagnostics, load_parity=parity, reviewer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(receipt=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
