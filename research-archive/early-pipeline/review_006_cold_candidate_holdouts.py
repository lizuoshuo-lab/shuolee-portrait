"""Read-only native posed holdout comparison for the fresh 006 six-worker candidate."""
import argparse
import ast
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import review_006_area_formal_holdouts as prior
from review_bound_alpha_ab import pin, read, require, sha
from review_bound_scale_coverage import scale_pixels


RUN = prior.LAB/'shuolee-smpl-sixworker-opacity-006/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-sixworker-opacity')
COMMIT = '213ad16c035ead820947549f2783d6f7cea1f771'
VIDEO_SHA = 'cb11ad325834b399c655371dc5d161e1801a3719b0892da959dc1e57e764572f'
AREA_REVIEW = prior.LAB/'shuolee-smpl-alpha006-area-formal-paired-retry/renders'
AREA_REVIEW_SHA = '2d6e278e1899df784db6a40159ac84f6943cb912e3444d21ce39296b33426b46'
NATIVE_FIELDS = ('frame', 'source_frame_index', 'source_sha256', 'wh', 'K_integer_centers', 'K', 'w2c',
                 'initial_box_xyxy', 'time_seconds')
RESET_STEPS = [1800, 3600, 5400, 7200]


def source_rows(root, path):
    return {row['frame']: row for row in read(root/path)['records']}


def archived_posed(root, record, original, crop):
    path = root/record['image']
    pin(path, record['image_sha256'])
    with Image.open(path) as source:
        panel = source.convert('RGB')
    w, h = original.size
    require(panel.size == (3*w, h) and np.array_equal(np.asarray(panel.crop((0,0,w,h))), np.asarray(original)),
            f'Archived panel uses different pixels or crop: {record["frame"]}, {crop}')
    return panel.crop((w,0,2*w,h))


def frozen_reference():
    state = prior.preflight()
    pin(AREA_REVIEW/'result.json', AREA_REVIEW_SHA)
    area = read(AREA_REVIEW/'result.json')
    require(area['state'] == 'formal_heldout_replay_complete' and area['protected_frames'] == prior.HELD
            and area['area_model_sha256'] == prior.AREA_PLY and area['baseline006_model_sha256'] == prior.BASE_PLY
            and area['gaussian_optimizer_updates'] == area['evaluation_refits'] == 0
            and area['evaluation_fitted_sha256'] == prior.FIXED_PINS[state.pose_archive/'evaluation-fitted.npz'],
            'Different archived Area replay or evaluation fitting.')
    return state, {row['frame']: row for row in area['records']}


def compare_observations(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return sorted(name for name in set(a.files)|set(b.files)
                      if name not in a.files or name not in b.files or
                      a[name].dtype != b[name].dtype or a[name].shape != b[name].shape
                      or a[name].tobytes() != b[name].tobytes())


def verify_mask_boundary(project, state):
    archive_inputs = read(state.pose_archive/'inputs.json')['input_sha256']
    code = {
        'official_sequence.py': ('checked_image', 'load_inputs', '_frame_fit'),
        'official_smplifyx.py': ('fit_frame', 'fit_one'),
    }
    source_pins = {}
    for file, names in code.items():
        old, = [Path(name) for name in archive_inputs if Path(name).name == file]
        current = project/'src/shuolee_portrait'/file
        source_pins[str(old)] = pin(old, archive_inputs[str(old)])
        source_pins[str(current)] = sha(current)
        definitions = []
        for path in (old, current):
            module = ast.parse(path.read_text(encoding='utf-8-sig'))
            definitions.append({node.name: ast.dump(node) for node in module.body
                                if isinstance(node, ast.FunctionDef) and node.name in names})
        require(set(definitions[0]) == set(names) and definitions[0] == definitions[1],
                f'The archived RGB/keypoints/K fitting boundary changed: {file}')
    return dict(archive_evaluator_sha256=prior.FIXED_PINS[prior.EVALUATOR], source_pins=source_pins,
                scope='checked_image reads mask bytes for SHA256/dimensions and returns only RGB. The original CPU evaluator passes RGB, keypoints/weights, K, beta and shared pose. Mask pixels do not enter the solver; scoring uses archived 269 masks.')


def preflight(run, project):
    status = read(run/'status.json')
    if status['state'] != 'completed':
        return dict(state='terminal_not_ready', run_state=status['state'], run_stage=status['stage']), None
    old, area = frozen_reference()
    mask_boundary = verify_mask_boundary(project, old)
    launch = read(run.parent/'preregistration.json')
    meta, cfg, result = (read(run/f'{name}.json') for name in ('run', 'config', 'result'))
    require(Path(launch['output']).resolve() == run and launch['candidate_commit'] == COMMIT
            and launch['cpu_workers'] == 6 and launch['source_sha256'] == meta['video_sha256'] == VIDEO_SHA
            and launch['expected_holdouts'] == list(prior.HELD)
            and launch['expected_held_source_indices'] == list(prior.HELD.values())
            and pin(run.parent/'camera.json', launch['camera_config_sha256']) == meta['camera_config_sha256'],
            'Wrong registered cold 006 candidate, video, camera or protected roster.')
    prior.check_pins({project/name: digest for name, digest in launch['code_sha256'].items()})
    pin(Path(meta['video']), VIDEO_SHA)

    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import official_sequence as sequence, runtime as rt
    from shuolee_portrait.bound_replay import _inputs
    from shuolee_portrait.training import camera, verify_official_fit

    require(rt.PROJECT == project, 'Wrong native renderer checkout.')
    train = run/'gaussian-run'
    train_result, profile = read(train/'result.json'), read(train/'config.json')
    require(result['state'] == train_result['state'] == 'completed' and result['format_passed'] is True
            and result['from_scratch'] is train_result['from_scratch'] is True
            and result['steps'] == train_result['steps'] == 10000
            and result['sh_degree'] == train_result['sh_degree'] == 0
            and profile['opacity_reset_steps'] == RESET_STEPS
            and profile['sampling_policy'] == 'seeded_equal_occupied_source_bin_cycles'
            and profile['alpha_loss_normalization'] == 'foreground_area'
            and profile['optimizer_initial_state_entries'] == 0, 'Wrong cold training profile or incomplete terminal.')
    for name, key in (('export/point_cloud.ply', 'sha256'), ('export/scene.zip', 'scene_sha256'),
                      (result['bound_replay'], 'bound_replay_sha256'), (result['bound_panels'], 'bound_panels_sha256')):
        pin(run/name, result[key])
    snapshot = train/'step-010000'
    bound_path = snapshot/'bound-replay.json'
    bound = read(bound_path)
    pin(bound_path, read(run/result['bound_replay'])['bound_replay_sha256'])
    pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    require(bound['schema'] == 'shuolee-smplx-bound-v1' and bound['bound_ply'] == 'bound-local.ply'
            and bound['points'] == result['points'] == train_result['points']
            and bound['sh_degree'] == bound['optimizer_updates'] == 0
            and bound['reference_frame_id'] == result['reference_frame_id'], 'Wrong terminal binding receipt.')
    asset = Path(meta['assets'])/'smplx/SMPLX_NEUTRAL.npz'
    fit, mesh, facts = _inputs(run, snapshot, asset, bound)
    official = read(run/'fit/official-fit-result.json')
    verify_official_fit(run)
    shared_path = run/'fit/shared-body.json'
    pin(shared_path, official['shared_body_sha256'])
    shared, base_shared = read(shared_path), read(prior.BASE/'fit/shared-body.json')
    selection = read(run/'fit/selection.json')
    native, training = (source_rows(run, path) for path in ('data/inputs.json', 'data/training-inputs.json'))
    base_training = source_rows(prior.BASE, 'data/training-inputs.json')
    train_obs, base_obs = run/'data/observations.npz', prior.BASE/'data/observations.npz'
    for root, path in ((run, train_obs), (prior.BASE, base_obs)):
        pin(path, read(root/'fit/official-fit-result.json')['source']['input_files'][str(path)])
    inputs_path = run/'data/inputs.json'
    pin(inputs_path, official['source']['input_files'][str(inputs_path)])
    inputs = sequence.load_inputs(inputs_path, train_obs, camera_length_contract=cfg['camera_length_contract'],
        reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    protected = set(prior.HELD)
    anchors = {frame for group in shared['anchor_groups'].values() for frame in group}
    actual_anchors = {path.name for path in (run/'fit/shape-anchors').iterdir() if path.is_dir()}
    fitted = {path.name for path in (run/'fit/frames').iterdir() if path.is_dir()}
    selected = set(selection['selected_frames'])
    require(official['execution_kind'] == 'fresh_official_sequence' and official['selection'] == selection
            and inputs['scope']['retained_holdouts'] == selection['retained_holdouts'] == list(prior.HELD)
            and anchors == actual_anchors <= fitted and fitted == set(inputs['scope']['fit_frame_ids'])
            and len(anchors) == launch['expected_shape_anchors'] and len(fitted) == launch['expected_fit_candidates']
            and len(native) == launch['expected_native_samples']
            and selected == set(training) == {frame['frame'] for frame in facts['frames']}
            and selected <= fitted and protected.isdisjoint(anchors | fitted | selected)
            and all(not inputs['targets'][inputs['ids'].index(frame)].any() for frame in protected),
            'Protected frames entered the cold fitting or training scope, or a receipt differs.')
    target_hashes = set()
    for frame in sorted(fitted):
        report = read(run/'fit/frames'/frame/'result.json')
        require(report['state'] == 'complete' and report['frame_id'] == frame
                and report['body_placement_parameters'] == 3, 'Original placement policy changed.')
        target_hashes.add(report['shared_body_target_sha256'])
    require(len(target_hashes) == 1, 'Candidate does not use one shared body target.')

    mismatches = {}
    if not np.array_equal(np.asarray(shared['betas'], dtype=np.float32), old.beta):
        mismatches['shared_betas'] = True
    if not np.array_equal(np.asarray(shared['pose_latent'], dtype=np.float32), np.asarray(base_shared['pose_latent'], dtype=np.float32)):
        mismatches['shared_pose_latent'] = True
    if target_hashes != {old.target_hash}:
        mismatches['shared_target_sha256'] = sorted(target_hashes)
    candidate_q90 = official['source']['confidence']['per_joint_q90']
    require(candidate_q90 == inputs['confidence']['per_joint_q90'], 'Candidate q90 does not replay from fit-only inputs.')
    if candidate_q90 != read(old.pose_archive/'inputs.json')['frozen_q90']:
        mismatches['fit_only_q90'] = True
    differences = compare_observations(base_obs, train_obs)
    if differences:
        mismatches['raw_observation_arrays'] = differences
    changed_masks = sorted(frame for frame in set(base_training)&selected
                           if base_training[frame]['mask_sha256'] != training[frame]['mask_sha256'])
    holdout_mask_provenance = {}
    for frame in prior.HELD:
        row, baseline = native[frame], old.baseline[frame]
        archived = next(item for item in old.subset['rows'] if item['frame'] == frame)
        changed = [key for key in NATIVE_FIELDS if row[key] != archived[key]]
        if changed:
            mismatches.setdefault('heldout_native_identity', {})[frame] = changed
        pin(Path(row['image']), row['source_sha256'])
        pin(Path(row['mask']), row['mask_sha256'])
        holdout_mask_provenance[frame] = dict(candidate_preprocessed_mask_sha256=row['mask_sha256'],
            archived_eval_fit_mask_sha256=baseline['fit_mask_sha256'],
            archived_eval_score_mask_sha256=baseline['scoring_mask_sha256'])
    changed_holdout_fit_masks = [frame for frame, facts in holdout_mask_provenance.items()
        if facts['candidate_preprocessed_mask_sha256'] != facts['archived_eval_fit_mask_sha256']]

    receipt = dict(state='frozen_formal_inputs_exact' if not mismatches else 'frozen_pose_reuse_unavailable',
                   mismatches=mismatches, protected_frames=prior.HELD, old_reference=old.result['reference_frame_id'],
                   candidate_reference=result['reference_frame_id'], candidate_result_sha256=sha(run/'result.json'),
                   candidate_bound_manifest_sha256=sha(bound_path), terminal_points=result['points'],
                   training_roster_added=sorted(selected-set(base_training)),
                   training_roster_removed=sorted(set(base_training)-selected),
                   changed_common_training_masks=changed_masks, changed_candidate_holdout_fit_masks=changed_holdout_fit_masks,
                   archived_scoring_masks_fixed=True, archived_fit_masks_fixed=True,
                   holdout_mask_provenance=holdout_mask_provenance, fitting_mask_boundary=mask_boundary,
                   raw_observation_npz_bytes_equal=sha(train_obs)==sha(base_obs), raw_observation_arrays_equal=not differences,
                   evaluation_fit_sha256=prior.FIXED_PINS[old.pose_archive/'evaluation-fitted.npz'],
                   old_area_review_sha256=AREA_REVIEW_SHA, script_sha256=sha(Path(__file__)),
                   optimizer_updates=0, evaluation_refits=0, quality_accepted=False,
                   scope='Complete candidate workflow compared with two archived results. No single-factor attribution.')
    if mismatches:
        receipt['followup']='Separate evaluation-only official fit with candidate beta and frozen original heldout RGB/K may be registered. Keep archived poses, scoring masks and metrics as separate historical evidence.'
        return receipt, None
    require(np.array_equal(fit['shared_betas'], old.beta), 'Terminal binding uses another shape.')
    with np.load(old.pose_archive/'evaluation-fitted.npz', allow_pickle=False) as archive:
        poses = dict(archive)
    records = []
    for i, (row, score_row, benchmark) in enumerate(zip(old.subset['rows'], old.old_rows, old.old_records, strict=True)):
        frame = row['frame']
        baseline, area_record = old.baseline[frame], area[frame]
        keys = ('frame','source_frame_index','native_source_sha256','scoring_mask_sha256','fit_mask_sha256',
                'pose_screen_passed','pose_screen_flags','posed_camera')
        require(all(baseline[key] == area_record[key] for key in keys), f'Archived baselines disagree: {frame}')
        pose = len(facts['frames'])+i
        expected = dict(baseline['posed_camera'], pose_index=pose)
        require(camera(row, i, pose_index=pose)[1] == expected, f'Native pose camera changed: {frame}')
        mask_path = old.helper['relocate'](score_row['mask'])
        with Image.open(row['image']) as source, Image.open(mask_path) as silhouette:
            require(list(source.size) == list(silhouette.size) == row['wh'], f'Scoring raster differs: {frame}')
            image, mask = source.convert('RGB'), np.asarray(silhouette.convert('L')) >= 128
        crop = tuple(benchmark['crop_xyxy'])
        original = image.crop(crop)
        records.append(SimpleNamespace(row=row, pose=pose, i=i, expected=expected, image=image, mask=mask,
            crop=crop, original=original, baseline=baseline, area=area_record,
            cold_picture=archived_posed(old.pose_archive/'renders', baseline, original, crop),
            area_picture=archived_posed(AREA_REVIEW, area_record, original, crop)))
    return receipt, SimpleNamespace(rt=rt, asset=asset, snapshot=snapshot, fit=fit, mesh=mesh, facts=facts,
        bound=bound, poses=poses, records=records, helper=old.helper, archive_pins=old.pins)


def replay(state, output):
    import torch
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import _frozen_smpl_surface, camera, render_pose

    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU.')
            state.rt.configure(state.asset.parent.parent)
            from gaussian_renderer import render

            path = state.snapshot/'bound-local.ply'
            pin(path, state.bound['bound_ply_sha256'])
            model, parity = _load_model(path, state.snapshot, state.fit, state.mesh, state.facts, state.asset)
            base, linear, clothing, _, _ = _frozen_smpl_surface(state.poses, state.asset)
            delta = state.mesh['static_offset']
            require(not np.any(delta[~clothing]), 'Candidate offset changed protected skin.')
            offset = torch.from_numpy(delta).cuda()[...,None]
            posed = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(base.cuda(),linear.cuda(),strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence, posed))
            model.num_timesteps = len(model.vertices_sequence)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            output.mkdir(parents=True, exist_ok=False)
            records = []
            for frame in state.records:
                cam, actual = camera(frame.row, frame.i, pose_index=frame.pose)
                require(actual == frame.expected, f'Native pose camera changed: {frame.row["frame"]}')
                pixels, error = scale_pixels(render, render_pose, cam, model, pipe, black, white, frame.pose, 1.)
                rgb, _, alpha = pixels
                current = state.helper['metrics'](rgb, alpha, np.asarray(frame.image, dtype=np.float32)/255, frame.mask)
                current['black_white_composition_error'] = error
                rendered = Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)).crop(frame.crop)
                pictures = (frame.original, frame.cold_picture, frame.area_picture, rendered)
                w,h = rendered.size
                panel = Image.new('RGB', (4*w,h+23), '#171717')
                draw = ImageDraw.Draw(panel)
                for i,(name, picture) in enumerate(zip(('Source','Cold placement','Same-fit Area','Cold 006 candidate'),pictures,strict=True)):
                    draw.text((i*w+4,5), name, fill='white')
                    panel.paste(picture,(i*w,23))
                target = output/(frame.row['frame']+'-source-cold-area-candidate.png')
                panel.save(target)
                facts = {key: frame.baseline[key] for key in ('frame','source_frame_index','native_source_sha256',
                    'scoring_mask_sha256','fit_mask_sha256','pose_screen_passed','pose_screen_flags')}
                records.append(dict(**facts, posed_camera=actual, metrics=current,
                    prior_cold_metrics=frame.baseline['metrics']['posed'], prior_area_metrics=frame.area['metrics']['posed'],
                    image=target.name, image_sha256=sha(target)))
            require(model.optimizer is None, 'Read-only replay created an optimizer.')
    prior.check_pins(state.archive_pins)
    return records, parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    run, project = args.run.resolve(), args.project.resolve()
    receipt, state = preflight(run, project)
    if args.preflight:
        print(json.dumps(dict(**receipt,gpu_calls=0), ensure_ascii=False))
        return
    require(state is not None, json.dumps(receipt, ensure_ascii=False))
    require(args.output is not None, 'Choose a fresh evaluation directory.')
    out = args.output.resolve()
    require(not out.exists() and not any(out.is_relative_to(path) for path in
            (run.parent, project, prior.BASE.parent, prior.AREA.parent, prior.POSE_ARCHIVE, AREA_REVIEW.parent)),
            'Choose an output outside all fitting, source and historical evaluation paths.')
    records, parity = replay(state, out)
    target = out/'result.json'
    state.rt.save(target, dict(receipt, state='cold_006_frozen_holdout_replay_complete', records=records,
                              load_parity=parity, fixed_static_comparison=False))
    print(json.dumps(dict(receipt=str(target),sha256=sha(target))))


if __name__ == '__main__':
    main()
