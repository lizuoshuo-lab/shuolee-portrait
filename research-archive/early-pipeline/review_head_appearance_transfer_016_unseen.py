"""Replay 016's registered Gaussian-unseen frame using its exact archived fitted pose."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

import launch_head_appearance_frozen_fit as launch
from review_head_appearance_transfer_001_holdouts import head_panel, region_metrics, regions, scale_pixels


need, read, sha, pin = launch.need, launch.read, launch.sha, launch.pin
ROOT = launch.root_for('016')
RUN = ROOT/'run'
FRAME, SOURCE_INDEX = 'frame_000016', 120
COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
STAGE_SHA = 'c76c6e808f87580e3186392b833e61c466b1c59740c15b4643f0f7adbdae50c3'
PRIOR_STAGE_SHA = 'f7e57677879cc357cd7ae5f68b62e6b3fdb32b3330bd8edd331532bda177f1fd'
STOCK, TAIL = 'step-010000', 'step-010000-head-000600'


def inputs():
    import smplx
    import torch
    from shuolee_portrait import official_sequence as seq, stable_selection as selector
    from shuolee_portrait.training import _frozen_smpl_surface

    torch.set_num_threads(1)
    pin(launch.ACCEPTANCE, launch.ACCEPTANCE_SHA)
    scope = read(launch.ACCEPTANCE)
    registered = scope['subjects']['016']
    need(scope['no_refit'] is True and registered['diagnostic_gaussian_unseen'] == [FRAME]
         and registered['formal_holdouts'] == [], 'Wrong registered 016 diagnostic.')
    for name in ('baseline_terminal', 'fit', 'training_inputs', 'selection'):
        item = registered[name]
        pin(Path(item['path']), item['sha256'])
    prior = Path(registered['baseline_terminal']['path']).parents[1]
    pin(prior/'preregistration.json', PRIOR_STAGE_SHA)
    prior_plan = read(prior/'preregistration.json')
    stage_path = ROOT/'preregistration.json'
    pin(stage_path, STAGE_SHA)
    stage = read(stage_path)
    need(stage['candidate_commit'] == COMMIT and stage['acceptance_plan_sha256'] == launch.ACCEPTANCE_SHA
         and stage['baseline_run'] == str(prior/'run')
         and stage['baseline_result_sha256'] == registered['baseline_terminal']['sha256']
         and stage['fit_calls'] == 0 and stage['fit_reused'] is True and stage['formal_holdouts'] == []
         and stage['selected_frames'] == registered['selected_frames'],
         'Current train lineage or protected diagnostic scope differs.')
    need(launch.check_source(COMMIT) == stage['changed_paths']
         and all(sha(launch.PROJECT/name) == digest for name,digest in stage['code_sha256'].items()),
         'Registered renderer source differs.')
    for name, key in (('fit/fitted.npz', 'fit'), ('data/training-inputs.json', 'training_inputs'),
                      ('fit/selection.json', 'selection')):
        pin(RUN/name, registered[key]['sha256'])

    old = Path(prior_plan['baseline_run'])
    winner_path = old/'fit/frames'/FRAME/'winner-camera-space.npz'
    report_path = winner_path.with_name('result.json')
    cfg = old/'config.json'
    source, observed = old/'data/inputs.json', old/'data/observations.npz'
    old_shared = old/'fit/shared-body.json'
    pins = {path:prior_plan['input_sha256'][str(path)] for path in
            (cfg, source, observed, old_shared, winner_path, report_path, old/'fit/selection.json')}
    for path,digest in pins.items():
        pin(path,digest)
    pin(RUN/'fit/shared-body.json', pins[old_shared])
    cfg = read(cfg)
    data = seq.load_inputs(source, observed, camera_length_contract=cfg['camera_length_contract'],
                           reference_frame=registered['reference_frame'], capture_layout=cfg['capture_layout'])
    slot = data['ids'].index(FRAME)
    source_row = data['rows'][slot]
    shared = read(old_shared)
    beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1, 10)
    chosen = stage['selected_frames']
    need(source_row['source_frame_index'] == SOURCE_INDEX and FRAME in data['scope']['fit_frame_ids']
         and FRAME not in chosen and FRAME not in data['scope']['retained_holdouts']
         and FRAME not in read(old/'fit/selection.json')['selected_frames']
         and FRAME not in {f for group in shared['anchor_groups'].values() for f in group},
         'The diagnostic was trained, used as a shape anchor or has a different identity.')
    pins.update({Path(source_row['image']):source_row['source_sha256'], Path(source_row['mask']):source_row['mask_sha256']})
    for name in ('image', 'mask'):
        path = Path(source_row[name])
        pin(path, pins[path])
    with np.load(winner_path, allow_pickle=False) as archive:
        winner = dict(archive)
    report = read(report_path)
    seq.check_result(report, source_row)
    train_receipts = read(RUN/'fit/official-fit-result.json')['frame_results']
    need({r['shared_body_target_sha256'] for r in train_receipts} == {report['shared_body_target_sha256']},
         'Historical fitted pose does not share the training target.')
    scores, flags, _, _ = selector._candidate(source_row, winner, data['targets'][slot], data['K'][slot],
                                               beta, None, report, screen_pose=True)
    need(not flags, 'The registered original pose screen rejected f016.')
    asset = launch.common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    pins[asset] = stage['asset_sha256'][str(asset)]
    pin(asset, pins[asset])
    body = smplx.SMPLX(str(asset), batch_size=1, num_betas=10, num_expression_coeffs=10,
        use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
        create_body_pose=False, create_transl=False, dtype=torch.float32).eval()
    subset = seq._selection_inputs(data, [FRAME], FRAME)
    pose, camera_rows = seq.assemble(subset, {FRAME:winner}, beta, seq.pelvis_basis(body))
    del body
    with np.load(RUN/'fit/fitted.npz', allow_pickle=False) as training:
        need(FRAME not in training['frame_ids'].tolist() and len(training['frame_ids']) == len(chosen)
             and np.array_equal(pose['shared_betas'], training['shared_betas'])
             and np.array_equal(pose['faces'], training['faces']), 'Same-shape stock pose assembly failed.')
    row, = camera_rows['records']
    need(np.array_equal(pose['vertices_camera'], winner['vertices_camera'])
         and row['K_integer_centers'] == source_row['K_integer_centers']
         and row['K'] == source_row['K'] and row['w2c'] == source_row['w2c'],
         'Archived vertices or native camera changed.')
    pose['w2c'] = pose['w2c'].copy()
    base, linear, clothing, _, max_error = _frozen_smpl_surface(pose, asset)
    with Image.open(row['image']) as image, Image.open(row['mask']) as image_mask:
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32)/255
        score = np.asarray(image_mask.convert('L'))
    need(rgb.shape[:2] == score.shape == tuple(row['wh'][::-1]), 'Wrong native diagnostic raster.')
    rois, support = regions(pose['observed_native_keypoints'][0], score, row['wh'])
    need(rois['face'].any(), 'No face pixels are supported in the registered diagnostic.')
    evidence = dict(state='diagnostic_inputs_verified', frame=FRAME, source_frame_index=SOURCE_INDEX,
        time_seconds=row['time_seconds'], scope='Gaussian-unseen selection candidate; source pose used fitted 2D observations.',
        formal_holdout=False, no_refit=True, plan_sha256=launch.ACCEPTANCE_SHA, stage_sha256=STAGE_SHA,
        winner_sha256=pins[winner_path], report_sha256=pins[report_path],
        source_image_sha256=row['source_sha256'], score_mask_sha256=row['mask_sha256'],
        shared_body_target_sha256=report['shared_body_target_sha256'], source_pins={str(p):h for p,h in pins.items()},
        native_K=row['K'], native_w2c=row['w2c'], native_wh=row['wh'], stock_screen_scores=scores,
        exact_shared_beta_and_topology=True, archived_LBS_replay_max_error_m=max_error,
        supported_face_slots=support, roi_pixels={k:int(v.sum()) for k,v in rois.items()})
    return SimpleNamespace(row=row, rgb=rgb, rois=rois, base=base, linear=linear, clothing=clothing,
                           asset=asset, pins=pins, stage=stage, evidence=evidence)


def terminal(state):
    from shuolee_portrait.bound_replay import _inputs
    from shuolee_portrait.head_appearance import snapshot_name

    marker = ROOT/'wrapper-terminal.json'
    if not marker.is_file():
        return None
    wrapper = read(marker)
    pin(RUN/'result.json', wrapper['result_sha256'])
    result = read(RUN/'result.json')
    train = read(RUN/'gaussian-run/result.json')
    need(wrapper['state'] == 'completed' and wrapper['candidate_commit'] == COMMIT
         and result['state'] == 'completed' and result['format_passed'] is True
         and result['fit_reused'] is True and result['fit_calls'] == 0
         and result['reused_fit_preregistration_sha256'] == STAGE_SHA
         and result['steps'] == train['steps'] == 10000 and train['stock_snapshot'] == STOCK
         and snapshot_name(train) == result['final_snapshot'] == wrapper['final_snapshot'] == TAIL
         and train['head_appearance'] == result['head_appearance'] == wrapper['head_appearance']
         and train['head_appearance']['frozen_ply_bytes_verified'] is True, 'Wrong stock/tail terminal.')
    for name,digest in state.stage['camera_pins'].items():
        pin(RUN/name, digest)
    paths = {key:RUN/'gaussian-run'/name for key,name in (('stock', STOCK), ('tail', TAIL))}
    delta = None
    for name,path in paths.items():
        manifest = read(path/'bound-replay.json')
        _, mesh, facts = _inputs(RUN, path, state.asset, manifest)
        pin(path/'bound-local.ply', manifest['bound_ply_sha256'])
        need(FRAME not in {row['frame'] for row in facts['frames']}, 'Diagnostic entered the Gaussian roster.')
        if name == 'stock':
            delta = mesh['static_offset']
        else:
            pin(path/'reference-mesh.npz', sha(paths['stock']/'reference-mesh.npz'))
            need(not np.any(delta[~state.clothing]), 'Geometry offsets crossed protected skin.')
    return paths, delta, wrapper['result_sha256']


def render(state, finished):
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    paths, delta, result_sha = finished
    output = ROOT/'native-unseen-f016'
    need(not output.exists(), 'Preserve any existing f016 review.')
    results, arrays, cameras = {}, {}, {}
    with launch.common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            need(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            rt.configure(state.asset.parents[1])
            from gaussian_renderer import render as raster

            posed = state.base.cuda() + (state.linear.cuda() @ torch.from_numpy(delta).cuda()[...,None]).squeeze(-1)
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            for arm,path in paths.items():
                model, manifest = load_bound_snapshot(RUN, path, asset_path=state.asset)
                pose_index = len(model.vertices_sequence)
                model.vertices_sequence = torch.cat((model.vertices_sequence, posed))
                model.num_timesteps = len(model.vertices_sequence)
                cam, cameras[arm] = camera(state.row, 0, pose_index=pose_index)
                (rgb, _, alpha), error = scale_pixels(raster, render_pose, cam, model, pipe, black, white, pose_index, 1.)
                results[arm] = dict(regions={name:region_metrics(rgb, alpha, state.rgb, roi)
                                             for name,roi in state.rois.items()}, composition_error=error)
                arrays[arm] = rgb, alpha, error
                need(model.optimizer is None and len(model.binding) == manifest['points'], 'Replay changed model.')
                del model
    need(cameras['stock'] == cameras['tail'], 'Native cameras differ.')
    changes = {}
    for name, before in results['stock']['regions'].items():
        after = results['tail']['regions'][name]
        if before is not None:
            changes[name] = dict(black_rgb_mae=after['black_rgb_mae']-before['black_rgb_mae'],
                alpha_below_095_fraction=after['alpha']['fraction_below_095']-before['alpha']['fraction_below_095'],
                mean_transmission=after['alpha']['mean_transmission']-before['alpha']['mean_transmission'])
    for path,digest in state.pins.items():
        pin(path,digest)
    pin(RUN/'result.json', result_sha)
    output.mkdir()
    panel = head_panel(output/'frame_000016-face.png', state.rgb, state.rois['face'], arrays)
    result = output/'result.json'
    launch.common.write(result, dict(state.evidence, state='native_gaussian_unseen_diagnostic_complete',
        camera=cameras['stock'], actual=results, tail_minus_stock=changes, native_face_panel=panel,
        candidate_terminal_sha256=result_sha,
        renderer_calls=6, optimizer_updates=0, refits=0, script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(result=str(result), sha256=sha(result))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'terminal-check', 'render'))
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(launch.PROJECT/'src'))
    state = inputs()
    finished = terminal(state)
    if args.action != 'check':
        need(finished is not None, 'Wait for the completed 016 common tail.')
    if args.action == 'render':
        render(state, finished)
    else:
        print(json.dumps(dict(state.evidence, terminal_ready=finished is not None, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()
