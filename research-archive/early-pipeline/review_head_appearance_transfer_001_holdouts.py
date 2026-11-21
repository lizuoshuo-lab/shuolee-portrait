"""Compare the 001 common tail with its own stock snapshot on protected native views."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

import launch_head_appearance_frozen_fit as launch
import review_unified_001_005_frozen_holdouts as frozen
from review_bound_scale_coverage import face_alpha, interior, scale_pixels


need, read, sha, pin = frozen.need, frozen.read, frozen.sha, frozen.pin
ROOT = frozen.common.LAB/'shuolee-smpl-head-appearance-001'
RUN = ROOT/'run'
PROJECT = launch.PROJECT
PLAN = frozen.five.ARCHIVE.parents[1]/'robust-portrait-122/smpl-head-appearance-transfer-preregistration.json'
PLAN_SHA = 'd8d9ff902f1f0d312233d14076548e48554ed766f2ba29de1d6382ccf4dde55b'
STAGE_SHA = '7e50af4da5f18277844dea8b60aaf6f2abbf27d40c63f1d38354225c6fda2a10'
FRAMES = frozen.PRIMARY['001']
LIMITS = dict(rgb=.003, face_rgb=.003, exterior=.05, far_added=50)
EYES = {'eye_36_41': slice(86, 92), 'eye_42_47': slice(92, 98)}
REGION_PINS = {
    'frame_000033': dict(face='b1b2de0accd4bad5561bbd14bdcb1629fdc56b2e055ca934f8b4303c18030551',
        eye_36_41='29eee076242568f5a6a1c99083fde42395c496d77c5cfc7ca41efdcb95faf44c',
        eye_42_47='cae6c7ba4b9c3eadb28050509b95b99b755244701395c5765ef110f88dc32afe'),
    'frame_000049': dict(face='284ba06c009dcea4aff60ac066d4b0fa2782562b4be5e204512d5454fc54a956',
        eye_36_41='d0b1357f7f8839c452e372a83532d2f096e8f01d0670a685e3c4745a9be25b66',
        eye_42_47='f60a21ace29f1f39ef9c3e95f8575652036d1ce6d7b3cad10fa5691f1c835ae5'),
}
STOCK, TAIL = 'step-010000', 'step-010000-head-000600'


def registration():
    pin(PLAN, PLAN_SHA)
    plan = read(PLAN)
    subject = plan['subjects']['001']
    need(plan['state'] == 'registered_before_common_candidate_training'
         and plan['baseline_commit'] == frozen.queue.COMMIT
         and plan['one_shared_profile'] is True and plan['no_refit'] is True
         and plan['body_steps'] == 10000 and plan['head_appearance_steps'] == 600
         and plan['face_weight'] == .05 and plan['sh_degree'] == 0
         and subject['formal_holdouts'] == FRAMES,
         'The registered 001 transfer changed.')
    for name in ('baseline_terminal', 'fit', 'training_inputs', 'selection', 'baseline_holdout_receipt'):
        item = subject[name]
        pin(Path(item['path']), item['sha256'])
    return subject


def regions(keypoints, score, wh):
    face = keypoints[67:118]
    supported = np.isfinite(face).all(axis=1) & (face[:, 2] >= .5)
    supported &= ((face[:, :2] >= 0) & (face[:, :2] < wh)).all(axis=1)
    hull = interior(face[supported, :2], wh)
    foreground = score >= 253
    output = dict(face=(hull & foreground) if hull is not None else np.zeros_like(foreground))
    if output['face'].sum() < 32:
        output['face'][:] = False
    # Source-only review confirmed both eye contours visible in the two fixed
    # frames. 033's far eye is foreshortened. Require the complete polygon.
    for name, slots in EYES.items():
        points = keypoints[slots]
        xy = points[:, :2]
        region = np.zeros_like(foreground)
        if (np.isfinite(points).all() and (points[:, 2] >= .5).all()
                and ((xy >= 0) & (xy < wh)).all()):
            polygon = Image.new('1', tuple(wh))
            ImageDraw.Draw(polygon).polygon([tuple(point) for point in xy], fill=1)
            whole = np.asarray(polygon)
            if not (whole & ~foreground).any():
                region = whole
        output[name] = region
    return output, int(supported.sum())


def native_views(spec, poses):
    for item in spec['frames']:
        pin(Path(item['image']), item['source_sha256'])
        pin(Path(item['mask']), item['score_mask_sha256'])
        with Image.open(item['image']) as source, Image.open(item['mask']) as mask:
            rgb = np.asarray(source.convert('RGB'), dtype=np.float32)/255.
            score = np.asarray(mask.convert('L'))
        need(rgb.shape[:2] == score.shape == tuple(item['camera']['wh'][::-1]),
             'Frozen source and score mask sizes differ.')
        slot = item['camera_index']
        need(poses['frame_ids'][slot] == item['frame'] and
             poses['source_frames'][slot] == item['source_frame_index'] and
             poses['image_sha256'][slot] == item['source_sha256'] and
             np.array_equal(poses['w2c'][slot], item['camera']['w2c']),
             'Frozen RGB, pose or camera slot differs.')
        rois, points = regions(poses['observed_native_keypoints'][slot], score, item['camera']['wh'])
        facts = dict(supported_face_slots=points, roi_pixels={k:int(v.sum()) for k,v in rois.items()},
                     roi_sha256={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in rois.items()})
        need(facts['roi_sha256'] == REGION_PINS[item['frame']], 'Registered native ROI pixels changed.')
        yield item, rgb, score >= 128, rois, facts


def inputs():
    subject = registration()
    evidence, _, state, archive, _, _ = frozen.preflight('001', True)
    prior = read(Path(subject['baseline_holdout_receipt']['path']))
    need(prior['state'] == 'formal_frozen_replay_complete' and prior['exact_shared_beta'] is True
         and prior['zero_refits'] is True and prior['evaluation_fit_sha256'] == evidence['evaluation_fit_sha256'],
         'The historical frozen review does not bind the current evaluation inputs.')
    records = []
    for slot, (row, scoring, record) in enumerate(zip(state.subset['rows'], state.old_rows, prior['records'], strict=True)):
        if row['frame'] not in FRAMES:
            continue
        frame = row['frame']
        pose_index = len(subject['selected_frames']) + slot
        camera = record['posed_camera']
        need(record['frame'] == frame and record['role'] == 'primary_pose_pass'
             and record['pose_screen_passed'] is True
             and camera['pose_index'] == pose_index and camera['K'] == row['K']
             and camera['w2c'] == row['w2c'] and camera['wh'] == row['wh']
             and record['native_source_sha256'] == row['source_sha256']
             and record['scoring_mask_sha256'] == scoring['mask_sha256'],
             'The native protected camera, scoring mask or pose-screen role differs.')
        records.append(dict(frame=frame, source_frame_index=row['source_frame_index'],
            image=row['image'], source_sha256=row['source_sha256'], mask=str(state.helper['relocate'](scoring['mask'])),
            score_mask_sha256=scoring['mask_sha256'], camera_index=slot, camera=camera))
    need([row['frame'] for row in records] == FRAMES, 'Wrong registered 001 holdout order.')
    pose = archive/'evaluation-fitted.npz'
    pin(pose, evidence['evaluation_fit_sha256'])
    spec = dict(state='registered_native_inputs_verified', plan_sha256=PLAN_SHA,
        script_sha256=sha(Path(__file__)), pose_archive=str(pose), pose_sha256=evidence['evaluation_fit_sha256'],
        fit_sha256=subject['fit']['sha256'], score_helper=str(frozen.five.METRIC_HELPER),
        score_helper_sha256='72fb6738b6b6807dc0fe6d8124be41791eb31cdd3fc478a4349a139809a29f8a',
        historical_review_sha256=subject['baseline_holdout_receipt']['sha256'],
        historical_review_is_input_provenance_only=True, frames=records, limits=LIMITS, gpu_calls=0)
    with np.load(pose, allow_pickle=False) as archive_npz:
        for item, _, _, _, facts in native_views(spec, archive_npz):
            item.update(facts)
    return spec


def candidate():
    subject = registration()
    stage = ROOT/'preregistration.json'
    wrapper = ROOT/'wrapper-terminal.json'
    if not stage.is_file():
        return None
    pin(stage, STAGE_SHA)
    plan = read(stage)
    need(plan['schema'] == 'shuolee-head-appearance-frozen-fit-transfer-v1'
         and plan['acceptance_plan_sha256'] == PLAN_SHA and plan['baseline_commit'] == frozen.queue.COMMIT
         and plan['baseline_result_sha256'] == subject['baseline_terminal']['sha256']
         and plan['subject'] == '001' and plan['formal_holdouts'] == FRAMES
         and plan['selected_frames'] == subject['selected_frames']
         and plan['reference_frame'] == subject['reference_frame']
         and plan['fit_reused'] is True and plan['fit_calls'] == 0
         and plan['head_appearance_steps'] == 600 and plan['head_appearance_weight'] == .05,
         'The candidate wrapper did not complete the registered frozen fit.')
    for name, key in (('fit/fitted.npz', 'fit'), ('data/training-inputs.json', 'training_inputs'),
                      ('fit/selection.json', 'selection')):
        pin(RUN/name, subject[key]['sha256'])
    if not wrapper.is_file():
        return None
    terminal = read(wrapper)
    need(terminal['state'] == 'completed' and terminal['candidate_commit'] == plan['candidate_commit']
         and terminal['selected_frames'] == plan['selected_frames'] and terminal['fit_calls'] == 0
         and terminal['gaussian_from_scratch'] is True and terminal['cold_video_clock'] is False,
         'Wrong completed candidate wrapper.')
    pin(RUN/'result.json', terminal['result_sha256'])
    result = read(RUN/'result.json')
    gaussian = read(RUN/'gaussian-run/result.json')
    need(result['state'] == 'completed' and result['format_passed'] is True
         and result['reused_fit_preregistration_sha256'] == sha(stage)
         and result['reference_frame_id'] == subject['reference_frame']
         and result['fit_reused'] is True and result['fit_calls'] == 0
         and result['gaussian_from_scratch'] is True and result['steps'] == gaussian['steps'] == 10000
         and result['sh_degree'] == 0 and result['final_snapshot'] == gaussian['final_snapshot'] == terminal['final_snapshot']
         and result['head_appearance'] == gaussian['head_appearance'] == terminal['head_appearance'],
         'The candidate terminal has a different fit, profile or snapshot.')
    need(set(plan['camera_pins']) == set(launch.CAMERA_FILES), 'Unexpected camera delivery roster.')
    for filename, digest in plan['camera_pins'].items():
        pin(RUN/filename, digest)
    return plan, terminal, result, gaussian


def snapshots(candidate_state, poses):
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.head_appearance import APPEARANCE_FIELDS, _vertices, editable_points, snapshot_name
    from shuolee_portrait.training import _frozen_smpl_surface, verify_official_fit

    plan, terminal, result, gaussian = candidate_state
    commit = terminal['candidate_commit']
    need(rt.PROJECT == PROJECT and launch.check_source(commit) == plan['changed_paths']
         and all(sha(PROJECT/n) == h for n,h in plan['code_sha256'].items())
         and verify_official_fit(RUN)['fit_sha256'] == sha(RUN/'fit/fitted.npz'),
         'Candidate renderer or frozen official fit differs.')
    appearance = gaussian['head_appearance']
    need(snapshot_name(gaussian) == TAIL and gaussian['stock_snapshot'] == STOCK
         and appearance['frozen_ply_bytes_verified'] is True and appearance['face_weight'] == .05,
         'A completed head-only snapshot and its stock control are required.')
    asset = frozen.common.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    paths = {key:RUN/'gaussian-run'/name for key,name in (('stock', STOCK), ('tail', TAIL))}
    replay_path = RUN/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    need(replay_path.parents[1] == paths['tail'], 'Published bound replay is outside the tail snapshot.')
    replay = read(replay_path)
    pin(paths['tail']/'bound-replay.json', replay['bound_replay_sha256'])
    manifests = {key:read(path/'bound-replay.json') for key,path in paths.items()}
    before, after = (_inputs(RUN, paths[key], asset, manifests[key]) for key in paths)
    train_fit, mesh, facts = before
    need(facts['frames'] == after[2]['frames']
         and np.array_equal(poses['faces'], train_fit['faces'])
         and np.array_equal(poses['shared_betas'], train_fit['shared_betas'])
         and set(poses['frame_ids'].tolist()).isdisjoint(train_fit['frame_ids'].tolist())
         and sha(paths['stock']/'reference-mesh.npz') == sha(paths['tail']/'reference-mesh.npz'),
         'The stock, tail or heldout mesh/fitting identity differs.')
    base, linear, clothing, protected, _ = _frozen_smpl_surface(poses, asset)
    delta = mesh['static_offset']
    need(not np.any(delta[~clothing]), 'Learned offsets crossed protected vertices.')
    for key, path in paths.items():
        pin(path/'bound-local.ply', manifests[key]['bound_ply_sha256'])
    binding, _ = inspect_bound_ply(paths['stock']/'bound-local.ply', len(poses['faces']))
    editable = editable_points(protected, poses['faces'], binding)
    need(appearance['editable_points'] == int(editable.sum()) and
         appearance['frozen_points'] == int((~editable).sum()), 'Strict ownership counts differ.')
    changed = {}
    for filename in ('bound-local.ply', 'world.ply'):
        stock, tail = (_vertices(paths[key]/filename) for key in paths)
        need(stock.dtype == tail.dtype and len(stock) == len(tail) == len(editable)
             and stock[~editable].tobytes() == tail[~editable].tobytes()
             and all(stock[k].tobytes() == tail[k].tobytes() for k in stock.dtype.names if k not in APPEARANCE_FIELDS),
             'Frozen geometry, binding or non-head Gaussian bytes changed.')
        changed[filename] = int(np.logical_or.reduce([stock[k] != tail[k] for k in APPEARANCE_FIELDS]).sum())
        need(0 < changed[filename] <= int(editable.sum()), 'No editable appearance change.')
    identity = dict(candidate_commit=commit, terminal_sha256=terminal['result_sha256'],
        stage_sha256=sha(ROOT/'preregistration.json'), snapshot_sha256={
            key:sha(path/'bound-replay.json') for key,path in paths.items()},
        verified_training_and_portable_camera_sha256=plan['camera_pins'],
        changed_head_rows=changed, geometry_and_nonhead_rows_bitwise=True, reference_mesh_bitwise=True)
    return rt, asset, paths, base, linear, delta, identity


def region_metrics(rgb, alpha, source, mask):
    if not mask.any():
        return None
    return dict(black_rgb_mae=float(np.abs(rgb[mask]-source[mask]).mean()),
                alpha=face_alpha(alpha, mask))


def compare(item, source, mask, rois, arrays, metrics):
    before_rgb, before_alpha, before_error = arrays['stock']
    after_rgb, after_alpha, after_error = arrays['tail']
    before, after = metrics(before_rgb, before_alpha, source, mask), metrics(after_rgb, after_alpha, source, mask)
    far = distance_transform_edt(~mask) > 24
    old, new = before_alpha >= .5, after_alpha >= .5
    added, removed = int((new & ~old & far).sum()), int((~new & old & far).sum())
    region_values = {}
    for name, roi in rois.items():
        baseline = region_metrics(before_rgb, before_alpha, source, roi)
        candidate = region_metrics(after_rgb, after_alpha, source, roi)
        region_values[name] = dict(stock=baseline, tail=candidate,
            black_rgb_mae_delta=candidate['black_rgb_mae']-baseline['black_rgb_mae'] if candidate else None)
    rgb_delta = after['foreground_rgb_mae']-before['foreground_rgb_mae']
    face_delta = region_values['face']['black_rgb_mae_delta']
    gates = dict(rgb=rgb_delta <= LIMITS['rgb'], face_rgb=None if face_delta is None else face_delta <= LIMITS['face_rgb'],
                 exterior=after['alpha_half_spill_pixels'] <= before['alpha_half_spill_pixels']*(1+LIMITS['exterior']),
                 far_added=added <= LIMITS['far_added'])
    return dict(frame=item['frame'], camera=item['camera'], stock=before, tail=after, foreground_rgb_mae_delta=rgb_delta,
        face_and_eyes=region_values, exterior_far=dict(stock=int((old & far).sum()), tail=int((new & far).sum()),
            added=added, removed=removed, net=added-removed),
        composition_max_abs=dict(stock=before_error, tail=after_error), gates=gates,
        passed=all(gate is True for gate in gates.values()))


def head_panel(path, source, region, arrays):
    ys, xs = np.nonzero(region)
    if not len(xs):
        return None
    x0, x1 = max(0, xs.min()-12), min(source.shape[1], xs.max()+13)
    y0, y1 = max(0, ys.min()-12), min(source.shape[0], ys.max()+13)
    before, a0, _ = arrays['stock']
    after, a1, _ = arrays['tail']
    w, h = x1-x0, y1-y0
    panel = Image.new('RGB', (5*w, h+16), '#171717')
    draw = ImageDraw.Draw(panel)
    for i,(name,array) in enumerate(zip(('Source', 'Stock RGB', 'Tail RGB', 'Stock alpha', 'Tail alpha'),
                                       (source, before, after, a0, a1), strict=True)):
        pixels = np.rint(np.clip(array[y0:y1, x0:x1], 0, 1)*255).astype(np.uint8)
        panel.paste(Image.fromarray(pixels).convert('RGB'), (i*w,16))
        draw.text((i*w+2,2), name, fill='white')
    panel.save(path)
    return dict(file=path.name, sha256=sha(path), native_crop_xyxy=[int(v) for v in (x0,y0,x1,y1)])


def render(spec, state, output):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose

    rt, asset, paths, base, linear, delta, identity = state
    pin(Path(spec['score_helper']), spec['score_helper_sha256'])
    metrics = runpy.run_path(spec['score_helper'])['metrics']
    pin(Path(spec['pose_archive']), spec['pose_sha256'])
    with np.load(spec['pose_archive'], allow_pickle=False) as poses:
        views = list(native_views(spec, poses))
    collected = {item['frame']:{} for item,*_ in views}
    with frozen.common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            need(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            rt.configure(asset.parents[1])
            from gaussian_renderer import render as raster

            offset = torch.from_numpy(delta).cuda()[..., None]
            world = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(base.cuda(), linear.cuda(), strict=True)])
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            for name, snapshot in paths.items():
                pin(snapshot/'bound-replay.json', identity['snapshot_sha256'][name])
                model, manifest = load_bound_snapshot(RUN, snapshot, asset_path=asset)
                first = len(model.vertices_sequence)
                model.vertices_sequence = torch.cat((model.vertices_sequence, world))
                model.num_timesteps = len(model.vertices_sequence)
                for item, _, _, _, facts in views:
                    need(all(item[k] == v for k,v in facts.items()), 'Registered native ROI changed.')
                    expected = item['camera']
                    pose = first+item['camera_index']
                    row = dict(frame=item['frame'], image=item['image'], wh=expected['wh'],
                               K=expected['K'], w2c=expected['w2c'])
                    cam, actual = camera(row, item['camera_index'], pose_index=pose)
                    need(actual == expected, 'Replay camera differs from the frozen evaluator.')
                    (rgb, _, alpha), error = scale_pixels(raster, render_pose, cam, model, pipe, black, white, pose, 1.)
                    collected[item['frame']][name] = rgb, alpha, error
                need(model.optimizer is None and len(model.binding) == manifest['points'], 'Replay changed model.')
                del model
    records = [compare(item, rgb, mask, rois, collected[item['frame']], metrics)
               for item,rgb,mask,rois,_ in views]
    output.mkdir(exist_ok=False)
    for record,(item,rgb,_,rois,_) in zip(records,views,strict=True):
        record['native_face_panel'] = head_panel(output/(item['frame']+'-face.png'),
                                                rgb, rois['face'], collected[item['frame']])
    frozen.common.write(output/'result.json', dict(spec, **identity, state='protected_transfer_review_complete',
        records=records, native_views=4, raster_calls=12, optimizer_updates=0, evaluation_refits=0,
        protected_gate_passed=all(record['passed'] for record in records), quality_accepted=False))
    return output/'result.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'terminal-check', 'render', '_inputs'))
    args = parser.parse_args()
    if args.action != 'render':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    if args.action == '_inputs':
        print(json.dumps(inputs()), flush=True)
        return
    payload = subprocess.check_output([sys.executable, '-B', __file__, '_inputs'],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), text=True)
    spec = json.loads(payload.splitlines()[-1])
    candidate_state = candidate()
    state = None
    if candidate_state is not None:
        sys.path.insert(0, str(PROJECT/'src'))
        with np.load(spec['pose_archive'], allow_pickle=False) as poses:
            state = snapshots(candidate_state, dict(poses))
    if args.action != 'check':
        need(state is not None, 'Wait for the completed common tail terminal.')
    if args.action == 'render':
        output = ROOT/'protected-holdout-transfer'
        need(not output.exists(), 'Preserve any existing transfer receipt.')
        result = render(spec, state, output)
        print(json.dumps(dict(result=str(result), sha256=sha(result))), flush=True)
    else:
        print(json.dumps(dict(spec, terminal_ready=state is not None,
                              terminal_identity=state[-1] if state is not None else None)), flush=True)


if __name__ == '__main__':
    main()
