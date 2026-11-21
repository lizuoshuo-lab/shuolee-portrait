"""Compare 008 same-run stock and head-tail at fixed native and synthetic cameras."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_frozen_fit_005_008 as launch
import review_face_photometric_006 as face
from audit_008_head_ownership import polygon
from review_bound_alpha_ab import mask_metrics, pin, read, require, sha, write_panel
from review_bound_scale_coverage import face_alpha, interior, scale_pixels


LAB = face.LAB
BASE = LAB/'shuolee-smpl-unified-008/run'
MATCHED = LAB/'shuolee-smpl-unified-008-matched-views'
OWNERSHIP = LAB/'shuolee-smpl-head-ownership-008/preregistration.json'
RUN = LAB/'shuolee-smpl-head-appearance-008/run'
OUT = RUN.parent/'reference-face-occiput'
FRONT, REAR = 'frame_000004', 'frame_000047'
CAMERA_KEYS = ('frame', 'pose_index', 'K', 'w2c', 'wh', 'pixel_convention')
ROI_PINS = {
    'native': (6159, '3e9cd35daa11c99c9e44863aaa3625189f5ba07827b3055b1aedbb243458b0a7'),
    'dolly': (25786, 'f4da8389c9b8e92bbcc5caee2b0715e518942adf56b516455a19dc7d7ae51560'),
    'occiput': (11310, 'e187f440f3552d3e2b2462dc2e5ae60c98e52bfe0da7218403ab077ec4cd02f4'),
}


def digest(mask):
    return hashlib.sha256(mask.tobytes()).hexdigest()


def native_view(row, pose, camera):
    require(pose == camera['pose_index'] and camera['pixel_convention'] == 'pixel_corners' and
            all(row[k] == camera[k] for k in ('frame', 'K', 'w2c', 'wh')), 'Native camera does not match its fit row.')
    pin(Path(row['image']), row['source_sha256'])
    pin(Path(row['mask']), row['mask_sha256'])
    with Image.open(row['image']) as im, Image.open(row['mask']) as matte:
        require(im.size == matte.size == tuple(row['wh']), 'Native source and mask shape differs.')
        rgb = np.asarray(im.convert('RGB'), dtype=np.uint8)
        mask = np.asarray(matte.convert('L'), dtype=np.float32)/255
    return SimpleNamespace(row=row, camera=camera, rgb=rgb, mask=mask, eyes={})


def inputs():
    plan = launch.acceptance(launch.ACCEPTANCE_SHA)
    spec = plan['subjects']['008']
    for key in ('baseline_terminal', 'baseline_wrapper', 'fit', 'training_inputs', 'selection',
                'training_cameras', 'portable_camera', 'prior_protected_diagnostics'):
        pin(Path(spec[key]['path']), spec[key]['sha256'])
    diagnostics = read(Path(spec['prior_protected_diagnostics']['path']))
    require(spec['formal_holdouts'] == diagnostics['formal_frames'] == [] and
            diagnostics['formal_score_computed'] is False and spec['failed_pose'] == diagnostics['diagnostic_frames']
            and set(spec['unavailable']) == set(diagnostics['unscored']), 'Unexpected protected diagnostic roles.')
    terminal = read(BASE/'result.json')
    replay_path = BASE/terminal['bound_replay']
    pin(replay_path, terminal['bound_replay_sha256'])
    replay = read(replay_path)
    rows = read(BASE/'data/training-inputs.json')['records']
    cameras = read(BASE/'gaussian-run/training-cameras.json')['records']
    require(len(rows) == len(cameras) == len(replay['records']) == 19 and spec['reference_frame'] == FRONT
            and spec['retained_holdouts'] == replay['retained_holdouts'] and
            set(spec['selected_frames']).isdisjoint(spec['retained_holdouts']), '008 frozen roster changed.')
    for row, camera, record in zip(rows, cameras, replay['records'], strict=True):
        require(all(row[k] == record[k] for k in ('frame', 'source_frame_index', 'source_sha256', 'mask_sha256'))
                and all(camera[k] == record['camera'][k] for k in CAMERA_KEYS), 'Source and replay identity differ.')
    front, rear = (native_view(rows[i], i, cameras[i]) for i in (0, 7))
    require(front.row['frame'] == FRONT and front.row['source_frame_index'] == 24 and
            rear.row['frame'] == REAR and rear.row['source_frame_index'] == 368, 'Fixed view/pose identities differ.')

    matched_sha = '9603e1b4d78b47661a4ccdaa3de108d0ed8a4da5f1ac51e2a48f0b684f851517'
    pin(MATCHED/'new-result.json', matched_sha)
    matched = read(MATCHED/'new-result.json')['records'][0]
    close = matched['closeup']
    require(matched['frame'] == FRONT and matched['source_sha256'] == front.row['source_sha256']
            and matched['own_mask_sha256'] == front.row['mask_sha256'] and close['kind'] == 'fixed_focal_dolly'
            and all(matched['native_camera'][k] == front.camera[k] for k in CAMERA_KEYS)
            and all(close['camera'][k] == front.camera[k] for k in CAMERA_KEYS if k != 'w2c')
            and close['min_depth_m'] > .2, 'Saved safe dolly and native camera differ.')
    pin(MATCHED/close['panel']['file'], close['panel']['sha256'])

    with np.load(BASE/'fit/fitted.npz', allow_pickle=False) as fit:
        require(fit['frame_ids'][0] == FRONT and fit['frame_ids'][7] == REAR and
                not (fit['observed_native_keypoints'][7, 67:118, 2] >= .5).any(), 'Rear face validity changed.')
        observed, joints = fit['observed_native_keypoints'][0, 67:118], fit['joints_camera'][0, 67:118]
    K = np.asarray(front.row['K'])
    good = (observed[:, 2] >= .5) & (joints[:, 2] > .2)
    uv = joints[good] @ K.T
    residual = np.linalg.norm(uv[:, :2]/uv[:, 2:]-.5-observed[good, :2], axis=1)
    change = np.asarray(close['camera']['w2c']) @ np.linalg.inv(np.asarray(front.row['w2c']))
    projected = (joints[good][residual <= 6] @ change[:3, :3].T + change[:3, 3]) @ K.T
    require(good.sum() == 51 and len(projected) == 49 and (projected[:, 2] > .2).all(), 'Saved face support differs.')
    pin(MATCHED/'plan.json', '29f9e2bdf63d335a703a6d54c4a14880bfb9b82ceb55d44386d7998ad48ce074')
    front.crop = read(MATCHED/'plan.json')['pairs'][0]['new']['head_crop']
    front.roi = interior(observed[good, :2], front.row['wh']) & (front.mask > .99)
    front.eyes, eyes_input_sha = face.eye_regions(SimpleNamespace(snapshot=replay_path.parents[1]), front.row,
                                                  front.rgb.astype(np.float32)/255, front.mask)
    near = SimpleNamespace(row=dict(front.row, w2c=close['camera']['w2c']), camera=close['camera'],
                           crop=close['panel']['crop_xyxy'], roi=interior(projected[:, :2]/projected[:, 2:]-.5, front.row['wh']))

    ownership_sha = '2aa4a97949b51799e01eb6b269714a1852ce431610b8369efd9e5f67ffbf9f23'
    pin(OWNERSHIP, ownership_sha)
    rear_region, = [row for row in read(OWNERSHIP)['views'] if row['frame'] == REAR]
    require(rear_region['role'] == 'train_occiput_control' and rear_region['source'] == rear.row['image']
            and all(rear_region['camera'][k] == rear.camera[k] for k in CAMERA_KEYS), 'The occiput control changed.')
    rear.crop = rear_region['crop_xyxy']
    x0, y0, x1, y1 = rear.crop
    rear.roi = np.zeros(rear.mask.shape, dtype=bool)
    rear.roi[y0:y1, x0:x1] = polygon(rear.crop, rear_region['polygon_xy'])
    require((rear.mask[rear.roi] > .99).all(), 'Occiput control includes non-foreground.')
    views = dict(native=front, dolly=near, occiput=rear)
    identities = {}
    for name, view in views.items():
        roi_sha = digest(view.roi)
        require((int(view.roi.sum()), roi_sha) == ROI_PINS[name], f'Registered native ROI changed: {name}')
        y, x = np.nonzero(view.roi)
        x0, y0, x1, y1 = view.crop
        require(x0 <= x.min() <= x.max() < x1 and y0 <= y.min() <= y.max() < y1, f'ROI leaves crop: {name}')
        identities[name] = dict(camera=view.camera, crop_xyxy=view.crop, roi_pixels=int(view.roi.sum()),
                                roi_sha256=roi_sha, source_sha256=view.row['source_sha256'],
                                mask_sha256=view.row['mask_sha256'])
    return spec, views, dict(acceptance_plan_sha256=launch.ACCEPTANCE_SHA, matched_view_sha256=matched_sha,
        region_plan_sha256=ownership_sha, views=identities, eye_observation_sha256=eyes_input_sha,
        eyes={key: dict(pixels=int(roi.sum()), sha256=digest(roi)) for key, roi in front.eyes.items()},
        formal_holdouts=[], invalid_pose_frames=spec['failed_pose'], unavailable=spec['unavailable'],
        dolly_rgb_truth_available=False, optimizer_updates=0, refits=0, quality_accepted=False)


def candidate(spec):
    wrapper = read(RUN.parent/'wrapper-terminal.json')
    pin(RUN/'result.json', wrapper['result_sha256'])
    result = read(RUN/'result.json')
    plan = read(RUN.parent/'preregistration.json')
    train = read(RUN/'gaussian-run/result.json')
    tail = train['head_appearance']
    require(wrapper['state'] == result['state'] == train['state'] == 'completed' and result['format_passed'] is True
            and wrapper['candidate_commit'] == plan['candidate_commit'] == launch.COMMIT
            and plan['acceptance_plan_sha256'] == launch.ACCEPTANCE_SHA
            and result['reused_fit_preregistration_sha256'] == sha(RUN.parent/'preregistration.json')
            and result['fit_calls'] == 0 and result['fit_reused'] is True
            and result['reference_frame_id'] == FRONT and result['frames'] == len(spec['selected_frames']) == 19
            and wrapper['selected_frames'] == plan['selected_frames'] == spec['selected_frames']
            and wrapper['head_appearance'] == result['head_appearance'] == tail
            and result['steps'] == train['steps'] == 10000 and result['sh_degree'] == train['sh_degree'] == 0
            and train['stock_snapshot'] == 'step-010000' and tail['state'] == 'completed' and tail['steps'] == 600
            and tail['frozen_ply_bytes_verified'] is True and train['final_snapshot'] == result['final_snapshot']
            == wrapper['final_snapshot'] == 'step-010000-head-000600', 'Expected the completed isolated 008 head tail.')
    for name, key in (('fit/fitted.npz', 'fit'), ('fit/selection.json', 'selection'),
                      ('data/training-inputs.json', 'training_inputs'), ('gaussian-run/training-cameras.json', 'training_cameras'),
                      ('export/cameras.json', 'portable_camera')):
        pin(RUN/name, spec[key]['sha256'])
    for key, digest_key in (('ply', 'sha256'), ('scene', 'scene_sha256'), ('bound_replay', 'bound_replay_sha256')):
        pin(RUN/result[key], result[digest_key])
    return result, wrapper['result_sha256'], dict(stock=train['stock_snapshot'], tail=train['final_snapshot'])


def render(spec, views, plan):
    result, result_sha, snapshots = candidate(spec)
    require(not OUT.exists(), 'Preserve existing 008 endpoint evidence.')
    launch.prior.check_source(launch.COMMIT)
    sys.path.insert(0, str(launch.prior.PROJECT/'src'))
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    rt.configure(face.ASSET.parents[1])
    from gaussian_renderer import render as raster

    reports, tiles, snapshot_hashes = {}, {}, {}
    with face.GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Use exactly one GPU.')
            OUT.mkdir()
            face.save(OUT/'plan.json', dict(plan, candidate_terminal_sha256=result_sha, snapshots=snapshots,
                                           script_sha256=sha(Path(__file__)), scale=1.))
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            replay_path = RUN/result['bound_replay']
            replay = read(replay_path)
            for arm, name in snapshots.items():
                snapshot = RUN/'gaussian-run'/name
                model, manifest = load_bound_snapshot(RUN, snapshot, asset_path=face.ASSET)
                require(model.optimizer is None and manifest['optimizer_updates'] == 0 and
                        manifest['reference_frame_id'] == FRONT and manifest['reference_pose_index'] == 0,
                        'Unexpected optimizer or reference pose.')
                if arm == 'tail':
                    pin(snapshot/'bound-replay.json', replay['bound_replay_sha256'])
                snapshot_hashes[arm] = dict(bound_manifest_sha256=sha(snapshot/'bound-replay.json'),
                                             bound_ply_sha256=manifest['bound_ply_sha256'])
                reports[arm], tiles[arm] = {}, {}
                for key, view in views.items():
                    pose = view.camera['pose_index']
                    cam, receipt = camera(view.row, pose, pose_index=pose)
                    require(all(receipt[k] == view.camera[k] for k in CAMERA_KEYS) and
                            manifest['frames'][pose]['source_sha256'] == view.row['source_sha256'], 'Render camera or frame differs.')
                    model.select_mesh_by_timestep(pose)
                    depth = model.get_xyz.new_tensor(view.row['w2c'][2])
                    min_depth = float((model.get_xyz @ depth[:3] + depth[3]).min())
                    require(key != 'dolly' or min_depth > .2, 'Dolly intersects a center cutoff.')
                    pixels, composition = scale_pixels(raster, render_pose, cam, model, pipe, black, white, pose, 1.)
                    metrics = dict(alpha=face_alpha(pixels[2], view.roi), camera=receipt, composition_error=composition,
                                   minimum_center_depth_m=min_depth)
                    if key != 'dolly':
                        rgb = view.rgb.astype(np.float32)/255
                        metrics.update(appearance=face.appearance(pixels[0], rgb, view.roi),
                                       foreground_rgb_mae=face.appearance(pixels[0], rgb, view.mask > .99)['black_rgb_mae'],
                                       full_raster_mask=mask_metrics(pixels[2], view.mask, np.zeros(view.mask.shape, bool)),
                                       eyes={label: face.appearance(pixels[0], rgb, roi) for label, roi in view.eyes.items()})
                        if arm == 'tail' or key == 'native':
                            record = replay['records'][pose]
                            archived = replay_path.parent/record['image'] if arm == 'tail' else snapshot/'reference-render.png'
                            if arm == 'tail':
                                pin(archived, record['image_sha256'])
                            with Image.open(archived) as saved:
                                expected = np.asarray(saved.convert('RGB'))
                            require(np.array_equal(expected, np.rint(np.clip(pixels[0], 0, 1)*255).astype(np.uint8)),
                                    'Native stock reference or final pose replay pixels changed.')
                            metrics['archived_native_max_abs_8bit'] = 0
                    reports[arm][key] = metrics
                    x0, y0, x1, y1 = view.crop
                    tiles[arm][key] = tuple(layer[y0:y1, x0:x1].copy() for layer in pixels)
                del model

    panels = {}
    for key, view in views.items():
        a, b = tiles['stock'][key], tiles['tail'][key]
        x0, y0, x1, y1 = view.crop
        if key == 'dolly':
            width, height = x1-x0, y1-y0
            page = Image.new('RGB', (3*width, 2*height))
            for i, layers in enumerate((a, b)):
                for j, layer in enumerate(layers):
                    tile = np.rint(np.clip(layer, 0, 1)*255).astype(np.uint8)
                    page.paste(Image.fromarray(tile).convert('RGB'), (j*width, i*height))
            target = OUT/'dolly.png'
            page.save(target)
            panels[key] = dict(file=target.name, sha256=sha(target))
        else:
            panels[key] = write_panel(OUT, f'{key}.png', view.rgb[y0:y1, x0:x1], view.mask[y0:y1, x0:x1],
                                      a, b, (0, 0, x1-x0, y1-y0))
        panels[key]['crop_xyxy'] = view.crop
    delta = {key: dict(alpha_below_095=reports['tail'][key]['alpha']['fraction_below_095']
                           - reports['stock'][key]['alpha']['fraction_below_095']) for key in views}
    for key in ('native', 'occiput'):
        a, b = (reports[arm][key] for arm in ('stock', 'tail'))
        delta[key].update(black_rgb_mae=b['appearance']['black_rgb_mae']-a['appearance']['black_rgb_mae'],
                          exterior_spill_at_half=b['full_raster_mask']['exterior_spill_pixels_at_half']
                          - a['full_raster_mask']['exterior_spill_pixels_at_half'])
    receipt = dict(plan, state='same_run_training_view_diagnostic_complete', snapshots=snapshots,
        candidate_terminal_sha256=result_sha, report_plan_sha256=sha(OUT/'plan.json'),
        snapshot_hashes=snapshot_hashes, reports=reports, panels=panels, tail_minus_stock=delta,
        render_calls=18, formal_score_computed=False, unresolved=['unseen-view appearance', 'held-out body and alpha spill',
        'head orientation on rejected poses', 'physical face correspondence'], visual_review='pending')
    face.save(OUT/'result.json', receipt)
    print(json.dumps(dict(result=str(OUT/'result.json'), sha256=sha(OUT/'result.json'), tail_minus_stock=delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    args = parser.parse_args()
    data = inputs()
    if args.action == 'check':
        ready = (RUN.parent/'wrapper-terminal.json').is_file()
        result_sha = candidate(data[0])[1] if ready else None
        print(json.dumps(dict(data[-1], gpu_calls=0, script_sha256=sha(Path(__file__)),
                              candidate_terminal_sha256=result_sha)))
    else:
        render(*data)


if __name__ == '__main__':
    main()
