"""Score the completed 001 opacity fork on the original two primary holdouts."""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

import review_001_opacity_tail_formal_holdouts as formal
from review_bound_alpha_ab import pin, read, require, sha
from review_bound_scale_coverage import scale_pixels
from review_opacity_fork001 import ARMS, validate_fork


LAB = Path('/opt/gs-slam-lab')
TRAIN_REVIEW_SHA = 'fbb17c51c48af2680f73dade316391c8ed73995a875fa77b03f686e0b77b80bb'
PRIMARY = ('frame_000033', 'frame_000049')
METRICS = ('foreground_rgb_mae', 'alpha_half_mask_iou', 'alpha_half_mask_coverage',
           'alpha_half_spill_pixels', 'mean_foreground_alpha', 'foreground_fraction_below_095')


def preflight(fork, project):
    reference = LAB/'shuolee-scale-coverage-001/result.json'
    rt, asset, _, arms, evidence = validate_fork(fork, project, reference)
    training = LAB/'shuolee-smpl-opacity-fork-001-review/result.json'
    pin(training, TRAIN_REVIEW_SHA)
    checked = read(training)
    require(checked['state'] == 'completed' and checked['optimizer_updates'] == 0
            and all(checked[key] == evidence[key] for key in
                    ('fork_result_sha256', 'checkpoint_sha256', 'camera_sha256', 'recovery_receipt_sha256'))
            and all(checked['records'][name]['terminal'] == arm.terminal
                    and checked['records'][name]['manifest_sha256'] == arm.manifest_sha256
                    for name, arm in arms.items()), 'The training-view review has a different terminal or fork.')

    _, _, protected, primary, outside, state = formal.preflight(True)
    require(tuple(primary) == PRIMARY and rt.PROJECT == project and state.asset.resolve() == asset,
            'Primary holdout roster, replay checkout or model asset changed.')
    from shuolee_portrait.training import camera

    with np.load(state.pose_archive/'evaluation-fitted.npz', allow_pickle=False) as source:
        poses = dict(source)
    require(poses['frame_ids'].tolist() == list(protected)
            and all(set(protected).isdisjoint(frame['frame'] for frame in arm.facts['frames'])
                    for arm in arms.values()), 'Heldout pose order changed or leaked into training.')
    records = {}
    for frame in PRIMARY:
        i = list(protected).index(frame)
        row, old, archived = state.subset['rows'][i], state.old_rows[i], state.old_records[i]
        prior = state.baseline[frame]
        pose = len(next(iter(arms.values())).facts['frames']) + i
        expected = prior['posed_camera']
        require(camera(row, i, pose_index=pose)[1] == expected
                and prior['native_source_sha256'] == row['source_sha256']
                and prior['scoring_mask_sha256'] == old['mask_sha256'], f'Archived source camera changed: {frame}')
        image_path, mask_path = Path(row['image']), state.helper['relocate'](old['mask'])
        pin(image_path, row['source_sha256'])
        pin(mask_path, old['mask_sha256'])
        with Image.open(image_path) as source, Image.open(mask_path) as silhouette:
            require(list(source.size) == list(silhouette.size) == row['wh'], f'Native raster changed: {frame}')
            image = source.convert('RGB')
            mask = np.asarray(silhouette.convert('L')) >= 128
        crop = tuple(archived['crop_xyxy'])
        x0, y0, x1, y1 = crop
        require(0 <= x0 < x1 <= row['wh'][0] and 0 <= y0 < y1 <= row['wh'][1] and mask.any(),
                f'Invalid archived scoring crop or mask: {frame}')
        records[frame] = SimpleNamespace(i=i, row=row, pose=pose, expected=expected, prior=prior,
                                         image=image, mask=mask, crop=crop)
    evidence.update(training_review_sha256=TRAIN_REVIEW_SHA, primary_frames=list(PRIMARY),
                    protected_frames=protected, outside_arc_not_rendered=outside, script_sha256=sha(Path(__file__)),
                    evaluation_fit_sha256=sha(state.pose_archive/'evaluation-fitted.npz'))
    return rt, asset, arms, state, poses, records, evidence


def panel(path, record, pictures):
    labels = ('Original RGB / score mask', 'Reset', 'Reset repeat', 'Skip step 9000 reset')
    columns = [(record.image.crop(record.crop), Image.fromarray(record.mask).convert('RGB').crop(record.crop))]
    columns.extend(pictures[name] for name in ARMS)
    w, h = columns[0][0].size
    header = 23
    sheet = Image.new('RGB', (4*w, 2*h+header), '#171717')
    draw = ImageDraw.Draw(sheet)
    for i, (label, (rgb, alpha)) in enumerate(zip(labels, columns, strict=True)):
        draw.text((i*w+5, 5), label, fill='white')
        sheet.paste(rgb, (i*w, header))
        sheet.paste(alpha, (i*w, header+h))
    sheet.save(path)


def replay(rt, asset, arms, state, poses, records, output):
    import torch
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import _frozen_smpl_surface, camera, render_pose

    values = {frame: {} for frame in PRIMARY}
    pictures = {frame: {} for frame in PRIMARY}
    parities = {}
    with formal.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU for the audit.')
            rt.configure(asset.parent.parent)
            from gaussian_renderer import render

            base, linear, clothing, _, _ = _frozen_smpl_surface(poses, asset)
            base, linear = base.cuda(), linear.cuda()
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
            for name, arm in arms.items():
                pin(arm.snap/'bound-replay.json', arm.manifest_sha256)
                pin(arm.snap/'bound-local.ply', arm.terminal['bound_ply_sha256'])
                pin(arm.snap/'world.ply', arm.terminal['world_ply_sha256'])
                model, parities[name] = _load_model(arm.snap/'bound-local.ply', arm.snap, arm.fitted,
                                                    arm.mesh, arm.facts, asset)
                delta = arm.mesh['static_offset']
                require(not np.any(delta[~clothing]), f'Offset changed protected skin: {name}')
                offset = torch.from_numpy(delta).cuda()[..., None]
                posed = torch.stack([v+(a@offset).squeeze(-1) for v,a in zip(base,linear,strict=True)])
                model.vertices_sequence = torch.cat((model.vertices_sequence, posed))
                model.num_timesteps = len(model.vertices_sequence)
                for frame, row in records.items():
                    cam, actual = camera(row.row, row.i, pose_index=row.pose)
                    require(actual == row.expected, f'Camera changed during replay: {frame}')
                    pixels, error = scale_pixels(render, render_pose, cam, model, pipe, black, white, row.pose, 1.)
                    rgb, _, alpha = pixels
                    source = np.asarray(row.image, dtype=np.float32)/255
                    value = state.helper['metrics'](rgb, alpha, source, row.mask)
                    value.update(mean_foreground_alpha=float(alpha[row.mask].mean()),
                                 foreground_fraction_below_095=float((alpha[row.mask] < .95).mean()),
                                 black_white_composition_error=error)
                    values[frame][name] = value
                    x0,y0,x1,y1 = row.crop
                    def raster(array):
                        return Image.fromarray(np.rint(np.clip(array[y0:y1,x0:x1],0,1)*255).astype(np.uint8)).convert('RGB')
                    pictures[frame][name] = (raster(rgb), raster(alpha))
                require(model.optimizer is None, f'Replay created an optimizer: {name}')
                del model
    formal.check_pins(state.pins)
    output.mkdir(parents=True, exist_ok=False)
    result = []
    for frame, row in records.items():
        path = output/(frame+'-source-three-posed-arms.png')
        panel(path, row, pictures[frame])
        stock, repeat, skip = [values[frame][arm] for arm in ARMS]
        deltas = {key: dict(repeat_minus_stock=repeat[key]-stock[key],
                            skip_minus_stock=skip[key]-stock[key],
                            skip_minus_repeat=skip[key]-repeat[key]) for key in METRICS}
        result.append(dict(frame=frame, source_frame_index=row.row['source_frame_index'],
                           source_sha256=row.row['source_sha256'], scoring_mask_sha256=row.prior['scoring_mask_sha256'],
                           fit_mask_sha256=row.prior['fit_mask_sha256'], posed_camera=row.expected,
                           pose_screen_passed=row.prior['pose_screen_passed'],
                           pose_screen_flags=row.prior['pose_screen_flags'], crop_xyxy=list(row.crop),
                           image=path.name, image_sha256=sha(path), metrics=values[frame], deltas=deltas))
    return result, parities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fork', type=Path, default=LAB/'shuolee-smpl-opacity-fork-001')
    parser.add_argument('--project', type=Path, default=Path('/mnt/d/Projects/Shuolee-smpl-opacity-fork'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    fork, project, output = (path.resolve() for path in (args.fork, args.project, args.output))
    require(not output.exists() and not any(output.is_relative_to(path) for path in
            (fork, fork.with_name(fork.name+'-recovered'), project, formal.POSES, formal.BASE.parent)),
            'Use a fresh output outside the model, evaluation and source roots.')
    rt, asset, arms, state, poses, rows, evidence = preflight(fork, project)
    if args.preflight:
        print(json.dumps(dict(state='primary_heldout_inputs_verified', arms=list(arms), gpu_calls=0,
                              evaluation_refits=0, optimizer_updates=0, **evidence)))
        return
    records, parities = replay(rt, asset, arms, state, poses, rows, output)
    target = output/'result.json'
    rt.save(target, dict(state='primary_heldout_replay_complete', records=records, load_parity=parities,
                         terminal_models={name: arm.terminal for name,arm in arms.items()},
                         optimizer_updates=0, evaluation_refits=0, quality_accepted=False,
                         scope='Two registered in-arc holdouts. All three terminal models use the exact archived independent pose and native camera. Metrics use the same archived scoring masks.',
                         panel_layout='Original, reset, reset repeat, skip reset. Top RGB, bottom scoring mask or accumulated alpha.',
                         **evidence))
    print(json.dumps(dict(receipt=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
