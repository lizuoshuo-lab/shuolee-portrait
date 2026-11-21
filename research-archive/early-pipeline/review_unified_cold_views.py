"""Render the delivered 016/006 cold runs and their archived baselines with no optimizer."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from review_bound_alpha_ab import delivered, mask_metrics, native_box, pin, read, require, sha
from review_bound_scale_coverage import face_dolly, scale_pixels


LAB = Path('/opt/gs-slam-lab')
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
GPU_LEASE = Path('/tmp/shuolee-native122-gpu.lock')
BASELINES = {
    '016': ('shuolee-smpl-bound-016', 'Shuolee-smpl-bound-replay',
            '9f1fae99da88a0c5f1c7a927bd58cef2b1dabb2d0f5d9f72a5071c91df86abaf',
            'shuolee-scale-coverage-016', 'a5aab4385cafd56fd771004fa5161d4a70e26d8159aa825a411e2d0e6e2cc96a'),
    '006': ('shuolee-smpl-sixworker-opacity-006', 'Shuolee-smpl-sixworker-opacity',
            '4b228f050170a172997a12fa7e4c2d06361877098c03232fa8011202e0f25683',
            'shuolee-scale-coverage-sixworker-006', '4f9dd0132733a716a212e33cf8cf0ceb68e8ee3cbc0740d271245be502ffc385'),
}
NATIVE_KEYS = ('frame', 'source_frame_index', 'source_sha256', 'K', 'w2c', 'wh')
POSE_KEYS = ('global_orient', 'body_pose_axis_angle', 'camera_translation', 'expression', 'jaw_pose',
             'left_hand_pose', 'right_hand_pose', 'leye_pose', 'reye_pose', 'vertices_world')


def save(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def planned_model(run, project, expected=None):
    terminal_path = run/'result.json'
    digest = pin(terminal_path, expected) if expected else sha(terminal_path)
    terminal = read(terminal_path)
    require(terminal['state'] == 'completed' and terminal['format_passed'] and terminal['steps'] == 10000,
            f'Expected a delivered 10,000-step result in {run}')
    rows_path = run/'data/training-inputs.json'
    rows = read(rows_path)['records']
    by_frame = {row['frame']: row for row in rows}
    holds = read(run/'fit/selection.json')['retained_holdouts']
    require(len(rows) == len(by_frame) and set(holds).isdisjoint(by_frame), 'Invalid protected training roster.')
    replay_path = run/terminal['bound_replay']
    pin(replay_path, terminal['bound_replay_sha256'])
    replay = read(replay_path)
    require(replay['retained_holdouts'] == holds, 'Terminal roster protection changed.')
    bound_path = replay_path.parents[1]/'bound-replay.json'
    pin(bound_path, replay['bound_replay_sha256'])
    bound = read(bound_path)
    pin(rows_path, bound['inputs_sha256']['training_rows'])
    for name, digest_source in bound['source_files_sha256'].items():
        pin(project/name, digest_source)
    fit_path = run/'fit/fitted.npz'
    pin(fit_path, bound['inputs_sha256']['fitted'])
    with np.load(fit_path, allow_pickle=False) as archive:
        fit = {name: archive[name] for name in (*POSE_KEYS, 'frame_ids', 'faces', 'shared_betas')}
    slots = {name: index for index, name in enumerate(fit['frame_ids'].tolist())}
    require(set(slots) == set(by_frame), 'Frozen pose slots differ from selected source frames.')
    provenance = dict(run=str(run), project=str(project), terminal_sha256=digest,
                      training_inputs_sha256=bound['inputs_sha256']['training_rows'],
                      fitted_sha256=bound['inputs_sha256']['fitted'], protected_holdouts=holds,
                      reference_frame_id=terminal['reference_frame_id'], points=terminal['points'],
                      gaussian_profile=terminal['gaussian_profile'], from_scratch_receipt=terminal['from_scratch'])
    return SimpleNamespace(provenance=provenance, rows=by_frame, fit=fit, slots=slots)


def pose_digest(model, frame):
    index = model.slots[frame]
    digest = hashlib.sha256()
    arrays = (model.fit['faces'], model.fit['shared_betas'], *(model.fit[key][index] for key in POSE_KEYS))
    for array in arrays:
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def audit_plan(args):
    old_dir, old_project, old_sha, audit_dir, audit_sha = BASELINES[args.subject]
    old = planned_model(LAB/old_dir/'run', Path('/mnt/d/Projects')/old_project, old_sha)
    new = planned_model(args.run, args.project)
    require(all(not args.output.is_relative_to(Path(old.provenance[key])) for key in ('run', 'project')),
            'Keep the comparison outside the frozen baseline and checkout.')
    require(args.preflight or args.run != Path(old.provenance['run']), 'A cold comparison needs its own delivered run.')
    audit_path = LAB/audit_dir/'result.json'
    pin(audit_path, audit_sha)
    audit = read(audit_path)
    require(audit['terminal_sha256'] == old_sha, 'The archived camera audit belongs to a different result.')
    common = {name for name in old.rows.keys() & new.rows.keys()
              if all(old.rows[name][key] == new.rows[name][key] for key in NATIVE_KEYS)}
    preferred = audit['records']
    require(len(preferred) == 3, 'Expected three locked reference/early/late audit frames.')
    labels = ('reference', 'early', 'late')
    unlocked_common = common - {saved['frame'] for saved in preferred}
    pairs = []
    for label, saved in zip(labels, preferred, strict=True):
        locked_frame = saved['frame']
        old_row = old.rows[locked_frame]
        require(all(saved[key] == old_row[key] for key in ('source_frame_index', 'source_sha256', 'mask_sha256')),
                'The old fixed camera audit is linked to different source bytes.')
        left_frame = locked_frame
        if locked_frame in common:
            right_frame = locked_frame
        elif unlocked_common:
            left_frame = right_frame = min(unlocked_common, key=lambda name: (
                abs(old.rows[name]['source_frame_index']-old_row['source_frame_index']), name))
            old_row = old.rows[left_frame]
            unlocked_common.remove(left_frame)
        elif label == 'reference':
            right_frame = new.provenance['reference_frame_id']
        else:
            ordered = sorted(new.rows.values(), key=lambda row: row['source_frame_index'])
            right_frame = ordered[0 if label == 'early' else -1]['frame']
        native_match = left_frame == right_frame and left_frame in common
        frames = {'old': left_frame, 'new': right_frame}
        sides = {}
        pose_shas = {side: pose_digest(model, frames[side]) for side, model in (('old', old), ('new', new))}
        for side, model in (('old', old), ('new', new)):
            row = model.rows[frames[side]]
            box_source = old_row if native_match else row
            x0, y0, x1, y1 = box_source['initial_box_xyxy']
            mask_row = old_row if native_match else row
            sides[side] = dict(frame=row['frame'], crop=native_box([x0-64, y0-64, x1+64, y1+64], row['wh']),
                               score_mask=mask_row['mask'], score_mask_sha256=mask_row['mask_sha256'],
                               own_mask_sha256=row['mask_sha256'], pose_sha256=pose_shas[side])
            if label == 'reference' and box_source.get('face_box_xyxy'):
                a, b, c, d = box_source['face_box_xyxy']
                sides[side]['head_crop'] = native_box([a-24, b-24, c+24, d+24], row['wh'])
        near = saved['closeup_1x']
        if label == 'reference':
            sides['old']['dolly'] = 'own'
            sides['new']['dolly'] = 'old' if native_match else 'own'
            if left_frame == locked_frame and near is not None:
                recorded = near['camera']
                require(all(recorded[key] == old_row[key] for key in ('frame', 'K', 'wh')),
                        'Archived dolly changed source pixel convention or focal length.')
                archived_panel, = [item for item in saved['panels'] if item['file'].endswith('-dolly.png')]
                pin(audit_path.parent/archived_panel['file'], archived_panel['sha256'])
                sides['old']['dolly'] = dict(w2c=recorded['w2c'], crop=archived_panel['crop_xyxy'])
                if native_match:
                    sides['new']['dolly'] = sides['old']['dolly']
        pairs.append(dict(label=label, archived_preferred_frame=locked_frame,
                          exact_native_source_and_camera=native_match,
                          exact_fitted_pose=native_match and pose_shas['old'] == pose_shas['new'], **sides))
    return dict(schema='shuolee-unified-cold-views-v1', subject=args.subject, asset=str(args.asset),
                old=old.provenance, new=new.provenance, baseline_camera_audit_sha256=audit_sha,
                exact_fitted_archive=old.provenance['fitted_sha256'] == new.provenance['fitted_sha256'],
                exact_native_common_training_frames=sorted(common), pairs=pairs)


def panel(path, crop, pixels, source=None, mask=None):
    x0, y0, x1, y1 = crop
    w, h = x1-x0, y1-y0
    require(w > 0 and h > 0, 'Empty diagnostic crop.')
    first = [source, mask] if source is not None else []
    items = first + [value for at_scale in pixels for value in at_scale]
    labels = (['Source RGB', 'Locked target mask'] if first else []) + [
        '1.0 / black', '1.0 / white', '1.0 / alpha', '0.5 / black', '0.5 / white', '0.5 / alpha']
    columns = 4 if first else 3
    image = Image.new('RGB', (columns*w, 2*(h+21)), '#171717')
    draw = ImageDraw.Draw(image)
    for i, (value, label) in enumerate(zip(items, labels, strict=True)):
        sample = value[y0:y1, x0:x1]
        if sample.dtype != np.uint8:
            sample = np.rint(np.clip(sample, 0, 1)*255).astype(np.uint8)
        x, y = (i % columns)*w, (i//columns)*(h+21)
        image.paste(Image.fromarray(sample).convert('RGB'), (x, y+21))
        draw.text((x+4, y+4), label, fill='white')
    image.save(path)
    return dict(file=path.name, sha256=sha(path), crop_xyxy=list(crop))


def worker(plan_path, side, lease_fd):
    require(os.path.samestat(os.fstat(lease_fd), GPU_LEASE.stat()), 'Worker did not inherit the native GPU lease.')
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(plan_path)
    job, out = plan[side], plan_path.parent
    root, asset = Path(job['run']), Path(plan['asset'])
    pin(root/'result.json', job['terminal_sha256'])
    pin(root/'data/training-inputs.json', job['training_inputs_sha256'])
    pin(root/'fit/fitted.npz', job['fitted_sha256'])
    sys.path.insert(0, str(Path(job['project'])/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _load_model
    from shuolee_portrait.training import camera, render_pose

    rows = read(root/'data/training-inputs.json')['records']
    saved = delivered(root, asset, rows, job['protected_holdouts'])
    require(saved.facts['source_files_sha256'] == saved.bound['source_files_sha256'], 'Renderer source changed.')
    by_frame = {row['frame']: (i, row) for i, row in enumerate(rows)}
    import torch
    from gaussian_renderer import render

    rt.configure(asset.parents[1])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black, white = torch.zeros(3, device='cuda'), torch.ones(3, device='cuda')
    evidence = []
    with torch.no_grad():
        model, parity = _load_model(saved.snapshot/'bound-local.ply', saved.snapshot, saved.fit,
                                    saved.mesh, saved.facts, asset)
        for entry in plan['pairs']:
            role, spec = entry['label'], entry[side]
            i, row = by_frame[spec['frame']]
            pose = saved.facts['frames'][i]['pose_index']
            model.select_mesh_by_timestep(pose)
            points = model.get_xyz.clone()
            pin(Path(row['image']), row['source_sha256'])
            pin(Path(row['mask']), row['mask_sha256'])
            pin(Path(spec['score_mask']), spec['score_mask_sha256'])
            with Image.open(row['image']) as im, Image.open(spec['score_mask']) as target:
                require(list(im.size) == list(target.size) == row['wh'], 'Native comparison image size changed.')
                rgb = np.array(im.convert('RGB'))
                mask = np.asarray(target.convert('L'), dtype=np.float32)/255.

            def raster(camera_row):
                cam, receipt = camera(camera_row, i, pose_index=pose)
                first, error_one = scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, 1.)
                half, error_half = scale_pixels(render, render_pose, cam, model, pipe, black, white, pose, .5)
                return (first, half), receipt, [error_one, error_half]

            pixels, native, composition = raster(row)
            archive = saved.replay['records'][i]
            with Image.open(saved.snapshot/'posed-replay'/archive['image']) as im:
                prior = np.asarray(im.convert('RGB'), dtype=np.int16)
            error = int(abs(np.rint(np.clip(pixels[0][0], 0, 1)*255).astype(np.int16)-prior).max())
            require(error <= 1, 'The 1.0 replay differs from its delivered posed archive.')
            zero = np.zeros(mask.shape, dtype=bool)
            stats = [{k: v for k, v in mask_metrics(value[2], mask, zero).items() if not k.startswith('dark_eye_')}
                     for value in pixels]
            native_panel = panel(out/f'{role}-{side}-native.png', spec['crop'], pixels, rgb, mask)
            head = (dict(kind='native_head_crop', camera=native,
                         panel=panel(out/f'{role}-{side}-head.png', spec['head_crop'], pixels, rgb, mask))
                    if 'head_crop' in spec else None)
            del pixels
            close = spec.get('dolly')
            if close == 'old':
                original = read(out/'old-result.json')['records'][0]['closeup']
                close = (dict(w2c=original['camera']['w2c'], crop=original['panel']['crop_xyxy'])
                         if original and original['kind'] == 'fixed_focal_dolly' else None)
            if close == 'own':
                dolly = face_dolly(row, saved.fit['observed_native_keypoints'][pose, 67:118],
                                   saved.fit['joints_camera'][pose, 67:118], points.cpu().numpy())
                close = dict(w2c=dolly[0]['w2c'], crop=dolly[2]) if dolly else None
            near = None
            if close:
                matrix = points.new_tensor(close['w2c'])
                min_depth = float((points @ matrix[2, :3] + matrix[2, 3]).min())
                if min_depth > .2:
                    pixels, near_camera, near_composition = raster(dict(row, w2c=close['w2c']))
                    near = dict(kind='fixed_focal_dolly',
                                panel=panel(out/f'{role}-{side}-dolly.png', close['crop'], pixels),
                                camera=near_camera, composition_max_abs=near_composition, min_depth_m=min_depth)
                    del pixels
            require(torch.equal(model.get_xyz, points) and model.optimizer is None, 'Render-only check changed geometry.')
            evidence.append(dict(role=role, frame=row['frame'], source_frame_index=row['source_frame_index'],
                                 source_sha256=row['source_sha256'], own_mask_sha256=row['mask_sha256'],
                                 score_mask_sha256=spec['score_mask_sha256'], pose_sha256=spec['pose_sha256'],
                                 native_camera=native, native_panel=native_panel, closeup=near or head,
                                 closeup_fallback_reason=('dolly_cutoff_or_unavailable' if head and not near else None),
                                 native_archive_max_abs_8bit=error, native_composition_max_abs=composition,
                                 native_mask_metrics_by_scale=dict(zip(('1.0', '0.5'), stats, strict=True))))
    save(out/f'{side}-result.json', dict(model_parity=parity, optimizer_updates=0, records=evidence))


def comparison(output, entry, left, right, key, new_label):
    first = left['native_panel'] if key == 'native' else (left['closeup'] or {}).get('panel')
    second = right['native_panel'] if key == 'native' else (right['closeup'] or {}).get('panel')
    if first is None or second is None:
        return None
    same_camera = entry['exact_native_source_and_camera']
    if key == 'closeup':
        same_camera = (same_camera and left['closeup']['kind'] == right['closeup']['kind']
                       and all(left['closeup']['camera'][key] == right['closeup']['camera'][key]
                               for key in ('K', 'w2c', 'wh')))
    scope = ('Exact source camera and fitted pose' if same_camera and entry['exact_fitted_pose'] else
             'Exact camera, separately fitted poses' if same_camera else 'Different views, visual reference only')
    with Image.open(output/first['file']) as a, Image.open(output/second['file']) as b:
        page = Image.new('RGB', (max(a.width, b.width), a.height+b.height+64), '#262626')
        draw = ImageDraw.Draw(page)
        draw.text((8, 6), f'{scope} | Old {left["frame"]}', fill='white')
        page.paste(a, (0, 28))
        draw.text((8, a.height+33), f'{new_label} {right["frame"]}', fill='white')
        page.paste(b, (0, a.height+56))
        path = output/f'{entry["label"]}-old-new-{key}.png'
        page.save(path)
    return dict(file=path.name, sha256=sha(path), camera_matched=same_camera,
                pose_matched=same_camera and entry['exact_fitted_pose'], scope=scope)


def main():
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', required=True, choices=BASELINES)
    for key in ('run', 'project', 'output'):
        parser.add_argument('--'+key, required=True, type=lambda value: Path(value).resolve())
    parser.add_argument('--asset', type=Path, default=ASSET)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_relative_to(args.run)
            and not args.output.is_relative_to(args.project), 'Use a fresh output separate from the run and checkout.')
    plan = audit_plan(args)
    if args.preflight:
        print(json.dumps(dict(plan=plan, gpu_calls=0)))
        return
    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output.mkdir(parents=True, exist_ok=False)
        plan_path = args.output/'plan.json'
        save(plan_path, plan)
        for side in ('old', 'new'):
            with (args.output/f'{side}.log').open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-B', __file__, '--worker', str(plan_path), side, str(lease.fileno())],
                               pass_fds=(lease.fileno(),), stdout=log, stderr=subprocess.STDOUT, check=True)
    old, new = (read(args.output/f'{side}-result.json') for side in ('old', 'new'))
    comparisons = []
    new_label = 'Cold run' if plan['new']['from_scratch_receipt'] else 'Frozen fit, fresh Gaussians'
    for entry, left, right in zip(plan['pairs'], old['records'], new['records'], strict=True):
        comparisons.append(dict(label=entry['label'],
                                native=comparison(args.output, entry, left, right, 'native', new_label),
                                closeup=comparison(args.output, entry, left, right, 'closeup', new_label)))
    report = args.output/'result.json'
    save(report, dict(state='completed', plan_sha256=sha(plan_path), reviewer_sha256=sha(Path(__file__)),
                      old=old, new=new, comparisons=comparisons, optimizer_updates=0, quality_accepted=False,
                      scope='Native training views and synthetic fixed-focal dolly. No held-out validation or single-variable attribution.'))
    print(json.dumps(dict(comparison_receipt=str(report), sha256=sha(report))), flush=True)


if __name__ == '__main__':
    main()
