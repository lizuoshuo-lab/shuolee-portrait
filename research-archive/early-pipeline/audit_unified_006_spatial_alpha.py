"""Localize 006's matched held-out alpha spill without fitting or changing either model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

import review_006_cold_candidate_holdouts as cold
import review_unified_006_holdouts as unified
from review_bound_alpha_ab import pin, read, require, sha


LAB = cold.prior.LAB
DEFAULT = LAB/'shuolee-smpl-unified-006-spatial-alpha'
FRAMES = unified.PRIMARY
ARCHIVES = {
    'old': ('shuolee-smpl-sixworker-opacity-006-formal-frozen',
            'c35c4399b71f9687fe23b29f14684fddf119f44146f2f42658551622d6d0cdaf'),
    'new': ('shuolee-smpl-unified-006-protected-holdouts',
            'b557e5484eec35117a52d67f04a8fc005d87435c9268e38e9b511c694b3793d8'),
}
EDGES = (0, 2, 8, 24)
THRESHOLD = .5
ROWS = ('above_shoulders', 'shoulders_to_wrists', 'wrists_to_ankles', 'below_ankles')
LANDMARKS = ((2, 5), (4, 7), (11, 14))


def save(path, data):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False, ensure_ascii=False)+'\n', encoding='utf-8')
    pending.replace(path)


def sources():
    return {str(path): sha(path) for path in (Path(__file__), Path(cold.__file__), Path(unified.__file__),
            Path(cold.prior.__file__), Path(cold.scale_pixels.__code__.co_filename))}


def inspect(side):
    if side == 'old':
        receipt, state = cold.preflight(cold.RUN, cold.PROJECT)
        rows_key, terminal_key, bound_key = 'records', 'candidate_result_sha256', 'candidate_bound_manifest_sha256'
    else:
        receipt, state = unified.preflight()
        rows_key, terminal_key, bound_key = 'primary_records', 'terminal_result_sha256', 'terminal_bound_manifest_sha256'
    require(state is not None and receipt['state'] == 'frozen_formal_inputs_exact',
            f'Frozen holdout inputs unavailable for {side}.')
    name, expected = ARCHIVES[side]
    archive_path = LAB/name/'result.json'
    pin(archive_path, expected)
    archive = read(archive_path)
    for key in (terminal_key, bound_key, 'evaluation_fit_sha256'):
        require(receipt[key] == archive[key], f'Render archive no longer binds {side}: {key}.')
    archived = {item['frame']: item for item in archive[rows_key]}
    records = []
    state.records = [item for item in state.records if item.row['frame'] in FRAMES]
    require(tuple(item.row['frame'] for item in state.records) == FRAMES, 'Wrong protected comparison roster.')
    for item in state.records:
        frame = item.row['frame']
        past = archived[frame]
        pin(archive_path.parent/past['image'], past['image_sha256'])
        require(past['posed_camera'] == item.expected and past['pose_screen_passed'] is True
                and past['source_frame_index'] == item.row['source_frame_index']
                and past['native_source_sha256'] == item.row['source_sha256']
                and past['scoring_mask_sha256'] == item.baseline['scoring_mask_sha256'],
                f'Archive pose, image, mask or camera differs: {side}/{frame}.')
        pose_slot = state.poses['frame_ids'].tolist().index(frame)
        k = state.poses['observed_native_keypoints'][pose_slot]
        cuts = []
        for left, right in LANDMARKS:
            pair = k[[left, right]]
            require(np.isfinite(pair).all() and (pair[:, 2] >= .5).all(), 'Unsupported native ROI row boundary.')
            cuts.append(int(np.rint(pair[:, 1].mean())))
        require(0 < cuts[0] < cuts[1] < cuts[2] < item.mask.shape[0], 'Invalid native ROI partition.')
        mask = [Path(path) for path, digest in state.archive_pins.items()
                if digest == past['scoring_mask_sha256'] and Path(path).is_file()]
        require(len(mask) == 1, f'Expected one archived score mask for {frame}: {mask}.')
        records.append(dict(frame=frame, source_frame_index=past['source_frame_index'], image=item.row['image'],
                            source_sha256=past['native_source_sha256'], score_mask=str(mask[0]),
                            score_mask_sha256=past['scoring_mask_sha256'], camera=past['posed_camera'],
                            crop=list(item.crop), row_cuts=cuts, archived_metrics=past['metrics']))
    return dict(renderer_checkout=str(state.rt.PROJECT), source_review_sha256=expected,
                terminal_result_sha256=receipt[terminal_key], bound_manifest_sha256=receipt[bound_key],
                frozen_evaluation_fit_sha256=receipt['evaluation_fit_sha256'], records=records), state


def capture(output, side):
    spec, state = inspect(side)
    prereg = read(output/'preregistration.json')
    require(spec == prereg['arms'][side] and prereg['code_sha256'] == sources(),
            f'Preregistered source or model changed for {side}.')
    alpha_frames = []
    metric = state.helper['metrics']

    def capture_alpha(rgb, alpha, source, mask):
        alpha_frames.append(np.array(alpha, dtype=np.float32, copy=True))
        return metric(rgb, alpha, source, mask)

    state.helper = dict(state.helper, metrics=capture_alpha)
    destination = output/f'{side}-replay'
    rows, parity = cold.replay(state, destination)
    require(len(alpha_frames) == len(FRAMES) and tuple(row['frame'] for row in rows) == FRAMES,
            'Incomplete native alpha captures.')
    for row, locked in zip(rows, spec['records'], strict=True):
        actual, expected = row['metrics'], locked['archived_metrics']
        tolerance = dict(alpha_half_spill_pixels=4, alpha_half_mask_coverage=2e-5,
                         alpha_half_mask_iou=2e-5, foreground_rgb_mae=2e-5)
        require(all(abs(actual[key]-expected[key]) <= bound for key, bound in tolerance.items()),
                f'Native replay does not reproduce archived metrics for {side}/{row["frame"]}.')
    alpha_path = destination/'native-alpha.npz'
    np.savez_compressed(alpha_path, **dict(zip(FRAMES, alpha_frames, strict=True)))
    save(destination/'capture.json', dict(native_alpha_sha256=sha(alpha_path), optimizer_updates=0,
         refits=0, records=rows, reference_parity=parity, preregistration_sha256=sha(output/'preregistration.json')))


def comparison(old, new, mask, cut_rows):
    distance = distance_transform_edt(~mask)
    old_visible, new_visible = old >= THRESHOLD, new >= THRESHOLD
    added, removed = new_visible & ~old_visible, old_visible & ~new_visible

    def stats(region):
        old_n, new_n = int((old_visible & region).sum()), int((new_visible & region).sum())
        gain, loss = int((added & region).sum()), int((removed & region).sum())
        require(new_n-old_n == gain-loss, 'Thresholded accounting failed.')
        return dict(old=old_n, new=new_n, net=new_n-old_n, added=gain, removed=loss,
                    alpha_sum_old=float(old[region].sum(dtype=np.float64)),
                    alpha_sum_new=float(new[region].sum(dtype=np.float64)))

    def detached(visible):
        groups, _ = label(visible, structure=np.ones((3, 3), dtype=bool))
        connected = np.unique(groups[visible & mask])
        return visible & ~mask & ~np.isin(groups, connected)

    y = np.arange(mask.shape[0])[:, None]
    bands = [(f'gt_{low}_le_{high}', (distance > low) & (distance <= high))
             for low, high in zip(EDGES, EDGES[1:])]
    bands.append((f'gt_{EDGES[-1]}', distance > EDGES[-1]))
    old_detached, new_detached = detached(old_visible), detached(new_visible)
    require(sum(stats(region)['new'] for _, region in bands) == int((new_visible & ~mask).sum()),
            'The distance bands do not partition the exterior.')
    cuts = (0, *cut_rows, mask.shape[0])
    rows = {name: (y >= low) & (y < high) for name, low, high in zip(ROWS, cuts, cuts[1:])}
    result = dict(exterior=stats(~mask), foreground=stats(mask),
                  detached_from_rendered_foreground=dict(old=int(old_detached.sum()), new=int(new_detached.sum()),
                      added=int((new_detached & ~old_detached).sum()), removed=int((old_detached & ~new_detached).sum())),
                  exterior_by_distance={name: stats(region) for name, region in bands},
                  by_source_row={name: dict(exterior=stats(row & ~mask), foreground=stats(row & mask),
                     exterior_by_distance={name: stats(row & region) for name, region in bands}) for name, row in rows.items()})
    return result, added & ~mask, removed & ~mask


def plot(path, item, old, new, mask, added, removed):
    with Image.open(item['image']) as rgb:
        source = np.array(rgb.convert('RGB'))
    highlighted = source.copy()
    highlighted[added] = [255, 55, 55]
    highlighted[removed] = [31, 234, 239]
    crop = tuple(item['crop'])
    x0, y0, x1, y1 = crop
    w, h = x1-x0, y1-y0
    omitted = int(added.sum()+removed.sum()-added[y0:y1, x0:x1].sum()-removed[y0:y1, x0:x1].sum())
    page = Image.new('RGB', (5*w, h+23), '#171717')
    draw = ImageDraw.Draw(page)
    labels = ('Source', 'Archived score mask', '213ad16 alpha', '79b906f alpha', 'Exterior change: red added, cyan removed')
    values = (source, mask.astype(np.uint8)*255, np.rint(old*255).astype(np.uint8),
              np.rint(new*255).astype(np.uint8), highlighted)
    for i, (name, pixels) in enumerate(zip(labels, values, strict=True)):
        page.paste(Image.fromarray(pixels).convert('RGB').crop(crop), (i*w, 23))
        draw.text((i*w+4, 5), name, fill='white')
    page.save(path)
    return dict(file=path.name, sha256=sha(path), native_crop_xyxy=list(crop), exterior_changes_outside_crop=omitted)


def register(args):
    payloads = {}
    for side in ARCHIVES:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        child = subprocess.run([sys.executable, '-B', __file__, '_inspect', '--side', side],
                               env=env, text=True, capture_output=True, check=False)
        require(child.returncode == 0, child.stderr+child.stdout)
        payloads[side] = json.loads(child.stdout.splitlines()[-1])
    old, new = (payloads[side] for side in ARCHIVES)
    require(old['frozen_evaluation_fit_sha256'] == new['frozen_evaluation_fit_sha256'], 'Different fitted holdout poses.')
    for a, b in zip(old['records'], new['records'], strict=True):
        require(all(a[key] == b[key] for key in a if key not in ('archived_metrics', 'image')),
                'Mismatched native identity, score mask, camera, or ROI.')
    report = dict(schema='shuolee-006-spatial-alpha-v1', arms=payloads, alpha_threshold=THRESHOLD,
                  distance_band_edges_native_px=EDGES, row_bands=ROWS, native_joint_pairs=LANDMARKS,
                  view_scope='Native posed 1.0 scale. Two protected holdouts, report-only. Foreground in archived269 mask. '
                             'Row bins are coarse source-defined regions, not anatomical assignments.',
                  detached_definition='Eight-connected rendered components that never touch any archived foreground pixel.',
                  code_sha256=sources(), optimizer_updates=0, refits=0, gpu_calls=0)
    if args.action == 'register':
        args.output.mkdir(exist_ok=False)
        save(args.output/'preregistration.json', report)
    print(json.dumps(report), flush=True)


def run(output):
    prereg = output/'preregistration.json'
    plan = read(prereg)
    require(plan['code_sha256'] == sources() and not (output/'result.json').exists(), 'Unexpected diagnostic state.')
    receipts, alphas = {}, {}
    for side in ARCHIVES:
        destination = output/f'{side}-replay'
        if not (destination/'capture.json').exists():
            subprocess.run([sys.executable, '-B', __file__, '_capture', '--output', str(output), '--side', side], check=True)
        receipt = read(destination/'capture.json')
        require(receipt['preregistration_sha256'] == sha(prereg) and receipt['optimizer_updates'] == receipt['refits'] == 0,
                f'Invalid saved replay: {side}.')
        receipts[side] = sha(destination/'capture.json')
        with np.load(destination/'native-alpha.npz', allow_pickle=False) as planes:
            pin(destination/'native-alpha.npz', receipt['native_alpha_sha256'])
            alphas[side] = {frame: planes[frame] for frame in FRAMES}
    rows = []
    for item in plan['arms']['old']['records']:
        frame = item['frame']
        old, new = (alphas[side][frame] for side in ARCHIVES)
        pin(Path(item['image']), item['source_sha256'])
        pin(Path(item['score_mask']), item['score_mask_sha256'])
        with Image.open(item['score_mask']) as image:
            mask = np.asarray(image.convert('L')) >= 128
        require(old.shape == new.shape == mask.shape and np.isfinite(old).all() and np.isfinite(new).all(),
                'Native alpha dimensions or values differ.')
        values, added, removed = comparison(old, new, mask, item['row_cuts'])
        panel = plot(output/f'{frame}-alpha-spill.png', item, old, new, mask, added, removed)
        rows.append(dict(frame=frame, **values, panel=panel))
    result = output/'result.json'
    save(result, dict(state='completed', preregistration_sha256=sha(prereg), render_receipts_sha256=receipts,
                      records=rows, optimizer_updates=0, refits=0, quality_accepted=False))
    print(json.dumps(dict(receipt=str(result), sha256=sha(result))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'register', 'render', '_inspect', '_capture'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--side', choices=ARCHIVES)
    args = parser.parse_args()
    args.output = args.output.resolve()
    require(args.output.parent == LAB and args.output.name.startswith('shuolee-smpl-unified-006-spatial-alpha'),
            'Choose a dedicated spatial-alpha lab output.')
    if args.action == '_inspect':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Inspection must hide CUDA.')
        print(json.dumps(inspect(args.side)[0]), flush=True)
    elif args.action == '_capture':
        capture(args.output, args.side)
    elif args.action in ('check', 'register'):
        register(args)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
