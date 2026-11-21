"""Verify and append image-only records for the pinned head-appearance transfers."""

import argparse
import json
from pathlib import Path
import re
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from archive_unified_bound_records import record_jpeg, require, sha
from archive_unified_queue_records import contact_sheet


COMMIT = '762d6d0b2a5db804d8d4de94c53779b2a5eb80b7'
TAG = '头部精修762d6d0'
FINAL = 'step-010000-head-000600'
LAB = Path('/opt/gs-slam-lab')
RECORDS = Path('/mnt/d/00/记录')
FRAME_COUNTS = {'001': 24, '005': 23, '008': 19, '016': 24}
POSE_KEYS = ('frame', 'pose_index', 'source_frame_index', 'source_sha256', 'mask_sha256')
CAMERA_KEYS = ('frame', 'pose_index', 'wh', 'K', 'w2c', 'pixel_convention')


def comparison_jpeg(sources, crop, wh, target=None):
    images = []
    for path in sources:
        with Image.open(path) as image:
            require(image.size == tuple(wh), f'Native reference raster differs: {path}')
            images.append(image.convert('RGB'))
    require(ImageChops.difference(images[2], images[3]).getbbox() is None,
            'Saved final reference does not match exact posed replay.')
    if target is not None:
        x0, y0, x1, y1 = crop
        width, height = x1 - x0, y1 - y0
        result = Image.new('RGB', (3 * width, height))
        for i, image in enumerate(images[:3]):
            result.paste(image.crop(crop), (i * width, 0))
        result.save(target, quality=95, subsampling=0)


def publish_images(files, pins):
    prepared = []
    for source, target in files:
        with Image.open(source) as image:
            require(image.format == 'JPEG', f'Expected a JPEG: {source}')
            image.verify()
        require(not target.exists() and not target.is_symlink(), f'Existing record preserved: {target}')
        prepared.append((source, target, sha(source)))
    require(all(sha(path) == digest for path, digest in pins.items()), 'A source changed during conversion.')
    created = []
    try:
        for source, target, digest in prepared:
            with source.open('rb') as original, target.open('xb') as copy:
                created.append(target)
                copyfileobj(original, copy)
            require(sha(target) == digest, f'Record hash changed: {target}')
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return [dict(path=str(path), sha256=digest) for _, path, digest in prepared]


def archive(subject, approved_result_sha256=None, *, root=None, expected_count=None, tag=TAG):
    count = FRAME_COUNTS[subject] if expected_count is None else expected_count
    root = LAB/f'shuolee-smpl-head-appearance-{subject}' if root is None else root
    run = root/'run'
    dest = RECORDS/subject
    require(dest.resolve(strict=True) == dest and dest.is_dir(), 'Use the existing subject record directory.')
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha(path)
        require((expected is None or expected == actual) and pins.get(path, actual) == actual,
                f'Hash mismatch: {path}')
        pins[path] = actual
        return path

    def read(path, expected=None):
        return json.loads(pin(path, expected).read_text(encoding='utf-8-sig'))

    terminal = read(root/'wrapper-terminal.json')
    plan = read(root/'preregistration.json')
    result_sha256 = terminal['result_sha256']
    require(approved_result_sha256 is None or approved_result_sha256 == result_sha256,
            'The approved terminal SHA does not match the current run.')
    result = read(run/'result.json', result_sha256)
    status = read(run/'status.json')
    training = read(run/'gaussian-run/result.json')
    tail = training['head_appearance']
    require(terminal['state'] == status['state'] == result['state'] == training['state'] == 'completed'
            and plan['schema'] == 'shuolee-head-appearance-frozen-fit-transfer-v1' and plan['subject'] == subject
            and terminal['candidate_commit'] == plan['candidate_commit'] == COMMIT
            and terminal['fit_calls'] == result['fit_calls'] == plan['fit_calls'] == 0
            and terminal['fit_reused'] is result['fit_reused'] is plan['fit_reused'] is True
            and result['from_scratch'] is False and result['gaussian_from_scratch'] is True
            and result['reused_fit_preregistration_sha256'] == pins[root/'preregistration.json']
            and result['format_passed'] is status['format_passed'] is True and status['sha256'] == result['sha256']
            and result['steps'] == training['steps'] == 10000 and result['frames'] == count
            and terminal['final_snapshot'] == result['final_snapshot'] == training['final_snapshot'] == tail['snapshot'] == FINAL
            and terminal['head_appearance'] == result['head_appearance'] == tail
            and tail['state'] == 'completed' and tail['steps'] == 600 and tail['frozen_ply_bytes_verified'] is True,
            'Delivery does not bind the completed head-only snapshot and frozen fit.')

    baseline = Path(plan['baseline_run'])
    fit = read(run/'fit/official-fit-result.json', plan['staged_sha256']['fit/official-fit-result.json'])
    selection = read(run/'fit/selection.json', plan['input_sha256'][str(baseline/'fit/selection.json')])
    fit_path = pin(run/'fit/fitted.npz', plan['input_sha256'][str(baseline/'fit/fitted.npz')])
    rows_path = run/'data/training-inputs.json'
    rows = read(rows_path, plan['input_sha256'][str(baseline/'data/training-inputs.json')])['records']
    require(fit['selection'] == selection and fit['fit_sha256'] == pins[fit_path]
            and fit['training_inputs_sha256'] == pins[rows_path], 'The saved official fit or training rows differ.')
    metadata = read(run/'run.json', plan['staged_sha256']['run.json'])
    pin(metadata['video'], plan['source_video_sha256'])
    require(set(plan['camera_pins']) == {'gaussian-run/training-cameras.json', 'export/cameras.json'}
            and result['camera'] == 'export/cameras.json' and result['scene'] == 'export/scene.zip'
            and Path(result['ply']).parent == Path('export'), 'The portable delivery paths or camera pins differ.')
    for name, digest in plan['camera_pins'].items():
        pin(run/name, digest)
    pin(run/result['ply'], result['sha256'])
    pin(run/result['scene'], result['scene_sha256'])

    snapshot = run/'gaussian-run'/FINAL
    replay_path = snapshot/'posed-replay/result.json'
    panel_path = run/'audit/bound-panels/result.json'
    require(replay_path == run/result['bound_replay'] and panel_path == run/result['bound_panels'],
            'The selected snapshot and panel paths differ.')
    replay = read(replay_path, result['bound_replay_sha256'])
    panels = read(panel_path, result['bound_panels_sha256'])
    bound = read(snapshot/'bound-replay.json', replay['bound_replay_sha256'])
    pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    camera_path = run/'gaussian-run/training-cameras.json'
    cameras = read(camera_path, replay['training_camera_sha256'])
    require(replay['mesh_file'] == 'meshes.npz' and replay['mesh_sha256'] == panels['mesh_sha256'],
            'The mesh identity in the review receipts differs.')
    mesh_path = pin(replay_path.parent/'meshes.npz', replay['mesh_sha256'])
    bound_inputs = bound['inputs_sha256']
    pin(snapshot/'world.ply', bound_inputs['world_ply'])
    pin(snapshot/'reference-mesh.npz', bound_inputs['reference_mesh'])
    pin(snapshot/'native-ply-parity.json', bound_inputs['native_ply_parity'])

    ref = plan['reference_frame']
    slot = replay['reference_pose_index']
    selected = plan['selected_frames']
    order = [ref] + [frame for frame in selected if frame != ref]
    parity = replay['reference_reload_parity']
    require(len(order) == len(set(order)) == count and ref in selected and slot == 0
            and selected == terminal['selected_frames'] == selection['selected_frames']
            and order == [r['frame'] for r in rows] == [r['frame'] for r in replay['records']]
            == [r['frame'] for r in panels['panels']] == [r['frame'] for r in bound['frames']]
            == [r['frame'] for r in cameras['records']]
            and not set(order) & set(plan['retained_holdouts'])
            and plan['retained_holdouts'] == selection['retained_holdouts'] == replay['retained_holdouts']
            and result['reference_frame_id'] == replay['reference_frame_id'] == panels['reference_frame_id'] == ref
            == bound['reference_frame_id'] == terminal['reference_frame'] == selection['reference_frame']
            and slot == bound['reference_pose_index'] == panels['reference_pose_index']
            and replay['state'] == panels['state'] == 'completed'
            and replay['optimizer_updates'] == panels['optimizer_updates'] == bound['optimizer_updates'] == 0
            and bound['schema'] == 'shuolee-smplx-bound-v1' and bound['bound_ply'] == 'bound-local.ply'
            and bound['sh_degree'] == training['sh_degree'] == 0 and bound['points'] == training['points']
            and bound_inputs['fitted'] == pins[fit_path] and bound_inputs['training_rows'] == pins[rows_path]
            == cameras['source_sha256'] == replay['training_inputs_sha256']
            and panels['same_frame_bound_gaussians'] is panels['native_crops'] is True
            and panels['replay_receipt_sha256'] == result['bound_replay_sha256']
            and 0 <= parity['mean_abs'] < 2e-6 and 0 <= parity['max_abs'] < .005 and parity['archived_max_abs_8bit'] == 0,
            'Pose, mesh, camera, held-out frame, or replay evidence differs.')

    records = []
    with np.load(fit_path, allow_pickle=False) as fitted, np.load(mesh_path, allow_pickle=False) as mesh:
        for key in ('frame_ids', 'source_frames', 'image_sha256', 'mask_sha256', 'faces'):
            require(np.array_equal(fitted[key], mesh[key]), f'Fitted and posed mesh identity differs: {key}')
        require(np.array_equal(mesh['frame_ids'], order) and np.array_equal(mesh['pose_indices'], np.arange(count)),
                'Mesh does not have exactly the selected frame order.')
        for index, (row, splat, panel, bind, camera) in enumerate(zip(
                rows, replay['records'], panels['panels'], bound['frames'], cameras['records'], strict=True)):
            frame = row['frame']
            require(re.fullmatch(r'frame_[0-9]{6}', frame) and index == bind['pose_index'] == panel['pose_index']
                    and all(splat[key] == bind[key] for key in POSE_KEYS)
                    and all(row[key] == splat[key] for key in ('frame', 'source_frame_index', 'source_sha256', 'mask_sha256'))
                    and all(row[key] == panel[key] for key in ('frame', 'source_frame_index', 'source_sha256'))
                    and all(camera[key] == splat['camera'][key] for key in CAMERA_KEYS)
                    and camera['pose_index'] == index and camera['pixel_convention'] == 'pixel_corners'
                    and all(row[key] == camera[key] for key in ('frame', 'wh', 'K', 'w2c'))
                    and np.array_equal(fitted['K_integer_centers'][index], row['K_integer_centers'])
                    and np.array_equal(fitted['w2c'][index], row['w2c'])
                    and int(mesh['source_frames'][index]) == row['source_frame_index']
                    and mesh['image_sha256'][index] == row['source_sha256'] and mesh['mask_sha256'][index] == row['mask_sha256']
                    and splat['image'] == panel['panel'] == frame + '.png' and panel['gaussian_sha256'] == splat['image_sha256']
                    and 0 <= panel['projection_error_px'] < .001, f'Mixed frame, camera, geometry or observations: {frame}')
            pin(row['image'], row['source_sha256'])
            pin(row['mask'], row['mask_sha256'])
            pin(replay_path.parent/splat['image'], splat['image_sha256'])
            source = pin(panel_path.parent/panel['panel'], panel['panel_sha256'])
            crop = panel['crop_xyxy']
            x0, y0, x1, y1 = crop
            width, height = row['wh']
            require(0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, f'Invalid native crop: {frame}')
            with Image.open(source) as image:
                require(image.size == (4 * (x1 - x0), y1 - y0 + 26), f'Unexpected panel size: {frame}')
            records.append((source, dest/f'3DGS_{subject}_{tag}_原图-高斯-Mesh-特征点_{frame}.jpg', crop))

    first = run/'gaussian-run/step-001000'
    first_parity = read(first/'native-ply-parity.json')
    require(first_parity['passed'] is True and first_parity['reference_frame_id'] == ref
            and first_parity['reference_pose_index'] == slot and len(first_parity['checks']) == 1
            and 0 <= first_parity['checks'][0]['mean_abs'] < 2e-6
            and 0 <= first_parity['checks'][0]['max_abs'] < .005, 'The saved 1,000-step reference parity differs.')
    pin(first/'world.ply', first_parity['sha256'])
    pin(first/'reference-mesh.npz', first_parity['mesh_sha256'])
    reference_sources = (Path(rows[0]['image']), pin(first/'reference-render.png'),
                         pin(snapshot/'reference-render.png'), replay_path.parent/(ref + '.png'))
    comparison_jpeg(reference_sources, records[0][2], rows[0]['wh'])

    report = dict(state='verified', subject=subject, commit=COMMIT, frames=len(records), reference_frame=ref,
                  final_snapshot=FINAL, fit_reused=True, cold_video_clock=False, quality_accepted=False,
                  selected=[dict(frame=row['frame'], source_frame_index=row['source_frame_index']) for row in rows],
                  result_sha256=result_sha256, panel_receipt_sha256=result['bound_panels_sha256'],
                  comparison=dict(order=['source', 'step-001000', FINAL], crop_xyxy=records[0][2],
                                  first_png_sha256=pins[reference_sources[1]], saved_first_ply_parity=True,
                                  first_png_independently_replayed=False, final_png_matches_pose_replay=True), images=[])
    if approved_result_sha256 is None:
        return report

    with TemporaryDirectory(prefix=f'head-appearance-{subject}-record-') as temp:
        staging = Path(temp)
        files = []
        for source, target, crop in records:
            item = staging/target.name
            record_jpeg(source, item, crop)
            files.append((item, target))
        summary = dest/f'分析_{subject}_{tag}_入选四合一总览_{count}帧.jpg'
        contact_sheet(rows, [item for item, _ in files], staging/summary.name)
        compare = dest/f'分析_{subject}_{tag}_原图-1000步-最终_{ref}.jpg'
        comparison_jpeg(reference_sources, records[0][2], rows[0]['wh'], staging/compare.name)
        files.extend((staging/target.name, target) for target in (summary, compare))
        report['images'] = publish_images(files, pins)
    report['state'] = 'archived'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('subject', choices=FRAME_COUNTS)
    parser.add_argument('--publish-confirmed-sha256', metavar='SHA256', help='Write records only for this confirmed result.')
    args = parser.parse_args()
    print(json.dumps(archive(args.subject, args.publish_confirmed_sha256), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
