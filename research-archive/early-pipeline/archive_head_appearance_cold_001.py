"""Verify fresh 001 video-to-export lineage and archive only native review JPEGs."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_head_appearance_cold_001 as cold
from archive_head_appearance_records import (
    CAMERA_KEYS, FINAL, FRAME_COUNTS, POSE_KEYS, RECORDS, comparison_jpeg,
    contact_sheet, publish_images, record_jpeg, require, sha,
)


TAG = '原视频计时762d6d0'
SUBJECT = '001'


def sheet(rows, panels, target):
    if len(rows) in FRAME_COUNTS.values():
        contact_sheet(rows, panels, target)
        return
    # The cold selector can legitimately admit fewer frames than the old 24.
    width, height = 900, 325
    page = Image.new('RGB', (3 * width, ((len(rows)+2)//3) * height), '#161616')
    ordered = sorted(zip(rows, panels, strict=True), key=lambda pair: pair[0]['source_frame_index'])
    for i, (_, path) in enumerate(ordered):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert('RGB'), (width-8, height-8), Image.Resampling.LANCZOS)
        page.paste(tile, (i%3*width + (width-tile.width)//2, i//3*height + (height-tile.height)//2))
    page.save(target, quality=95, subsampling=0)


def archive(approved_result_sha256=None, timing_receipt_sha256=None):
    root, run = cold.ROOT, cold.RUN
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve(strict=True)
        actual = sha(path)
        require((expected is None or actual == expected) and pins.get(path, actual) == actual,
                f'Cold archive input hash differs: {path}')
        pins[path] = actual
        return path

    def read(path, expected=None):
        return json.loads(pin(path, expected).read_text(encoding='utf-8-sig'))

    launch = read(root/'launch.json')
    plan = read(root/'preregistration.json', launch['preregistration_sha256'])
    status = read(run/'status.json')
    require(plan['schema'] == 'shuolee-head-appearance-cold-video-v1' and plan['subject'] == SUBJECT
            and plan['candidate_commit'] == cold.COMMIT and plan['automatic_reference'] is True
            and plan['no_preprocessing_fit_or_gaussian_reuse'] is True
            and plan['output'] == str(run) and launch['entry_command'] == plan['entry_command'] == cold.command(),
            'Unexpected raw-video registration or launch.')
    pin(Path(cold.__file__), plan['wrapper_sha256'])
    wrapper_path = root/'wrapper-terminal.json'
    if not wrapper_path.is_file():
        require(approved_result_sha256 is None and timing_receipt_sha256 is None,
                'The cold wrapper terminal has not been verified.')
        return dict(state='awaiting_cold_terminal', subject=SUBJECT, stage=status['stage'],
                    process_state=status['state'], images=[])

    wrapper = read(wrapper_path)
    require(wrapper['returncode'] == 0 and wrapper['candidate_commit'] == cold.COMMIT
            and wrapper['process_status'] == status and wrapper['command'] == launch['entry_command']
            and wrapper['prerequisite_008'] == launch['prerequisite_008']
            and wrapper['preregistration_sha256'] == launch['preregistration_sha256'],
            'The original cold wrapper identity differs.')
    receipt = wrapper
    monotonic_elapsed = wrapper['command_to_cli_end_seconds']
    elapsed = monotonic_elapsed
    if timing_receipt_sha256 is not None:
        receipt = read(root/'timing-revalidated.json', timing_receipt_sha256)
        require(wrapper['state'] == 'incomplete' and wrapper['validation_error'] ==
                'Command-spawn, 1k, static export, or terminal panel timing is inconsistent.'
                and receipt['state'] == 'completed' and receipt['original_wrapper_sha256'] == pins[wrapper_path]
                and all(receipt[key] == wrapper[key] for key in ('subject', 'candidate_commit', 'returncode', 'command',
                    'command_spawned_at', 'command_to_cli_end_seconds', 'prerequisite_008', 'preregistration_sha256',
                    'process_status', 'application_reported_static_export_seconds')),
                'The timing receipt does not resolve the original clock-only validation failure.')
        elapsed = receipt['utc_elapsed_upper_bound_seconds']
        require(math.isfinite(elapsed) and elapsed > 0, 'The UTC timing upper bound is invalid.')
    require(receipt['state'] == 'completed' and receipt['from_scratch'] is True
            and receipt['fixed_fit_reused'] is receipt['checkpoint_reused'] is False,
            'The cold run needs a completed terminal or its pinned independent timing receipt.')
    result_hash = receipt['result_sha256']
    require(approved_result_sha256 in (None, result_hash), 'Approved result SHA differs from this cold terminal.')
    result = read(run/'result.json', result_hash)
    verified = cold.validate(plan, status, datetime.fromisoformat(wrapper['command_spawned_at']), elapsed)
    require(all(receipt[key] == value for key, value in verified.items()), 'Cold terminal validation no longer agrees.')

    snapshot = run/'gaussian-run'/FINAL
    replay_path = snapshot/'posed-replay/result.json'
    panel_path = run/'audit/bound-panels/result.json'
    require(result['final_snapshot'] == FINAL and run/result['bound_replay'] == replay_path
            and run/result['bound_panels'] == panel_path and result['camera'] == 'export/cameras.json',
            'Output must use the exact completed head snapshot and portable camera.')
    for artifact in verified['artifacts'].values():
        pin(artifact['path'], artifact['sha256'])
    replay, panels = read(replay_path, result['bound_replay_sha256']), read(panel_path, result['bound_panels_sha256'])
    manifest = read(snapshot/'bound-replay.json', replay['bound_replay_sha256'])
    pin(snapshot/'bound-local.ply', manifest['bound_ply_sha256'])
    bound = manifest['inputs_sha256']
    pin(snapshot/'world.ply', bound['world_ply'])
    pin(snapshot/'reference-mesh.npz', bound['reference_mesh'])
    pin(snapshot/'native-ply-parity.json', bound['native_ply_parity'])
    fit = pin(run/'fit/fitted.npz', bound['fitted'])
    rows = read(run/'data/training-inputs.json', bound['training_rows'])['records']
    cameras = read(run/'gaussian-run/training-cameras.json', replay['training_camera_sha256'])
    require(replay['mesh_file'] == 'meshes.npz' and replay['mesh_sha256'] == panels['mesh_sha256'],
            'Replay and panel mesh IDs differ.')
    posed_mesh = pin(replay_path.parent/'meshes.npz', replay['mesh_sha256'])

    ref, selected = verified['reference_frame'], verified['selected_frames']
    order = [ref] + [frame for frame in selected if frame != ref]
    count = len(order)
    require(0 < count <= 24 and len(set(order)) == count and ref in selected
            and order == [row['frame'] for row in rows] == [row['frame'] for row in cameras['records']]
            == [row['frame'] for row in replay['records']] == [row['frame'] for row in panels['panels']]
            == [row['frame'] for row in manifest['frames']]
            and plan['expected_holdouts'] == replay['retained_holdouts'] and set(order).isdisjoint(replay['retained_holdouts'])
            and manifest['reference_frame_id'] == replay['reference_frame_id'] == panels['reference_frame_id'] == ref
            and manifest['reference_pose_index'] == replay['reference_pose_index'] == panels['reference_pose_index'] == 0
            and manifest['optimizer_updates'] == replay['optimizer_updates'] == panels['optimizer_updates'] == 0
            and manifest['schema'] == 'shuolee-smplx-bound-v1' and manifest['points'] == result['points']
            and manifest['sh_degree'] == 0 and replay['state'] == panels['state'] == 'completed'
            and panels['replay_receipt_sha256'] == result['bound_replay_sha256']
            and bound['training_rows'] == replay['training_inputs_sha256'] == cameras['source_sha256']
            and panels['same_frame_bound_gaussians'] is panels['native_crops'] is True
            and replay['reference_reload_parity']['archived_max_abs_8bit'] == 0,
            'Saved same-frame pose/camera, selected roster, or final replay verification differs.')

    records = []
    with np.load(fit, allow_pickle=False) as fitted, np.load(posed_mesh, allow_pickle=False) as posed:
        require(np.array_equal(posed['frame_ids'], order) and np.array_equal(posed['pose_indices'], np.arange(count))
                and all(np.array_equal(fitted[k], posed[k]) for k in ('frame_ids','source_frames','image_sha256','mask_sha256','faces')),
                'Official pose and Gaussian-render mesh use different frame identities.')
        for i, (row, camera, bind, splat, panel) in enumerate(zip(
                rows, cameras['records'], manifest['frames'], replay['records'], panels['panels'], strict=True)):
            frame = row['frame']
            require(re.fullmatch(r'frame_[0-9]{6}', frame) and i == bind['pose_index'] == panel['pose_index'] == camera['pose_index']
                    and all(bind[k] == splat[k] for k in POSE_KEYS)
                    and all(row[k] == bind[k] for k in ('frame','source_frame_index','source_sha256','mask_sha256'))
                    and all(row[k] == panel[k] for k in ('frame','source_frame_index','source_sha256'))
                    and all(camera[k] == splat['camera'][k] for k in CAMERA_KEYS)
                    and all(camera[k] == row[k] for k in ('frame','wh','K','w2c')) and camera['pixel_convention'] == 'pixel_corners'
                    and np.array_equal(fitted['K_integer_centers'][i], row['K_integer_centers'])
                    and np.array_equal(fitted['w2c'][i], row['w2c'])
                    and (int(posed['source_frames'][i]), posed['image_sha256'][i], posed['mask_sha256'][i])
                    == (row['source_frame_index'], row['source_sha256'], row['mask_sha256'])
                    and splat['image'] == panel['panel'] == frame+'.png' and panel['gaussian_sha256'] == splat['image_sha256']
                    and 0 <= panel['projection_error_px'] < .001, f'Cold panel identities differ: {frame}')
            for field, digest in (('image','source_sha256'), ('mask','mask_sha256')):
                source_path = pin(row[field], row[digest])
                require(source_path.is_relative_to(run), f'Cold data escaped fresh run: {source_path}')
            pin(replay_path.parent/splat['image'], splat['image_sha256'])
            source = pin(panel_path.parent/panel['panel'], panel['panel_sha256'])
            x0,y0,x1,y1 = panel['crop_xyxy']
            require(0 <= x0 < x1 <= row['wh'][0] and 0 <= y0 < y1 <= row['wh'][1], f'Invalid native crop: {frame}')
            with Image.open(source) as image:
                require(image.size == (4*(x1-x0), y1-y0+26), f'Four-column image size differs: {frame}')
            records.append((source, f'3DGS_{SUBJECT}_{TAG}_原图-高斯-Mesh-特征点_{frame}.jpg', panel['crop_xyxy']))

    first = run/'gaussian-run/step-001000'
    parity = read(first/'native-ply-parity.json', receipt['snapshot_1000_parity_sha256'])
    require(parity['passed'] is True and parity['reference_frame_id'] == ref and parity['reference_pose_index'] == 0
            and len(parity['checks']) == 1 and 0 <= parity['checks'][0]['mean_abs'] < 2e-6
            and 0 <= parity['checks'][0]['max_abs'] < .005, 'No verified 1,000-step render for the selected pose.')
    pin(first/'world.ply', parity['sha256'])
    pin(first/'reference-mesh.npz', parity['mesh_sha256'])
    reference = (Path(rows[0]['image']), pin(first/'reference-render.png'), pin(snapshot/'reference-render.png'),
                 replay_path.parent/(ref+'.png'))
    crop = records[0][2]
    comparison_jpeg(reference, crop, rows[0]['wh'])

    report = dict(state='verified', subject=SUBJECT, commit=cold.COMMIT, from_scratch=True, fit_reused=False,
        quality_accepted=False, result_sha256=result_hash, wrapper_sha256=pins[wrapper_path],
        frames=count, reference_frame=ref, final_snapshot=FINAL, panel_receipt_sha256=result['bound_panels_sha256'],
        raw_video_to_static_export_seconds=verified['raw_video_to_static_export_seconds'],
        command_to_cli_end_seconds=monotonic_elapsed, timing_receipt_sha256=timing_receipt_sha256,
        utc_elapsed_upper_bound_seconds=elapsed if timing_receipt_sha256 else None,
        selected=[dict(frame=row['frame'], source_frame_index=row['source_frame_index']) for row in rows],
        comparison=dict(order=['source','step-001000',FINAL], crop_xyxy=crop, first_png_sha256=pins[reference[1]],
                        saved_first_ply_parity=True, first_png_independently_replayed=False, final_png_matches_pose_replay=True), images=[])
    if approved_result_sha256 is None:
        return report

    dest = RECORDS/SUBJECT
    require(dest.resolve(strict=True) == dest and dest.is_dir(), 'Use the existing 001 image-record directory.')
    with TemporaryDirectory(prefix='cold-001-record-') as temporary:
        work = Path(temporary)
        files = []
        for source, name, box in records:
            record_jpeg(source, work/name, box)
            files.append((work/name, dest/name))
        summary = f'分析_{SUBJECT}_{TAG}_入选四合一总览_{count}帧.jpg'
        sheet(rows, [path for path, _ in files], work/summary)
        compare = f'分析_{SUBJECT}_{TAG}_原图-1000步-最终_{ref}.jpg'
        comparison_jpeg(reference, crop, rows[0]['wh'], work/compare)
        files.extend((work/name, dest/name) for name in (summary, compare))
        report['images'] = publish_images(files, pins)
    report['state'] = 'archived'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-confirmed-sha256', metavar='SHA256', help='Write JPEGs only for this confirmed result.')
    parser.add_argument('--timing-receipt-sha256', metavar='SHA256', help='Use the pinned UTC revalidation receipt.')
    args = parser.parse_args()
    print(json.dumps(archive(args.publish_confirmed_sha256, args.timing_receipt_sha256), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
