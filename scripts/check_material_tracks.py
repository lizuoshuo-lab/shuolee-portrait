"""Actual SIFT extraction on known image warps; not a real-human quality test."""
import argparse
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt, cli
from shuolee_portrait.material_tracks import extract_material_tracks, mutual_matches, cycle_components
from shuolee_portrait.fixed_tracks import evaluate_fixed_tracks
from shuolee_portrait.observations import face_region_box


def fixture(root, moving=False, blank=False, write_config=True):
    root.mkdir(parents=True, exist_ok=True);(root / 'data').mkdir(exist_ok=True)
    width, height = 480, 360
    rng = np.random.default_rng(1234)
    texture = rng.integers(0, 256, (height, width), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (0, 0), 1.)
    for _ in range(170):
        xy = tuple(rng.integers([10, 10], [width - 10, height - 10]).tolist())
        cv2.circle(texture, xy, int(rng.integers(2, 7)), int(rng.integers(0, 256)), -1)
    mask = np.zeros((height, width), np.uint8);mask[45:320, 130:350] = 255
    if blank:texture[:] = 100
    rows, transforms = [], []
    for i, camera_x in enumerate([-.16, -.08, 0., .08, .16]):
        background_dx = -300 * camera_x / 3.
        subject_dx = 4. if moving and i == 2 else 0.
        H = np.array([[1., 0., background_dx], [0., 1., 0.]])
        F = H.copy();F[0, 2] += subject_dx
        image = cv2.warpAffine(texture, H, (width, height), borderValue=100)
        person = cv2.warpAffine(texture, F, (width, height), borderValue=100)
        foreground = cv2.warpAffine(mask, F, (width, height), flags=cv2.INTER_NEAREST)
        image[foreground > 127] = person[foreground > 127]
        rgb_path, mask_path = root / f'f{i}.png', root / f'f{i}_mask.png'
        Image.fromarray(image).convert('RGB').save(rgb_path);Image.fromarray(foreground).save(mask_path)
        E = np.eye(4);E[0, 3] = -camera_x
        face = [175 + background_dx + subject_dx, 65, 315 + background_dx + subject_dx, 185]
        rows.append(dict(frame=f'f{i}', time_seconds=i * .1, wh=[width, height],
                         K=[[300., 0., 240.5], [0., 300., 180.5], [0., 0., 1.]], w2c=E.tolist(),
                         image=str(rgb_path), source_sha256=rt.sha(rgb_path),
                         mask=str(mask_path), mask_sha256=rt.sha(mask_path),
                         face_box_xyxy=face, other_person_boxes_xyxy=[[30 + background_dx, 25, 70 + background_dx, 100]]))
        transforms.append(dict(background_dx=background_dx, subject_dx=subject_dx))
    rt.save(root / 'data/training-inputs.json', dict(records=rows, pixel_domain='undistorted_integer_centers', intrinsic_origin='COLMAP_pixel_corners'))
    if write_config:rt.save(root / 'config.json', dict(material_tracks=dict(max_features=1200)))
    return rows, transforms


def correspondence_error(bank, transforms):
    errors = []
    for track in bank['tracks']:
        if track['region'] == 'unknown':continue
        foreground = track['region'] in ('face', 'body')
        canonical = []
        for observation in track['observations']:
            i = int(observation['frame'][1:]);warp = transforms[i]
            delta = warp['background_dx'] + (warp['subject_dx'] if foreground else 0)
            canonical.append(np.asarray(observation['uv']) - [delta, 0])
        errors.extend(np.linalg.norm(np.asarray(canonical) - canonical[0], axis=1).tolist())
    return dict(count=len(errors), median=float(np.median(errors)), p95=float(np.quantile(errors, .95)), maximum=float(max(errors)))


def rejected(call):
    try:call()
    except (ValueError, FileExistsError):return
    raise AssertionError('Invalid input was accepted.')


def check():
    cases, evidence = [], {}
    repeated = np.ones((4, 128), np.float32)
    match = mutual_matches(repeated, repeated)
    assert len(match['matches']) == 0 and np.all(match['forward_decision'] == 2)
    rng = np.random.default_rng(10)
    a = rng.normal(size=(12, 128)).astype('float32');order = rng.permutation(len(a))
    unique = mutual_matches(a, a[order])
    assert len(unique['matches']) == len(a) and all(order[j] == i for i, j in unique['matches'])
    assert not len(mutual_matches(a, a[:1])['matches'])
    cases.append('Actual nearest-neighbor matching rejects repeated zero-distance descriptors and insufficient neighbors, while preserving unique bidirectional identities.')

    simple = {(0, 1): {0: 0}, (1, 2): {0: 0}, (0, 2): {0: 1}}
    accepted, rejected_components, stats = cycle_components(simple, [(0, 1, 2)])
    assert not accepted and stats['third_edge_identity_mismatch_or_absent'] == 1
    conflicting = {(0, 1): {0: 0}, (1, 2): {0: 0}, (0, 2): {0: 0},
                   (1, 3): {0: 0}, (2, 3): {0: 0}, (0, 3): {1: 0},
                   (0, 4): {1: 0}, (3, 4): {0: 0}}
    accepted, rejected_components, stats = cycle_components(conflicting, [(0, 1, 2), (1, 2, 3), (0, 3, 4)])
    assert not accepted and len(rejected_components) == 1
    cases.append('A broken third edge is rejected; locally closed cycles with conflicting identities in one frame cannot merge into a false global track.')

    kp = np.zeros((133, 2));scores = np.zeros(133)
    assert face_region_box(kp, scores, [480, 360]) is None
    kp[23:91] = np.column_stack([np.linspace(100, 200, 68), np.linspace(30, 130, 68)]);scores[23:91] = 1
    assert face_region_box(kp, scores, [480, 360]) == [100., 30., 200., 130.]
    kp[23:91] += 1000
    assert face_region_box(kp, scores, [480, 360]) is None
    cases.append('Face keypoints supply only a bounded region; missing or off-image detections produce no face ROI.')

    with tempfile.TemporaryDirectory(prefix='shuolee-material-check-') as temp:
        base = Path(temp);run = base / 'static';rows, transforms = fixture(run)
        before = {p.relative_to(run).as_posix(): rt.sha(p) for p in run.rglob('*') if p.is_file()}
        output = base / 'static_tracks'
        summary = extract_material_tracks(run, output, dict(max_features=1200))
        bank = rt.read(output / 'tracks.json');error = correspondence_error(bank, transforms)
        assert summary['track_count'] > 30 and summary['frames_with_tracks'] == 5, summary
        assert summary['tracks_by_region'].get('face', 0) >= 5
        assert error['p95'] < .5, error
        assert before == {p.relative_to(run).as_posix(): rt.sha(p) for p in run.rglob('*') if p.is_file()}
        assert bank['camera_fit_observation_ids'] is None and not summary['independent_camera_validation']
        numerical = evaluate_fixed_tracks(rows, bank)
        assert numerical['held_error_px']['p90'] < .5, numerical['held_error_px']
        evidence['static_known_warp'] = dict(tracks=summary['track_count'], regions=summary['tracks_by_region'],
                                            correspondence_error_px=error, held_projection_px=numerical['held_error_px'])
        cases.append('Actual SIFT/mutual/cycle extraction on five views of a rendered plane agrees with known pixel warps, feeds the fixed-point module, and leaves source files unchanged.')
        original_manifest=rt.read(run / 'data/training-inputs.json')
        changed_manifest=json.loads(json.dumps(original_manifest))
        for row in changed_manifest['records']:
            row['K'][0][0]*=1.3;row['w2c'][1][3]+=.2
        rt.save(run / 'data/training-inputs.json',changed_manifest)
        extract_material_tracks(run,base / 'changed_cameras_tracks',dict(max_features=1200))
        assert rt.read(base / 'changed_cameras_tracks/tracks.json')['tracks']==bank['tracks']
        rt.save(run / 'data/training-inputs.json',original_manifest)
        assert before == {p.relative_to(run).as_posix():rt.sha(p) for p in run.rglob('*') if p.is_file()}
        cases.append('Changing only synthetic K and extrinsics leaves the extracted appearance tracks identical; no camera residual selects matches.')

        moving = base / 'moving';moving_rows, moving_transforms = fixture(moving, moving=True)
        moving_summary = extract_material_tracks(moving, base / 'moving_tracks', dict(max_features=1200))
        moving_bank = rt.read(base / 'moving_tracks/tracks.json');motion_error = correspondence_error(moving_bank, moving_transforms)
        assert motion_error['p95'] < .5, motion_error
        motion_eval = evaluate_fixed_tracks(moving_rows, moving_bank)
        # First/last fitting frames are static; the middle foreground alone
        # moves four pixels. All-five tracks are chosen by coverage, not error.
        full_tracks = [t for t in motion_eval['tracks'] if len(t['held_frames']) == 3 and t['status'] == 'measured']
        subject = [p['residual_px'] for t in full_tracks if t['region'] in ('face', 'body') for p in t['projections'] if p['frame'] == 'f2']
        background = [p['residual_px'] for t in full_tracks if t['region'] == 'static_scene_candidate' for p in t['projections'] if p['frame'] == 'f2']
        assert len(subject) >= 5 and len(background) >= 5
        assert 3.5 < np.median(subject) < 4.5 and np.median(background) < .5
        assert not moving_summary['subject_staticity_verified'] and not motion_eval['root_cause_resolved']
        evidence['known_foreground_motion'] = dict(correspondence_error_px=motion_error,
            subject_midframe_median_px=float(np.median(subject)), static_background_median_px=float(np.median(background)),
            subject_tracks=len(subject), background_tracks=len(background))
        cases.append('Correct texture correspondence is preserved when the foreground moves; fixed-camera projection separates its known motion residual from the static control without claiming real-world causality.')

        cli_run = base / 'cli';fixture(cli_run)
        process = subprocess.run([sys.executable, str(PROJECT / 'run.py'), '_stage', '--stage', 'material-tracks',
                                  '--output', str(cli_run), '--assets', str(base / 'missing_assets')], capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        assert rt.read(cli_run / 'data/material-tracks/summary.json')['track_count'] > 0
        assert not (cli_run / 'fit').exists() and not (cli_run / 'gaussian-run').exists()
        rejected(lambda: extract_material_tracks(run, output))
        bad = rt.read(run / 'data/training-inputs.json');bad.pop('pixel_domain')
        rt.save(run / 'data/training-inputs.json', bad)
        rejected(lambda: extract_material_tracks(run, base / 'bad_domain'))
        bad['pixel_domain'] = 'undistorted_integer_centers';rt.save(run / 'data/training-inputs.json', bad)
        Image.new('L', (480, 360), 0).save(rows[0]['mask'])
        rejected(lambda: extract_material_tracks(run, base / 'bad_hash'))
        cases.append('The actual CLI stage runs without human weights; existing output, missing pixel-domain contracts and changed mask bytes are rejected.')

        blank_run = base / 'blank';fixture(blank_run, blank=True)
        process = subprocess.run([sys.executable, str(PROJECT / 'run.py'), '_stage', '--stage', 'material-tracks',
                                  '--output', str(blank_run), '--assets', str(base / 'missing_assets')], capture_output=True, text=True)
        assert process.returncode == 3, process.stderr
        blank_summary = rt.read(blank_run / 'data/material-tracks/summary.json')
        assert blank_summary['track_count'] == 0 and not blank_summary['training_admitted']
        cases.append('Textureless input retains a no-tracks report and stops with exit3; semantic landmarks are not substituted.')

        # Earlier heavy stages are simulated only; material extraction is real.
        parent_output = base / 'parent';video = base / 'unused_fixture.mp4';video.write_bytes(b'not decoded in this test')
        real_run = subprocess.run;calls = []
        def dispatch(command, *args, **kwargs):
            stage = command[command.index('--stage') + 1];calls.append(stage)
            if stage == 'material-tracks':return real_run(command, *args, **kwargs)
            if stage not in ('extract', 'sfm', 'observations'):raise AssertionError('Ran fitting/training after no-tracks failure.')
            if stage == 'observations':fixture(parent_output, blank=True, write_config=False)
            return subprocess.CompletedProcess(command, 0)
        argv = ['run.py', 'reconstruct', '--video', str(video), '--output', str(parent_output), '--config', str(PROJECT / 'configs/smoke.json')]
        with patch.object(sys, 'argv', argv), patch.object(cli, 'doctor', return_value={}), patch.object(cli, 'gpu_lock', return_value=nullcontext()), patch.object(cli.shutil, 'which', return_value='fixture'), patch.object(cli.subprocess, 'run', side_effect=dispatch), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            try:cli.main()
            except SystemExit as exc:assert exc.code == 3
            else:raise AssertionError('Parent did not stop.')
        status = rt.read(parent_output / 'status.json')
        assert status['state'] == 'alignment_not_admitted' and not status['training_started']
        assert status['alignment_report'] == 'data/material-tracks/summary.json'
        assert calls == ['extract', 'sfm', 'observations', 'material-tracks']
        cases.append('Parent execution records the actual material failure report and stops before prior, fitting and training; preceding stages were simulated only.')
    assert 'torch' not in sys.modules
    return dict(passed=True, cases=cases, evidence=evidence, synthetic_images_only=True,
                actual_sift_extraction=True, automatic_candidate_generation_implemented=True,
                real_person_tested=False, camera_independence_validated=False,
                ghosting_resolved=False, gpu_task=False, training_started=False, quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();result = check();rt.save(args.output, result)
    print(json.dumps(result, indent=2))
