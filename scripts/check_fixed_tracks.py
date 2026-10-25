"""Known-geometry regression tests; no captured person, matching or calibration fit.

OpenCV independently projects the fixture observations. Counterexamples include
camera/motion ambiguity and an incorrect common focal length with zero residual.
These tests validate numerical behavior, not the real capture or reconstruction.
"""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt
from shuolee_portrait.fixed_tracks import cameras_from_records, evaluate_fixed_tracks, project


def records(rotated=True):
    rows = []
    for i, x in enumerate([-.5, 0., .5]):
        R, _ = cv2.Rodrigues(np.array([0., (.09 * (i - 1)) if rotated else 0., 0.]))
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3] = -R @ np.array([x, 0., 0.])
        rows.append(dict(frame=f'f{i}', wh=[640, 960], source_sha256=f'{i:064x}',
                         K=[[800., 0., 320.3], [0., 830., 480.7], [0., 0., 1.]],
                         w2c=E.tolist()))
    return rows


def reference_pixels(row, xyz):
    E = np.asarray(row['w2c'], float)
    rv, _ = cv2.Rodrigues(E[:3, :3])
    pixels, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), rv, E[:3, 3],
                                 np.asarray(row['K']), np.zeros(5))
    return pixels.reshape(-1, 2) - .5


def bank_for(rows, xyz):
    return dict(schema='shuolee-fixed-tracks-v1', pixel_domain='undistorted_integer_centers',
                camera_fit_observation_ids=None,
                tracks=[dict(track_id='physical_fixture_0', identity_basis='material_candidate',
                             region='face', triangulation_frames=['f0', 'f2'],
                             observations=[dict(frame=r['frame'], observation_id=f'feature_{i}_0',
                                                image_sha256=r['source_sha256'],
                                                uv=reference_pixels(r, xyz)[0].tolist())
                                           for i, r in enumerate(rows)])])


def rejected(operation):
    try:
        operation()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('Invalid input was silently accepted.')


def check():
    cases, evidence = [], {}
    rows = records()
    xyz = np.array([.1, .15, 3.])
    bank = bank_for(rows, xyz)
    cameras = cameras_from_records(rows)
    rng = np.random.default_rng(117)
    points = rng.uniform([-.3, -.4, 2.5], [.3, .4, 4.], size=(37, 3))
    maximum = max(float(np.max(np.abs(np.stack([project(cameras[r['frame']], p)[0] for p in points]) -
                                     reference_pixels(r, points)))) for r in rows)
    assert maximum < 1e-9
    result = evaluate_fixed_tracks(rows, bank)
    recovered = result['tracks'][0]
    assert np.linalg.norm(np.array(recovered['xyz']) - xyz) < 1e-10
    assert result['held_error_px']['maximum'] < 1e-9
    assert not result['training_admitted'] and not result['independent_camera_validation']
    evidence['opencv_projection_max_px'] = maximum
    cases.append('Rotated OpenCV camera projections, COLMAP half-pixel conversion and held-view triangulation agree.')

    changed = deepcopy(bank)
    changed['tracks'][0]['observations'][1]['uv'][0] += 19.
    mismatch = evaluate_fixed_tracks(rows, changed)
    assert abs(mismatch['held_error_px']['maximum'] - 19.) < 1e-9
    assert np.allclose(mismatch['tracks'][0]['xyz'], xyz, atol=1e-10)
    assert len(mismatch['tracks']) == 1 and not mismatch['root_cause_resolved']
    cases.append('A wrong held-view correspondence produces its full error; it cannot move the point or be pruned away.')

    reused = deepcopy(bank)
    reused['camera_fit_observation_ids'] = ['feature_1_0']
    used = evaluate_fixed_tracks(rows, reused)['tracks'][0]
    assert used['camera_validation_independence'] == 'not_independent'
    assert used['camera_fit_observation_overlap'] == ['feature_1_0']
    reused['camera_fit_observation_ids'] = []
    assert evaluate_fixed_tracks(rows, reused)['tracks'][0]['camera_validation_independence'] == 'id_disjoint_only_unverified'
    semantic = deepcopy(bank)
    semantic['tracks'][0]['identity_basis'] = 'semantic_landmark'
    assert not evaluate_fixed_tracks(rows, semantic)['tracks'][0]['physical_material_identity_verified']
    cases.append('Held point observations reused by the camera solver are marked non-independent; semantic IDs and disjoint IDs never prove material identity.')

    # In this parallel rig, multiplying shared focal length and scene depth by
    # the same factor leaves every image point unchanged. A low residual alone
    # must never be treated as a physically validated camera.
    parallel = records(rotated=False)
    parallel_bank = bank_for(parallel, xyz)
    wrong_k = deepcopy(parallel)
    for row in wrong_k:
        row['K'][0][0] *= 1.2
        row['K'][1][1] *= 1.2
    ambiguous_k = evaluate_fixed_tracks(wrong_k, parallel_bank)
    bad_depth = ambiguous_k['tracks'][0]['xyz'][2]
    assert ambiguous_k['held_error_px']['maximum'] < 1e-9
    assert abs(bad_depth / xyz[2] - 1.2) < 1e-10
    assert not ambiguous_k['independent_camera_validation'] and not ambiguous_k['training_admitted']
    evidence['shared_focal_counterexample'] = dict(focal_ratio=1.2, recovered_depth_ratio=bad_depth / xyz[2],
        held_error_px=ambiguous_k['held_error_px']['maximum'], training_admitted=False)
    cases.append('An incorrect common focal length can have zero held error in a degenerate rig; it remains unaccepted.')

    moved_camera = deepcopy(parallel)
    moved_camera[1]['w2c'][0][3] += .05
    camera_case = evaluate_fixed_tracks(moved_camera, parallel_bank)
    moved_subject = deepcopy(parallel_bank)
    moved_subject['tracks'][0]['observations'][1]['uv'] = reference_pixels(parallel[1], xyz - [.05, 0, 0])[0].tolist()
    motion_case = evaluate_fixed_tracks(parallel, moved_subject)
    camera_error = camera_case['held_error_px']['maximum']
    motion_error = motion_case['held_error_px']['maximum']
    assert abs(camera_error - motion_error) < 1e-9 and camera_error > 10
    assert not camera_case['root_cause_resolved'] and not motion_case['root_cause_resolved']
    evidence['camera_motion_counterexample'] = dict(camera_error_px=camera_error, subject_error_px=motion_error)
    cases.append('Camera translation error and subject displacement can give the same residual; the diagnostic does not invent a unique cause.')

    coincident = records(rotated=False)
    for row in coincident:
        row['w2c'] = np.eye(4).tolist()
    degenerate = evaluate_fixed_tracks(coincident, bank_for(coincident, xyz))
    assert degenerate['tracks'][0]['reason'] == 'coincident_camera_centers'
    assert degenerate['held_error_px']['count'] == 0 and degenerate['held_error_px']['median'] is None
    narrow = records(rotated=False)
    for row in narrow:
        row['w2c'][0][3] *= 1e-5
    assert evaluate_fixed_tracks(narrow, bank_for(narrow, xyz))['tracks'][0]['reason'] == 'insufficient_triangulation_angle'
    behind = deepcopy(rows)
    R, _ = cv2.Rodrigues(np.array([0., np.pi, 0.]))
    behind[1]['w2c'] = np.block([[R, np.zeros((3, 1))], [np.array([[0., 0., 0., 1.]])]]).tolist()
    assert evaluate_fixed_tracks(behind, bank)['tracks'][0]['reason'] == 'non_positive_depth'
    cases.append('Coincident centers, insufficient parallax and negative depth remain explicit unusable observations, not zero-error successes.')

    bad = deepcopy(bank);bad['pixel_domain'] = 'distorted_raw'
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    bad = deepcopy(bank);bad['tracks'][0]['observations'][1]['image_sha256'] = 'f' * 64
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    bad = deepcopy(bank);bad['tracks'][0]['observations'][1]['uv'][0] = float('nan')
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    bad = deepcopy(bank);bad['tracks'][0]['triangulation_frames'] = ['f0', 'f1', 'f2']
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    bad = deepcopy(bank);duplicate = deepcopy(bad['tracks'][0]);duplicate['track_id'] = 'copied_track';bad['tracks'].append(duplicate)
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    for o in duplicate['observations']:
        o['observation_id'] += '_alias'
    rejected(lambda: evaluate_fixed_tracks(rows, bad))
    transposed = deepcopy(rows);transposed[0]['w2c'] = np.asarray(transposed[0]['w2c']).T.tolist()
    rejected(lambda: evaluate_fixed_tracks(transposed, bank))
    cases.append('Wrong pixel domain, image identities, nonfinite pixels, missing holdout, duplicate observations and transposed camera matrices are rejected.')

    # Exercise the actual read-only CLI using PNG fixtures, not a real subject.
    with tempfile.TemporaryDirectory(prefix='shuolee-fixed-track-check-') as temporary:
        base = Path(temporary);run = base / 'run';(run / 'data').mkdir(parents=True)
        fixture_rows = deepcopy(rows)
        for i, row in enumerate(fixture_rows):
            p = run / f'{row["frame"]}.png'
            Image.new('RGB', tuple(row['wh']), (30 + i, 70, 120)).save(p)
            row.update(image=str(p), source_sha256=rt.sha(p))
        manifest = run / 'data/training-inputs.json'
        rt.save(manifest, dict(records=fixture_rows))
        fixture_bank = bank_for(fixture_rows, xyz)
        fixture_bank['camera_manifest_sha256'] = rt.sha(manifest)
        bank_path = base / 'tracks.json';report = base / 'result.json'
        rt.save(bank_path, fixture_bank)
        before = {p.name: rt.sha(p) for p in run.rglob('*') if p.is_file()}
        script = PROJECT / 'scripts/evaluate_fixed_tracks.py'
        command = [sys.executable, str(script), '--run', str(run), '--tracks', str(bank_path), '--report', str(report)]
        proc = subprocess.run(command, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        actual = rt.read(report)
        assert actual['image_bytes_verified'] and actual['held_error_px']['maximum'] < 1e-9
        assert before == {p.name: rt.sha(p) for p in run.rglob('*') if p.is_file()}
        spec = importlib.util.spec_from_file_location('fixed_track_cli', script)
        module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        rejected(lambda: module.evaluate(run, bank_path, report))
        rejected(lambda: module.evaluate(run, bank_path, run / 'overwrite.json'))
        fixture_rows[0]['K'][0][0] += 1
        rt.save(manifest, dict(records=fixture_rows))
        rejected(lambda: module.evaluate(run, bank_path, base / 'stale.json'))
        rt.save(manifest, dict(records=[dict(r) for r in fixture_rows]))
        fixture_bank['camera_manifest_sha256'] = rt.sha(manifest);rt.save(bank_path, fixture_bank)
        Image.new('RGB', (640, 960), (255, 0, 0)).save(fixture_rows[0]['image'])
        rejected(lambda: module.evaluate(run, bank_path, base / 'changed_image.json'))
    cases.append('Actual CLI verifies image bytes and the camera manifest, preserves the source run, and refuses stale inputs or report overwrite.')
    assert 'torch' not in sys.modules
    return dict(passed=True, cases=cases, evidence=evidence, synthetic_fixtures_only=True,
                matching_algorithm_implemented=False, captured_person_tested=False,
                independent_camera_validation=False, training_started=False,
                gpu_task=False, ghosting_resolved=False, quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check();rt.save(args.output, result)
    print(json.dumps(result, indent=2))
