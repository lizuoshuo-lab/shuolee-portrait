"""Known-geometry native COLMAP integration. No captured subject or GS training."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import pycolmap as pc

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import runtime as rt, video_sfm
from shuolee_portrait.sfm_holdout import prepare_holdout, audit_usage, _features, validate_settings


def fixture(root):
    root.mkdir();(root / 'raw').mkdir();(root / 'data').mkdir();(root / 'logs').mkdir()
    dbpath = root / 'fixture-features.db';db = pc.Database(str(dbpath))
    cam = pc.Camera(model='SIMPLE_RADIAL', width=640, height=480, params=[700., 320., 240., .035])
    camera_id = db.write_camera(cam)
    rng = np.random.default_rng(1771)
    xyz = rng.uniform([-1., -1., 3.4], [1., 1., 6.], size=(340, 3))
    desc = rng.uniform(0, 1, (356, 128));desc = np.rint(desc / np.linalg.norm(desc, axis=1)[:, None] * 512).astype('uint8')
    decoded = [];truth = {}
    for i, x in enumerate(np.linspace(-.65, .65, 5)):
        center = np.array([x, .08 * np.sin(i), 0.])
        z = np.array([0, 0, 4.5]) - center;z /= np.linalg.norm(z)
        right = np.cross([0, 1, 0], z);right /= np.linalg.norm(right)
        R = np.stack([right, np.cross(z, right), z]);t = -R @ center
        points = xyz @ R.T + t;xy = cam.img_from_cam(points)
        # Extra native rows near existing measurements exercise alias exclusion.
        xy = np.r_[xy, xy[:16] + [.6, .3]]
        assert ((xy >= .5) & (xy <= [639.5, 479.5])).all()
        name = f'frame_{i + 1:06d}.png';image_id = db.write_image(pc.Image(name=name, camera_id=camera_id))
        db.write_keypoints(image_id, xy.astype('float32'));db.write_descriptors(image_id, desc)
        pixels = rng.integers(30, 225, (480, 640, 3), dtype='uint8')
        Image.fromarray(pixels).save(root / 'raw' / name)
        decoded.append(dict(frame=Path(name).stem, name=name, source_frame_index=i,
                            time_seconds=.1 * i, wh=[640, 480], source_sha256=rt.sha(root / 'raw' / name)))
        truth[name] = dict(R=R, t=t, xyz=xyz, xy=xy)
    db.close();rt.save(root / 'decoded.json', dict(records=decoded))
    return dbpath, decoded, truth


def expect_error(call, types=(ValueError, RuntimeError, FileExistsError)):
    try:
        call()
    except types as exc:
        return str(exc)
    raise AssertionError('Invalid input unexpectedly admitted.')


def run():
    results = {};cfg = rt.read(PROJECT / 'configs/server.json')
    settings = dict(temporal_hops=[1, 2], alias_radius_px=1., min_tracks=12)
    with tempfile.TemporaryDirectory(prefix='shuolee-holdout-') as tmp:
        base = Path(tmp);root = base / 'native';source, decoded, truth = fixture(root)
        original_hash = rt.sha(source)
        split = prepare_holdout(source, root / 'train.db', decoded, root / 'raw', root / 'split', settings)
        assert 12 <= len(split['tracks']) < split['candidate_count']
        before = rt.read(root / 'split/usage-before-matching.json')
        assert before['reserved_used'] == 0 and not before['reconstruction_checked']
        assert sum(r['removed_rows'] for r in split['frames']) > sum(r['reserved_rows'] for r in split['frames'])
        assert original_hash == rt.sha(source)
        split2 = prepare_holdout(source, root / 'train2.db', decoded, root / 'raw', root / 'split2', settings)
        assert split['tracks'] == split2['tracks']
        results['pregeometry_split_and_alias_exclusion'] = dict(tracks=len(split['tracks']),
            reserved_observations=before['reserved_observations'],
            excluded_rows=sum(r['removed_rows'] for r in split['frames']), deterministic=True, source_unchanged=True)

        matched = root / 'already-matched.db';shutil.copy2(source, matched)
        db = pc.Database(str(matched));db.write_matches(1, 2, np.array([[0, 0]], np.uint32));db.close()
        results['retrospective_split_rejected'] = expect_error(lambda: prepare_holdout(
            matched, root / 'late-train.db', decoded, root / 'raw', root / 'late', settings))

        def mutate_case(name, mutation):
            altered = root / (name + '.db');shutil.copy2(root / 'train.db', altered)
            with sqlite3.connect(altered) as db:mutation(db)
            return expect_error(lambda: audit_usage(source, altered, root / 'split', report_name=name + '.json'))
        def change_pixels(db):
            key, _ = _features(db, 1);key[0, 0] += .1
            db.execute('UPDATE keypoints SET data=? WHERE image_id=1', (key.tobytes(),))
        def invalid_match(db):
            db.execute('INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,?,?)',
                       (2147483649, 1, 2, np.array([[99999, 0]], '<u4').tobytes()))
        results['native_feature_tampering_and_stale_indices_rejected'] = [
            mutate_case('altered-pixels', change_pixels), mutate_case('stale-index', invalid_match)]
        results['invalid_budget_rejected'] = expect_error(lambda: validate_settings(dict(fraction=1.)))
        empty = root / 'ambiguous.db';shutil.copy2(source, empty)
        with sqlite3.connect(empty) as db:
            db.execute('UPDATE descriptors SET data=zeroblob(rows*cols)')
        ambiguous_error = expect_error(lambda: prepare_holdout(empty, root / 'ambiguous-train.db',
            decoded, root / 'raw', root / 'ambiguous-split', settings))
        assert not (root / 'ambiguous-train.db').exists()
        assert rt.read(root / 'ambiguous-split/split.json')['state'] == 'insufficient_pregeometry_candidates'
        starvation_error = expect_error(lambda: prepare_holdout(source, root / 'starved-train.db',
            decoded, root / 'raw', root / 'starved', settings, min_training_features=1000))
        assert not (root / 'starved-train.db').exists()
        results['no_candidates_or_insufficient_solver_support_stop_before_geometry'] = [ambiguous_error, starvation_error]

        # This substitutes only extraction with known projected SIFT-layout rows.
        # Native matching, geometric verification, incremental mapping/BA,
        # image undistortion and the actual video_sfm integration all execute.
        def inject_known_features(database_path, *args, **kwargs):
            shutil.copy2(source, database_path)
        with patch.object(pc, 'extract_features', side_effect=inject_known_features):
            video_sfm.sfm(root, dict(cfg, sfm_holdout=settings))
        audit = rt.read(root / 'sfm/holdout/usage-after-mapping.json')
        camera = rt.read(root / 'camera-audit.json')
        assert audit['reserved_used'] == 0 and audit['reconstruction_checked']
        assert all(n > 0 for n in audit['usage_counts'].values()) and camera['registered'] == 5
        assert original_hash == rt.sha(source)
        results['native_matching_mapping_and_usage'] = dict(
            registered=camera['registered'], reserved_tracks=audit['reserved_tracks'],
            reserved_used=audit['reserved_used'], usage_counts=audit['usage_counts'],
            mean_training_reprojection_px=camera['mean_reprojection_error_px'],
            extraction_injected_known_measurements=True, native_geometry_actually_executed=True)

        conversion = rt.read(root / 'sfm/holdout/coordinate-conversion.json')
        assert conversion['tracks_exported'] > 0 and not conversion['rejected_tracks']
        assert max(o['raw_roundtrip_error_px'] for o in conversion['observations']) < 1e-5
        diagnostic_path = base / 'held-diagnostic.json'
        proc = subprocess.run([sys.executable, str(PROJECT / 'scripts/evaluate_fixed_tracks.py'),
            '--run', str(root), '--tracks', str(root / 'data/sfm-holdout-tracks.json'),
            '--report', str(diagnostic_path)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        measured = rt.read(diagnostic_path)
        assert measured['measured_tracks'] > 0 and measured['held_error_px']['p90'] < .1
        assert not measured['training_admitted'] and not measured['quality_accepted']
        results['raw_rectified_half_pixel_and_held_prediction'] = dict(
            raw_roundtrip_max_px=max(o['raw_roundtrip_error_px'] for o in conversion['observations']),
            held_error_px=measured['held_error_px'], training_admitted=False)

        # A manifest path must not turn the standalone diagnostic into reading
        # a camera set outside its declared source run.
        bank = rt.read(root / 'data/sfm-holdout-tracks.json')
        bank['camera_manifest_relative_path'] = '../unrelated.json';rt.save(base / 'bad-bank.json', bank)
        bad = subprocess.run([sys.executable, str(PROJECT / 'scripts/evaluate_fixed_tracks.py'),
            '--run', str(root), '--tracks', str(base / 'bad-bank.json'), '--report', str(base / 'bad-report.json')],
            capture_output=True, text=True)
        assert bad.returncode != 0 and 'must stay inside' in bad.stderr
        results['manifest_escape_rejected'] = True

        # Exercise real native extraction as well, without a reconstruction or
        # any captured images: a known synthetic texture undergoes translation.
        textured = base / 'texture';textured.mkdir();(textured / 'raw').mkdir()
        import cv2
        rng = np.random.default_rng(617);im = rng.integers(0, 255, (300, 400), dtype=np.uint8)
        im = cv2.GaussianBlur(im, (0, 0), .7);rows = []
        for i in range(5):
            name = f'frame_{i+1:06d}.png'
            Image.fromarray(cv2.warpAffine(im, np.float32([[1, 0, 3*i], [0, 1, 0]]), (400, 300))).save(textured / 'raw' / name)
            rows.append(dict(frame=Path(name).stem, name=name, time_seconds=.1*i, wh=[400, 300],
                             source_sha256=rt.sha(textured / 'raw' / name)))
        opt = pc.SiftExtractionOptions();opt.max_num_features=700;opt.num_threads=4
        pc.extract_features(str(textured / 'features.db'), str(textured / 'raw'),
            camera_mode=pc.CameraMode.SINGLE, camera_model='SIMPLE_RADIAL', sift_options=opt, device=pc.Device.cpu)
        native = prepare_holdout(textured / 'features.db', textured / 'train.db', rows,
            textured / 'raw', textured / 'split', settings)
        error = []
        for track in native['tracks']:
            p = track['observations'][0];i = int(p['frame'].split('_')[1])
            for q in track['observations'][1:]:
                j = int(q['frame'].split('_')[1])
                error.append(np.linalg.norm(np.asarray(q['raw_colmap_xy']) - p['raw_colmap_xy'] - [3*(j-i), 0]))
        assert len(error) and np.median(error) < .1
        results['actual_native_sift_on_known_image_translation'] = dict(
            tracks=len(native['tracks']), median_correspondence_error_px=float(np.median(error)),
            p90_correspondence_error_px=float(np.quantile(error,.9)))
    return dict(passed=True, groups=len(results), results=results,
        pycolmap=pc.__version__, synthetic_only=True, native_geometry_actually_executed=True,
        real_person_tested=False, physical_camera_calibration_validated=False,
        ghosting_resolved=False, training_started=False, original_video_reexecuted=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=run();rt.save(args.report,result)
    print(json.dumps(result,indent=2))
