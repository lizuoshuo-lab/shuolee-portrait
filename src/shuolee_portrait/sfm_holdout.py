"""Reserve native image measurements before any geometric camera estimation.

The audit proves exclusion of measured database rows, not complete independence
of physical points: untracked aliases, reflections and motion remain unknown.
No residual or camera parameter enters the appearance split.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from . import runtime as rt

DEFAULTS = dict(max_features=2000, temporal_hops=[1, 4], ratio=.75,
                fraction=.2, max_tracks=256, min_tracks=12, alias_radius_px=8.)
PAIR_BASE = 2147483647


def validate_settings(settings=None):
    s = dict(DEFAULTS, **(settings or {}))
    if set(s) != set(DEFAULTS):
        raise ValueError('Unknown pre-SfM holdout setting.')
    for name in ('max_features', 'max_tracks', 'min_tracks'):
        if type(s[name]) is not int or s[name] < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if not 8 <= s['max_features'] <= 8192 or s['max_tracks'] < s['min_tracks']:
        raise ValueError('Invalid feature or reserved-track budget.')
    for name in ('ratio', 'fraction'):
        if not isinstance(s[name], (float, int)) or not 0 < s[name] < 1:
            raise ValueError(f'{name} must lie in (0,1).')
    hops = s['temporal_hops']
    if (not isinstance(hops, list) or not hops or len(hops) > 4 or
            any(type(h) is not int or not 1 <= h <= 16 for h in hops) or hops != sorted(set(hops))):
        raise ValueError('Declare up to four distinct increasing temporal hops in [1,16].')
    if not isinstance(s['alias_radius_px'], (float, int)) or not np.isfinite(s['alias_radius_px']) or s['alias_radius_px'] <= 0:
        raise ValueError('Positive finite alias exclusion radius required.')
    return s


def _open(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def _array(db, table, image_id, dtype):
    row = db.execute(f'SELECT rows,cols,data FROM {table} WHERE image_id=?', (image_id,)).fetchone()
    if row is None or row[0] < 0 or row[1] < 1:
        raise ValueError(f'Missing or invalid {table} for image {image_id}.')
    a = np.frombuffer(row[2] or b'', dtype=dtype)
    if a.size != row[0] * row[1]:
        raise ValueError('Invalid native feature blob dimensions.')
    return a.reshape(row[0], row[1]).copy()


def _features(db, image_id):
    key = _array(db, 'keypoints', image_id, '<f4')
    desc = _array(db, 'descriptors', image_id, 'u1')
    if key.shape[1] not in (2, 4, 6) or desc.shape[1] != 128 or len(key) != len(desc) or not np.isfinite(key).all():
        raise ValueError('Expected native COLMAP SIFT keys (2/4/6 columns) and 128-byte descriptors.')
    return key, desc


def _observation(row, feature_index, xy):
    return dict(frame=row['frame'], image_id=row['image_id'], feature_index=int(feature_index),
                observation_id=f'colmap:{row["source_sha256"]}:{row["name"]}:{feature_index}',
                raw_image_sha256=row['source_sha256'], raw_colmap_xy=np.asarray(xy, float).tolist())


def prepare_holdout(source_db, training_db, decoded, raw, output, settings=None, min_training_features=50):
    """Copy an unmatched extraction database and remove reserved measurements."""
    import cv2
    from scipy.spatial import cKDTree
    from .material_tracks import mutual_matches, cycle_components, schedule
    source_db, training_db, raw, output = map(Path, (source_db, training_db, raw, output))
    s = validate_settings(settings)
    if output.exists() or training_db.exists():
        raise FileExistsError('Pre-SfM split and training database must be fresh.')
    if len(decoded) < 3 or len({r['name'] for r in decoded}) != len(decoded) or len({r['frame'] for r in decoded}) != len(decoded):
        raise ValueError('At least three unique real frames are required.')
    times = np.asarray([r['time_seconds'] for r in decoded], float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Use actual increasing capture order.')
    source_hash = rt.sha(source_db)
    with _open(source_db) as source:
        if any(source.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0] for t in ('matches', 'two_view_geometries')):
            raise ValueError('Source database has already been matched/verified; a retrospective split is forbidden.')
        images = dict(source.execute('SELECT name,image_id FROM images'))
        if set(images) != {r['name'] for r in decoded}:
            raise ValueError('Extracted database and decoded frame identity disagree.')
        rows = [dict(r, image_id=images[r['name']]) for r in decoded]
        for row in rows:
            if Path(row['name']).name != row['name'] or rt.sha(raw / row['name']) != row['source_sha256']:
                raise ValueError('Decoded image identity differs from extraction input.')
        triplets, pairs = schedule(len(rows), s['temporal_hops'])
        output.mkdir(parents=True);(output / 'pairs').mkdir();(output / 'row-maps').mkdir()
        rt.save(output / 'plan.json', dict(schema='shuolee-pre-sfm-plan-v1', settings=s,
            source_database_sha256=source_hash, frames=rows, triplets=triplets, pairs=pairs,
            implementation_sha256=rt.sha(Path(__file__)),
            matching_implementation_sha256=rt.sha(Path(__file__).with_name('material_tracks.py')),
            opencv_version=cv2.__version__, camera_parameters_used_for_split=False,
            geometry_or_residual_used_for_split=False, physical_identity_verified=False,
            selection='first native feature rows after exact-pixel dedup; SHA256-ranked whole cycle components',
            triangulation_views='first and last chronological observation, chosen before camera estimation'))
        @lru_cache(maxsize=12)
        def candidates(i):
            key, desc = _features(source, rows[i]['image_id'])
            wh = np.asarray(rows[i]['wh'], float)
            xy = key[:, :2] - .5
            valid = np.flatnonzero(((xy >= 0) & (xy <= wh - 1)).all(1))
            seen, ids = set(), []
            for idx in valid:
                location = tuple(key[idx, :2])
                if location not in seen:
                    seen.add(location);ids.append(int(idx))
                if len(ids) == s['max_features']:
                    break
            ids = np.asarray(ids, np.int64)
            return key[ids, :2], desc[ids], ids
        previous = cv2.getNumThreads();cv2.setNumThreads(4)
        pair_maps = {}
        try:
            for i, j in pairs:
                result = mutual_matches(candidates(i)[1], candidates(j)[1], s['ratio'])
                np.savez_compressed(output / 'pairs' / f'{i:06d}_{j:06d}.npz', **result)
                pair_maps[(i, j)] = {int(a): int(b) for a, b in result['matches']}
        finally:
            cv2.setNumThreads(previous)
        components, rejected, cycle_report = cycle_components(pair_maps, triplets)
        tracks = []
        for nodes in components:
            obs = [_observation(rows[i], int(candidates(i)[2][k]), candidates(i)[0][k]) for i, k in nodes]
            identity = hashlib.sha256(json.dumps([o['observation_id'] for o in obs]).encode()).hexdigest()
            tracks.append(dict(track_id='presfm_' + identity, observations=obs,
                               triangulation_frames=[obs[0]['frame'], obs[-1]['frame']]))
        tracks.sort(key=lambda t: t['track_id'])
        count = min(s['max_tracks'], int(len(tracks) * s['fraction']))
        selected = tracks[:count]
        split = dict(schema='shuolee-pre-sfm-split-v1', state='not_prepared', settings=s,
                     source_database_sha256=source_hash, plan_sha256=rt.sha(output / 'plan.json'),
                     candidate_count=len(tracks), candidate_track_ids=[t['track_id'] for t in tracks],
                     cycles=cycle_report, rejected_components=rejected, tracks=selected, frames=[],
                     camera_parameters_used_for_split=False, physical_track_independence_established=False,
                     limits=['Nearby alias exclusion is a fixed pixel-radius heuristic.',
                             'The same material may have untracked aliases in other views.',
                             'A closed appearance cycle does not establish staticity or physical identity.'])
        if count < s['min_tracks']:
            split['state'] = 'insufficient_pregeometry_candidates';rt.save(output / 'split.json', split)
            raise RuntimeError('Insufficient pre-SfM validation tracks. No matching or camera solve was started.')
        for row in rows:
            image_id = row['image_id'];key, desc = _features(source, image_id)
            chosen = sorted({o['feature_index'] for t in selected for o in t['observations'] if o['image_id'] == image_id})
            removed = np.zeros(len(key), bool)
            if chosen:
                removed = cKDTree(key[chosen, :2]).query(key[:, :2], k=1)[0] <= s['alias_radius_px']
            kept = np.flatnonzero(~removed);removed = np.flatnonzero(removed)
            map_path = f'row-maps/{image_id}.npz'
            np.savez_compressed(output / map_path, kept=kept, removed=removed,
                                reserved=np.asarray(chosen, np.int64))
            split['frames'].append(dict(**row, original_rows=len(key), training_rows=len(kept),
                removed_rows=len(removed), reserved_rows=len(chosen), row_map=map_path,
                row_map_sha256=rt.sha(output / map_path)))
        if min(r['training_rows'] for r in split['frames']) < min_training_features:
            split['state'] = 'insufficient_remaining_training_features';rt.save(output / 'split.json', split)
            raise RuntimeError('Holdout would leave insufficient camera-fit support; no fallback to all features.')
        if rt.sha(source_db) != source_hash:
            raise RuntimeError('Source database changed while preparing the split.')
        for row in rows:
            if rt.sha(raw / row['name']) != row['source_sha256']:
                raise RuntimeError('Decoded image changed while preparing the split.')
        with sqlite3.connect(training_db) as target:
            source.backup(target)
            for row in split['frames']:
                image_id = row['image_id'];key, desc = _features(source, image_id)
                with np.load(output / row['row_map'], allow_pickle=False) as m:
                    kept = m['kept']
                for table, a in [('keypoints', key[kept]), ('descriptors', desc[kept])]:
                    target.execute(f'UPDATE {table} SET rows=?,cols=?,data=? WHERE image_id=?',
                                   (len(a), a.shape[1], a.tobytes(), image_id))
        target.close()
    source.close()
    split['state'] = 'prepared_before_geometry';rt.save(output / 'split.json', split)
    audit_usage(source_db, training_db, output, report_name='usage-before-matching.json')
    return split


def audit_usage(source_db, training_db, output, reconstruction=None, report_name='usage-after-mapping.json'):
    """Audit native row identities through raw matches, verified matches and BA."""
    output = Path(output);split = rt.read(output / 'split.json')
    if split['state'] != 'prepared_before_geometry' or rt.sha(source_db) != split['source_database_sha256']:
        raise ValueError('Frozen extraction database or split state changed.')
    if rt.sha(output / 'plan.json') != split['plan_sha256']:
        raise ValueError('Frozen pregeometry plan changed.')
    frames, maps, use = {}, {}, {}
    with _open(source_db) as source, _open(training_db) as train:
        if dict(source.execute('SELECT image_id,name FROM images')) != dict(train.execute('SELECT image_id,name FROM images')):
            raise ValueError('Database image identity changed.')
        for row in split['frames']:
            image_id = row['image_id'];key, desc = _features(source, image_id)
            map_path = output / row['row_map']
            if rt.sha(map_path) != row['row_map_sha256']:
                raise ValueError('Frozen native row map changed.')
            with np.load(map_path, allow_pickle=False) as m:
                kept, removed, reserved = (m[k] for k in ('kept', 'removed', 'reserved'))
            if not np.array_equal(np.sort(np.r_[kept, removed]), np.arange(len(key))) or not set(reserved) <= set(removed):
                raise ValueError('Invalid row partition or reserved measurement retained.')
            k2, d2 = _features(train, image_id)
            if not np.array_equal(k2, key[kept]) or not np.array_equal(d2, desc[kept]):
                raise ValueError('Training keypoints/descriptors were changed or reserved rows reintroduced.')
            frames[image_id] = row;maps[image_id] = kept;use[image_id] = np.zeros(len(key), np.uint8)
        pair_counts = {}
        for table, bit in [('matches', 1), ('two_view_geometries', 2)]:
            pair_counts[table] = 0
            for pair, n, cols, blob in train.execute(f'SELECT pair_id,rows,cols,data FROM {table}'):
                if n == 0:
                    continue
                a, b = divmod(pair, PAIR_BASE)
                if a not in maps or b not in maps or a >= b or cols != 2:
                    raise ValueError('Invalid COLMAP pair identity.')
                match = np.frombuffer(blob, dtype='<u4')
                if len(match) != n * 2:
                    raise ValueError('Corrupt match blob.')
                match = match.reshape(n, 2)
                for side, image_id in enumerate((a, b)):
                    if (match[:, side] >= len(maps[image_id])).any():
                        raise ValueError('Match references an invalid compacted feature row.')
                    use[image_id][maps[image_id][match[:, side]]] |= bit
                pair_counts[table] += n
        if reconstruction is not None:
            for image_id, image in reconstruction.images.items():
                if image_id not in frames or image.name != frames[image_id]['name']:
                    raise ValueError('Reconstruction image identity changed.')
                key, _ = _features(train, image_id)
                if len(image.points2D) != len(key):
                    raise ValueError('Reconstruction feature index layout differs from training database.')
                for index, point in enumerate(image.points2D):
                    if not np.allclose(point.xy, key[index, :2], rtol=0, atol=1e-6):
                        raise ValueError('Reconstruction pixels differ from frozen native measurements.')
                    if point.has_point3D():
                        use[image_id][maps[image_id][index]] |= 4
    source.close();train.close()
    artifact = output / (Path(report_name).stem + '.npz')
    if artifact.exists() or (output / report_name).exists():
        raise FileExistsError('Usage reports are immutable; use a different named stage.')
    np.savez_compressed(artifact, **{str(k): v for k, v in use.items()})
    reserved = [(o['image_id'], o['feature_index']) for t in split['tracks'] for o in t['observations']]
    overlap = [(i, k) for i, k in reserved if use[i][k]]
    if overlap:
        raise ValueError('Reserved measurement reached native geometry.')
    report = dict(state='reserved_measurements_excluded', split_sha256=rt.sha(output / 'split.json'),
        source_database_sha256=split['source_database_sha256'], training_database_sha256=rt.sha(training_db),
        reserved_tracks=len(split['tracks']), reserved_observations=len(reserved), reserved_used=len(overlap),
        frames_without_reserved_measurements=[r['frame'] for r in split['frames'] if not r['reserved_rows']],
        pair_counts=pair_counts, reconstruction_checked=reconstruction is not None,
        usage_counts={name: int(sum(np.count_nonzero(v & bit) for v in use.values()))
                      for name, bit in [('descriptor_matches', 1), ('geometric_verification', 2), ('reconstructed_points', 4)]},
        usage_bitmask=dict(descriptor_matches=1, geometric_verification=2, reconstructed_points=4),
        usage_arrays_sha256=rt.sha(artifact), physical_track_independence_established=False,
        quality_accepted=False, training_admitted=False)
    rt.save(output / report_name, report)
    return report


def export_rectified_holdout(root, raw_rec, rectified_rec):
    """Transform frozen raw observations with the SAME native camera models."""
    root = Path(root);directory = root / 'sfm/holdout'
    split = rt.read(directory / 'split.json');audit = rt.read(directory / 'usage-after-mapping.json')
    if not audit['reconstruction_checked'] or audit['split_sha256'] != rt.sha(directory / 'split.json'):
        raise ValueError('Mapped-camera exclusion audit is missing or stale.')
    manifest = root / 'data/sfm-inputs.json';manifest_hash = rt.sha(manifest)
    rows = {r['frame']: r for r in rt.read(manifest)['records']}
    raw_images = {i.name: i for i in raw_rec.images.values()}
    new_images = {i.name: i for i in rectified_rec.images.values()}
    bank, rejected, conversion = [], [], []
    for track in split['tracks']:
        obs, errors = [], []
        for original in track['observations']:
            row = rows[original['frame']]
            old = raw_images[row['name']];new = new_images[row['name']]
            rawcam = raw_rec.cameras[old.camera_id];newcam = rectified_rec.cameras[new.camera_id]
            if (not np.allclose(old.cam_from_world.matrix(), new.cam_from_world.matrix(), rtol=0, atol=1e-9) or
                    not np.allclose(np.asarray(row['w2c'])[:3], new.cam_from_world.matrix(), rtol=0, atol=1e-9) or
                    not np.allclose(row['K'], newcam.calibration_matrix(), rtol=0, atol=1e-9)):
                raise ValueError('Raw, rectified and manifest camera transforms disagree.')
            xy = np.asarray(original['raw_colmap_xy'])
            ray = rawcam.cam_from_img(xy)
            uv = np.asarray(newcam.img_from_cam(ray)) - .5
            roundtrip = float(np.linalg.norm(rawcam.img_from_cam(ray) - xy))
            supported = (np.isfinite(uv).all() and np.isfinite(roundtrip) and roundtrip <= .01 and
                         (uv >= 0).all() and (uv <= np.asarray(row['wh']) - 1).all())
            conversion.append(dict(**original, rectified_uv=uv.tolist() if np.isfinite(uv).all() else None,
                                   raw_roundtrip_error_px=roundtrip if np.isfinite(roundtrip) else None,
                                   supported=bool(supported)))
            if not supported:
                errors.append(original['observation_id']);continue
            obs.append(dict(frame=row['frame'], observation_id=original['observation_id'],
                            image_sha256=row['source_sha256'], uv=uv.tolist()))
        if errors:
            rejected.append(dict(track_id=track['track_id'], reason='coordinate_conversion_unsupported',
                                 observation_ids=errors))
            continue
        bank.append(dict(track_id=track['track_id'], triangulation_frames=track['triangulation_frames'],
                         identity_basis='material_candidate', region='unknown', observations=obs))
    for original in split['frames']:
        if rt.sha(root / 'raw' / original['name']) != original['source_sha256']:
            raise ValueError('Raw image changed after the pregeometry split.')
    for row in rows.values():
        if rt.sha(row['image']) != row['source_sha256']:
            raise ValueError('Rectified image changed after camera export.')
    rt.save(root / 'data/sfm-holdout-tracks.json', dict(schema='shuolee-fixed-tracks-v1',
        pixel_domain='undistorted_integer_centers', camera_manifest_sha256=manifest_hash,
        camera_manifest_relative_path='data/sfm-inputs.json', camera_fit_observation_ids=None,
        tracks=bank, physical_material_identity_verified=False,
        camera_solver_provenance=dict(state='native_measurements_excluded_before_geometry',
            audit_relative_path='sfm/holdout/usage-after-mapping.json',
            audit_sha256=rt.sha(directory / 'usage-after-mapping.json'),
            physical_track_independence_established=False)))
    rt.save(directory / 'coordinate-conversion.json', dict(tracks_exported=len(bank),
        rejected_tracks=rejected, observations=conversion, residual_filtering=False,
        coordinate_contract='native raw COLMAP corners -> cam_from_img -> rectified img_from_cam -> subtract 0.5',
        camera_parameters_estimated_not_ground_truth=True))
    return len(bank)
