"""Appearance-only, cycle-checked texture candidates from the current video.

This is a CPU SIFT baseline, not proof of physical material identity. It never
uses K, R/t, epipolar residuals, fitted bodies, or learned Gaussians to select
matches. Camera-fit independence and subject staticity remain unverified.
"""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from . import runtime as rt


DEFAULTS = dict(max_features=2000, ratio=0.75, temporal_hops=[1, 2, 4],
                boundary_margin_px=4, sift_support_factor=3.0)
REGIONS = ('static_scene_candidate', 'body', 'face')
DECISIONS = {0: 'mutual_ratio_match', 1: 'insufficient_neighbors',
             2: 'ambiguous_descriptor_ratio', 3: 'reverse_ratio_rejected',
             4: 'reverse_nearest_identity_mismatch'}


def validate_settings(settings=None):
    result = dict(DEFAULTS, **(settings or {}))
    if set(result) != set(DEFAULTS):
        raise ValueError('Unknown material-track setting.')
    if type(result['max_features']) is not int or not 8 <= result['max_features'] <= 8192:
        raise ValueError('max_features must be an integer in [8,8192].')
    if not isinstance(result['ratio'], (float, int)) or not 0 < result['ratio'] < 1:
        raise ValueError('Descriptor ratio must be in (0,1).')
    hops = result['temporal_hops']
    if (not isinstance(hops, list) or not hops or len(hops) > 4 or
            any(type(h) is not int or not 1 <= h <= 16 for h in hops) or
            hops != sorted(set(hops))):
        raise ValueError('Use at most four increasing distinct integer temporal hops in [1,16].')
    for key in ('boundary_margin_px', 'sift_support_factor'):
        if not isinstance(result[key], (float, int)) or not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be positive and finite.')
    return result


def schedule(frame_count, hops):
    triplets = sorted({(i, i + h, i + 2 * h) for h in hops for i in range(frame_count - 2 * h)})
    pairs = sorted({pair for a, b, c in triplets for pair in [(a, b), (b, c), (a, c)]})
    if not triplets:
        raise ValueError('No actual three-view triplet fits the declared temporal hops.')
    return triplets, pairs


def _direction(a, b, ratio):
    states = np.full(len(a), 1, np.uint8)
    nearest = np.full(len(a), -1, np.int32)
    ratios = np.full(len(a), np.nan, np.float32)
    if not len(a) or len(b) < 2:
        return nearest, ratios, states
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a, b, k=2)
    for neighbors in matches:
        if len(neighbors) != 2:
            continue
        first, second = neighbors
        i = first.queryIdx
        states[i] = 2
        ratios[i] = first.distance / max(second.distance, 1e-12)
        # Two identical descriptors are ambiguous, including the zero/zero case.
        if second.distance > 1e-12 and first.distance < ratio * second.distance:
            nearest[i] = first.trainIdx
            states[i] = 0
    return nearest, ratios, states


def mutual_matches(a, b, ratio=.75):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Descriptor arrays must have matching vector dimensions.')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not 0 < ratio < 1:
        raise ValueError('Finite descriptors and ratio in (0,1) are required.')
    ab, ratio_ab, state_a = _direction(a, b, ratio)
    ba, ratio_ba, state_b = _direction(b, a, ratio)
    accepted = []
    for i, j in enumerate(ab):
        if j < 0:
            continue
        if ba[j] < 0:
            state_a[i] = 3
        elif ba[j] != i:
            state_a[i] = 4
        else:
            accepted.append((i, int(j)))
    for j, i in enumerate(ba):
        if i >= 0 and ab[i] != j:
            state_b[j] = 3 if ab[i] < 0 else 4
    return dict(matches=np.asarray(accepted, np.int32).reshape(-1, 2),
                forward_ratio=ratio_ab, reverse_ratio=ratio_ba,
                forward_decision=state_a, reverse_decision=state_b)


def _features(row, settings):
    gray = np.asarray(Image.open(row['image']).convert('L'))
    mask = np.asarray(Image.open(row['mask']).convert('L')) > 127
    height, width = gray.shape
    if list(row['wh']) != [width, height] or mask.shape != gray.shape:
        raise ValueError(f'Image, mask and camera dimensions disagree: {row["frame"]}')
    detector = cv2.SIFT_create(nfeatures=settings['max_features'], contrastThreshold=.02)
    keys, descriptors = detector.detectAndCompute(gray, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), np.float32)
    interior = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 5)
    exterior = cv2.distanceTransform((~mask).astype('uint8'), cv2.DIST_L2, 5)
    face_box = row.get('face_box_xyxy')
    if face_box is not None:
        face_box = np.asarray(face_box, float)
        if face_box.shape != (4,) or not np.isfinite(face_box).all() or (face_box[2:] <= face_box[:2]).any():
            raise ValueError('Invalid face region box; it is only an ROI, not a correspondence.')
    other_boxes = row.get('other_person_boxes_xyxy', [])
    if any(np.asarray(b).shape != (4,) or not np.isfinite(b).all() for b in other_boxes):
        raise ValueError('Invalid non-target person exclusion boxes.')
    selected, regions, rejected = [], [], Counter()
    seen = set()
    # SIFT may emit several orientations for exactly one source location. Keep
    # one deterministically before matching; do not count aliases as new points.
    order = sorted(range(len(keys)), key=lambda k: (-keys[k].response, keys[k].pt, keys[k].angle))
    for i in order:
        key = keys[i]
        x, y = key.pt
        location = (float(x), float(y))
        if location in seen:
            rejected['same_pixel_orientation_alias'] += 1
            continue
        seen.add(location)
        support = max(settings['boundary_margin_px'], settings['sift_support_factor'] * key.size)
        if min(x, y, width - 1 - x, height - 1 - y) < support:
            rejected['image_boundary_support'] += 1
            continue
        ix, iy = int(round(x)), int(round(y))
        if any(b[0] - support <= x <= b[2] + support and b[1] - support <= y <= b[3] + support for b in other_boxes):
            rejected['non_target_person_box'] += 1
            continue
        if min(interior[iy, ix], exterior[iy, ix]) != 0:
            raise AssertionError('Invalid binary-mask distance fields.')
        distance = interior[iy, ix] if mask[iy, ix] else exterior[iy, ix]
        if distance < support:
            rejected['foreground_boundary_support'] += 1
            continue
        region = 0
        if mask[iy, ix]:
            region = 1
            if face_box is not None and face_box[0] <= x <= face_box[2] and face_box[1] <= y <= face_box[3]:
                region = 2
        selected.append(i);regions.append(region)
    selected = np.asarray(selected, np.int32)
    return dict(xy=np.asarray([keys[i].pt for i in selected], np.float32).reshape(-1, 2),
                descriptor=descriptors[selected], region=np.asarray(regions, np.uint8),
                sift_index=selected), dict(detected=len(keys), retained=len(selected),
                    rejected=dict(rejected), non_target_person_detection_recorded='other_person_boxes_xyxy' in row,
                    face_roi_available=face_box is not None)


def cycle_components(pair_maps, triplets):
    parent = {}
    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    counters = Counter()
    for a, b, c in triplets:
        ab, bc, ac = pair_maps[(a, b)], pair_maps[(b, c)], pair_maps[(a, c)]
        for x, y in ab.items():
            counters['candidate_chains'] += 1
            z = bc.get(y)
            if z is None:
                counters['missing_second_edge'] += 1
            elif ac.get(x) != z:
                counters['third_edge_identity_mismatch_or_absent'] += 1
            else:
                counters['closed_cycles'] += 1
                union((a, x), (b, y));union((b, y), (c, z))
    components = {}
    for node in sorted(parent):
        components.setdefault(find(node), []).append(node)
    accepted, rejected = [], []
    for nodes in components.values():
        frames = [n[0] for n in nodes]
        if len(set(frames)) != len(frames):
            rejected.append(dict(nodes=nodes, reason='multiple_feature_identities_in_one_frame'))
        else:
            accepted.append(sorted(nodes))
    counters['conflicting_components'] = len(rejected)
    return sorted(accepted), rejected, dict(counters)


def extract_material_tracks(root, output, settings=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    settings = validate_settings(settings)
    if output.exists():
        raise FileExistsError(f'Fresh material-track output required: {output}')
    manifest = root / 'data/training-inputs.json'
    manifest_hash = rt.sha(manifest)
    input_manifest = rt.read(manifest)
    if input_manifest.get('pixel_domain') != 'undistorted_integer_centers':
        raise ValueError('The source manifest must explicitly identify the undistorted integer-centre domain.')
    rows = input_manifest['records']
    if len(rows) < 3 or len({r['frame'] for r in rows}) != len(rows):
        raise ValueError('Three or more distinct real frames are required.')
    times = np.asarray([r['time_seconds'] for r in rows], float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Frames must be in strictly increasing recorded-time order.')
    for row in rows:
        if not row.get('mask') or not row.get('mask_sha256'):
            raise ValueError('A verified subject mask is required for each frame.')
        for path_key, hash_key in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
            if rt.sha(row[path_key]) != row[hash_key]:
                raise ValueError(f'Input bytes differ from the manifest: {row["frame"]} {path_key}')
    triplets, pairs = schedule(len(rows), settings['temporal_hops'])
    output.mkdir(parents=True);(output / 'features').mkdir();(output / 'pairs').mkdir()
    rt.save(output / 'plan.json', dict(schema='shuolee-material-track-plan-v1',
        camera_manifest_sha256=manifest_hash, settings=settings, opencv_version=cv2.__version__,
        implementation_sha256=rt.sha(Path(__file__)),
        frames=[dict(frame=r['frame'], time_seconds=r['time_seconds'], image_sha256=r['source_sha256'], mask_sha256=r['mask_sha256']) for r in rows],
        triplets=triplets, pairs=pairs, pixel_domain='undistorted_integer_centers',
        camera_residual_selection=False, camera_fit_independence_established=False,
        triangulation_selection='first and last chronological observation; all intermediate observations held',
        projection_diagnostics_run=False, scope='appearance_candidates_not_physical_validation'))
    previous_threads = cv2.getNumThreads();cv2.setNumThreads(4)
    feature_reports, pair_reports, pair_maps = [], [], {}
    try:
        for i, row in enumerate(rows):
            data, info = _features(row, settings)
            np.savez_compressed(output / 'features' / f'{i:06d}.npz', **data)
            feature_reports.append(dict(frame=row['frame'], **info))
        @lru_cache(maxsize=12)
        def load(i):
            with np.load(output / 'features' / f'{i:06d}.npz', allow_pickle=False) as values:
                return {k: values[k] for k in values.files}
        for i, j in pairs:
            match = mutual_matches(load(i)['descriptor'], load(j)['descriptor'], settings['ratio'])
            np.savez_compressed(output / 'pairs' / f'{i:06d}_{j:06d}.npz', **match)
            pair_maps[(i, j)] = {int(a): int(b) for a, b in match['matches']}
            pair_reports.append(dict(frames=[rows[i]['frame'], rows[j]['frame']], matches=len(match['matches']),
                                     forward_decisions=dict(Counter(DECISIONS[int(v)] for v in match['forward_decision']))))
        components, rejected, cycle_report = cycle_components(pair_maps, triplets)
        tracks = []
        for nodes in components:
            regions = [REGIONS[int(load(i)['region'][k])] for i, k in nodes]
            region = regions[0] if len(set(regions)) == 1 else ('body' if set(regions) <= {'face', 'body'} else 'unknown')
            observations = [dict(frame=rows[i]['frame'],
                                 observation_id=f'sift:{rows[i]["frame"]}:{rows[i]["source_sha256"]}:{int(load(i)["sift_index"][k])}',
                                 image_sha256=rows[i]['source_sha256'], uv=load(i)['xy'][k].astype(float).tolist()) for i, k in nodes]
            identity = hashlib.sha256(json.dumps([o['observation_id'] for o in observations]).encode()).hexdigest()[:20]
            tracks.append(dict(track_id=f'texture_{identity}', identity_basis='material_candidate', region=region,
                               triangulation_frames=[observations[0]['frame'], observations[-1]['frame']],
                               observations=observations))
    finally:
        cv2.setNumThreads(previous_threads)
    if rt.sha(manifest) != manifest_hash:
        raise ValueError('Source manifest changed during extraction; do not use this candidate bank.')
    for row in rows:
        if rt.sha(row['image']) != row['source_sha256'] or rt.sha(row['mask']) != row['mask_sha256']:
            raise ValueError('Source image or mask changed during extraction; do not use this bank.')
    bank = dict(schema='shuolee-fixed-tracks-v1', pixel_domain='undistorted_integer_centers',
                camera_manifest_sha256=manifest_hash, camera_fit_observation_ids=None,
                tracks=tracks, generator='OpenCV SIFT mutual ratio and exact three-view feature cycles',
                physical_material_identity_verified=False,
                camera_solver_provenance=dict(state='not_independently_withheld',
                    reason='Cameras precede this extraction. Independently named OpenCV features may overlap the COLMAP observations; no independent validation is claimed.'))
    rt.save(output / 'tracks.json', bank)
    rt.save(output / 'rejected-components.json', dict(components=rejected, pair_decision_codes=DECISIONS))
    report = dict(schema='shuolee-material-track-summary-v1', settings=settings,
                  camera_manifest_sha256=manifest_hash, track_bank_sha256=rt.sha(output / 'tracks.json'),
                  frames=len(rows), track_count=len(tracks), tracks_by_region=dict(Counter(t['region'] for t in tracks)),
                  frames_with_tracks=len({o['frame'] for t in tracks for o in t['observations']}),
                  features=feature_reports, pairs=pair_reports, cycles=cycle_report,
                  state='candidate_tracks_generated' if tracks else 'no_cycle_consistent_tracks',
                  source_inputs_unchanged=True, camera_parameters_used_for_matching=False,
                  independent_camera_validation=False, subject_staticity_verified=False,
                  physical_material_identity_verified=False, training_admitted=False,
                  quality_accepted=False, ghosting_resolved=False,
                  limits=['Cycle-consistent appearance can still follow repeated patterns or reflections.',
                          'Weakly textured face/skin and occlusions may have no tracks; no semantic-point substitution.',
                          'Mask and detector boxes are estimates; background candidates are not proven static.',
                          'The SIFT support margin is a boundary heuristic, not a calibrated correspondence uncertainty.',
                          'Camera-use independence and micro-motion validation are not implemented by this extractor.'])
    rt.save(output / 'summary.json', report)
    return report
