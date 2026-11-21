"""Select native side-face observations from the protected, already fitted roster."""

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation

from . import runtime as rt
from .camera_world_units import normalize_camera_w2c
from .official_smplifyx import COCO, adapt_rtmw_observations


REGIONS = dict(right_eye=np.arange(86, 92), left_eye=np.arange(92, 98),
               nose=np.arange(77, 86), mouth=np.arange(98, 118),
               right_ear=np.asarray([17]), left_ear=np.asarray([18]))
REPLAY = ('global_orient', 'expression', 'jaw_pose', 'leye_pose', 'reye_pose',
          'left_hand_pose', 'right_hand_pose', 'body_pose_axis_angle', 'camera_translation')
CONFIG = dict(max_views=4, yaw_sector_deg=15., min_side_yaw_deg=25., max_side_yaw_deg=78.,
              max_pitch_deg=42., min_novel_yaw_deg=6.,
              min_supplement_spacing_s=.5, min_face_span_px=32., min_confidence=.5,
              min_raw_score=.35, max_flip_px=6., max_flip_over_span=.08,
              max_region_rms_px=4., max_region_rms_over_span=.055,
              max_region_peak_px=8., max_region_peak_over_span=.10,
              min_eye_points=4, min_nose_points=5, min_mouth_points=8,
              ear_visibility_min_depth_difference_m=.005, max_mesh_occlusion_m=.012, source_alpha_min=.99,
              support_padding_over_span=.075, min_roi_foreground_fraction=.90,
              min_luma_gradient=2./255, max_saturated_fraction=.30)


@dataclass(frozen=True)
class FaceSupplement:
    row: dict
    box: tuple
    valid: np.ndarray
    alpha: np.ndarray
    pose: dict


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _unique_rows(rows):
    names = [row['frame'] for row in rows]
    source = [row['source_frame_index'] for row in rows]
    _require(len(set(names)) == len(set(source)) == len(names) > 0
             and all(isinstance(name, str) and name for name in names)
             and all(type(i) is int and i >= 0 for i in source), 'Invalid native source roster.')
    return dict(zip(names, rows))


def _bound_sha(path, source, suffix):
    digest = rt.sha(path)
    expected = [value for key, value in source.items() if Path(key).as_posix().endswith('/'+suffix)]
    _require(len(expected) == 1 and digest == expected[0], 'Changed official source: '+suffix)
    return digest


def _load_inputs(root, rows):
    official = rt.read(root/'fit/official-fit-result.json')
    joint = official.get('joint_body_refinement')
    _require(official.get('state') == 'complete' and isinstance(joint, dict),
             'Face supplements require the complete named joint fit.')
    selection_path = root/'fit/selection.json'
    selection = rt.read(selection_path)
    _require(selection == official['selection'], 'The frozen full-body selection changed.')
    native_path = root/'data/inputs.json'
    observations_path = root/'data/observations.npz'
    paths = dict(native_inputs=native_path, observations=observations_path, selection=selection_path,
                 base_fit=root/'fit/fitted.npz', base_rows=root/'data/training-inputs.json',
                 joint_candidates=root/'fit/joint-candidates.npz', joint_fit=root/'fit/joint-body-refinement.json',
                 official_fit=root/'fit/official-fit-result.json')
    digests = {name: rt.sha(path) for name, path in paths.items()}
    source = official['source']['input_files']
    _bound_sha(native_path, source, 'data/inputs.json')
    _bound_sha(observations_path, source, 'data/observations.npz')
    expected_candidate = joint['candidate_sha256']
    if 'native_head_alignment' in official:
        from .head_fit_stage import verify_head_fit

        refinement = verify_head_fit(root, official)
        _require(refinement['source_sha256']['joint-candidates.npz'] == expected_candidate,
                 'The native head stage does not descend from the archived joint fit.')
        expected_candidate = refinement['active']['candidate_sha256']
        digests['native_head_alignment'] = official['native_head_alignment']['sha256']
    _require(digests['joint_candidates'] == expected_candidate and digests['joint_fit'] == joint['sha256']
             and digests['base_fit'] == official['fit_sha256']
             and digests['base_rows'] == official['training_inputs_sha256'],
             'A frozen fit input differs from its receipt.')
    roster = _unique_rows(rt.read(native_path)['records'])
    training = rt.read(paths['base_rows'])
    base = _unique_rows(rows)
    selected, held = set(selection['selected_frames']), set(selection['retained_holdouts'])
    _require(rows == training['records'] and selected == set(base) and not selected & held
             and held <= roster.keys(), 'Full-body roster or retained holdout identity changed.')
    scope = rt.read(paths['joint_fit'])
    with np.load(paths['joint_candidates'], allow_pickle=False) as archive:
        fit = dict(archive)
    names = fit['frame_ids'].astype(str).tolist()
    _require(names == scope['fit_frame_ids'] and set(scope['excluded_holdouts']) == held
             and len(set(names)) == len(names) and selected <= set(names) <= roster.keys()
             and not set(names) & held, 'Joint candidate roster differs from the held-out fit contract.')
    with np.load(paths['base_fit'], allow_pickle=False) as saved:
        _require(np.array_equal(fit['betas'], saved['shared_betas']) and np.array_equal(fit['faces'], saved['faces']),
                 'Face supplements changed the shared shape or topology.')
        slots = {name: i for i, name in enumerate(names)}
        full = [slots[name] for name in saved['frame_ids'].astype(str).tolist()]
        for key in (*REPLAY, 'K_integer_centers', 'vertices_camera', 'observed_native_keypoints'):
            _require(np.array_equal(fit[key][full], saved[key]), 'Candidate and base fit differ: '+key)
    candidates = [roster[name] for name in names]
    units = normalize_camera_w2c([row['w2c'] for row in candidates], training['source_camera_length_contract'])
    K = np.asarray([row['K_integer_centers'] for row in candidates])
    corner = K.copy()
    corner[:, :2, 2] += .5
    _require(np.array_equal(corner, [row['K'] for row in candidates])
             and np.array_equal(K, fit['K_integer_centers']) and np.isfinite(K).all()
             and np.all(K[:, (0, 1), (0, 1)] > 0)
             and np.all(K[:, 0, 1] == 0) and np.all(K[:, 1, 0] == 0)
             and np.all(K[:, 2] == [0, 0, 1]), 'Native source camera differs from fitted projection.')
    for i, row in enumerate(candidates):
        name = row['frame']
        _require(int(fit['source_frames'][i]) == row['source_frame_index']
                 and fit['image_sha256'][i] == row['source_sha256'] and fit['mask_sha256'][i] == row['mask_sha256']
                 and len(row['wh']) == 2 and all(type(v) is int and v > 0 for v in row['wh'])
                 and type(row['time_seconds']) in (int, float) and np.isfinite(row['time_seconds']),
                 'Native candidate source identity differs: '+name)
        if name in base:
            _require(np.array_equal(units.w2c_meters[i], base[name]['w2c'])
                     and np.array_equal(units.source_w2c[i], base[name]['source_w2c']),
                     'Base and candidate use different camera transforms: '+name)

    with np.load(observations_path, allow_pickle=False) as saved:
        observed = dict(saved)
    obs_names = observed['frame_ids'].astype(str).tolist()
    _require(len(set(obs_names)) == len(obs_names) and set(obs_names) == roster.keys(),
             'Native detector roster differs.')
    obs_slots = {name: i for i, name in enumerate(obs_names)}
    order = [obs_slots[name] for name in names]
    fields = ('keypoints', 'scores', 'boxes', 'flip_disagreement')
    points, raw, boxes, flips = [observed[key][order] for key in fields]
    targets, _ = adapt_rtmw_observations(points, raw, boxes, flips)
    dimensions = np.asarray([row['wh'] for row in candidates])[:, None, :]
    on_screen = ((targets[..., :2] >= -.5) & (targets[..., :2] < dimensions-.5)).all(-1)
    targets[..., 2] *= on_screen
    face = np.r_[17, 18, np.arange(67, 118)]
    _require(np.array_equal(targets[:, face], fit['observed_native_keypoints'][:, face]),
             'The archived face measurements differ from the native detector observations.')
    dst, src = np.r_[COCO, np.arange(67, 118)], np.r_[np.arange(17), np.arange(40, 91)]
    native_raw, native_flip = np.zeros_like(targets[..., 2]), np.zeros_like(targets[..., 2])
    native_raw[:, dst], native_flip[:, dst] = raw[:, src], flips[:, src]
    projected = fit['joints_camera'].astype(np.float64) @ K.transpose(0, 2, 1)
    _require(all(np.isfinite(fit[key]).all() for key in (*REPLAY, 'vertices_camera', 'joints_camera',
                                                       'joints_projected_native'))
             and np.all(fit['vertices_camera'][..., 2] > .05) and np.all(projected[..., 2] > .05)
             and np.allclose(projected[..., :2]/projected[..., 2:3], fit['joints_projected_native'], atol=.001, rtol=0),
             'Archived candidate geometry fails exact native projection replay.')
    return roster, candidates, selection, fit, units, native_raw, native_flip, digests


def _directions(fit, reference):
    root = Rotation.from_rotvec(fit['global_orient']).as_matrix()
    relative = root @ root[reference].T
    root_yaw = np.degrees(np.arctan2(relative[:, 0, 2], relative[:, 2, 2]))
    head = root
    body = fit['body_pose_axis_angle'].reshape(-1, 21, 3)
    # Global pelvis, then the canonical SMPL-X spine, neck and head chain.
    for joint in (3, 6, 9, 12, 15):
        head = head @ Rotation.from_rotvec(body[:, joint-1]).as_matrix()
    to_camera = np.einsum('nij,ni->nj', head, -fit['joints_camera'][:, 0])
    yaw = np.degrees(np.arctan2(to_camera[:, 0], to_camera[:, 2]))
    pitch = np.degrees(np.arctan2(to_camera[:, 1], np.linalg.norm(to_camera[:, [0, 2]], axis=1)))
    return yaw, pitch, root_yaw


def _sector(yaw):
    size = CONFIG['yaw_sector_deg']
    return int(size*np.floor((yaw+.5*size)/size))


def _read_pixels(row):
    decoded = []
    for key, digest, mode in (('image', 'source_sha256', 'RGB'), ('mask', 'mask_sha256', 'L')):
        path = Path(row[key])
        _require(path.is_absolute(), 'Candidate input is not a native absolute path.')
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == row[digest], 'Changed candidate '+key+': '+row['frame'])
        with Image.open(io.BytesIO(raw)) as image:
            _require(list(image.size) == row['wh'], 'Candidate native raster dimensions differ: '+row['frame'])
            decoded.append(np.asarray(image.convert(mode)))
    return decoded


def _pixel_support(xy, mask):
    h, w = mask.shape
    inside = ((xy >= -.5) & (xy < [w-.5, h-.5])).all(1)
    x, y = np.rint(xy).astype(int).T
    return inside & (mask[y.clip(0, h-1), x.clip(0, w-1)] >= 255*CONFIG['source_alpha_min'])


def _front_depth(points, vertices, faces, K):
    """Native ray intersection with the frozen fitted mesh, using perspective-correct depth."""
    homogeneous = vertices.astype(np.float64) @ K.T
    screen = homogeneous[:, :2]/homogeneous[:, 2:3]
    triangles = screen[faces]
    low, high = triangles.min(1), triangles.max(1)
    depth = 1./vertices[faces, 2]
    edge_a, edge_b = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
    det = edge_a[:, 0]*edge_b[:, 1]-edge_a[:, 1]*edge_b[:, 0]
    output = np.full(len(points), np.inf)
    for i, point in enumerate(points):
        possible = ((point >= low-1e-4) & (point <= high+1e-4)).all(1) & (abs(det) > 1e-9)
        if not possible.any():
            continue
        delta = point-triangles[possible, 0]
        a, b = edge_a[possible], edge_b[possible]
        u = (delta[:, 0]*b[:, 1]-delta[:, 1]*b[:, 0])/det[possible]
        v = (a[:, 0]*delta[:, 1]-a[:, 1]*delta[:, 0])/det[possible]
        weights = np.stack((1-u-v, u, v), axis=1)
        covered = (weights >= -1e-5).all(1)
        if covered.any():
            output[i] = 1./np.max(np.sum(weights[covered]*depth[possible][covered], axis=1))
    return output


def _hull(points):
    # Andrew's monotone chain on the integer native landmark pixels.
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])

    coords = sorted(set(map(tuple, np.rint(points).astype(int).tolist())))
    if len(coords) < 3:
        return coords
    lower, upper = [], []
    for order, out in ((coords, lower), (reversed(coords), upper)):
        for point in order:
            while len(out) >= 2 and cross(out[-2], out[-1], point) <= 0:
                out.pop()
            out.append(point)
    return lower[:-1]+upper[:-1]


def _source_region_masks(observed, groups, unavailable, span, image, mask, row):
    padding = max(2, round(span*CONFIG['support_padding_over_span']))
    points = observed[np.concatenate(list(groups.values())), :2]
    left, top = np.maximum(np.floor(points.min(0)).astype(int)-padding-2, 0)
    right, bottom = np.minimum(np.ceil(points.max(0)).astype(int)+padding+3, row['wh'])
    box = (int(left), int(top), int(right), int(bottom))
    y, x = slice(top, bottom), slice(left, right)
    shape = (bottom-top, right-left)
    origin = np.array([left, top])
    raster = Image.new('1', (shape[1], shape[0]))
    draw = ImageDraw.Draw(raster)
    core = np.concatenate([indices for name, indices in groups.items() if not name.endswith('_ear')])
    outline = _hull(observed[core, :2]-origin)
    if len(outline) >= 3:
        draw.polygon(outline, fill=1)
    for name, ids in groups.items():
        if name.endswith('_ear'):
            pixel = tuple(np.rint(observed[ids[0], :2]-origin).astype(int))
            draw.point(pixel, fill=1)
    grid = np.arange(-padding, padding+1)
    disk = grid[:, None]**2+grid[None, :]**2 <= padding**2
    support = binary_dilation(np.asarray(raster), structure=disk)
    if len(unavailable):
        excluded = Image.new('1', raster.size)
        draw = ImageDraw.Draw(excluded)
        radius = max(2, round(.55*padding))
        for x0, y0 in np.rint(observed[unavailable, :2]-origin).astype(int):
            draw.ellipse((x0-radius, y0-radius, x0+radius, y0+radius), fill=1)
        support &= ~np.asarray(excluded)
    alpha = mask[y, x].astype(np.float32)/255
    valid = support & (alpha >= CONFIG['source_alpha_min'])
    for x0, y0, x1, y1 in row.get('other_person_boxes_xyxy', ()):
        xx0, yy0 = np.maximum(np.floor([x0, y0]).astype(int)-origin, 0)
        xx1, yy1 = np.minimum(np.ceil([x1, y1]).astype(int)-origin, [shape[1], shape[0]])
        if xx0 < xx1 and yy0 < yy1:
            valid[yy0:yy1, xx0:xx1] = False
    luminance = image[y, x].astype(np.float32) @ np.asarray([.2126, .7152, .0722], np.float32)/255
    interior = binary_erosion(valid)[1:-1, 1:-1]
    dx = (luminance[1:-1, 2:]-luminance[1:-1, :-2])*.5
    dy = (luminance[2:, 1:-1]-luminance[:-2, 1:-1])*.5
    sharpness = float(np.sqrt(np.mean(dx[interior]**2+dy[interior]**2))) if interior.any() else 0.
    saturation = float(((luminance[valid] <= 2/255) | (luminance[valid] >= 253/255)).mean()) if valid.any() else 1.
    receipt = dict(box=list(box), valid_pixels=int(valid.sum()), region_pixels=int(support.sum()),
                   foreground_fraction=float(valid.sum()/max(1, support.sum())),
                   luma_gradient_rms=sharpness, saturated_fraction=saturation)
    return box, valid, alpha, receipt


def _candidate(row, fit, index, yaw, pitch, raw, flip):
    c = CONFIG
    if not c['min_side_yaw_deg'] <= abs(yaw) <= c['max_side_yaw_deg'] or abs(pitch) > c['max_pitch_deg']:
        return None, dict(status='rejected', reasons=['no_reliable_side_face_pose'])
    observed = fit['observed_native_keypoints'][index]
    projected = fit['joints_projected_native'][index]
    joints = fit['joints_camera'][index]
    confidence = (observed[:, 2] >= c['min_confidence']) & (raw >= c['min_raw_score'])
    inner = observed[67:, :2][confidence[67:]]
    span = float(np.ptp(inner, axis=0).max()) if len(inner) >= 8 else 0.
    if span < c['min_face_span_px']:
        return None, dict(status='rejected', reasons=['face_native_span_too_small'], face_span_px=span)

    image, mask = _read_pixels(row)
    usable = confidence & (flip <= min(c['max_flip_px'], c['max_flip_over_span']*span))
    usable &= _pixel_support(observed[:, :2], mask) & _pixel_support(projected, mask)
    for x0, y0, x1, y1 in row.get('other_person_boxes_xyxy', ()):
        xy = observed[:, :2]
        usable &= ~((xy >= [x0, y0]) & (xy < [x1, y1])).all(1)
    near = 'right' if joints[17, 2] < joints[18, 2] else 'left'
    depth_gap = float(abs(joints[17, 2]-joints[18, 2]))
    regions, accepted = {}, {}
    tested = np.unique(np.concatenate(list(REGIONS.values())))
    mesh_depth = _front_depth(projected[tested], fit['vertices_camera'][index], fit['faces'],
                              fit['K_integer_centers'][index])
    depth_delta = np.full(len(observed), -np.inf)
    depth_delta[tested] = joints[tested, 2]-mesh_depth
    mesh_supported = np.isfinite(depth_delta)
    mesh_occluded = depth_delta > c['max_mesh_occlusion_m']
    error = np.linalg.norm(projected-observed[:, :2], axis=1)
    for name, ids in REGIONS.items():
        available = usable[ids] & mesh_supported[ids] & ~mesh_occluded[ids]
        selected = ids[available]
        ear = name.endswith('_ear')
        count = 1 if ear else c['min_eye_points'] if name.endswith('_eye') else c['min_'+name+'_points']
        rms = float(np.sqrt(np.mean(error[selected]**2))) if len(selected) else None
        peak = float(error[selected].max()) if len(selected) else None
        if ear and (name != near+'_ear' or depth_gap < c['ear_visibility_min_depth_difference_m']):
            reason = 'fitted_side_occlusion_or_depth_ambiguity'
        elif int(usable[ids].sum()) < count:
            reason = 'insufficient_detector_and_mask_support'
        elif len(selected) < count:
            reason = 'occluded_or_unsupported_by_frozen_mesh'
        elif rms > min(c['max_region_rms_px'], c['max_region_rms_over_span']*span) or peak > min(
                c['max_region_peak_px'], c['max_region_peak_over_span']*span):
            reason = 'native_local_reprojection_exceeds_limit'
        else:
            reason = 'admitted'
            accepted[name] = selected
        regions[name] = dict(status=reason, supported_points=selected.tolist(), rms_px=rms, peak_px=peak,
                             mesh_occluded_points=ids[mesh_occluded[ids]].tolist(),
                             mesh_unresolved_points=ids[~mesh_supported[ids]].tolist(),
                             mesh_depth_delta_mm=[float(depth_delta[i]*1000) if mesh_supported[i] else None for i in ids])
    required = {near+'_eye', 'nose', 'mouth'}
    receipt = dict(face_span_px=span, near_side_from_fit=near, ear_depth_gap_m=depth_gap, regions=regions)
    if not required <= accepted.keys():
        return None, dict(receipt, status='rejected', reasons=['required_face_regions_unavailable'])
    supported = np.concatenate(list(accepted.values()))
    unavailable = np.setdiff1d(tested[confidence[tested]], supported)
    box, valid, alpha, pixels = _source_region_masks(observed, accepted, unavailable, span, image, mask, row)
    reasons = []
    if pixels['foreground_fraction'] < c['min_roi_foreground_fraction']:
        reasons.append('face_region_mask_disagreement')
    if pixels['luma_gradient_rms'] < c['min_luma_gradient']:
        reasons.append('insufficient_native_face_detail')
    if pixels['saturated_fraction'] > c['max_saturated_fraction']:
        reasons.append('excess_saturated_native_face')
    receipt.update(status='rejected' if reasons else 'face_region_admitted', reasons=reasons, native_roi=pixels)
    if reasons:
        return None, receipt
    pixels['valid_sha256'] = hashlib.sha256(valid.tobytes()).hexdigest()
    pixels['alpha_sha256'] = hashlib.sha256(alpha.tobytes()).hexdigest()
    return (box, valid, alpha), receipt


def _select(records, base):
    c = CONFIG
    chosen = []
    current = list(base)
    eligible = [record for record in records if record['status'] == 'face_region_admitted']
    while eligible and len(chosen) < c['max_views']:
        sectors = {item['head_yaw_sector_deg'] for item in current}
        bins = {item['source_bin'] for item in current}
        roots = {item['root_yaw_sector_deg'] for item in current}
        choices = []
        for row in eligible:
            time = row['time_seconds']
            yaw = row['fitted_head_yaw_proxy_deg']
            if any(abs(time-item['time_seconds']) < c['min_supplement_spacing_s'] for item in chosen):
                continue
            yaw_gap = min(abs(yaw-item['fitted_head_yaw_proxy_deg']) for item in current)
            time_gap = min(abs(time-item['time_seconds']) for item in current)
            new_yaw = row['head_yaw_sector_deg'] not in sectors
            new_source = row['source_bin'] not in bins
            new_root = row['root_yaw_sector_deg'] not in roots
            if not (new_yaw or new_source or new_root or yaw_gap >= c['min_novel_yaw_deg']):
                continue
            rms = max(value['rms_px'] for name, value in row['regions'].items()
                      if value['status'] == 'admitted' and not name.endswith('_ear'))
            priority = (not new_yaw, not new_root, not new_source, -yaw_gap, -time_gap, rms, row['source_index'])
            proof = dict(new_head_yaw_sector=new_yaw, new_source_bin=new_source, new_root_yaw_sector=new_root,
                         nearest_previous_head_yaw_deg=yaw_gap, nearest_previous_time_s=time_gap)
            choices.append((priority, row, proof))
        if not choices:
            break
        _, best, evidence = min(choices, key=lambda value: value[0])
        best.update(status='selected_face_supplement', selection=evidence)
        chosen.append(best)
        current.append(best)
        eligible.remove(best)
    return chosen


def prepare_face_supplements(root, rows):
    """Bind auxiliary RGB/alpha to a native facial ROI and one frozen SMPL-X pose."""
    root = Path(root)
    receipt = root/'fit/official-fit-result.json'
    official = rt.read(receipt) if receipt.is_file() else {}
    if 'joint_body_refinement' not in official:
        return [], dict(schema='shuolee-face-supplement/1', state='skipped_missing_joint_candidate_contract',
                        chosen_frame_ids=[], full_body_frame_ids=[row['frame'] for row in rows],
                        retained_holdouts=[], heldout_pixels_read=False)
    roster, candidates, selection, fit, units, raw, flip, digests = _load_inputs(root, rows)
    names = fit['frame_ids'].astype(str).tolist()
    selected, held = set(selection['selected_frames']), set(selection['retained_holdouts'])
    yaw, pitch, roots = _directions(fit, names.index(selection['reference_frame']))
    protected_digests = {roster[name]['source_sha256'] for name in selected | held}
    records, base, targets = [], [], {}
    candidate_slots = {name: i for i, name in enumerate(names)}
    for i, row in enumerate(candidates):
        name = row['frame']
        original = selection['reasons'][name]
        record = dict(frame=name, source_index=row['source_frame_index'], time_seconds=row['time_seconds'],
                      source_bin=original['source_bin'], full_body_status=original['status'],
                      fitted_head_yaw_proxy_deg=float(yaw[i]), fitted_head_pitch_proxy_deg=float(pitch[i]),
                      fitted_root_relative_yaw_proxy_deg=float(roots[i]), head_yaw_sector_deg=_sector(yaw[i]),
                      root_yaw_sector_deg=_sector(roots[i]), image_sha256=row['source_sha256'],
                      mask_sha256=row['mask_sha256'])
        if name in selected:
            base.append(dict(record, status='existing_full_body'))
            continue
        if row['source_sha256'] in protected_digests:
            record.update(status='rejected', reasons=['duplicate_protected_source_bytes'])
        else:
            target, evidence = _candidate(row, fit, i, yaw[i], pitch[i], raw[i], flip[i])
            record.update(evidence)
            if target is not None:
                targets[name] = target
        records.append(record)
    comparison = None
    if 'native_head_alignment' in official:
        # Isolate fitting on the same four identities. A candidate must still
        # pass the per-frame native visibility and face region checks.
        reference = rt.read(root/official['native_head_alignment']['file'])
        frozen_auxiliary = reference['frozen_auxiliary_frames']
        by_frame = {record['frame']: record for record in records}
        choices = [by_frame[frame] for frame in frozen_auxiliary if frame in targets]
        comparison = dict(original_face_only_frames=frozen_auxiliary,
            readmitted_frame_ids=[record['frame'] for record in choices],
            not_readmitted=[dict(frame=frame, status=by_frame[frame]['status'],
                reasons=by_frame[frame].get('reasons', []), regions=by_frame[frame].get('regions', {}))
                for frame in frozen_auxiliary if frame not in targets])
        for choice in choices:
            choice.update(status='selected_face_supplement',
                          selection=dict(frozen_comparison_identity=True, per_frame_admission_recomputed=True))
    else:
        choices = _select(records, base)
    supplements = []
    for choice in choices:
        name = choice['frame']
        index = candidate_slots[name]
        source = roster[name]
        E = units.w2c_meters[index]
        vertex = (fit['vertices_camera'][index].astype(np.float64)-E[:3, 3]) @ E[:3, :3]
        row = dict(source, w2c=E.tolist(), source_w2c=units.source_w2c[index].tolist())
        pose = dict(frame_ids=np.asarray(name), vertices_world=vertex, w2c=E,
                    **{key: fit[key][index] for key in REPLAY})
        box, valid, alpha = targets[name]
        choice['native_camera'] = dict(K=row['K'], K_integer_centers=row['K_integer_centers'], w2c=row['w2c'],
                                       source_w2c=row['source_w2c'], wh=row['wh'])
        supplements.append(FaceSupplement(row, box, valid, alpha, pose))

    def coverage(items):
        return dict(source_bins=sorted({row['source_bin'] for row in items}),
                    head_yaw_sectors_deg=sorted({row['head_yaw_sector_deg'] for row in items}),
                    root_relative_yaw_sectors_deg=sorted({row['root_yaw_sector_deg'] for row in items}))

    audit = dict(schema='shuolee-face-supplement/1', state='admission_complete', config=CONFIG.copy(), input_sha256=digests,
                 full_body_frame_ids=[row['frame'] for row in rows], retained_holdouts=selection['retained_holdouts'],
                 fit_candidate_frame_ids=names, chosen_frame_ids=[row['frame'] for row in choices],
                 candidates=records, full_body_coverage=coverage(base), supplemented_coverage=coverage(base+choices),
                 direction_basis='Fitted camera-relative SMPL-X head direction and root yaw proxies',
                 physical_angle_calibration=False,
                 supervision='Independent native mask within observed facial ROI; all other pixels unknown',
                 visibility_basis='Detector agreement and independent source foreground, constrained by the nearest frozen '
                                  'mesh triangle depth. Fitted visibility is not independent physical ground truth.',
                 heldout_pixels_read=False, frozen_shared_shape_and_poses=True)
    if comparison is not None:
        audit['fixed_identity_comparison'] = comparison
    return supplements, audit
