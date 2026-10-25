"""Region-resolved diagnostics of the frozen pre-SfM observations.

ROI labels are not anatomical/material truth. This stage never changes camera
parameters, measured pixels, track membership or triangulation-view selection.
"""
from collections import Counter
import copy
import html
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from . import runtime as rt
from .fixed_tracks import evaluate_fixed_tracks, cameras_from_records, _summary
from .scene_support import analyze_scene_support, specification as scene_specification
from .material_review import write_material_review, specification as material_review_specification

REGIONS = ('face', 'body', 'static_scene_candidate', 'unknown')
REGION_LABELS = dict(face='面部ROI', body='非脸部前景（含衣物与手部）',
                     static_scene_candidate='背景候选（未证实静止）', unknown='未知／边界／区域不一致')
DEFAULTS = dict(boundary_margin_px=4., min_parallax_degrees=1., review_frames=5,
                displayed_tracks_per_region=12)


def validate_settings(settings=None):
    s = dict(DEFAULTS, **(settings or {}))
    if set(s) != set(DEFAULTS):
        raise ValueError('Unknown alignment-diagnostic setting.')
    for key in ('boundary_margin_px', 'min_parallax_degrees'):
        if not isinstance(s[key], (int, float)) or not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError('Finite positive diagnostic margins are required.')
    if s['min_parallax_degrees'] >= 90:
        raise ValueError('Triangulation-angle limit must be below 90 degrees.')
    for key, maximum in [('review_frames', 5), ('displayed_tracks_per_region', 24)]:
        if type(s[key]) is not int or not 1 <= s[key] <= maximum:
            raise ValueError('Review size exceeds the declared bounded budget.')
    return s


def _box(box, wh):
    if box is None:
        return None
    a = np.asarray(box, float)
    if (a.shape != (4,) or not np.isfinite(a).all() or (a[:2] < 0).any() or
            (a[2:] <= a[:2]).any() or (a[2:] > np.asarray(wh)).any()):
        raise ValueError('Invalid pixel-domain ROI box.')
    return a


def _inside(xy, box, margin=0.):
    return box is not None and np.all(xy >= box[:2] - margin) and np.all(xy <= box[2:] + margin)


def verify_camera_identity(frozen, observed):
    """Mask inference may append fields; the full-image camera must not change."""
    cameras_from_records(frozen);cameras_from_records(observed)
    a, b = ({r['frame']: r for r in rows} for rows in (frozen, observed))
    if list(a) != list(b):
        raise ValueError('Observed-frame order differs from the frozen SfM camera set.')
    for frame, old in a.items():
        new = b[frame]
        for key in ('name', 'source_frame_index', 'time_seconds', 'wh', 'source_sha256'):
            if old[key] != new[key]:
                raise ValueError(f'Camera/source identity changed during person observation: {frame} {key}')
        if not np.array_equal(old['K'], new['K']) or not np.array_equal(old['w2c'], new['w2c']):
            raise ValueError(f'Full-image camera changed during person observation: {frame}')


def annotate_tracks(rows, bank, settings=None):
    """Add labels only; every measurement, including uncertain ones, persists."""
    import cv2
    s = validate_settings(settings);by_frame = {r['frame']: r for r in rows}
    def mask_data(frame):
        row = by_frame[frame]
        if not row.get('mask') or not row.get('mask_sha256') or rt.sha(row['mask']) != row['mask_sha256']:
            raise ValueError(f'Missing or changed mask bytes: {frame}')
        mask = np.asarray(Image.open(row['mask']).convert('L'))
        if list(mask.shape[::-1]) != row['wh'] or not np.isin(mask, [0, 255]).all():
            raise ValueError('Expected the exact full-image binary SAM mask, not resized/soft data.')
        if not (mask == 0).any() or not (mask == 255).any():
            raise ValueError('Mask must contain both foreground and background.')
        foreground = mask > 127
        inside = cv2.distanceTransform(foreground.astype('uint8'), cv2.DIST_L2, 5)
        outside = cv2.distanceTransform((~foreground).astype('uint8'), cv2.DIST_L2, 5)
        return foreground, inside + outside
    annotated = copy.deepcopy(bank)
    # Process every image once. Track-major iteration would repeatedly decode
    # and distance-transform a long video's masks when the tracks span frames.
    observations_by_frame = {r['frame']: [] for r in rows}
    for track in annotated['tracks']:
        for obs in track['observations']:
            observations_by_frame[obs['frame']].append(obs)
    for frame, observations in observations_by_frame.items():
        row = by_frame[frame];mask, distance = mask_data(frame)
        face = _box(row.get('face_box_xyxy'), row['wh'])
        others = [_box(b, row['wh']) for b in row.get('other_person_boxes_xyxy', [])]
        for obs in observations:
            xy = np.asarray(obs['uv'], float)
            if xy.shape != (2,) or not np.isfinite(xy).all() or (xy < 0).any() or (xy > np.asarray(row['wh']) - 1).any():
                raise ValueError('Observation outside the unchanged full-image integer-centre domain.')
            x, y = np.floor(xy + .5).astype(int)
            margin = s['boundary_margin_px'];label = 'unknown'
            if any(_inside(xy, b, margin) for b in others):
                reason = 'known_non_target_person_box'
            elif distance[y, x] < margin:
                reason = 'uncertain_mask_boundary'
            elif not mask[y, x]:
                label = 'static_scene_candidate';reason = 'outside_target_mask_staticity_unverified'
            elif face is None:
                reason = 'foreground_without_face_roi'
            elif _inside(xy, face, margin):
                if _inside(xy, face, -margin):
                    label = 'face';reason = 'inside_estimated_face_roi'
                else:
                    reason = 'uncertain_face_roi_boundary'
            else:
                label = 'body';reason = 'foreground_outside_face_not_clothing_segmentation'
            obs['region_annotation'] = dict(region=label, reason=reason, foreground=bool(mask[y, x]),
                                           distance_to_mask_boundary_px=float(distance[y, x]))
    for track in annotated['tracks']:
        labels=[o['region_annotation']['region'] for o in track['observations']]
        reasons=[o['region_annotation']['reason'] for o in track['observations']]
        track['region'] = labels[0] if len(set(labels)) == 1 else 'unknown'
        track['region_annotation'] = dict(consistent=len(set(labels)) == 1,
            reason='same_estimated_region' if len(set(labels)) == 1 else 'cross_view_region_disagreement',
            observed_regions=labels, observed_reasons=reasons, physical_identity_verified=False)
    return annotated


def measure_regions(rows, annotated, settings=None):
    s = validate_settings(settings)
    if annotated['tracks']:
        measurement = evaluate_fixed_tracks(rows, annotated, s['min_parallax_degrees'])
    else:
        measurement = dict(tracks=[], measured_tracks=0, not_observable_tracks=0,
                           quality_accepted=False, training_admitted=False, root_cause_resolved=False)
    measured = {t['track_id']: t for t in measurement['tracks']}
    frame_ids = [r['frame'] for r in rows];row_map = {r['frame']: r for r in rows}
    regions = {}
    for region in REGIONS:
        selected = [t for t in annotated['tracks'] if t['region'] == region]
        results = [measured[t['track_id']] for t in selected]
        errors = [e for r in results if r['status'] == 'measured' for e in r['held_error_px']]
        per_frame = []
        for frame in frame_ids:
            observations = [(t, o) for t in selected for o in t['observations'] if o['frame'] == frame]
            row = row_map[frame];box = row.get('face_box_xyxy') if region == 'face' else None
            box = _box(box, row['wh'])
            if box is None:box = np.r_[0., 0., np.asarray(row['wh']) - 1.]
            cells = {tuple(np.clip(np.floor((np.asarray(o['uv']) - box[:2]) / (box[2:] - box[:2]) * 4), 0, 3).astype(int)) for _, o in observations}
            held = [p['residual_px'] for t, _ in observations for p in measured[t['track_id']].get('projections', [])
                    if p['frame'] == frame and frame not in t['triangulation_frames'] and
                    measured[t['track_id']]['status'] == 'measured']
            per_frame.append(dict(frame=frame, observations=len(observations),
                held_observations=sum(frame not in t['triangulation_frames'] for t, _ in observations),
                occupied_cells_4x4=len(cells), cells=[list(map(int,c)) for c in sorted(cells)],
                grid_domain='face_roi' if region == 'face' else 'full_image',
                held_error_px=_summary(held)))
        regions[region] = dict(label=REGION_LABELS[region], tracks=len(selected),
            observable_tracks=sum(r['status'] == 'measured' for r in results),
            unobservable_reasons=dict(Counter(r['reason'] for r in results if r['status'] != 'measured')),
            held_error_px=_summary(errors), frames_with_observations=sum(bool(r['observations']) for r in per_frame),
            frames_with_held_predictions=sum(bool(r['held_error_px']['count']) for r in per_frame),
            frames_without_observations=[r['frame'] for r in per_frame if not r['observations']], frames=per_frame,
            triangulation_angle_degrees=_summary([r['max_triangulation_angle_degrees'] for r in results]),
            scene_staticity_verified=False)
    summary = dict(state='diagnostic_computed' if measurement['measured_tracks'] else 'no_observable_reserved_tracks',
        total_tracks=len(annotated['tracks']), measured_tracks=measurement['measured_tracks'], regions=regions,
        region_disagreement_tracks=sum(not t['region_annotation']['consistent'] for t in annotated['tracks']),
        membership_or_pixel_changes=False, residual_based_filtering=False,
        root_cause_resolved=False, physical_track_independence_established=False,
        camera_calibration_verified=False, subject_staticity_verified=False, quality_accepted=False, training_admitted=False,
        missing=['Physical material identity and unknown aliases.',
                 'Known-static scene support at the subject depth and camera uncertainty/observability.',
                 'A trustworthy motion negative control before attributing subject residuals.'],
        interpretation='Region residuals describe conditional errors; background, face and body have different depth/visibility, so numerical differences are not a causal comparison.')
    return summary, measurement


def _source_evidence(root):
    paths = ['data/sfm-inputs.json', 'data/training-inputs.json', 'data/sfm-holdout-tracks.json',
             'sfm/holdout/usage-after-mapping.json', 'sfm/holdout/usage-after-mapping.npz',
             'sfm/holdout/split.json', 'sfm/holdout/plan.json', 'sfm/holdout/coordinate-conversion.json',
             'sfm/raw-features.db', 'sfm/database.db']
    hashes = {p: rt.sha(root / p) for p in paths}
    bank = rt.read(root / paths[2]);audit = rt.read(root / paths[3]);split = rt.read(root / paths[5])
    conversion = rt.read(root / paths[7]);provenance = bank['camera_solver_provenance']
    if (bank.get('camera_manifest_relative_path') != paths[0] or bank['camera_manifest_sha256'] != hashes[paths[0]] or
            bank.get('pixel_domain') != 'undistorted_integer_centers' or
            provenance['audit_sha256'] != hashes[paths[3]] or provenance.get('conversion_sha256') != hashes[paths[7]] or
            audit['split_sha256'] != hashes[paths[5]] or audit['source_database_sha256'] != hashes[paths[8]] or
            audit['training_database_sha256'] != hashes[paths[9]] or audit['usage_arrays_sha256'] != hashes[paths[4]] or
            split['plan_sha256'] != hashes[paths[6]] or not audit['reconstruction_checked'] or audit['reserved_used'] != 0):
        raise ValueError('Frozen camera/split/conversion/native usage evidence is missing, stale or inconsistent.')
    original = {t['track_id']: t for t in split['tracks']}
    conversions = {o['observation_id']: o for o in conversion['observations']}
    rejected = {t['track_id'] for t in conversion['rejected_tracks']}
    bank_ids=[t['track_id'] for t in bank['tracks']]
    if len(set(bank_ids))!=len(bank_ids) or set(bank_ids)&rejected or set(original) != set(bank_ids) | rejected:
        raise ValueError('Tracks were dropped after the pregeometry split.')
    for track in bank['tracks']:
        prior = original[track['track_id']]
        if track['triangulation_frames'] != prior['triangulation_frames'] or [o['observation_id'] for o in track['observations']] != [o['observation_id'] for o in prior['observations']]:
            raise ValueError('Reserved track membership or triangulation selection changed.')
        for o, raw in zip(track['observations'], prior['observations']):
            mapped = conversions[o['observation_id']]
            if (o['frame'] != raw['frame'] or not mapped['supported'] or
                    not np.array_equal(o['uv'], mapped['rectified_uv']) or
                    not np.array_equal(raw['raw_colmap_xy'], mapped['raw_colmap_xy'])):
                raise ValueError('Reserved source or converted pixel changed.')
    return hashes, bank, len(rejected)


def write_reviews(output, rows, annotated, measurement, settings):
    """Unwarped originals and separate marker overlays; no text in PNG pixels."""
    output.mkdir();measurements = {t['track_id']: t for t in measurement['tracks']}
    chosen_frames = sorted(set(np.rint(np.linspace(0, len(rows) - 1, min(settings['review_frames'], len(rows)))).astype(int)))
    reviews = [];cards = []
    for index in chosen_frames:
        row = rows[index];frame = row['frame']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',frame):
            raise ValueError('Frame identity is not a safe filename.')
        image = Image.open(row['image']).convert('RGB');overlay = image.copy();draw = ImageDraw.Draw(overlay)
        selected = []
        for region in REGIONS:
            candidates = sorted([t for t in annotated['tracks'] if t['region'] == region and any(o['frame'] == frame for o in t['observations'])], key=lambda t: t['track_id'])
            selected.extend(candidates[:settings['displayed_tracks_per_region']])
        ids = []
        for track in selected:
            result = measurements[track['track_id']]
            o = next(o for o in track['observations'] if o['frame'] == frame);x, y = o['uv']
            draw.ellipse((x-2, y-2, x+2, y+2), outline=(40, 240, 100), width=2)
            projection = next((p for p in result.get('projections', []) if p['frame'] == frame), None)
            held = frame not in track['triangulation_frames']
            visible=False
            if result['status']=='measured' and projection and projection['predicted_uv'] is not None and held:
                a, b = projection['predicted_uv']
                visible=bool(0<=a<image.width and 0<=b<image.height)
                if visible:
                    draw.line((x,y,a,b),fill=(240,100,80),width=1)
                    draw.line((a-3,b,a+3,b),fill=(255,70,60),width=2)
                    draw.line((a,b-3,a,b+3),fill=(255,70,60),width=2)
            ids.append(dict(track_id=track['track_id'], region=track['region'], held=held,
                            held_prediction_visible=visible))
        crops = [('full', (0, 0, image.width, image.height))]
        box = _box(row.get('face_box_xyxy'), row['wh'])
        if box is not None:
            margin = .25 * max(box[2:] - box[:2])
            lo = np.maximum(np.floor(box[:2] - margin), 0).astype(int)
            hi = np.minimum(np.ceil(box[2:] + margin + 1), row['wh']).astype(int)
            crops.append(('face', tuple(np.r_[lo, hi])))
        for label, crop in crops:
            first=f'{frame}_{label}_original.png';second=f'{frame}_{label}_held_projection.png'
            image.crop(crop).save(output/first);overlay.crop(crop).save(output/second)
            reviews.append(dict(frame=frame, region=label, original=first, overlay=second, crop_xyxy=list(map(int,crop)),
                                original_image_sha256=row['source_sha256'], displayed_tracks=ids,
                                selection='fixed temporal positions; ascending track IDs per region, never residual rank',
                                text_burned_into_images=False, image_warped=False))
            cards.append(f'<article><h2>{html.escape(frame)} · {label}</h2><div class="pair"><img src="{first}"><img src="{second}"></div></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>冻结观测诊断</title>'
        '<style>body{background:#101615;color:#eee;font:16px sans-serif;margin:24px} .pair{display:flex;gap:12px}.pair img{width:calc(50% - 6px);height:auto;object-fit:contain;align-self:start} article{margin:30px 0}</style>'
        '<h1>原图与求解前留出观测</h1><p>左：未变形原图。右：绿圆为冻结观测，红叉为未参与三角化的视角预测。'
        '三角化使用的视角不画红叉，越界与不可测投影保留在JSON中。标记不是物理真值。相同区域与裁剪比例；图片像素不含文字。</p>' + ''.join(cards),encoding='utf-8')
    rt.save(output/'index.json', dict(reviews=reviews))
    return len(reviews)


def diagnose(root, output, settings=None):
    root, output = Path(root).resolve(), Path(output).resolve();settings=validate_settings(settings)
    if output.exists():raise FileExistsError('Diagnostic output must be fresh.')
    hashes, bank, coordinate_rejections = _source_evidence(root)
    frozen = rt.read(root/'data/sfm-inputs.json')['records'];rows = rt.read(root/'data/training-inputs.json')['records']
    verify_camera_identity(frozen, rows)
    for row in rows:
        if rt.sha(row['image']) != row['source_sha256'] or rt.sha(row['mask']) != row['mask_sha256']:
            raise ValueError('Source image or mask hash changed.')
    output.mkdir(parents=True)
    rt.save(output/'plan.json',dict(settings=settings,source_hashes=hashes,
        implementation_sha256=rt.sha(Path(__file__)),regions=REGION_LABELS,
        scene_support_implementation_sha256=rt.sha(Path(__file__).with_name('scene_support.py')),
        scene_support_specification=scene_specification(),
        material_review_implementation_sha256=rt.sha(Path(__file__).with_name('material_review.py')),
        material_review_specification=material_review_specification(),
        masks=[dict(frame=r['frame'],mask_sha256=r['mask_sha256']) for r in rows],
        region_labels_do_not_change_membership=True,thresholds_are_not_quality_acceptance=True))
    annotated=annotate_tracks(rows,bank,settings);summary,measurement=measure_regions(rows,annotated,settings)
    support=analyze_scene_support(rows,annotated,measurement)
    rt.save(output/'scene-support.json',support)
    summary['scene_support']=dict(state=support['state'], report='scene-support.json',
        scene_candidate_tracks=support['scene_candidate_tracks'],
        frames_without_scene_observations=support['frames_without_scene_observations'],
        physical_calibration_verified=False, no_camera_optimization=True)
    summary.update(coordinate_rejected_tracks=coordinate_rejections,
                   source_split_tracks=summary['total_tracks']+coordinate_rejections,
                   source_hashes=hashes,source_inputs_modified=False)
    count=write_reviews(output/'reviews',rows,annotated,measurement,settings)
    summary['material_source_review']=write_material_review(output/'material-review',rows,annotated,hashes)
    page=output/'reviews/index.html'
    page.write_text(page.read_text(encoding='utf-8')+'<p><a href="../scene-support.json">场景覆盖与局部相机可辨识性</a>：'
        '固定外参的内参敏感度，与允许外参和三维点同时变化的敏感度分别记录。背景候选没有被确认为静止，数值满秩不代表物理标定通过。</p>'
        '<p><a href="../material-review/index.html">逐条轨迹审阅全部原图观测</a>：局部窗口只便于检查材料，不能把窗口居中看成几何对齐。</p>',encoding='utf-8')
    summary['review_pairs']=count
    for relative, expected in hashes.items():
        if rt.sha(root/relative)!=expected:raise ValueError('Frozen diagnostic source changed during evaluation.')
    for row in rows:
        if rt.sha(row['image'])!=row['source_sha256'] or rt.sha(row['mask'])!=row['mask_sha256']:
            raise ValueError('Image or mask changed during evaluation.')
    rt.save(output/'annotated-tracks.json',annotated)
    rt.save(output/'measurements.json',measurement)
    rt.save(output/'summary.json',summary)
    return summary
