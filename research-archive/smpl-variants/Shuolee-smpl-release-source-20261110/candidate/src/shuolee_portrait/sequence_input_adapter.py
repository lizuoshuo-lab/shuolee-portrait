"""CPU-only, frame-identified inputs for one shared-identity sequence fit.

This adapter does not run a detector, VPoser, SMPL-X, a solver or a renderer.
It preserves source pixels and conditional cameras, joins original incam priors
through their recorded crop slots, and leaves unreviewed face points inactive.
Fresh inference additionally requires a hash-bound producer receipt. Cached
inference is never described as fresh inference or an optimized fit.
"""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .alignment_inputs import COCO133_NAMES, need, numeric, read, sha, validate_camera
from .camera_world_units import validate_camera_length_contract


SCHEMA = 'shuolee-sequence-inputs/1'
COCO25_BODY = np.array([0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11])
SOURCE_CHANNELS = np.r_[np.arange(23), np.arange(91, 133), np.arange(40, 91)]
TARGET_CHANNELS = np.r_[COCO25_BODY, np.arange(19, 25), np.arange(25, 67), np.arange(67, 118)]


def _frozen(value):
    a = np.ascontiguousarray(value)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def relative_detector_confidence(scores, boxes, flip):
    """Pinned existing score/flip heuristic, not visibility or probability."""
    scores, boxes, flip = (np.asarray(a, dtype=np.float64) for a in (scores, boxes, flip))
    need(scores.ndim == 2 and boxes.shape == (len(scores), 4) and flip.shape == scores.shape,
         'RTMW score/box/flip shape mismatch.')
    need(all(np.isfinite(a).all() for a in (scores, boxes, flip)), 'Nonfinite detector reliability input.')
    heights = boxes[:, 3] - boxes[:, 1]
    need(np.all(heights > 0) and np.all(boxes[:, 2] > boxes[:, 0]) and
         np.all(scores >= 0) and np.all(flip >= 0), 'Invalid detector reliability input.')
    q90 = np.maximum(np.quantile(scores, .9, axis=0), 1e-6)
    # Keep the prior adapter's float32 rounding before the FP64 objective.
    confidence = (np.clip(scores / q90, 0, 1) *
                  np.exp(-.25 * (flip / (.012 * heights[:, None])) ** 2)).astype(np.float32)
    return confidence, q90


@dataclass(frozen=True, slots=True, eq=False)
class SequenceInputBundle:
    frame_ids: tuple
    source_frames: np.ndarray
    wh: np.ndarray
    K_integer_centers: np.ndarray
    source_w2c: np.ndarray
    prior_slots: np.ndarray
    body_pose: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    targets: np.ndarray
    confidence: np.ndarray
    face_visibility: np.ndarray
    face_split: np.ndarray
    face_semantic_verified: np.ndarray
    physical_face_verified: np.ndarray
    face_family_ids: np.ndarray
    initial_shared_betas: np.ndarray
    reference_frame_id: str
    camera_length_contract: object
    bound_files: tuple
    _report_json: str

    def assert_unchanged(self):
        for path, expected in self.bound_files:
            need(sha(path) == expected, 'Sequence source changed: ' + str(path))

    def report(self):
        return json.loads(self._report_json)

    def make_observations(self, *, device='cpu', dtype=None):
        """Create the existing observation type without any model forward."""
        import torch
        from .sequence_fitting import SequenceObservations
        dtype = torch.float64 if dtype is None else dtype
        self.assert_unchanged()
        return SequenceObservations(
            self.frame_ids, torch.tensor(self.targets.copy(), dtype=dtype, device=device),
            torch.tensor(self.confidence.copy(), dtype=dtype, device=device),
            self.face_visibility.copy(), self.face_split.copy(), self.face_semantic_verified.copy(),
            self.physical_face_verified.copy(), self.face_family_ids.copy())


def load_sequence_inputs(manifest_path, *, source_path_resolver=None, decode_images=False):
    """Bind full roster inputs; no old fitted state or latent can enter.

    ``source_path_resolver`` may relocate a recorded absolute path to the same
    file on another host. SHA256 and native dimensions still must match. It is
    not a pixel transform. Missing review records become explicit inactive
    unknown states, never confidence-derived visibility or fitting permission.
    """
    manifest_path = Path(manifest_path).resolve()
    m = read(manifest_path)
    need(m.get('schema') == SCHEMA, 'Unsupported sequence-input schema.')
    need(m.get('purpose') in ('real_research_adaptation', 'synthetic_contract_test'), 'Explicit purpose required.')
    need(m.get('pixel_domain') == 'rectified_pinhole' and m.get('pixel_convention') == 'integer_centers',
         'This sequence adapter requires native rectified integer-centre pixels.')
    need(m.get('root_policy') == 'provided_initialization', 'Explicit provided_initialization policy required.')
    kind = m.get('initialization_kind')
    need(kind in ('cached_same_video_gvhmr_incam', 'fresh_same_video_gvhmr_incam'),
         'Explicitly identified same-video original incam priors required, not fitted checkpoints.')
    fresh = kind == 'fresh_same_video_gvhmr_incam'
    need(isinstance(m.get('initialization_evidence'), str) and bool(m['initialization_evidence'].strip()),
         'Original prior provenance is required.')
    length = validate_camera_length_contract(m.get('camera_length_contract'))
    if m['purpose'] == 'real_research_adaptation':
        need(length['basis'] != 'synthetic_known', 'Synthetic scale cannot be assigned to real data.')
    bound = [(manifest_path, sha(manifest_path))]

    def resolve(path):
        value = Path(path)
        if source_path_resolver is not None:
            value = Path(source_path_resolver(str(value)))
        return value if value.is_absolute() else (manifest_path.parent / value).resolve()

    def file(desc, label):
        need(isinstance(desc, dict) and set(desc) == {'path', 'sha256'}, 'Explicit hash-bound ' + label + ' required.')
        path = resolve(desc['path'])
        need(path.is_file() and sha(path) == desc['sha256'], label + ' source/hash mismatch.')
        bound.append((path, desc['sha256']))
        if path.suffix.lower() == '.npz':
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}
        return read(path)

    rows = file(m.get('frames'), 'frames')['records']
    need(isinstance(rows, list) and bool(rows), 'Full frame roster required.')
    ids = tuple(str(r['frame']) for r in rows)
    n = len(ids)
    need(len(set(ids)) == n, 'Duplicate frame identity.')
    need(m.get('reference_frame_id') in ids, 'Reference frame must be an actual roster ID.')
    source_frames = []
    for r in rows:
        need(type(r.get('source_frame_index')) is int and r['source_frame_index'] >= 0,
             'Explicit source-video frame index required.')
        source_frames.append(r['source_frame_index'])
        need(len(r['wh']) == 2 and all(type(v) is int and v > 0 for v in r['wh']), 'Invalid native dimensions.')
        need(r.get('camera_registration', {}).get('interpolation_used') is False,
             'Every camera must be registered; no interpolated or unknown camera status.')
    need(len(set(source_frames)) == n, 'Duplicate source-video frame identity.')
    cameras = [validate_camera(r) for r in rows]
    need(all(c is not None for c in cameras), 'Complete K/E required for every frame.')
    obs = file(m.get('observations'), 'observations')
    need(obs['frame_ids'].shape == (n,) and tuple(obs['frame_ids'].astype(str)) == ids,
         'Observation frame identities/order differ; positional joining forbidden.')
    need(obs['joint_names'].astype(str).tolist() == COCO133_NAMES, 'Exact COCO133 semantic labels required.')
    xy = numeric(obs['keypoints'], (n, 133, 2), 'Invalid native keypoints.')
    scores = numeric(obs['scores'], (n, 133), 'Invalid scores.')
    boxes = numeric(obs['boxes'], (n, 4), 'Invalid boxes.')
    flip = numeric(obs['flip_disagreement'], (n, 133), 'Invalid flip errors.')
    relative, q90 = relative_detector_confidence(scores, boxes, flip)

    pred = file(m.get('prior'), 'prior')
    allowed = {'incam_body_pose', 'incam_betas', 'incam_global_orient', 'incam_transl', 'incam_joints'}
    need(set(pred) <= allowed and {'incam_body_pose', 'incam_global_orient', 'incam_transl'} <= set(pred),
         'Prior must contain original incam prediction arrays, never optimized pose/latent/geometry.')
    prior = {key: numeric(pred['incam_' + key], (n, width), 'Invalid incam ' + key)
             for key, width in [('body_pose', 63), ('global_orient', 3), ('transl', 3)]}
    crop = file(m.get('crop_mapping'), 'crop mapping')['records']
    need(len(crop) == n, 'Crop roster and prediction count differ.')
    slots = [r.get('official_prediction_slot') for r in rows]
    need(all(type(s) is int for s in slots) and sorted(slots) == list(range(n)),
         'Prior slots must explicitly cover the entire prediction roster once.')
    for row, slot in zip(rows, slots):
        c = crop[slot]
        need(str(c['frame']) == str(row['frame']) and c['source_sha256'] == row['source_sha256']
             and c['source_wh'] == row['wh'], 'Cached prior slot/crop/image identity mismatch.')
        a = numeric(c['source_to_crop'], (3, 3), 'Invalid source-to-crop transform.')
        b = numeric(c['crop_to_source'], (3, 3), 'Invalid crop-to-source transform.')
        need(np.allclose(a @ b, np.eye(3), atol=1e-7, rtol=0), 'Recorded crop transforms are not inverse.')
        need(c.get('pixel_convention') == 'Integer pixel centres in both source image and crop',
             'Explicit prior crop pixel convention required.')
    if fresh:
        receipt = file(m.get('prior_provenance'), 'fresh prior provenance')
        need(isinstance(receipt, dict), 'Fresh prior receipt must be an object.')
        need(receipt.get('prediction_variant') == 'original_incam' and
             receipt.get('postprocessing_applied') is False and
             receipt.get('fresh_inference') is True and receipt.get('from_this_video') is True and
             receipt.get('previous_person_parameters_loaded') is False,
             'Fresh original incam prediction required; postprocessed or old-person state is not interchangeable.')
        video_hash = m.get('video_sha256')
        need(isinstance(video_hash, str) and len(video_hash) == 64 and
             all(c in '0123456789abcdef' for c in video_hash) and receipt.get('video_sha256') == video_hash,
             'Fresh prior receipt must match this video SHA256.')
        need(receipt.get('prediction_sha256') == m['prior']['sha256'],
             'Fresh prior receipt prediction SHA256 differs from its actual arrays.')
        need(receipt.get('inputs_sha256') == m['frames']['sha256'],
             'Fresh prior receipt input roster SHA256 differs; use the exact producer inputs.json.')
        need('postprocess' not in str(m['prior']['path']).replace('\\', '/').rsplit('/', 1)[-1].lower(),
             'A postprocessed prediction file cannot be consumed as fresh original incam.')
        if 'crop_mapping_sha256' in receipt:
            need(receipt['crop_mapping_sha256'] == m['crop_mapping']['sha256'],
                 'Fresh prior crop-mapping SHA256 differs.')
        for key in ('frame_ids', 'source_frames', 'image_sha256'):
            need(isinstance(receipt.get(key), list) and len(receipt[key]) == n,
                 'Fresh prior receipt requires the complete prediction-order ' + key + ' list.')
        for row, slot in zip(rows, slots):
            need(receipt['frame_ids'][slot] == str(row['frame']) and
                 type(receipt['source_frames'][slot]) is int and
                 receipt['source_frames'][slot] == row['source_frame_index'] and
                 receipt['image_sha256'][slot] == row['source_sha256'],
                 'Fresh prior receipt frame/source-frame/image identity mismatch.')
            if 'video_sha256' in row:
                need(row['video_sha256'] == video_hash, 'Prepared frame belongs to a different video.')
    # No crop transform is applied: RTMW points and K are already in native pixels.
    targets = np.zeros((n, 118, 2), dtype=np.float64)
    confidence = np.zeros((n, 118), dtype=np.float64)
    targets[:, TARGET_CHANNELS] = xy[:, SOURCE_CHANNELS]
    confidence[:, TARGET_CHANNELS] = relative[:, SOURCE_CHANNELS]
    visibility = np.full((n, 51), 'unknown', dtype='<U10')
    split = np.full((n, 51), 'UNASSIGNED', dtype='<U10')
    semantic = np.zeros((n, 51), bool)
    physical = np.zeros((n, 51), bool)
    families = [[f'unreviewed:{name}:ibug68:{j}' for j in range(17, 68)] for name in ids]
    review_records = [] if m.get('reviews') is None else file(m['reviews'], 'reviews')['records']
    lookup, seen, frozen_splits = {f: i for i, f in enumerate(ids)}, set(), {}
    for r in review_records:
        name = str(r['frame']); anchor = r.get('anchor', r); j = anchor['ibug68_index']
        need(name in lookup and type(j) is int and 17 <= j < 68, 'Review frame/anchor is outside roster.')
        i, k = lookup[name], j - 17
        need('anchor_id' not in anchor or anchor['anchor_id'] == f'ibug68:{j}',
             'Review anatomical anchor ID and index disagree.')
        if 'source_frame_index' in r:
            need(r['source_frame_index'] == source_frames[i], 'Review source-video frame identity mismatch.')
        need((i, k) not in seen, 'Duplicate reviewed observation.')
        seen.add((i, k))
        need(np.array_equal(numeric(r['uv_native'], (2,), 'Invalid reviewed pixel.'), xy[i, j + 23]),
             'Review coordinates do not match this observation version.')
        v, s = r['visibility'], r['split']
        need(v in ('observed', 'unknown', 'occluded', 'outside') and s in ('FIT', 'HELD', 'UNASSIGNED'),
             'Invalid explicit visibility/split.')
        family = r.get('physical_family_id', r.get('family_id'))
        need(isinstance(family, str) and bool(family), 'Reviewed family identity required.')
        need(family not in frozen_splits or frozen_splits[family] == s, 'Face-family holdout leakage.')
        frozen_splits[family] = s
        verified = r.get('semantic_identity_verified') is True
        if v == 'observed':
            need(r.get('visibility_source') in ('assistant_raw_image_review', 'image_review',
                 'independent_annotation', 'synthetic_known'), 'Detector confidence cannot establish visibility.')
            need(m['purpose'] == 'synthetic_contract_test' or r['visibility_source'] != 'synthetic_known',
                 'Synthetic review cannot authorize real observations.')
            need(bool(r.get('review_evidence') or r.get('source_image')), 'Observed review requires evidence provenance.')
            need(0 <= xy[i, j + 23, 0] < rows[i]['wh'][0] and
                 0 <= xy[i, j + 23, 1] < rows[i]['wh'][1], 'Observed face target is outside image.')
        visibility[i, k], split[i, k], semantic[i, k] = v, s, verified
        physical[i, k] = r.get('physical_identity_verified') is True
        families[i][k] = family

    image_count = decoded = 0
    from PIL import Image
    for row in rows:
        for key, hashkey in [('image', 'source_sha256'), ('mask', 'mask_sha256')]:
            path = resolve(row[key])
            need(path.is_file() and sha(path) == row[hashkey], key + ' hash mismatch.')
            with Image.open(path) as im:
                need(im.size == tuple(row['wh']), key + ' native dimensions mismatch.')
                if decode_images:
                    im.load(); decoded += 1
            bound.append((path, row[hashkey])); image_count += 1
    active = (visibility == 'observed') & semantic
    report = dict(schema=SCHEMA, frame_count=n, reference_frame_id=m['reference_frame_id'],
        initialization_kind=m['initialization_kind'], initialization_evidence=m['initialization_evidence'],
        root_policy=m['root_policy'], cached_prior_is_fresh_inference=False, old_optimized_fit_loaded=False,
        prior_is_fresh_inference=fresh, prior_prediction_variant='original_incam',
        prior_provenance_sha256=(m.get('prior_provenance') or {}).get('sha256'),
        video_sha256=m.get('video_sha256'),
        initial_identity='one zero beta(1,10); per-frame prior betas deliberately unused',
        vposer_encoding_pending=True, raw_body_pose_shape=[n, 63],
        full_K_preserved=True, source_E_unchanged=True, camera_translation_scale_applications=0,
        camera_length_contract=dict(length), camera_normalization_owner='SharedIdentitySequence constructor once',
        original_source_pixels_preserved=True, prior_crop_transform_reapplied=False,
        mapped_observations_per_frame=116, face_records_received=len(seen),
        face_records_missing_inactive=n * 51 - len(seen), visibility_counts=dict(Counter(visibility.ravel())),
        split_counts=dict(Counter(split.ravel())), active_face_fit=int((active & (split == 'FIT')).sum()),
        active_face_held=int((active & (split == 'HELD')).sum()),
        relative_score_q90=q90.tolist(), detector_weights_are_physical_visibility=False,
        hand_policy='prior_only', image_mask_hash_and_header_count=image_count, complete_decode_count=decoded,
        sources=[dict(path=str(p), sha256=h) for p, h in bound],
        training_admitted=False, fitting_admitted=False, quality_accepted=False,
        physical_material_correspondence_validated=False, physical_camera_accepted=False,
        engineering_input_complete=True, model_forwards=0, GPU_tasks=0)
    result = SequenceInputBundle(ids, _frozen(source_frames), _frozen([r['wh'] for r in rows]),
        _frozen(np.stack([c[0] for c in cameras])), _frozen(np.stack([c[1] for c in cameras])),
        _frozen(slots), _frozen(prior['body_pose'][slots]), _frozen(prior['global_orient'][slots]),
        _frozen(prior['transl'][slots]), _frozen(targets), _frozen(confidence), _frozen(visibility),
        _frozen(split), _frozen(semantic), _frozen(physical), _frozen(np.asarray(families)),
        _frozen(np.zeros((1, 10))), m['reference_frame_id'], MappingProxyType(dict(length)),
        tuple(bound), json.dumps(report, sort_keys=True, allow_nan=False))
    result.assert_unchanged()
    return result
