"""Run the same native intake, shared-shape SMPLify-X and screen for each video."""
import argparse
from datetime import datetime, timezone
from difflib import unified_diff
from hashlib import sha256
import os
from pathlib import Path
from shutil import copyfile
import sys
import time
import types

import numpy as np
from PIL import Image

from pose_training_gate151 import TARGETS, require, select_fit
from review_full_sequence123 import bound_file, crop_box, relocate
from update_leafwoman129_state import atomic, read, sha


SHAPE_TARGETS = {'front': 24, 'rear': 12}
ANKLE_FOOT_TARGETS = {'left_foot': (14, 19, 20, 21), 'right_foot': (11, 22, 23, 24)}


def configured(function, **inputs):
    return types.FunctionType(function.__code__, dict(function.__globals__, **inputs), function.__name__)


def intake(video, report, data):
    import prepare_core_subjects124 as core
    from run_official_gaussians123 import sha as stream_sha

    digest = stream_sha(video)
    number = video.stem[5:].zfill(3) if video.stem.startswith('video') else video.stem
    for stage in (core.decode, core.observe):
        configured(stage, paths=lambda _: (video, data, report, number, digest),
                   __file__=__file__)(video.stem)


def fit_shard(folder, shard):
    from fit_video21_native148 import original

    archive = folder.parent / 'fit-observations.npz'
    contacts = folder.parent / 'contacts.json'
    raw = {r['source_frame_index']: r['paths']['body'] for r in read(contacts)['records']}

    def decoded_sources(indices, box, reverse):
        require(not reverse, 'Unexpected reverse source decode.')
        for index in indices:
            source = raw[index]
            require(tuple(source['crop_original_xyxy']) == box, 'Unexpected worker pixel domain.')
            with Image.open(bound_file(source)) as im:
                require(im.size == box[2:], 'Source RGB differs from the registered native size.')
                yield index, np.asarray(im.convert('RGB'))

    def snapshot_orientation_results(root):
        source, diff = original.entry.adapted_entry_source(root)
        marker = "            results.append({'loss': final_loss_val,"
        require(source.count(marker) == 1, 'Cannot locate the upstream orientation result.')
        # CPU NumPy views otherwise share storage with parameters reset by the next orientation.
        saved = source.replace(marker, '            result = {key: value.copy() for key, value in result.items()}\n' + marker)
        return saved, diff + ''.join(unified_diff(source.splitlines(True), saved.splitlines(True),
                                                  fromfile='input-output-adapter', tofile='orientation-snapshot'))

    solver = configured(original.entry.fit_one, adapted_entry_source=snapshot_orientation_results)

    def fit_with_snapshots(**kwargs):
        arrays, receipt = solver(**kwargs)
        receipt['independent_orientation_snapshots'] = True
        return arrays, receipt

    scope = dict(ARCHIVE=archive, CONTACTS=contacts, decode=decoded_sources, __file__=__file__)
    shared_pose = read(folder/'plan.json').get('shared_body_constraint')
    fit_one = fit_with_snapshots
    if shared_pose is not None:
        require(all(sha(Path(__file__).with_name(name)) == digest
                    for name, digest in shared_pose['loss_source_sha256'].items()),
                'The shared pose loss implementation changed.')
        def anchored_fit(**kwargs):
            import torch
            from smplify_seeded_refine124 import load_with_anchor

            vposer = kwargs['vposer2']
            with torch.no_grad():
                latent = torch.as_tensor(shared_pose['latent'], dtype=torch.float32)
                target = vposer.decode(latent)['pose_body'].reshape(1, 63).cpu().numpy()
            require(np.isfinite(target).all(), 'The shared pose must decode to finite body rotations.')
            frame = kwargs['result_path'].parent.name
            root_anchor = shared_pose.get('root_targets_camera', {}).get(frame)
            if root_anchor is not None:
                root_anchor = np.asarray(root_anchor, dtype=np.float32)
                require(root_anchor.shape == (1, 3) and np.isfinite(root_anchor).all(),
                        'Expected one finite camera-frame root rotation.')
            log = {}
            shared_solver = configured(solver,
                load_official=lambda root: load_with_anchor(root, target, log, root_anchor=root_anchor))
            arrays, receipt = shared_solver(**kwargs)
            require(log.get('calls', 0) > 0 and torch.isfinite(log['_last_anchor']).item(),
                    'The shared body loss was not evaluated.')
            receipt.update(upstream_loss_monitor_optimizer_unchanged=False,
                original_optimizer_and_monitor=True, shared_body_pose_soft=True,
                shared_body_weight=30., shared_root_weight=30. if root_anchor is not None else 0.,
                shared_body_loss_calls=log['calls'],
                last_evaluated_body_loss=float(log['_last_anchor']),
                shared_body_target_sha256=sha256(target.tobytes()).hexdigest(),
                independent_orientation_snapshots=True)
            return arrays, receipt

        fit_one = anchored_fit
    scope['entry'] = types.SimpleNamespace(**dict(vars(original.entry), fit_one=fit_one))
    worker = configured(original.fit_frames, **scope)
    return configured(original.fit_shard, fit_frames=worker, **scope)(folder, shard)


def source_view_groups(names, source, arc):
    rear = source['rear_coverage_observed']
    require(type(rear) is bool and (arc == 'half' or rear), 'Full-circle input requires observed rear coverage.')
    front_ids = source.get('front_frame_ids', names if not rear else [])
    rear_ids = source.get('rear_frame_ids', [])
    require(len(set(front_ids)) == len(front_ids) and len(set(rear_ids)) == len(rear_ids)
            and not set(front_ids) & set(rear_ids) and (set(front_ids) | set(rear_ids)) <= set(names)
            and bool(rear_ids) == rear, 'Require disjoint verified source front/rear groups.')
    return {'front': set(front_ids), 'rear': set(rear_ids) if arc == 'full' else set()}


def shape_anchors(names, candidates, groups, reliability, targets, *, automatic=False):
    anchors = {}
    for side, target in targets.items():
        pool = [j for j in candidates if names[j] in groups[side]]
        if target == 0:
            anchors[side] = []
            continue
        if automatic:
            target = min(target, len(pool))
            require(target > 0, f'No usable {side} source samples.')
        else:
            require(len(pool) >= target, f'{side} source coverage has {len(pool)}/{target} usable shape samples.')
        anchors[side] = [int(max(b, key=lambda j: (reliability[j], -j)))
                         for b in np.array_split(pool, target)]
    return anchors


def abstain_reviewed_source_visibility(source, source_indices, confidence):
    """Apply the reviewed source-domain exclusions before the joint mapping."""
    masked = confidence.copy()
    for item in source.get('source_visibility_exclusions', ()):
        first, last = item['sample_index_inclusive']
        points = item['coco133_indices']
        require(type(first) is int and type(last) is int and 1 <= first <= last <= source['native_samples']
                and points and all(type(i) is int and 0 <= i < masked.shape[1] for i in points),
                'Invalid source visibility exclusion.')
        slots = np.flatnonzero((source_indices >= first) & (source_indices <= last))
        masked[np.ix_(slots, points)] = 0.
    return masked


def face_counts(source, rows):
    binding = source.get('face_support_receipt')
    if binding is None:
        return None
    path, review = relocate(binding['path']), relocate(binding['review_path'])
    require(sha(path) == binding['sha256'] and sha(review) == binding['review_sha256']
            and read(review)['actual_visual_review'] is True, 'Missing verified face-support evidence.')
    records = [r for r in read(path)['records'] if r['subject'] == binding['subject']]
    by_name = {r['frame']: r for r in records}
    require(len(records) == len(by_name) == len(rows)
            and all(r['frame'] in by_name and r['source_sha256'] == by_name[r['frame']]['source_sha256']
                    for r in rows), 'Face-support evidence does not match the native image roster.')
    return np.asarray([len(by_name[r['frame']]['associated_indices']) for r in rows], dtype=np.int64)


def abstain_reviewed_ankle_feet(source, manifest_sha, rows, candidates, landmarks):
    binding = source.get('joint_group_support_receipt')
    if binding is None:
        return {}
    path = relocate(binding['path'])
    require(sha(path) == binding['sha256'], 'Joint visibility evidence changed.')
    evidence = read(path)
    require(evidence['actual_visual_review'] is True
            and evidence['observation_manifest_sha256'] == manifest_sha
            and evidence['observations_sha256'] == source['observations_sha256']
            and evidence['source_video_sha256'] == source['video_sha256'],
            'Joint visibility evidence must match these reviewed native observations.')
    records = evidence['records']
    by_frame = {row['frame']: row for row in records}
    require(len(records) == len(by_frame) == len(rows)
            and all(row['frame'] in by_frame
                    and row['source_sha256'] == by_frame[row['frame']]['source_sha256'] for row in rows),
            'Joint visibility evidence does not match the native image roster.')
    require(all(isinstance(row['absent_groups'], list)
                and set(row['absent_groups']) <= ANKLE_FOOT_TARGETS.keys() for row in records),
            'Only reviewed ankle and foot groups can be withheld.')
    candidate_frames = [rows[i]['frame'] for i in candidates]
    for group, indices in ANKLE_FOOT_TARGETS.items():
        slots = [i for i, frame in enumerate(candidate_frames) if group in by_frame[frame]['absent_groups']]
        landmarks[np.ix_(slots, indices, [2])] = 0.
    return {frame: by_frame[frame]['absent_groups'] for frame in candidate_frames if by_frame[frame]['absent_groups']}


def prepare_fit(native_receipt, source_review, out, arc, *, automatic=False,
                front_last_source_index=None, capture_arcs=None):
    import fit_video21_native148 as adapter
    from smplify_face_support124 import abstain_inner_face
    from smplify_observation_validity124 import mask_outside_raster
    from smplify_score_contract123 import relative_confidence

    receipt = read(native_receipt)
    require(receipt['observations_ready'] is True, 'Need completed native observations.')
    require(capture_arcs is None or (automatic and front_last_source_index is None),
            'A capture layout is only for automatic fitting and replaces the single boundary.')
    source = None if automatic else read(source_review)
    if not automatic:
        require(source['actual_source_and_observation_review'] is True
                and all(source[k] == receipt[k] for k in ('video_sha256', 'observations_sha256')),
                'Source coverage belongs to different video observations.')
    root = relocate(receipt['data_root']) / 'data'
    manifest, observations = root/'inputs.json', root/'observations.npz'
    require(sha(manifest) == receipt['observation_manifest_sha256']
            and sha(observations) == receipt['observations_sha256'], 'Native input receipt mismatch.')
    rows = read(manifest)['records']
    names = [r['frame'] for r in rows]
    if capture_arcs is not None:
        from capture_arc_layout import source_layout

        source = source_layout(read(capture_arcs), rows, receipt['video_sha256'], arc)
    elif automatic:
        require(arc == 'half' or front_last_source_index is not None,
                'A full capture needs a capture layout or a source index separating front and rear arcs.')
        front = names if arc == 'half' else [r['frame'] for r in rows if r['source_frame_index'] <= front_last_source_index]
        rear = [f for f in names if f not in front]
        require(front and (arc == 'half' or rear), 'The capture boundary must leave frames on both full-arc sides.')
        source = dict(front_frame_ids=front, rear_frame_ids=rear, rear_coverage_observed=bool(rear))
    targets = dict(SHAPE_TARGETS)
    if arc == 'half':
        targets['rear'] = 0
    target = sum(targets.values())
    require(len(names) >= (1 if automatic else target + 4) and len(set(names)) == len(names),
            'Insufficient unique 3 fps samples.')
    native_wh = rows[0]['wh']
    K = np.asarray(rows[0]['K_integer_centers'])
    require(len(native_wh) == 2 and all(type(v) is int and v > 0 for v in native_wh)
            and all(r['wh'] == native_wh and np.array_equal(r['K_integer_centers'], K) for r in rows),
            'Expected one native frame size and the same full K for the video.')
    with np.load(observations, allow_pickle=False) as data:
        require(data['frame_ids'].tolist() == names, 'Observation order differs from native frames.')
        xy, scores, boxes, flip = [data[k] for k in ('keypoints', 'scores', 'boxes', 'flip_disagreement')]
    rejected = set(source.get('source_rejected_frames', ()))
    reserved = set(source.get('retained_holdouts', ()))
    require((rejected | reserved) <= set(names) and not rejected & reserved,
            'Source rejections and existing holdouts must be disjoint known frames.')
    groups = source_view_groups(names, source, arc)
    source_pool = [j for j, name in enumerate(names) if name not in rejected and
                   (name in groups['front'] or name in groups['rear'])]
    if not automatic:
        require(len(source_pool) >= target + max(4, len(reserved)), 'Insufficient usable source frames.')
    bins = []
    for side, count in targets.items():
        if count:
            pool = [j for j in source_pool if names[j] in groups[side]]
            if not automatic:
                require(len(pool) >= count, f'{side} source coverage has {len(pool)}/{count} samples.')
            bins.extend(list(map(int, b)) for b in np.array_split(pool, count))
    if reserved:
        held = {j for j, name in enumerate(names) if name in reserved}
    elif automatic:
        hold_bins = [target * k // 16 for k in (2, 6, 10, 14)]
        hold_bins = [i for i in hold_bins if len(bins[i]) > 1]
        held = {bins[i][len(bins[i])//2] for i in hold_bins}
    else:
        hold_bins = [target * k // 16 for k in (2, 6, 10, 14)]
        held = {bins[i][len(bins[i])//2] for i in hold_bins}
    eligible = [[j for j in b if j not in held] for b in bins]
    if not automatic:
        require(all(eligible), 'A source interval has no fit candidate after reserving holdouts.')
    candidates = [j for b in eligible for j in b]
    reliability = scores[:, :23].sum(1) / (1. + flip[:, :23].mean(1))
    anchor_groups = shape_anchors(names, candidates, groups, reliability, targets, automatic=automatic)
    anchors = [j for group in anchor_groups.values() for j in group]
    confidence, q90 = relative_confidence(scores[candidates], boxes[candidates], flip[candidates])
    if source.get('source_visibility_exclusions'):
        require(source['native_samples'] == len(rows), 'Visibility evidence uses a different source roster.')
        supported_confidence = abstain_reviewed_source_visibility(
            source, np.asarray(candidates) + 1, confidence)
    else:
        supported_confidence = confidence
    landmarks = np.zeros((len(candidates), 118, 3), np.float32)
    landmarks[:, adapter.TARGET_IDX, :2] = xy[candidates][:, adapter.SOURCE_IDX]
    landmarks[:, adapter.TARGET_IDX, 2] = supported_confidence[:, adapter.SOURCE_IDX]
    landmarks, inside = mask_outside_raster(landmarks, rows[0]['wh'])
    counts = face_counts(source, rows)
    face_absence = []
    if counts is not None:
        observed_counts = counts[candidates]
        landmarks = abstain_inner_face(landmarks, observed_counts)
        face_absence = [names[j] for j, n in zip(candidates, observed_counts, strict=True) if n == 0]
    foot_absence = abstain_reviewed_ankle_feet(source, receipt['observation_manifest_sha256'],
                                              rows, candidates, landmarks)
    contacts, crops = [], []
    for j in candidates:
        row = rows[j]
        image, mask = relocate(row['image']), relocate(row['mask'])
        require(sha(image) == row['source_sha256'] and sha(mask) == row['mask_sha256'],
                f"Source RGB or mask changed for {row['frame']}.")
        raw = dict(path=str(image), sha256=row['source_sha256'], native_integer_crop=True,
                   resized=False, crop_original_xyxy=[0, 0, *row['wh']])
        contacts.append(dict(row, paths={'body': raw}))
        crops.append(dict(frame=row['frame'], crop_xyxy=crop_box(row['initial_box_xyxy'], row['wh'], padding=24)))

    out.mkdir(parents=True, exist_ok=False)
    archive, contact_path = out/'fit-observations.npz', out/'contacts.json'
    frame_ids = [names[j] for j in candidates]
    adapter.save_npz(archive, frame_ids=np.asarray(frame_ids), observed_native_keypoints=landmarks,
                     K_integer_centers=K, confidence_q90=q90)
    atomic(contact_path, dict(records=contacts))
    atomic(out/'observation-review/inputs-review.json', dict(rows=crops))
    plan = dict(updated=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
        worker_sha256=sha(adapter.original.__file__), fit_entry_sha256=sha(adapter.entry.__file__),
        source_video_sha256=receipt['video_sha256'], native_receipt_sha256=sha(native_receipt),
        source_observations_sha256=receipt['observations_sha256'],
        source_review_sha256=None if automatic else sha(source_review),
        automatic_delivery=automatic, front_last_source_index=front_last_source_index,
        source_frames=frame_ids, native_indices=[r['source_frame_index'] for r in contacts],
        source_view_bins=[[names[j] for j in b] for b in bins], retained_holdouts=[names[j] for j in sorted(held)],
        view_targets=targets,
        reviewed_source_weights_withheld=int(np.count_nonzero(
            confidence[:, adapter.SOURCE_IDX] != supported_confidence[:, adapter.SOURCE_IDX])),
        source_rejected_frames=[f for f in names if f in rejected],
        raster_unsupported_frames=[names[j] for i, j in enumerate(candidates) if not inside[i].all()],
        face_absence_frames=face_absence, ankle_foot_absence=foot_absence,
        shape_anchors=[names[j] for j in anchors], rear_coverage_observed=source['rear_coverage_observed'],
        requested_shape_samples=targets, shape_samples={s: [names[j] for j in g] for s, g in anchor_groups.items()},
        baseline_observation_sha256=sha(archive), contacts_sha256=sha(contact_path),
        native_wh=native_wh, K_integer_centers=K.tolist(), conditional_K=True,
        shape_policy='Select best detector reliability in each chronological bin within its verified side. '
                     'Median of independent official beta fits, then freeze for every candidate.',
        model_unit_scale=1., per_frame_rescale=False, subject_specific_fit_overrides=False, gaussian_started=False)
    if capture_arcs is not None:
        plan.update(capture_arcs_path=str(capture_arcs), capture_arcs_sha256=sha(capture_arcs),
                    capture_arc_layout_code_sha256=sha(Path(__file__).with_name('capture_arc_layout.py')))
    atomic(out/'plan.json', plan)
    return plan


def reuse_free_fits(old_stage, out, plan):
    """Reuse only independent zero-start fits with bit-identical solver inputs."""
    if old_stage is None:
        return {}
    old_plan = read(old_stage/'plan.json')
    old_result = read(old_stage/'result.json')
    require(old_result['plan_sha256'] == sha(old_stage/'plan.json') and old_plan['shared_shape'] is None
            and all(old_plan[k] == plan[k] for k in ('worker_sha256', 'fit_entry_sha256', 'source_video_sha256')),
            'Reuse requires the same original independent free-fit stage.')
    contacts = {r['frame']: r for r in read(out/'contacts.json')['records']}
    old_contacts_path = old_stage.parent/'contacts.json'
    require(sha(old_contacts_path) == old_plan['contacts_sha256'], 'Cached source rows changed.')
    old_contacts = {r['frame']: r for r in read(old_contacts_path)['records']}
    reused = {}
    with np.load(out/'fit-observations.npz', allow_pickle=False) as data:
        ids = data['frame_ids'].tolist()
        for r in old_result['frames']:
            frame = r['frame']
            if frame not in plan['shape_anchors'] or (r.get('orientation_count') != 1
                                                     and r.get('independent_orientation_snapshots') is not True):
                continue
            require(r['freeze_betas'] is False and r['use_shared_betas'] is False
                    and contacts[frame]['source_sha256'] == old_contacts[frame]['source_sha256']
                    and contacts[frame]['source_frame_index'] == r['source_index']
                    and sha(r['output']) == r['sha256'], f'Cached free fit changed for {frame}.')
            with np.load(r['output'], allow_pickle=False) as fit:
                i = ids.index(frame)
                if (not np.array_equal(fit['K_integer_centers'], data['K_integer_centers'])
                        or not np.array_equal(fit['observed_native_keypoints'], data['observed_native_keypoints'][i:i+1])):
                    continue
            reused[frame] = r
    return reused


def fit(native_receipt, source_review, out, arc, reuse_anchors, *, automatic=False,
        front_last_source_index=None, capture_arcs=None):
    import fit_video21_native148 as adapter

    started = time.monotonic()
    plan = prepare_fit(native_receipt, source_review, out, arc, automatic=automatic,
                       front_last_source_index=front_last_source_index, capture_arcs=capture_arcs)
    atomic(out/'launch.json', dict(command=sys.argv, pid=os.getpid()))
    stage = configured(adapter.stage, OUT=out, fit_shard=fit_shard)
    reused = reuse_free_fits(reuse_anchors, out, plan)
    missing = [plan['source_frames'].index(f) for f in plan['shape_anchors'] if f not in reused]
    print('SHAPE_ANCHORS', {k: len(v) for k, v in plan['shape_samples'].items()},
          'reused', len(reused), 'new', len(missing), flush=True)
    fitted = dict(reused)
    if missing:
        fitted.update({r['frame']: r for r in stage('shape-anchors', plan, missing, None)})
    shapes, poses = [], []
    for f in plan['shape_anchors']:
        r = fitted[f]
        with np.load(r['output'], allow_pickle=False) as a:
            if automatic and not (np.isfinite(a['vertices_camera']).all() and (a['vertices_camera'][..., 2] > .05).all()):
                continue
            shapes.append(a['betas'])
            poses.append(a['pose_embedding'])
    require(shapes, 'No finite positive-depth result for the shared body shape.')
    beta = np.median(shapes, axis=0).astype(np.float32)
    require(beta.shape == (1, 10) and np.isfinite(beta).all(), 'Invalid shared body shape.')
    atomic(out/'shared-shape.json', dict(betas=beta.tolist(), model_unit_scale=1.,
        samples=plan['shape_samples'], reused_frames=list(reused),
        frame_fit_receipts=[fitted[f] for f in plan['shape_anchors']]))
    if automatic:
        target = np.median(poses, axis=0).astype(np.float32)
        require(target.shape == (1, 32) and np.isfinite(target).all(), 'Invalid shared body target.')
        plan['shared_body_constraint'] = dict(latent=target.tolist(), body_weight=30., root_weight=0.,
            loss_source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                                ('smplify_seeded_refine124.py','smplify_temporal124.py')},
            estimator='Componentwise median of original shape-anchor pose latents with positive depth.')
        atomic(out/'plan.json',plan)
    fit_poses(out, arc, plan, beta, len(fitted), len(reused), started)


def fit_poses(out, arc, plan, beta, anchor_count, reused_anchor_count, started, placements=None):
    import fit_video21_native148 as adapter
    import review_video21_fit148 as review

    stage = configured(adapter.stage, OUT=out, fit_shard=fit_shard)
    rows = stage('shared-cpu', plan, list(range(len(plan['source_frames']))), dict(vector=beta.tolist()))
    for r in rows:
        with np.load(r['output'], allow_pickle=False) as a:
            require(r['freeze_betas'] is True and np.array_equal(a['betas'], beta)
                    and not any(k in a for k in ('scale', 'body_scale', 'global_scale')),
                    f"Body shape or scale changed in {r['frame']}.")
            if placements is not None:
                require(np.array_equal(a['camera_translation'], placements[r['frame']]),
                        f"Camera initialization changed for {r['frame']}.")
    if 'shared_body_constraint' in plan:
        require(len({r['shared_body_target_sha256'] for r in rows}) == 1,
                'Workers did not use one shared body pose.')
    atomic(out/'result.json', dict(plan_sha256=sha(out/'plan.json'), frames=rows, fixed_shape=beta.tolist(),
        model_unit_scale=1., anchor_count=anchor_count, reused_anchor_count=reused_anchor_count,
        shared_shape_sha256=sha(out/'shared-shape.json'), fitting_complete=True, seconds=time.monotonic()-started,
        gaussian_started=False, quality_accepted=False))
    if plan.get('automatic_delivery'):
        return
    configured(review.review, OUT=out, NATIVE=out, CONTACTS=out/'contacts.json')(False)
    screening = select_fit(out, arc)
    atomic(out/'review/pose-training-selection.json', screening)
    print('FIXED_FLOW_SCREEN', len(rows), len(screening['selected_frames']), screening['target_frames'], flush=True)


def refit_shared_body(base, out, arc):
    import fit_video21_native148 as adapter

    started = time.monotonic()
    plan_path, result_path = base/'plan.json', base/'result.json'
    plan, baseline = read(plan_path), read(result_path)
    gate = read(base/'review/pose-training-selection.json')
    names = plan['source_frames']
    require(baseline['fitting_complete'] is True and baseline['plan_sha256'] == sha(plan_path)
            and gate['arc'] == arc and gate['source_fit_sha256'] == sha(result_path)
            and [row['frame'] for row in baseline['frames']] == names
            and not set(names) & set(plan['retained_holdouts']) and 'shared_body_constraint' not in plan,
            'Require one completed original baseline and its exact non-holdout roster.')
    require(plan['worker_sha256'] == sha(adapter.original.__file__)
            and plan['fit_entry_sha256'] == sha(adapter.entry.__file__)
            and plan['baseline_observation_sha256'] == sha(base/'fit-observations.npz')
            and plan['contacts_sha256'] == sha(base/'contacts.json')
            and baseline['shared_shape_sha256'] == sha(base/'shared-shape.json'),
            'The frozen baseline inputs have changed.')
    beta = np.asarray(baseline['fixed_shape'], np.float32)
    poses, placements = {}, {}
    for row in baseline['frames']:
        frame = row['frame']
        path = relocate(row['output'])
        require(row['freeze_betas'] is True and sha(path) == row['sha256'], f'Baseline changed for {frame}.')
        require(row.get('orientation_count') == 1 or row.get('independent_orientation_snapshots') is True,
                f'Recompute the baseline for {frame} with independent orientation snapshots.')
        with np.load(path, allow_pickle=False) as a:
            latent = a['pose_embedding']
            require(latent.shape == (1, 32) and np.isfinite(latent).all()
                    and np.array_equal(a['betas'], beta) and np.array_equal(a['K_integer_centers'], plan['K_integer_centers']),
                    f'Invalid baseline pose, shape or K for {frame}.')
            poses[frame] = latent[0]
            placements[frame] = a['camera_translation']
    groups = [[poses[frame] for frame in group if frame in poses] for group in plan['source_view_bins']]
    require(all(groups), 'A shared body estimate needs a baseline pose in every source interval.')
    consensus = np.median([np.median(group, axis=0) for group in groups], axis=0).astype(np.float32)
    out.mkdir(parents=True, exist_ok=False)
    (out/'observation-review').mkdir()
    for name in ('fit-observations.npz', 'contacts.json', 'shared-shape.json', 'observation-review/inputs-review.json'):
        copyfile(base/name, out/name)
    plan.update(updated=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
        shared_body_constraint=dict(latent=consensus[None].tolist(), baseline_plan_sha256=sha(plan_path),
            baseline_fit_sha256=sha(result_path), body_weight=30., root_weight=0.,
            loss_source_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                ('smplify_seeded_refine124.py', 'smplify_temporal124.py')},
            estimator='Median within each source interval, then median across intervals.'))
    atomic(out/'plan.json', plan)
    atomic(out/'launch.json', dict(command=sys.argv, pid=os.getpid()))
    fit_poses(out, arc, plan, beta, baseline['anchor_count'], baseline['anchor_count'], started, placements)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    p = stages.add_parser('intake')
    for name in ('video', 'report', 'data'):
        p.add_argument(name, type=relocate)
    p = stages.add_parser('fit')
    p.add_argument('native_receipt', type=relocate)
    p.add_argument('out', type=relocate)
    p.add_argument('--source-review', type=relocate)
    p.add_argument('--automatic', action='store_true')
    layout = p.add_mutually_exclusive_group()
    layout.add_argument('--front-last-source-index', type=int)
    layout.add_argument('--capture-arcs', type=relocate, help='Video-bound JSON with disjoint front/rear source intervals')
    p.add_argument('--reuse-anchors', type=relocate)
    p.add_argument('--arc', choices=tuple(TARGETS), required=True)
    p = stages.add_parser('shared-body')
    p.add_argument('baseline', type=relocate)
    p.add_argument('out', type=relocate)
    p.add_argument('--arc', choices=tuple(TARGETS), required=True)
    args = parser.parse_args()
    if args.stage == 'intake':
        intake(args.video, args.report, args.data)
    elif args.stage == 'fit':
        require(args.automatic != (args.source_review is not None), 'Provide --automatic or --source-review.')
        fit(args.native_receipt, args.source_review, args.out, args.arc, args.reuse_anchors,
            automatic=args.automatic, front_last_source_index=args.front_last_source_index,
            capture_arcs=args.capture_arcs)
    else:
        refit_shared_body(args.baseline, args.out, args.arc)
