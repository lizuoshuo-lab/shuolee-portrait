"""Track short-interval static-background candidates in the native 021 frames."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

import fit_video21_native148 as portrait
from review_full_sequence123 import relocate
from track_native_continuity122 import controls, step
from update_leafwoman129_state import atomic, read, sha


OUT = portrait.OUT / 'static-background-tracks154'
MIN_LENGTH = 3
MAX_ACTIVE = 640
SEPARATION = 28
FAMILY_RADIUS = 12


def load_gray(record, wh):
    data = relocate(record['image']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == record['source_sha256']
    rgb = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert rgb.shape == (wh[1], wh[0], 3)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


def mask_for(record, exclusion, wh):
    w, h = wh
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in (exclusion, *record['other_person_boxes_xyxy']):
        mask[max(0, int(y0)-16):min(h, int(np.ceil(y1))+16),
             max(0, int(x0)-16):min(w, int(np.ceil(x1))+16)] = 0
    mask[:16] = mask[-16:] = 0
    mask[:, :16] = mask[:, -16:] = 0
    return mask


def seed_points(gray, mask, existing):
    available = mask.copy()
    for x, y in np.rint(existing).astype(int):
        cv2.circle(available, (x, y), SEPARATION, 0, -1)
    w, h = gray.shape[::-1]
    points = cv2.goodFeaturesToTrack(gray, maxCorners=MAX_ACTIVE,
                                    qualityLevel=.01, minDistance=SEPARATION, mask=available)
    if points is None:
        return np.empty((0, 2), np.float32)
    counts = Counter((min(7, int(x*8/w)), min(4, int(y*5/h))) for x, y in existing)
    new = []
    for x, y in points[:, 0]:
        cell = (min(7, int(x*8/w)), min(4, int(y*5/h)))
        if counts[cell] >= 16:
            continue
        counts[cell] += 1
        new.append((x, y))
    return np.asarray(new[:MAX_ACTIVE-len(existing)], dtype=np.float32).reshape(-1, 2)


def family_split(uv):
    count = uv.shape[1]
    parents = list(range(count))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    visible = np.isfinite(uv[..., 0])
    for row, present in zip(uv, visible, strict=True):
        ids = np.flatnonzero(present)
        for a, b in cKDTree(row[ids]).query_pairs(FAMILY_RADIUS, output_type='ndarray'):
            left, right = root(int(ids[a])), root(int(ids[b]))
            parents[max(left, right)] = min(left, right)
    families = np.array([root(i) for i in range(count)])
    lengths = visible.sum(axis=0)
    canonical = {}
    for i, family in enumerate(families):
        if lengths[i] >= MIN_LENGTH and (family not in canonical or lengths[i] > lengths[canonical[family]]):
            canonical[family] = i
    ids = sorted(canonical.values(), key=lambda i: hashlib.sha256(
        np.r_[np.flatnonzero(visible[:, i])[0], uv[visible[:, i], i][0]].astype('<f8').tobytes()).digest())
    split = np.full(count, -1, dtype=np.int8)
    split[ids] = 0
    split[ids[::4]] = 1
    return families, split


def main():
    assert not OUT.exists()
    native = read(portrait.NATIVE/'result.json')
    fit_plan = read(portrait.OUT/'plan.json')
    wide = portrait.OUT/'static-background-pairs153'
    wide_decision = read(wide/'visual-decision.json')
    assert wide_decision['actual_source_exclusion_review'] and not wide_decision['trajectory_admitted']
    inputs = relocate(native['data_root'])/'data/inputs.json'
    assert sha(inputs) == native['observation_manifest_sha256']
    native_plan = read(portrait.NATIVE/'plan.json')
    assert native_plan['video_sha256'] == native['video_sha256'] == sha(relocate(native_plan['source_video']))
    rows = read(inputs)['records']
    held = set(fit_plan['retained_holdouts'])
    assert len(rows) == 59 and len(held) == 4
    rows = [r for r in rows if r['frame'] not in held]
    frames = [r['frame'] for r in rows]
    assert len(frames) == 55 and set(fit_plan['source_frames']) <= set(frames)
    wh = rows[0]['wh']
    K = fit_plan['K_integer_centers']
    assert all(r['wh'] == wh and r['K_integer_centers'] == K for r in rows)
    boxes = np.array([r['initial_box_xyxy'] for r in rows])
    exclusion = [int(boxes[:, 0].min())-300, int(boxes[:, 1].min())-120,
                 int(np.ceil(boxes[:, 2].max()))+300, wh[1]]
    plan_path = OUT/'plan.json'
    atomic(plan_path, dict(updated=datetime.now(timezone.utc).isoformat(), frames=frames,
        native_indices=[r['source_frame_index'] for r in rows], candidate_frames=fit_plan['source_frames'],
        retained_portrait_holdouts=sorted(held), wh=wh, fixed_K=K, excluded_subject_chair_xyxy=exclusion,
        exclude_recorded_bystanders=True, input_sha256=sha(inputs), source_video_sha256=native['video_sha256'],
        wide_result_sha256=sha(wide/'result.json'), worker_sha256=sha(__file__),
        tracker_sha256=sha(Path(__file__).with_name('track_native_continuity122.py')),
        hypothesis='Native neighboring intervals support continuous static background feature families across the half-orbit.',
        method='GFTT seeds capped per image cell, original native forward/backward LK. Stop individual tracks on first rejection. Whole-family split before any camera fit.',
        active_limit=MAX_ACTIVE, minimum_length=MIN_LENGTH, seed_separation_px=SEPARATION,
        whole_family_covisible_radius_px=FAMILY_RADIUS, split='Hash order, 1 in 4 canonical tracks held out. Co-visible neighbors stay in one family.',
        calibration_admitted=False, gaussian_started=False))
    cv2.setNumThreads(4)
    cv2.setRNGSeed(154)
    started = time.monotonic()
    control = controls()
    histories, counts = [], []
    active = np.empty(0, dtype=int)
    previous = None
    active_xy = np.empty((0, 2), np.float32)
    for i, record in enumerate(rows):
        gray = load_gray(record, wh)
        mask = mask_for(record, exclusion, wh)
        accepted = 0
        if len(active):
            xy, _, _, _, rejection = step(previous, gray, active_xy)
            safe = np.flatnonzero(rejection == 0)
            rounded = np.rint(xy[safe]).astype(int)
            keep = safe[mask[rounded[:, 1], rounded[:, 0]] != 0]
            active, active_xy = active[keep], xy[keep]
            accepted = len(active)
        new = seed_points(gray, mask, active_xy)
        new_ids = np.arange(len(histories), len(histories)+len(new))
        histories.extend([] for _ in new)
        active = np.concatenate([active, new_ids])
        active_xy = np.concatenate([active_xy, new], axis=0)
        for j, point in zip(active, active_xy, strict=True):
            histories[j].append((i, point.copy()))
        counts.append(dict(frame=record['frame'], propagated=accepted, seeds=len(new), active=len(active)))
        previous = gray
        if i%8 == 0 or i == len(rows)-1:
            atomic(OUT/'progress.json', dict(pid=os.getpid(), state='tracking', frames=i+1, total=len(rows)))
            print('BACKGROUND_TRACKS', i+1, len(rows), accepted, len(new), flush=True)
    uv = np.full((len(rows), len(histories), 2), np.nan, np.float32)
    for j, history in enumerate(histories):
        for i, point in history:
            uv[i, j] = point
    families, split = family_split(uv)
    dest = OUT/'tracks.npz'
    np.savez_compressed(dest, frames=np.asarray(frames), uv=uv, family=families, split=split)
    visible = np.isfinite(uv[..., 0])
    support = []
    for i in range(len(rows)-1):
        for j in range(i+1, min(i+3, len(rows))):
            shared = visible[i] & visible[j]
            support.append(dict(indices=[i, j], frames=[frames[i], frames[j]],
                fit=int(np.count_nonzero(shared & (split == 0))), heldout=int(np.count_nonzero(shared & (split == 1)))))
    atomic(OUT/'result.json', dict(seconds=time.monotonic()-started, plan_sha256=sha(plan_path),
        tracks_sha256=sha(dest), frames=len(rows), candidates=int((split>=0).sum()),
        fit_tracks=int((split==0).sum()), heldout_tracks=int((split==1).sum()),
        controls=control, per_frame=counts, pair_support=support, material_review_complete=False,
        calibrated_camera=False, gaussian_started=False))
    atomic(OUT/'progress.json', dict(pid=None, state='complete', frames=len(rows), total=len(rows)))
    print('SHORT_BACKGROUND_TRACKS_COMPLETE', len(rows), len(histories), int((split>=0).sum()), flush=True)


if __name__ == '__main__':
    main()
