"""Protect 021 background identity using native patch and independent two-hop checks."""
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np

from fit_video21_native148 import NATIVE
from review_full_sequence123 import relocate
from track_native_continuity122 import step
from track_video21_background154 import OUT as TRACKS, load_gray
from update_leafwoman129_state import atomic, read, sha


OUT = TRACKS.with_name('source-screened-background156')
GRID_Y, GRID_X = np.mgrid[:17, :17]


def patch_pair(first, second, a, b):
    h, w = first.shape
    if any(not (20 <= x < w-20 and 20 <= y < h-20) for x, y in (a, b)):
        return np.nan, np.nan, np.nan, False
    template = cv2.getRectSubPix(first, (21, 21), tuple(map(float, a)))
    region = cv2.getRectSubPix(second, (37, 37), tuple(map(float, b)))
    if template.std() < 3 or region[8:29, 8:29].std() < 3:
        return np.nan, np.nan, np.nan, False
    corr = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(np.argmax(corr), corr.shape)
    gap = float(corr[y, x]-corr[(GRID_Y-y)**2+(GRID_X-x)**2>9].max())
    central = float(corr[8, 8])
    shift = float(np.hypot(x-8, y-8))
    return central, gap, shift, central >= .85 and gap >= .05 and shift <= 2.


def longest_segment(verified_triples):
    start = best_start = best_length = 0
    for i, ok in enumerate(np.r_[verified_triples, False]):
        if ok:
            continue
        if i-start > best_length:
            best_start, best_length = start, i-start
        start = i+1
    return slice(best_start, best_start+best_length+2) if best_length else slice(0, 0)


def main():
    assert not OUT.exists()
    source_plan = read(TRACKS/'plan.json')
    source_result = read(TRACKS/'result.json')
    archive = TRACKS/'tracks.npz'
    assert source_result['plan_sha256'] == sha(TRACKS/'plan.json')
    assert source_result['tracks_sha256'] == sha(archive)
    with np.load(archive, allow_pickle=False) as z:
        frames, uv, family, split = (z[key] for key in ('frames', 'uv', 'family', 'split'))
    native = read(NATIVE/'result.json')
    intake = relocate(native['data_root'])/'data/inputs.json'
    assert sha(intake) == source_plan['input_sha256']
    rows = {r['frame']:r for r in read(intake)['records']}
    assert frames.tolist() == source_plan['frames']
    assert not set(frames) & set(source_plan['retained_portrait_holdouts'])
    cv2.setNumThreads(4)
    noise = np.random.default_rng(156).integers(0, 256, (96, 96), dtype=np.uint8)
    periodic = np.tile(noise[:7, :7], (14, 14))
    assert patch_pair(noise, noise, (48, 48), (48, 48))[3]
    assert not patch_pair(periodic, periodic, (48, 48), (48, 48))[3]
    assert longest_segment([True, True, False, True]) == slice(0, 4)
    plan_path = OUT/'plan.json'
    atomic(plan_path, dict(source_plan, updated=datetime.now(timezone.utc).isoformat(),
        worker_sha256=sha(__file__), source_track_sha256=sha(archive), source_result_sha256=sha(TRACKS/'result.json'),
        hypothesis='Native appearance and independent two-hop agreement reject ambiguous matches while preserving enough rigid support for fixed-K SfM.',
        patch_screen=dict(template_wh=21, search_wh=37, min_std=3, min_ncc=.85, min_nonlocal_peak_gap=.05,
                          max_peak_shift_px=2., excluded_peak_radius_px=3),
        independent_hop=dict(initial_flow='zero; no chained target supplied', previous_LK_forward_backward_limit_px=.75,
                             composed_endpoint_disagreement_limit_px=2.),
        segment_policy='Only the longest contiguous segment supported by valid triples, earliest tie. At least three original observations. Family and fit/holdout role inherited.',
        geometry_residual_used=False, thresholds_predeclared=True,
        precedent_sha256=sha(Path(__file__).with_name('audit_material117.py'))))
    started = time.monotonic()
    n, m = uv.shape[:2]
    visible = np.isfinite(uv[..., 0]) & (split >= 0)[None]
    appearance = np.full((n-1,m,3), np.nan, np.float32)
    patch_ok = np.zeros((n-1,m), bool)
    triple_ok = np.zeros((n-2,m), bool)
    direct_error = np.full((n-2,m), np.nan, np.float32)
    first = second = None
    for i, frame in enumerate(frames):
        third = load_gray(rows[frame], source_plan['wh'])
        if i:
            for j in np.flatnonzero(visible[i-1] & visible[i]):
                a, b, c, ok = patch_pair(second, third, uv[i-1,j], uv[i,j])
                appearance[i-1,j] = [a,b,c]
                patch_ok[i-1,j] = ok
        if i >= 2:
            ids = np.flatnonzero(patch_ok[i-2] & patch_ok[i-1])
            if len(ids):
                pred, _, _, _, rejected = step(first, third, uv[i-2,ids])
                error = np.linalg.norm(pred-uv[i,ids], axis=1)
                direct_error[i-2,ids[rejected==0]] = error[rejected==0]
                triple_ok[i-2,ids] = (rejected == 0) & (error <= 2.)
        first, second = second, third
        if i % 12 == 0:
            print('BACKGROUND_SOURCE_CHECK', i+1, n, flush=True)
    screened_uv = np.full_like(uv, np.nan)
    screened_split = np.full_like(split, -1)
    segments = []
    for j in np.flatnonzero(split >= 0):
        segment = longest_segment(triple_ok[:,j])
        if segment.start == segment.stop:
            continue
        screened_uv[segment,j] = uv[segment,j]
        screened_split[j] = split[j]
        segments.append(dict(track=int(j), family=int(family[j]), split=int(split[j]),
                             start=frames[segment.start], end=frames[segment.stop-1],
                             length=segment.stop-segment.start))
    present = np.isfinite(screened_uv[..., 0])
    assert np.array_equal(screened_uv[present], uv[present])
    assert np.array_equal(screened_split[screened_split>=0],split[screened_split>=0])
    dest = OUT/'tracks.npz'
    np.savez_compressed(dest, frames=frames, uv=screened_uv, family=family, split=screened_split)
    np.savez_compressed(OUT/'checks.npz', appearance=appearance, patch_ok=patch_ok,
                        direct_error_px=direct_error, triple_ok=triple_ok)
    support = []
    for i in range(n-1):
        for j in range(i+1,min(i+3,n)):
            shared = present[i] & present[j]
            support.append(dict(indices=[i,j],frames=[frames[i],frames[j]],
                fit=int((shared & (screened_split==0)).sum()), heldout=int((shared & (screened_split==1)).sum())))
    atomic(OUT/'result.json', dict(seconds=time.monotonic()-started, plan_sha256=sha(plan_path),
        tracks_sha256=sha(dest), checks_sha256=sha(OUT/'checks.npz'), frames=n,
        fit_tracks=int((screened_split==0).sum()),heldout_tracks=int((screened_split==1).sum()),
        segments=segments,pair_support=support, material_identity_manually_verified=False,
        calibrated_camera=False,gaussian_started=False))
    print('SOURCE_SCREEN_FINISHED', int((screened_split==0).sum()),int((screened_split==1).sum()), flush=True)


if __name__ == '__main__':
    main()
