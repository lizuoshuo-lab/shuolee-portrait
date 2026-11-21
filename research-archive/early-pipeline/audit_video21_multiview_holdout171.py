"""Compare two-view and all-other-view predictions at the frozen 021 cameras."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np

from fit_video21_native148 import OUT as BASE
from shared_background122 import project, triangulate
from update_leafwoman129_state import atomic, read, sha


OUT = BASE / 'background-multiview-check171'
REFERENCE = BASE / 'fixedk-dense159-reference'
CAMERA = REFERENCE / 'seated021/global-fixedk'
TRACKS = BASE / 'dense-camera-tracks158'
OLD_COVERAGE = BASE / 'shoe-depth-xy167/review/source-pool-camera-coverage.json'


def statistics(errors):
    a = np.asarray(errors)
    if not len(a):
        return dict(n=0, median=None, p90=None)
    median,p90 = map(float, np.percentile(a,[50,90]))
    return dict(n=len(a), median=median, p90=p90)


def passes(stats):
    return stats['n'] >= 10 and stats['median'] <= 1. and stats['p90'] <= 3.


def main():
    assert not OUT.exists()
    before = read(CAMERA/'result.json')
    assert read(REFERENCE/'process-exit.json')['result_sha256'] == sha(CAMERA/'result.json')
    track_receipt = read(TRACKS/'result.json')
    archive = TRACKS/'tracks.npz'
    assert track_receipt['tracks_sha256'] == sha(archive) == read(REFERENCE/'global-fixedk-plan.json')['tracks_sha256']
    with np.load(archive,allow_pickle=False) as z:
        names, uv, family, split = (z[key] for key in ('frames','uv','family','split'))
    names = names.tolist()
    index = {f:i for i,f in enumerate(names)}
    pose_plan = read(BASE/'plan.json')
    assert len(names) == len(index) == 113 == before['registered_count']
    assert not set(names) & set(pose_plan['retained_holdouts'])
    assert set(family[split==0]).isdisjoint(family[split==1])
    prior = read(OLD_COVERAGE)
    assert prior['camera_result_sha256'] == sha(CAMERA/'result.json')
    assert prior['background_predictions_sha256'] == sha(CAMERA/'heldout-predictions.json')
    K = np.asarray(before['K_integer_centers'])
    assert np.array_equal(K, pose_plan['K_integer_centers']) and K[0,1] == K[1,0] == 0
    cameras = {}
    for row in read(CAMERA/'cameras-diagnostic.json'):
        i = index[row['frame']]
        assert i == row['frame_index'] and np.array_equal(K,row['K_integer_centers'])
        R,C = np.asarray(row['R_w2c']),np.asarray(row['C_world'])
        assert np.allclose(R @ R.T, np.eye(3), rtol=0, atol=1e-9) and abs(np.linalg.det(R)-1) < 1e-9
        E = np.eye(4)
        E[:3,:3], E[:3,3] = R, -R@C
        cameras[i] = E
    assert len(cameras) == len(names)
    P = {i:K @ E[:3] for i,E in cameras.items()}
    paths = [Path(__file__),Path(triangulate.__code__.co_filename),CAMERA/'cameras-diagnostic.json',
             CAMERA/'heldout-predictions.json',CAMERA/'result.json',archive,OLD_COVERAGE,BASE/'plan.json']
    plan = OUT/'plan.json'
    atomic(plan,dict(updated=datetime.now(timezone.utc).isoformat(), source_hashes={str(p):sha(p) for p in paths},
        hypothesis='Using just the widest pair can make the held-out prediction unstable despite many other observations.',
        comparison='Keep the same old valid targets and reconstruct with all other source observations of that held-out track. No robust weighting or residual-based point/view selection.',
        gauge='Original reference world gauge, complete K and exact archived poses in both arms.',
        protection='The target frame is absent from its 3D solve. Whole held-out families remain separate from camera FIT. Any new invalidity counts against coverage. Original unusable rows stay reported.',
        threshold=dict(n_min=10, median_max_px=1., p90_max_px=3.), required_3fps_bins=16,
        pose_holdouts=pose_plan['retained_holdouts'], camera_updates=0, gaussian_started=False))
    seen = np.isfinite(uv).all(2)
    support = {int(j):np.flatnonzero(seen[:,j]) for j in np.flatnonzero(split==1)}
    records, per_frame = [],defaultdict(list)
    worst_replay=0.
    start=time.monotonic()
    source=read(CAMERA/'heldout-predictions.json')
    for row in source:
        j,i=row['track_id'],row['frame_index']
        assert split[j]==1 and names[i]==row['frame'] and np.array_equal(uv[i,j],row['observed_uv'])
        rec=dict(frame=row['frame'],frame_index=i,track_id=j,family=int(family[j]),
                 observed_uv=row['observed_uv'],old_state=row['state'])
        if row['state']!='predicted':
            records.append(rec)
            continue
        a,b=(index[f] for f in row['triangulation_frames'])
        assert a!=i and b!=i and seen[a,j] and seen[b,j]
        h=cv2.triangulatePoints(P[a],P[b],uv[a,j].astype(float).reshape(2,1),uv[b,j].astype(float).reshape(2,1))[:,0]
        original=project(K,cameras[i],(h[:3]/h[3])[None])[0][0]
        worst_replay=max(worst_replay,float(np.max(np.abs(original-row['predicted_uv']))))
        others=support[j][support[j]!=i]
        X=triangulate(K,cameras,[(k,uv[k,j]) for k in others])
        depth=np.asarray([cameras[k][2,:3] @ X+cameras[k][2,3] for k in np.r_[others,i]])
        pred=None
        if np.isfinite(X).all() and np.all(depth>0):
            xy=project(K,cameras[i],X[None])[0][0]
            if np.isfinite(xy).all():
                pred=xy.tolist()
        rec.update(old_error_px=row['error_px'],old_uv=row['predicted_uv'],other_observations=len(others),
                   widest_pair_angle_deg=row['parallax_degrees'],all_other_uv=pred,
                   all_other_error_px=None if pred is None else float(np.linalg.norm(np.asarray(pred)-uv[i,j])))
        records.append(rec)
        per_frame[rec['frame']].append(rec)
    assert worst_replay < 1e-6
    summary={}
    for f in names:
        group=per_frame[f]
        original=statistics([r['old_error_px'] for r in group])
        evaluated=[r for r in group if r['all_other_uv'] is not None]
        updated=statistics([r['all_other_error_px'] for r in evaluated])
        # Missing projections must not be hidden by computing statistics on a smaller set.
        summary[f]=dict(widest_pair=original,all_other_views=updated,invalidated=len(group)-len(evaluated),
                        old_pass=passes(original),new_pass=len(group)==len(evaluated) and passes(updated))
    eligible=[r for r in prior['rows'] if r['passed']]
    assert all(summary[r['frame']]['old_pass']==r['passed'] for r in prior['rows'])
    bins={r['bin'] for r in eligible}
    new=[r for r in prior['rows'] if summary[r['frame']]['new_pass']]
    new_bins={r['bin'] for r in new}
    old=[r for r in records if 'old_error_px' in r]
    common=[r for r in old if r['all_other_uv'] is not None]
    paths_digest=read(plan)['source_hashes']
    assert all(sha(path)==digest for path,digest in paths_digest.items())
    atomic(OUT/'predictions.json',records)
    atomic(OUT/'result.json',dict(plan_sha256=sha(plan), predictions_sha256=sha(OUT/'predictions.json'),
        source_pose_sha256=sha(CAMERA/'cameras-diagnostic.json'), replay_max_abs_px=worst_replay,
        seconds=time.monotonic()-start, source_rows=len(records), original_unusable=len(records)-len(old),
        valid_count=len(old), new_invalid_count=len(old)-len(common),
        matched_old=statistics([r['old_error_px'] for r in common]),
        matched_new=statistics([r['all_other_error_px'] for r in common]),
        old_3fps_camera_frames=len(eligible), new_3fps_camera_frames=len(new),
        old_3fps_bins=sorted(bins), new_3fps_bins=sorted(new_bins),
        missing_3fps_bins=sorted(set(range(1,17))-new_bins), per_frame=summary,
        diagnostic_only=True, camera_updates=0, physical_camera_accepted=False, gaussian_started=False))
    print('MULTIVIEW_CAMERA_DIAGNOSTIC',len(eligible),len(new),len(bins),len(new_bins),
          len(old)-len(common),worst_replay,flush=True)


if __name__=='__main__':
    main()
