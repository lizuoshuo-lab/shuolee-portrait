"""Small roster-only selector: screen, rank, balance views, require pose review.

No fitting/training and no frame-specific thresholds. The ranking is the
existing eight-frame formula. A shortlist is never a training admission.
"""
import math
import numpy as np


POLICY = dict(target=8,
    score='0.45 rank(body RMS)+0.25 rank(face RMS)+0.20 rank(-face sharpness)+0.10 rank(-body confidence)',
    body_max_over_height=0.03, feet_max_over_height=0.03,
    joint_speed_deg_per_second=135.0,
    threshold_source='Existing screen_demo_body_quality123.POLICY; no outcome tuning',
    mesh_mask_iou_gate=False,
    mesh_mask_note='An unclothed initialization need not fill a clothing silhouette; do not reject it for that alone.',
    coverage='Equal-angle bins over the recorded camera-relative fitted-yaw arc; missing bins stay empty.',
    yaw_is_proxy_not_ground_truth=True,
    review='Explicit pose acceptance including supported limb depth. Unknown and rejected poses cannot train.',
    minimum_required_all_sequence_frames=0,
    source='One coherent fit archive per subject. No mixing experiments per frame.')


def rank(values):
    a = np.asarray(values, dtype=float)
    return np.array([(np.count_nonzero(a < v)+.5*np.count_nonzero(a == v))/len(a) for v in a])


def select(records, decisions=None, target=8, admitted_only=True):
    decisions = decisions or {}
    assert target > 0 and records
    ids = [r['frame'] for r in records]
    assert len(ids) == len(set(ids))
    assert len({r['source_fit_sha256'] for r in records}) == 1, 'Never mix fitted runs'
    assert set(decisions) <= set(ids)
    work = [dict(r) for r in records]
    keys = ('body_weighted_rms_px', 'face_weighted_rms_px', 'face_sharpness', 'body_confidence')
    numeric = [r for r in work if all(math.isfinite(r[k]) for k in keys+('yaw_proxy_deg',))]
    if numeric:
        score = sum(w*rank([sgn*r[k] for r in numeric]) for k,w,sgn in
                    [(keys[0],.45,1),(keys[1],.25,1),(keys[2],.20,-1),(keys[3],.10,-1)])
        for r,s in zip(numeric,score): r['quality_score'] = float(s)
    yaw = [r['yaw_proxy_deg'] for r in numeric]
    lo, hi = (min(yaw), max(yaw)) if yaw else (0., 0.)
    selected, coverage = [], []
    for r in work:
        d = decisions.get(r['frame'], dict(status='unreviewed'))
        assert d['status'] in ('accepted','rejected','uncertain','unreviewed')
        if d['status'] == 'accepted':
            assert d.get('evidence_sha256') and d.get('depth_reviewed'), 'Score is not a depth review'
        flags = list(r.get('screen_flags', []))
        if 'quality_score' not in r: flags.append('nonfinite_or_missing_measurement')
        r.update(review=d, selected=False, screen_flags=flags)
        r['coverage_bin'] = min(target-1, int(target*(r['yaw_proxy_deg']-lo)/(hi-lo))) if hi>lo and math.isfinite(r['yaw_proxy_deg']) else 0
        r['eligible'] = (not flags and d['status'] != 'rejected'
                         and (d['status'] == 'accepted' if admitted_only else d['status'] != 'uncertain'))
    for b in range(target):
        eligible = [r for r in work if r['coverage_bin'] == b and r['eligible']]
        best = min(eligible, key=lambda r:(r['quality_score'],r['frame'])) if eligible else None
        if best:
            best['selected'] = True; selected.append(best)
        coverage.append(dict(bin=b, frame=best['frame'] if best else None,
                             eligible_count=len(eligible)))
    return dict(selected_frames=[r['frame'] for r in sorted(selected,key=lambda r:r['time_seconds'])],
        selected_poses_reviewed=admitted_only and bool(selected),
        training_admission=False, training_admission_note='Selected-set coverage and output-pose review still required; count alone never launches training.', target_frames=target,
        actual_count=len(selected), missing_bins=[r['bin'] for r in coverage if r['frame'] is None],
        observed_yaw_proxy_range_deg=[lo,hi], coverage=coverage, records=work,
        requires_all_frames_good=False, shortlisted_only=not admitted_only)


def self_test():
    rows=[dict(frame=f'{i:03d}',time_seconds=i,source_fit_sha256='one',yaw_proxy_deg=i*10.,
        body_weighted_rms_px=1.,face_weighted_rms_px=1.,face_sharpness=10.,body_confidence=1.,screen_flags=[])
        for i in range(16)]
    yes={r['frame']:dict(status='accepted',depth_reviewed=True,evidence_sha256='evidence') for r in rows}
    assert select(rows)['actual_count']==0
    assert select(rows,yes)['actual_count']==8
    assert select(list(reversed(rows)),yes)['selected_frames']==select(rows,yes)['selected_frames']
    no=dict(yes);no['000']=dict(status='rejected');no['001']=dict(status='rejected')
    s=select(rows,no);assert s['actual_count']==7 and 0 in s['missing_bins']
    flagged=[dict(r,screen_flags=['implausible_pose']) if i<2 else r for i,r in enumerate(rows)]
    assert select(flagged,yes)['actual_count']==7
    assert select(rows,admitted_only=False)['actual_count']==8
    assert not select(rows,admitted_only=False)['training_admission']
    shifted=[dict(r,yaw_proxy_deg=r['yaw_proxy_deg']+360) for r in rows]
    assert select(shifted,yes)['selected_frames']==select(rows,yes)['selected_frames']
    for bad in [dict(yes, **{'000':dict(status='accepted')}), dict(yes, **{'000':dict(status='wrong')})]:
        try: select(rows,bad)
        except AssertionError: pass
        else: raise AssertionError('Invalid review accepted')
    try: select([dict(rows[0],source_fit_sha256='other')]+rows[1:],yes)
    except AssertionError: pass
    else: raise AssertionError('Mixed fit sources accepted')
    return dict(controls=11,passed=True,scope='Selector safety and determinism; no physical geometry claim')


if __name__ == '__main__': print(self_test())
