"""Quality first, prefer covered directions, then fill up to about eight.

Angular bins are a diversity preference, not a reason to throw away an
otherwise eligible frame. Empty angles remain explicitly undisclosed coverage.
"""
import numpy as np
from select_quality_frames124 import select

def select_budget(records,decisions,target=8):
    result=select(records,decisions,target,admitted_only=True)
    rows=result['records'];chosen=[r for r in rows if r['selected']]
    while len(chosen)<target:
        rest=[r for r in rows if r['eligible'] and not r['selected']]
        if not rest:break
        def score(r):
            distance=min((abs((r['yaw_proxy_deg']-s['yaw_proxy_deg']+180)%360-180) for s in chosen),default=180.)
            return (-distance,r['quality_score'],r['frame'])
        best=min(rest,key=score);best['selected']=True;best['selection_role']='eligible_budget_fill';chosen.append(best)
    result.update(selected_frames=[r['frame'] for r in sorted(chosen,key=lambda r:(r['time_seconds'],r['frame']))],
        actual_count=len(chosen),selection_policy='Original quality/acceptance gates; one best per angular bin, then remaining eligible frames by largest angular distance, quality score and stable frame id.',
        fills_from_eligible_only=True,missing_bins=[b for b in range(target) if not any(r['coverage_bin']==b for r in chosen)],
        training_admission=False,coverage_is_recorded_yaw_proxy_only=True)
    return result

def verify():
    rows=[dict(frame=f'{i:03d}',time_seconds=i,source_fit_sha256='same',yaw_proxy_deg=x,
        body_weighted_rms_px=1.,face_weighted_rms_px=1.,face_sharpness=10.,body_confidence=1.,screen_flags=[])
        for i,x in enumerate([0,1,2,3,4,5,6,90,110,120])]
    yes={r['frame']:dict(status='accepted',depth_reviewed=True,evidence_sha256='saved') for r in rows}
    result=select_budget(rows,yes);assert result['actual_count']==8
    assert result['missing_bins'] and not result['training_admission']
    assert result['selected_frames']==select_budget(list(reversed(rows)),yes)['selected_frames']
    assert select_budget(rows,{})['actual_count']==0
    sparse=[dict(r,screen_flags=['bad']) if i<7 else r for i,r in enumerate(rows)]
    result=select_budget(sparse,yes);assert result['actual_count']==3
    rejected=dict(yes);rejected['007']=dict(status='rejected')
    assert select_budget(sparse,rejected)['actual_count']==2
    unknown=dict(yes);unknown['008']=dict(status='uncertain')
    assert '008' not in select_budget(rows,unknown)['selected_frames']
    try:select_budget([dict(rows[0],source_fit_sha256='other')]+rows[1:],yes)
    except AssertionError:pass
    else:raise AssertionError('Mixed runs admitted')
    return dict(controls=8,passed=True,thresholds_changed=False,eligibility_unchanged=True)

if __name__=='__main__':print(verify())
