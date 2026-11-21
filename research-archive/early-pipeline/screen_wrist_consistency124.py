"""Prefit screen only: flag contradictory aliases, never rewrite a target."""
import numpy as np
from select_quality_frames124 import POLICY

PAIRS=(('left',7,25,20),('right',4,46,21))


def screen(mapped,heights,mapper,frame_ids):
    x=np.asarray(mapped,dtype=np.float32)
    h=np.asarray(heights,dtype=np.float64)
    ids=list(frame_ids);m=np.asarray(mapper)
    if x.ndim!=3 or x.shape[1:]!=(118,3) or h.shape!=(len(x),) or len(ids)!=len(x):
        raise ValueError('Expected matched N1183 input, heights and frame identities.')
    if len(set(ids))!=len(ids) or not np.isfinite(x).all() or not np.isfinite(h).all() or np.any(h<=0):
        raise ValueError('Finite observations, positive heights and unique identities required.')
    if np.any(x[:,:,2]<0) or np.any(x[:,:,2]>1) or m.shape!=(118,):
        raise ValueError('Use existing normalized confidence and original118joint mapping.')
    if any(m[b]!=j or m[t]!=j for _,b,t,j in PAIRS):
        raise ValueError('Original wrist aliases changed.')
    weights=x[:,:,2]**2
    rows=[]
    for i,f in enumerate(ids):
        pairs=[];flags=[]
        for side,b,t,j in PAIRS:
            gap=float(np.linalg.norm(x[i,b,:2].astype(float)-x[i,t,:2]))
            both=bool(weights[i,b]>0 and weights[i,t]>0)
            limit=2*POLICY['body_max_over_height']*h[i]
            conflict=bool(both and gap>limit)
            if conflict:flags.append('contradictory_duplicate_'+side+'_wrist')
            pairs.append(dict(side=side,smplx_joint=j,body_index=b,hand_index=t,
                gap_px=gap,simultaneous_tolerance_limit_px=float(limit),
                body_effective_weight=float(weights[i,b]),hand_effective_weight=float(weights[i,t]),
                status='conflict' if conflict else ('consistent_within_tolerance' if both else 'insufficient_duplicate_support'),
                anatomical_correctness_verified=False))
        rows.append(dict(frame=f,screen_flags=flags,pairs=pairs,pose_quality_accepted=False))
    return rows


def apply_to_candidates(records,screen_rows):
    """Append input flags by identity; do not overwrite pose flags or acceptance."""
    evidence={r['frame']:r for r in screen_rows}
    if len(evidence)!=len(screen_rows):raise ValueError('Duplicate screening frame identities.')
    if not all(r['frame'] in evidence for r in records):raise ValueError('Missing prefit screen for a candidate.')
    result=[]
    for row in records:
        r=dict(row);r['screen_flags']=list(dict.fromkeys(list(r.get('screen_flags',[]))+evidence[r['frame']]['screen_flags']))
        r['input_consistency']=evidence[r['frame']];result.append(r)
    return result


def verify(mapper):
    x=np.zeros((4,118,3),np.float32);x[:,:,2]=1
    x[0,25,0]=6;x[1,25,0]=6.01;x[2,25,0]=100;x[2,7,2]=1e-23;x[3,25,0]=100;x[3,7,2]=0
    before=x.copy();ids=['boundary','conflict','underflow','missing']
    rows=screen(x,[100]*4,mapper,ids)
    checks=dict(boundary_allowed=not rows[0]['screen_flags'],conflict_rejected=bool(rows[1]['screen_flags']),
        zero_effective_weight_not_conflict=not rows[2]['screen_flags'] and rows[2]['pairs'][0]['status']=='insufficient_duplicate_support',
        missing_support_explicit=rows[3]['pairs'][0]['status']=='insufficient_duplicate_support',
        original_targets_unchanged=np.array_equal(before,x))
    bigger=x.copy();bigger[:,:,:2]*=4
    checks['scale_invariance']=[r['screen_flags'] for r in screen(bigger,[400]*4,mapper,ids)]==[r['screen_flags'] for r in rows]
    records=[dict(frame='conflict',screen_flags=['existing_pose_failure'],quality_score=.25)]
    combined=apply_to_candidates(records,rows)
    checks['prior_flags_preserved']=combined[0]['screen_flags']==['existing_pose_failure','contradictory_duplicate_left_wrist'] and records[0]['screen_flags']==['existing_pose_failure']
    bad=np.array(mapper);bad[25]=19
    try:screen(x,[100]*4,bad,ids)
    except ValueError:checks['wrong_mapper_rejected']=True
    else:checks['wrong_mapper_rejected']=False
    invalid=x.copy();invalid[0,0,0]=np.nan
    try:screen(invalid,[100]*4,mapper,ids)
    except ValueError:checks['nonfinite_rejected']=True
    else:checks['nonfinite_rejected']=False
    assert all(checks.values()),checks
    return checks
