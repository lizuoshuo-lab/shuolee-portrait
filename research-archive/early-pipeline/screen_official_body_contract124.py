"""Separate the original fitter's supported body points from review-only ones.

This changes a screening measurement, never fitting targets or weights. A
source/side-pose review is still required; a zero fitting weight is not proof
that the corresponding anatomy is correct.
"""
import numpy as np
from select_quality_frames124 import POLICY

BODY=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
IGNORED=(1,9,12)
OLD_FLAG='body_reprojection_over_existing3percent_height'
NEW_FLAG='original_supported_body_reprojection_over_existing3percent_height'


def measure(target,projected,height,executed_joint_weights):
    target=np.asarray(target);projected=np.asarray(projected);jw=np.asarray(executed_joint_weights)
    if target.shape!=(118,3) or projected.shape!=(118,2) or jw.shape!=(118,):
        raise ValueError('Require one native118point observation, projection, and executed joint-weight vector.')
    if not np.isfinite(target).all() or not np.isfinite(projected).all() or not np.isfinite(jw).all() or not np.isfinite(height) or height<=0:
        raise ValueError('Nonfinite or invalid measurement input.')
    expected=np.ones(118);expected[list(IGNORED)]=0
    if not np.array_equal(jw,expected):
        raise ValueError('Only the verified original COCO25 joint-weight contract is supported; do not infer exclusions from residuals.')
    if np.any(target[:,2]<0) or np.any(target[:,2]>1):
        raise ValueError('Require normalized original relative confidence.')
    error=np.linalg.norm(projected-target[:,:2],axis=1)
    supported=BODY[(jw[BODY]>0)&(target[BODY,2]**2>0)]
    diagnostic=BODY[jw[BODY]==0]
    if len(supported)==0:
        raise ValueError('No supported body measurements; missing evidence cannot pass.')
    maximum=float(error[supported].max())
    return dict(contract='original_coco25_loss_support_v1',supported_body_indices=supported.tolist(),
        original_ignored_body_indices=diagnostic.tolist(),
        supported_body_max_px=maximum,supported_body_max_over_height=maximum/height,
        supported_body_gate_failed=maximum/height>POLICY['body_max_over_height'],
        ignored_body_residuals=[dict(index=int(i),error_px=float(error[i]),relative_confidence=float(target[i,2]),
                                    measurement='detector hypothesis, not independently localized anatomy') for i in diagnostic],
        threshold=POLICY['body_max_over_height'],requires_original_image_and_depth_review=True,
        all_observations_retained=True)


def apply(record,measurement):
    out=dict(record)
    flags=[f for f in out.get('screen_flags',[]) if f!=OLD_FLAG]
    if measurement['supported_body_gate_failed']:flags.append(NEW_FLAG)
    out['screen_flags']=flags
    out['original_body_measurement_contract']=measurement
    out['previous_screen_flags']=list(record.get('screen_flags',[]))
    # Keep the previous ranking metric unchanged: gate support is the only change.
    return out


def verify():
    obs=np.zeros((118,3));obs[:,2]=1;uv=np.zeros((118,2));jw=np.ones(118);jw[list(IGNORED)]=0
    before=obs.copy();uv[9,0]=500
    measured=measure(obs,uv,1000,jw)
    assert not measured['supported_body_gate_failed'] and {x['index']:x['error_px'] for x in measured['ignored_body_residuals']}[9]==500
    assert np.array_equal(obs,before)
    uv[3,0]=31;measured=measure(obs,uv,1000,jw);assert measured['supported_body_gate_failed']
    row={'screen_flags':[OLD_FLAG,'feet_bad','ownership_unresolved','temporal_bad'],'body_weighted_rms_px':123.}
    new=apply(row,measured)
    assert set(new['screen_flags'])=={NEW_FLAG,'feet_bad','ownership_unresolved','temporal_bad'}
    assert new['body_weighted_rms_px']==123. and row['screen_flags'][0]==OLD_FLAG
    bad=jw.copy();bad[3]=0
    try:measure(obs,uv,1000,bad)
    except ValueError:pass
    else:raise AssertionError('Arbitrary excluded elbow admitted')
    missing=obs.copy();missing[:,2]=0
    try:measure(missing,uv,1000,jw)
    except ValueError:pass
    else:raise AssertionError('Absent observations admitted')
    return dict(controls=6,passed=True,scope='Ignored residual retained; shoulder/elbow, feet, ownership and temporal guards preserved; original inputs unchanged; arbitrary exclusions and absent evidence rejected.')


if __name__=='__main__':print(verify())
