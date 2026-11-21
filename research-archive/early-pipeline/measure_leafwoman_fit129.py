"""Read-only CPU measurements of one completed129 official fit24.

Audit uses synthetic arrays only. Measure requires actual exit0 and writes only
an explicitly requested NEW measurement/draft file, never final GS selection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import select_leafwoman_half12_sources129 as selector

ROOT = selector.ROOT
FIT = ROOT/'leafwoman-half12-pose129'
INPUT = ROOT/'leafwoman-half12-input129'
INPUT_SHA = '0115b08fed63b9c93042ee00f985dedf81f2634ead26fac0c79ea78044ff2373'
SELECTOR_SHA = '8571cffee9ba56a134bf246cdd318cca97720c8818bc92f0526cd6bc04806e1f'
BODY = np.arange(25)
FACE = np.arange(67,118)
FACE_VISIBILITY_TARGET = np.r_[0,np.arange(15,19),FACE]
GROUP_TARGET = dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24],
                    left_wrist=[7],right_wrist=[4],left_elbow=[6],right_elbow=[3])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def region_measurement(projected,observed,indices):
    """No residual rejection: support is exactly saved confidence>0."""
    projected=np.asarray(projected,dtype=np.float64)
    observed=np.asarray(observed,dtype=np.float64)
    ids=np.asarray(indices,dtype=np.int64)
    assert projected.shape==(118,2) and observed.shape==(118,3)
    assert np.isfinite(projected).all() and np.isfinite(observed).all()
    assert (observed[:,2]>=0).all() and (observed[:,2]<=1).all()
    errors=np.linalg.norm(projected[ids]-observed[ids,:2],axis=1)
    confidence=observed[ids,2]
    weights=confidence**2
    supported=weights>0
    count=int(supported.sum());weight_sum=float(weights.sum())
    return dict(rms_px=float(np.sqrt(np.sum(weights*errors**2)/weight_sum)) if count else None,
        count=count,weight_sum=weight_sum,supported_indices=ids[supported].tolist(),
        supported_max_px=float(errors[supported].max()) if count else None,
        empty_support_is_null=not bool(count),residual_rejection_performed=False)


def verify_saved_visibility(observed,row):
    face=row['face_region_is_observed'];groups=row['source_joint_group_supported']
    assert type(face) is bool and set(groups)==set(GROUP_TARGET)
    assert all(type(v) is bool for v in groups.values())
    if not face:assert np.all(observed[FACE_VISIBILITY_TARGET,2]==0)
    for name,ids in GROUP_TARGET.items():
        if not groups[name]:assert np.all(observed[ids,2]==0),(row['frame'],name)


def measurement_rows(ids,projected,observed,rows,features,digest):
    assert len(ids)==len(set(ids))==24
    assert projected.shape==(24,118,2) and observed.shape==(24,118,3)
    assert set(ids)==set(rows) and set(ids)<=set(features)
    result=[]
    for i,frame in enumerate(ids):
        row=rows[frame];verify_saved_visibility(observed[i],row)
        body=region_measurement(projected[i],observed[i],BODY)
        face=region_measurement(projected[i],observed[i],FACE)
        result.append(dict(frame=frame,time_seconds=row['time_seconds'],source_fit_sha256=digest,
            supported_body_rms_px=body['rms_px'],supported_body_count=body['count'],
            supported_body_weight_sum=body['weight_sum'],supported_body_indices=body['supported_indices'],
            supported_body_max_px=body['supported_max_px'],
            supported_face_rms_px=face['rms_px'],supported_face_count=face['count'],
            supported_face_weight_sum=face['weight_sum'],supported_face_indices=face['supported_indices'],
            supported_face_max_px=face['supported_max_px'],
            face_sharpness=features[frame]['face_sharpness'],face_region_is_observed=row['face_region_is_observed'],
            visible_eyes_open=row['visible_eyes_open'],source_joint_group_supported=row['source_joint_group_supported'],
            side_review_status='unreviewed',actual_visual_review=False,physical_joint_ground_truth=False))
    return result


def measure():
    """Call only after successful fit; no source observations/held poses read."""
    assert sha(selector.__file__)==SELECTOR_SHA
    paths=[FIT/'process-exit.json',FIT/'result.json',FIT/'plan.json',FIT/'selected-fit.npz',INPUT/'plan.json']
    assert all(p.is_file() for p in paths),'Actual completed fit files missing; do not measure a partial run.'
    exit_record=load(paths[0]);assert exit_record['exit_code']==0,'Actual fit must exit0 first.'
    hashes={str(p):sha(p) for p in paths}
    result,plan,source=load(paths[1]),load(paths[2]),load(paths[4])
    assert hashes[str(INPUT/'plan.json')]==INPUT_SHA
    assert result['fit_sha256']==hashes[str(FIT/'selected-fit.npz')] and result['frames_count']==24
    if exit_record.get('result_sha256'):assert exit_record['result_sha256']==hashes[str(FIT/'result.json')]
    assert Path(plan['source_selection'])==INPUT/'plan.json'
    assert plan['frames']==source['training_frames'] and len(plan['frames'])==24
    assert plan['review_only_excluded']==source['heldout_frames'] and len(source['heldout_frames'])==4
    reserve=[f for r in source['holdout_reservation'] for f in r['candidate_order']]
    assert len(reserve)==len(set(reserve))==8 and not set(reserve)&set(plan['frames'])
    selector_bindings=[v for k,v in source['source_hashes'].items() if k.endswith('/select_leafwoman_half12_sources129.py')]
    assert selector_bindings==[SELECTOR_SHA]
    rows={r['frame']:r for r in plan['rows']};assert set(rows)==set(plan['frames'])
    source_rows={r['frame']:r for r in source['rows']};assert set(rows)==set(source_rows)
    for frame,row in rows.items():
        for k in ('K_integer_centers','face_region_is_observed','visible_eyes_open','source_joint_group_supported'):
            assert row[k]==source_rows[frame][k],(frame,k)
    # Read only this archive's24 training observations, never raw/held NPZ rows.
    with np.load(FIT/'selected-fit.npz',allow_pickle=False) as z:
        ids=z['frame_ids'].astype(str).tolist();assert ids==plan['frames']
        projected=z['joints_projected_native'].copy()
        observed=z['observed_native_keypoints'].copy()
        K=z['K_integer_centers'].copy();assert K.shape==(3,3)
        assert all(np.array_equal(K,np.asarray(r['K_integer_centers'])) for r in rows.values())
    features={r['frame']:r for r in source['features'] if r['frame'] in set(ids)}
    records=measurement_rows(ids,projected,observed,rows,features,hashes[str(FIT/'selected-fit.npz')])
    draft=selector.postfit_shortlist(records,ids,reserve,source['fixed_time_range_seconds'])
    assert draft['gaussian_admitted'] is False
    assert all(sha(p)==hashes[str(p)] for p in paths),'Input changed during measurement.'
    return dict(updated=datetime.now(timezone.utc).isoformat(),state='measurements_and_preregistered_ranking_draft_only',
        subject='leafwoman',helper_sha256=sha(__file__),selector_sha256=SELECTOR_SHA,
        source_plan=str(FIT/'plan.json'),source_plan_sha256=hashes[str(FIT/'plan.json')],
        input_source_plan=str(INPUT/'plan.json'),input_source_plan_sha256=INPUT_SHA,
        source_fit_sha256=hashes[str(FIT/'selected-fit.npz')],source_hashes=hashes,
        training_frame_count=24,heldout_frame_count=4,heldout_or_backup_observations_read=False,
        frame_ids=ids,records=records,selection_draft=draft,reference_frame=None,
        metric_contract='Native integer-center saved joints_projected_native minus observed XY; Euclidean distance; weights=saved confidence squared. BODY25[0:25], face[67:118]. Saved source/raster confidence gates preserved. No residual threshold, trimming or point deletion.',
        body25_original_loss_ignored_indices=[1,9,12],
        metric_limit='BODY25 report includes original-loss-ignored neck1/hips9,12 if their saved confidence is positive. This descriptive detector-agreement RMS is not the original loss, physical joint/material error, pose-depth proof or geometry acceptance.',
        shared_fit_scope='All24 influence shared shape/body pose; GSabout12 is a subset of this one archive, not an independent12-frame fit. Held4 have no fitted poses and are not geometric validation.',
        actual_visual_review=False,reference_requires_selected_source_open_eyes_and_actual_side_review=True,
        final_selection_created=False,gaussian_admitted=False,quality_accepted=False,
        input_or_fit_archives_changed=False,GPU_started=False,new_fit_started=False)


def audit():
    """Pure in-memory contract checks. Does not open FIT or create any output."""
    assert sha(selector.__file__)==SELECTOR_SHA
    observed=np.zeros((118,3));projected=np.zeros((118,2))
    observed[2,2]=1.;projected[2]=[3.,4.]
    observed[3,2]=.5;projected[3]=[0.,10.]
    untouched=observed.copy();b=region_measurement(projected,observed,BODY)
    assert b['count']==2 and b['weight_sum']==1.25 and abs(b['rms_px']-np.sqrt(40.))<1e-12
    assert b['supported_indices']==[2,3] and np.array_equal(observed,untouched)
    f=region_measurement(projected,observed,FACE)
    assert f['count']==0 and f['weight_sum']==0 and f['rms_px'] is None and f['supported_max_px'] is None
    # A huge residual on zero-confidence/unsupported joint must not alter RMS;
    # a huge supported residual must remain, rather than being trimmed away.
    projected[4]=[1e8,1e8];assert region_measurement(projected,observed,BODY)['rms_px']==b['rms_px']
    observed[4,2]=1.;assert region_measurement(projected,observed,BODY)['supported_max_px']>1e8
    bad=observed.copy();bad[4,2]=-1
    try:region_measurement(projected,bad,BODY)
    except AssertionError:pass
    else:raise AssertionError('Negative confidence must fail closed.')
    ids=[f'frame_{i+1:06d}' for i in range(24)]
    rows={f:dict(frame=f,time_seconds=i/3,face_region_is_observed=True,visible_eyes_open=True,
                  source_joint_group_supported={k:True for k in GROUP_TARGET}) for i,f in enumerate(ids)}
    features={f:dict(face_sharpness=1.) for f in ids}
    records=measurement_rows(ids,np.zeros((24,118,2)),np.zeros((24,118,3)),rows,features,'synthetic_single_fit')
    assert len(records)==24 and all(r['supported_body_rms_px'] is None and r['supported_face_rms_px'] is None for r in records)
    unsupported=dict(rows[ids[0]],source_joint_group_supported={k:False for k in GROUP_TARGET})
    invalid=np.zeros((118,3));invalid[7,2]=1.
    try:verify_saved_visibility(invalid,unsupported)
    except AssertionError:pass
    else:raise AssertionError('Unsupported saved wrist cannot carry confidence.')
    source=selector.audit()
    return dict(CPU_synthetic_contract_passed=True,helper_sha256=sha(__file__),selector_static_contract=source,
        RMS_formula_checked=True,confidence_squared_checked=True,empty_support_null_checked=True,
        zero_confidence_outlier_ignored=True,supported_outlier_preserved=True,all24_rows_schema_checked=True,
        unsupported_saved_visibility_rejected=True,inputs_unmodified=True,
        actual_fit_read=False,output_created=False,GS_selection_created=False,GPU_started=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','measure']);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.mode=='audit':
        assert args.output is None,'Audit creates no files.';print(json.dumps(audit(),ensure_ascii=False,indent=2))
    else:
        assert args.output is not None,'Specify a NEW measurement/draft JSON path explicitly.'
        assert not args.output.exists(),'Never overwrite evidence or an existing selection.'
        data=measure();assert args.output.parent.is_dir(),'Do not create a GS-selection/fit directory here.'
        with args.output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),records=len(data['records']),
            draft_frames=data['selection_draft']['selected_candidate_frames'],reference_frame=None,
            gaussian_admitted=False,quality_accepted=False),ensure_ascii=False))
