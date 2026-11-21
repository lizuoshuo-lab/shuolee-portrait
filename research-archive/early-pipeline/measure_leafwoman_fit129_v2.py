"""129 CPU measurement-contract correction; frozen v1 code/JSON preserved.

Only BODY25's original fixed zero-loss indices1/9/12 are excluded. No fit,
archive, confidence, observedXY, face metric, bin or admission rule changes.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import measure_leafwoman_fit129 as v1

V1_CODE_SHA='2768d29991094de6c62b9f7cd1620d76c12527c49bec22e60947e613aaec283c'
V1_RESULT_SHA='caa0d3eb11544ccca61d28044f066a5302f846c813bf6b00dddd6b86a3a8987d'
IGNORED=(1,9,12)
BODY_ACTIVE=np.asarray([i for i in range(25) if i not in IGNORED],dtype=np.int64)
V1_RESULT=v1.FIT/'postfit-measurements-draft.json'
OUT=v1.FIT/'postfit-measurements-v2.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_contract():
    """Verify frozen executed sources, not a new data/error-derived mask."""
    plan=v1.load(v1.FIT/'plan.json')
    suffixes=['/official_smplifyx_entry123.py','/official/cfg_files/fit_smplx.yaml',
              '/official/smplifyx/fitting.py','/official/smplifyx/fit_single_frame.py',
              '/smplify_selected_views124.py']
    files={}
    for suffix in suffixes:
        matches=[(Path(p),h) for p,h in plan['source_hashes'].items() if p.endswith(suffix)]
        assert len(matches)==1,suffix
        p,expected=matches[0];actual=sha(p);assert actual==expected,(p,actual,expected)
        files[suffix]=dict(path=str(p),sha256=actual,matches_executed_fit_plan=True)
    entry=Path(files[suffixes[0]]['path']).read_text(encoding='utf-8-sig')
    config=Path(files[suffixes[1]]['path']).read_text(encoding='utf-8-sig')
    loss=Path(files[suffixes[2]]['path']).read_text(encoding='utf-8-sig')
    phases=Path(files[suffixes[3]]['path']).read_text(encoding='utf-8-sig')
    assert 'jw[:,[1,9,12]] = 0' in entry
    assert re.search(r'joints_to_ign:\s+- 1\s+- 9\s+- 12',config)
    assert re.search(r'use_joints_conf:\s*True',config)
    assert 'weights = (joint_weights * joints_conf' in loss and 'torch.sum(weights ** 2 * joint_diff)' in loss
    assert "joint_weights[:, 25:67] = curr_weights['hand_weight']" in phases
    assert "joint_weights[:, 67:] = curr_weights['face_weight']" in phases
    # A read-only list of the actual phase assignments guards against a new
    # phase secretly re-enabling the ignored BODY25 indices.
    assignments=[line.strip() for line in phases.splitlines() if re.search(r'joint_weights\s*\[.*\]\s*=',line)]
    assert assignments==["joint_weights[:, 25:67] = curr_weights['hand_weight']",
                         "joint_weights[:, 67:] = curr_weights['face_weight']"]
    return dict(ignored_body25_indices=list(IGNORED),active_body25_indices=BODY_ACTIVE.tolist(),
        fixed_before_final_GS_selection=True,source_hashes=files,phase_joint_weight_assignments=assignments,
        config_joints_to_ign=[1,9,12],entry_zero_weight_indices=[1,9,12],
        original_data_weight_formula='(joint_weights * joints_conf)^2; robustifier and stage data scalar remain original.',
        line_evidence={'entry_zero_and_fit_argument':[214,217], 'official_config_ignore':[2,5],
                       'official_config_use_joints_conf':16,'original_loss_weights_and_joint_loss':[371,379],
                       'phase_hand_and_face_only':[420,422]})


def measure():
    assert sha(v1.__file__)==V1_CODE_SHA and sha(V1_RESULT)==V1_RESULT_SHA
    previous=v1.load(V1_RESULT);contract=weight_contract()
    before=v1.BODY.copy()
    try:
        # In-memory parameter to the unchanged read-only measurement core.
        # Neither v1 source nor any archive is rewritten.
        v1.BODY=BODY_ACTIVE.copy()
        data=v1.measure()
    finally:v1.BODY=before
    assert data['source_fit_sha256']==previous['source_fit_sha256']
    assert data['source_plan_sha256']==previous['source_plan_sha256']
    assert data['input_source_plan_sha256']==previous['input_source_plan_sha256']
    old={r['frame']:r for r in previous['records']}
    face_keys=['supported_face_rms_px','supported_face_count','supported_face_weight_sum',
               'supported_face_indices','supported_face_max_px']
    for row in data['records']:
        assert all(row[k]==old[row['frame']][k] for k in face_keys),'Face measurement must not change.'
        assert not set(row['supported_body_indices'])&set(IGNORED)
        assert row['side_review_status']=='unreviewed'
    assert sha(v1.__file__)==V1_CODE_SHA and sha(V1_RESULT)==V1_RESULT_SHA
    data.update(updated=datetime.now(timezone.utc).isoformat(),measurement_version=2,
        state='fixed_original_weight_contract_corrected_measurements_and_ranking_draft_only',
        measurement_core_helper_sha256=V1_CODE_SHA,helper_sha256=sha(__file__),
        preserved_v1_helper=str(Path(v1.__file__)),preserved_v1_helper_sha256=V1_CODE_SHA,
        preserved_v1_result=str(V1_RESULT),preserved_v1_result_sha256=V1_RESULT_SHA,
        correction_contract=contract,
        correction_reason='V1 treated all saved positive BODY25 confidences as active support; original executed loss permanently zeroes neck1/hips9/12. The hidden hip detector hypotheses therefore dominated v1 body RMS despite not contributing to fitting. Correct this reporting/support contract before final GS selection; this is not a fitted-parameter, confidence, residual-threshold or loss change.',
        metric_contract='Native saved projection minus observed XY; Euclidean residual; unchanged saved confidence squared. BODY25 fixed original active indices[0:25] excluding[1,9,12], and face[67:118] unchanged. Existing source/raster confidence gates preserved; empty support=null. No residual threshold, trimming or per-frame index selection.',
        metric_limit='Fixed-active supported detector-agreement RMS, not the original robust/stage-scaled training loss, physical material/joint error, pose-depth proof or geometry acceptance. Remaining positive source-region confidence is not independent anatomical ground truth.',
        face_measurements_bit_equal_to_v1=True,source_fit_archive_same_as_v1=True,
        source_plan_same_as_v1=True,source_XY_confidence_and_masks_unchanged=True,
        selector_bin_tie_and_threshold_logic_unchanged=True,final_GS_selection_precedes_correction=False,
        correction_is_outcome_threshold_scan=False,reference_frame=None,gaussian_admitted=False,
        final_selection_created=False,quality_accepted=False)
    return data


def audit():
    assert sha(v1.__file__)==V1_CODE_SHA
    base=v1.audit();assert BODY_ACTIVE.tolist()==[0,2,3,4,5,6,7,8,10,11,13,14,15,16,17,18,19,20,21,22,23,24]
    obs=np.zeros((118,3));proj=np.zeros((118,2))
    obs[[1,9,12],2]=1.;proj[[1,9,12]]=[1e8,1e8]
    obs[2,2]=.5;proj[2]=[3.,4.]
    active=v1.region_measurement(proj,obs,BODY_ACTIVE)
    assert active['rms_px']==5. and active['count']==1 and active['weight_sum']==.25
    assert v1.region_measurement(proj,obs,np.arange(25))['rms_px']>1e8
    obs[2,2]=0.;assert v1.region_measurement(proj,obs,BODY_ACTIVE)['rms_px'] is None
    return dict(CPU_synthetic_contract_passed=True,helper_sha256=sha(__file__),v1_core_sha256=V1_CODE_SHA,
        active_body25_indices=BODY_ACTIVE.tolist(),fixed_original_ignore_indices=list(IGNORED),
        ignored_residual_magnitude_cannot_change_active_RMS=True,face_and_core_unchanged=True,
        v1_synthetic_contract=base,actual_fit_read=False,output_created=False,GPU_started=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['audit','measure']);a=parser.parse_args()
    if a.mode=='audit':print(json.dumps(audit(),ensure_ascii=False,indent=2))
    else:
        assert not OUT.exists(),'Keep every previous receipt; never overwrite.'
        data=measure()
        with OUT.open('x',encoding='utf-8') as stream:stream.write(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),helper_sha256=sha(__file__),
            draft_frames=data['selection_draft']['selected_candidate_frames'],reference_frame=None,
            gaussian_admitted=False,quality_accepted=False),ensure_ascii=False,indent=2))
