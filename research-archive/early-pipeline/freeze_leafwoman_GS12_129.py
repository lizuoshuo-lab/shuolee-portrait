"""One measured subset of the same fit24; visible-region1000 exploration only."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
FIT=ROOT/'leafwoman-half12-pose129';INPUT=ROOT/'leafwoman-half12-input129'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    plan=read(FIT/'plan.json');source=read(INPUT/'plan.json');r=read(FIT/'result.json')
    assert read(FIT/'process-exit.json')['exit_code']==0
    assert sha(FIT/'selected-fit.npz')==r['fit_sha256']=='d6e98d30409f2886279676240e144ef36cd94e1ea1865a94818c9f14682f4b5e'
    assert sha(INPUT/'plan.json')=='0115b08fed63b9c93042ee00f985dedf81f2634ead26fac0c79ea78044ff2373'
    measure_path=FIT/'postfit-measurements-v2.json'
    assert sha(measure_path)=='4476ba790a96101221112bd8f69b08a67b8b2bb278fd14c5c1a73e378e25d91d'
    m=read(measure_path);ids=m['selection_draft']['selected_candidate_frames']
    assert len(ids)==len(set(ids))==12 and set(ids)<=set(source['training_frames'])
    reserved={f for slot in source['holdout_reservation'] for f in slot['candidate_order']}
    assert not set(ids)&reserved
    stamp=datetime.now(timezone.utc).isoformat()
    pages=[FIT/f'review/page-{i:02d}.png' for i in range(1,7)]
    native=[Path('D:/00/记录/022')/f'smpl-x_持叶候选24帧129_frame_{i:06d}_{region}_原图_当前mesh.png'
            for i,region in [(4,'face'),(4,'arms'),(24,'arms'),(37,'arms'),(28,'face')]]
    review=dict(updated=stamp,actual_visual_review=True,reviewer='/root',all24_native_fourpanels_and_four_synthetic_sides_reviewed=True,
        actual_image_hashes={str(p):sha(p) for p in pages+native},source_plan_sha256=sha(INPUT/'plan.json'),
        saved_fit_sha256=r['fit_sha256'],review_generation_sha256=sha(FIT/'review/result.json'),
        findings=['Both forearms and hands stay in the broad chest-held-prop region; no128-style thigh-front dropped arms.',
            'Wrist shifts, spread/intersecting fingers and face/eye/mouth-width errors remain. Hand2D loss remains original uniform0, not repaired.',
            'Bare mesh does not cover hat, shawl or long skirt. This is not a clothing-surface fit or subpixel face acceptance.',
            'All synthetic-body meshes lean in the body-view gauge; not evidence of recovered gravity or true video trajectory.',
            'Knees/feet are unobserved under skirt/outside native frame; no foot crop or negative supervision is invented.',
            'Source1/2/4 wrists/elbows are unsupported; lower RMS has fewer supported joints and does not establish more accurate hand geometry.'],
        reference_warning='Frozen proxy ranking may favor fewer supported observations; output reference is a label, not a new pose seed or physical quality acceptance.',
        per_frame={f:dict(status='exploratory_visible_region_only',actual_review=True,physical_pose_accepted=False) for f in source['training_frames']},
        Gaussian_quality_accepted=False,face_subpixel_accepted=False,physical_calibration_accepted=False,
        exploratory_gaussian_admitted=True,initial_Gaussian_max_step=1000,continue_to18000_admitted=False)
    save(FIT/'root-visual-review.json',review)
    conclusion=dict(updated=stamp,actual_visual_review=True,exploratory_gaussian_admitted=True,
        max_gaussian_steps_before_review=1000,continue_to18000_admitted=False,
        body_region_scope='Observed upper body and skirt image coverage only; hands remain imperfect, lower joints unknown.',
        saved_fit_sha256=r['fit_sha256'],root_review_sha256=sha(FIT/'root-visual-review.json'),
        known_errors_preserved=True,quality_accepted=False,physical_face_or_pose_accepted=False,
        opacity_causal_candidate_admitted=False,stop='Actual1000-step native RGB/alpha/mesh and unknown-direction review before continuation.')
    save(FIT/'conclusion.json',conclusion)
    candidates=[x for x in m['records'] if x['frame'] in ids and x['face_region_is_observed'] and x['visible_eyes_open']]
    assert candidates
    reference=sorted(candidates,key=lambda x:(x['supported_body_rms_px'],x['supported_face_rms_px'] if x['supported_face_rms_px'] is not None else float('inf'),-x['face_sharpness'],x['frame']))[0]['frame']
    folder=FIT/'gs-selection';assert not folder.exists();folder.mkdir()
    selection=dict(updated=stamp,training_frames=ids,reference_frame=reference,
        fit_sha256=r['fit_sha256'],source_plan_sha256=sha(FIT/'plan.json'),input_source_plan_sha256=sha(INPUT/'plan.json'),
        heldout_frames=source['heldout_frames'],excluded_holdout_reserves=sorted(reserved),
        postfit_measurements=str(measure_path),postfit_measurements_sha256=sha(measure_path),
        original_fit_candidate_count=24,selected_Gaussian_count=12,all24_still_influence_shared_shape_and_pose=True,
        fit_reference_frame=source['reference_frame'],reference_changed_only_as_output_label=True,
        root_visual_review_sha256=sha(FIT/'root-visual-review.json'),fit_conclusion_sha256=sha(FIT/'conclusion.json'),
        source_masks_or_XY_or_NPZ_edited=False,shared_betas_or_pose_refit=False,heldout_used_for_selection=False,
        ranking='Preregistered12realPTS bins; within-bin active supportedBODY25 RMS, faceRMS, distance to center, ID. Fixed original loss-zero1/9/12 measurement correction preserved as v2.',
        reference_rule=source['postfit_selection_protocol']['reference'],
        reference_supported_observation_warning='Reference4 has no directly supported wrists/elbows. It wins the frozen proxy with fewer observed body joints; no hand-depth or final quality claim.',
        admitted_scope='1000-step visible-region exploration only',continue_to18000_admitted=False,quality_accepted=False)
    save(folder/'selection.json',selection)
    print('LEAFWOMAN129_GS12_FROZEN_DIAGNOSTIC1000_ONLY',ids,reference,sha(folder/'selection.json'))


if __name__=='__main__':main()
