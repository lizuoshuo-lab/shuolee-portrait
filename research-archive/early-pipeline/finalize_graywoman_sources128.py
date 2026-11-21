"""One CPU-only freeze of root-reviewed128 source12+4, then image-only record015.

No detector, camera solve, fitting, confidence mutation, mask correction or GPU.
The source support booleans describe observable regions, not anatomical truth.
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import inspect
import json
import math
import textwrap

import select_graywoman_half12_sources128 as selector
import publish_officewoman_source_record127 as publisher

ROOT=selector.OUT
RECORD=Path('/mnt/d/00/记录/015')
TRAIN=[4,11,13,22,23,28,34,38,45,51,54,63]
HOLD=[9,25,40,56]
ELBOW_REGION={22,23,25,26,28,30,33,34,38,39,40,41}
GROUP_SOURCE=dict(left_knee=[13],right_knee=[14],left_foot=[15,17,18,19],right_foot=[16,20,21,22],
                  left_wrist=[9],right_wrist=[10],left_elbow=[7],right_elbow=[8])
GROUP_TARGET=dict(left_knee=[13],right_knee=[10],left_foot=[14,19,20,21],right_foot=[11,22,23,24],
                  left_wrist=[7],right_wrist=[4],left_elbow=[6],right_elbow=[3])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def now(): return datetime.now(timezone.utc).isoformat()
def fid(index): return 'frame_%06d'%index


def support(frame):
    elbow=int(frame.rsplit('_',1)[1]) in ELBOW_REGION
    return dict(left_knee=True,right_knee=True,left_foot=True,right_foot=True,
                left_wrist=False,right_wrist=False,left_elbow=elbow,right_elbow=elbow)


def freeze():
    before=ROOT/'plan-before-source-review.json';plan=load(before);crops_path=ROOT/'source-review/source-crops.json';crops=load(crops_path)
    independent_path=ROOT/'source-review/independent-last16-review128.json';independent=load(independent_path)
    assert not (ROOT/'plan.json').exists(),'Frozen source plan exists; do not reselect or overwrite.'
    assert sha(before)=='adb4a79e26bd3c063694eab52cf12304669b9cddae7e06cb70cceed4b4e7385b'
    assert sha(crops_path)=='b7443de6d4e0fc376457b321649c19a163c6703bcb806377428f7ed7ff09f672'
    assert sha(independent_path)=='9089b06ad3a3d1520eb8bb1f234348946766072af60be216e5fb006300ee7db9'
    assert independent['actual_visual_review'] is True and independent['source_only'] is True
    train,held=[fid(i) for i in TRAIN],[fid(i) for i in HOLD]
    assert train==[x['candidate_order'][0] for x in plan['training_candidate_slots']]
    assert held==[x['candidate_order'][0] for x in plan['holdout_reservation']]
    reserved={f for x in plan['holdout_reservation'] for f in x['candidate_order']}
    assert len(reserved)==8 and not set(train)&reserved and set(held)<=reserved
    assert not plan['training_frames'] and not plan['heldout_frames'] and plan['reference_frame'] is None
    source={r['frame']:r for r in plan['source_rows']};cr={r['frame']:r for r in crops['source_rows']}
    candidates=sorted(plan['source_review_template']);assert len(candidates)==32
    for frame in candidates:
        assert sha(source[frame]['image'])==source[frame]['source_sha256'] and sha(source[frame]['mask'])==source[frame]['mask_sha256']
        for item in cr[frame]['paths'].values():assert sha(item['path'])==item['sha256']
    root_path=ROOT/'source-review/root-first16-review128.json'
    first16=candidates[:16];assert first16==[f for x in crops['pages'][:4] for f in x['frames']]
    save(root_path,dict(schema='graywoman128-root-first16-source-review-v1',updated=now(),reviewer='/root',
        record_basis='Root task message: actual contacts00..03 and native1/28/23 head,1/28/63 arms-wrists reviewed before this receipt; this writer does not impersonate that review.',
        actual_visual_review=True,source_only=True,fitting_residuals_used=False,
        source_draft_sha256=sha(before),source_crops_sha256=sha(crops_path),reviewed_frame_ids=first16,
        reviewed_pages=[dict(path=x['path'],sha256=x['sha256'],frames=x['frames']) for x in crops['pages'][:4]],
        additional_native_review=[dict(frame=fid(f),kind=k,**cr[fid(f)]['paths'][k]) for f,k in [(1,'head'),(28,'head'),(23,'head'),(1,'arms_wrists'),(28,'arms_wrists'),(63,'arms_wrists')]],
        findings=dict(mask_long_shadow_gross_leak_observed=False,subpixel_mask_boundary_accepted=False,
            wrists='Hands overlap behind lower back; neither anatomical wrist is independently localized.',
            elbows='Near-front sleeve elbow regions may retain estimated body observations; profile side/center unsupported.',
            feet='Both shoe regions in raster, cuffs cover ankles and far shoes partly self-occlude; not each foot landmark ground truth.',
            reference='23 near-frontal/three-quarter native face, both eyes open and clear; not first frame.',
            unknown_item='Rear-hip gray/beige rectangular piece: unknown person-adjacent accessory/object/pocket, not certified equipment-free.')))
    final_path=ROOT/'source-review/final-source-review.json';selected=sorted(train+held)
    source_reviews=[]
    for frame in candidates:
        n=int(frame.rsplit('_',1)[1]);groups=support(frame)
        source_reviews.append(dict(frame=frame,source_image=source[frame]['image'],source_sha256=source[frame]['source_sha256'],
            source_mask=source[frame]['mask'],source_mask_sha256=source[frame]['mask_sha256'],
            body_source_supported=True,mask_source_supported=True,face_region_is_observed=True,visible_eyes_open=True,
            source_joint_group_supported=groups,physical_joint_ground_truth=False,
            eye_scope='Visible eye(s) open; profiles do not establish far-eye ground truth.',
            arm_scope=('Left/right clothed sleeve-elbow regions source-localizable near front; detector-estimated anatomical centers remain uncertain.' if groups['left_elbow'] else 'Profile/overlap prevents independent bilateral sleeve-elbow localization; preserve unsupported/unknown.'),
            wrist_scope='Both wrists unsupported in all candidates: hands overlapped/occluded behind lower back; fingers unknown.',
            feet_scope='Knee and shoe regions in raster; pants/cuffs hide exact joints, far shoe self-occlusion possible. Not all ankle/toe/heel points ground truth.',
            mask_long_shadow_gross_leak_observed=False,mask_subpixel_boundary_accepted=False,
            person_adjacent_item='Gray/beige rear-hip rectangular object/accessory/pocket category unknown.' if n>=51 else 'Not independently categorized; no equipment-free certification.',
            source_view_usable=True,review_basis=str(root_path if frame in first16 else independent_path)))
    save(final_path,dict(schema='graywoman128-final-source-review-v1',updated=now(),actual_source_visual_review=True,
        source_only=True,source_draft_sha256=sha(before),source_crops_sha256=sha(crops_path),
        root_first16_review=dict(path=str(root_path),sha256=sha(root_path)),
        independent_last16_review=dict(path=str(independent_path),sha256=sha(independent_path)),
        finite_candidate_count=32,source_review_records=source_reviews,training_frames=train,heldout_frames=held,
        all_heldout_reserved_frames=sorted(reserved),reference_frame=fid(23),
        region_support_adjudication='Root final policy supports specified near-front clothed sleeve-elbow REGIONS. Independent last16 exact anatomical-center uncertainty remains preserved; no physical joint truth or subpixel assurance follows.',
        original_XY_NPZ_masks_and_source_pixels_unchanged=True,fit_residuals_used=False,
        source_quality_entrance_only=True,quality_accepted=False,gaussian_admitted=False,GPU_or_fitting_used=False))
    face={f:True for f in selected};groups={f:support(f) for f in selected}
    def row(frame):
        r=copy.deepcopy(source[frame]);b=r['initial_box_xyxy']
        r.update(face_region_is_observed=True,visible_eyes_open=True,source_joint_group_supported=groups[frame],
                 foot_out_of_frame=False,source_visibility_review=str(final_path),
                 source_body_crop_xyxy=[max(0,math.floor(b[0])-24),max(0,math.floor(b[1])-24),min(1520,math.ceil(b[2])+24),min(2704,math.ceil(b[3])+24)])
        return r
    features={r['frame']:r for r in plan['features']};eligible=[features[f] for f in train if f!=train[0]];ranked=[]
    for r in eligible:
        a=sum(t['face_sharpness']>r['face_sharpness'] for t in eligible);b=sum(t['raw_body_score']>r['raw_body_score'] for t in eligible)
        ranked.append(dict(frame=r['frame'],source_quality_numerator=3*a+2*b,source_quality_denominator=5*(len(eligible)-1),face_sharpness=r['face_sharpness'],raw_body_score=r['raw_body_score']))
    ranked.sort(key=lambda r:(r['source_quality_numerator'],r['frame']));assert ranked[0]['frame']==fid(23)
    plan.update(updated=now(),state='frozen_new_person12_plus4_source_roster_ready',training_frames=train,frames=train,
        heldout_frames=held,review_only_frames=held,rows=[row(f) for f in train],heldout_rows=[row(f) for f in held],
        reference_frame=fid(23),reference_order=ranked,reference_source_eyes_open=True,reference_native_face=cr[fid(23)]['paths']['head']['path'],
        reference_source_finding='Source23 is actual native clear/open-eyes near-front/three-quarter and minimum frozen source .6/.4 rank, excluding first train4. Output reference identity, not permanently fixed shape or old pose initializer.',
        face_visibility=face,joint_group_visibility=groups,source_review_records=source_reviews,
        actual_source_review=True,actual_source_visual_review=True,source_mask_actual_visual_review=True,
        training_roster_ready=True,static_gesture_plausible=True,static_gesture_note='Standing gesture with hands held behind lower back is broadly stable; small head/body motion remains. Not physical static certification.',
        actual_source_review_path=str(final_path),visual_review_path=str(final_path),source_review_sha256=sha(final_path),
        source_rejected_frames=[],no_fallback_required=True,source_images=[source[f]['image'] for f in selected],
        source_mask_finding='No gross long-shadow inclusion seen across actual32 contacts; raw masks retained. Hair/cuffs/near-object boundaries not certified subpixel.',
        source_body_review='Root contacts00..03 plus selected native head/arms; independent agent contacts04..07 and native patches. Region support is reviewed input gating, not exact physical joint ground truth.',
        known_unobserved='Rear surface, both hidden wrists/fingers, profile elbow identity, exact joints under clothing, accessory category, metric calibration.',
        source_coverage='Qualitative side/front/opposite-side half-circle, no rear;12time bins do not certify180degrees or equal yaw.',
        confidence_normalization_split=dict(train_count=12,held_count=4,statistics_train_only=True,source_visibility_on_score_copy_before_q90=True,heldout_uses_frozen_train_q90=True,
            face_source_indices_to_zero_if_false=list(range(5))+list(range(40,91)),joint_groups_source_indices=GROUP_SOURCE,joint_groups_target_indices=GROUP_TARGET,original_formula_unchanged=True),
        holdout_roles={f:dict(fit_excluded=True,training_statistics_excluded=True,physical_joint_ground_truth=False,source_entrance_reviewed=True,face_region_is_observed=True,source_joint_group_supported=groups[f],scope='Current-fit-excluded source pixels/observations; not an unseen independent video or3D truth.') for f in held},
        holdout_history='New person source observations are entrance-screened before fitting. Held4 and all8 reserve IDs never enter this fit or q90; not new-video generalization.',
        hand_policy='All original hand2D-data stage weights stay zero. Unsupported BODY25 wrist/elbow observations gate a scores COPY only; body/hand pose priors remain.',
        new_fit_calls=0,new_gaussian_steps=0,fitting_started=False,training_started=False,gaussian_admitted=False,quality_accepted=False,end_to_end_15min_accepted=False,
        original_source_points_or_confidences_edited=False,source_masks_edited=False,
        record_rules='D:/00/记录/015: same-frame native integer crop image composites only, no embedded text. No fabricated3D panel.')
    for frame,item in plan['source_review_template'].items():
        item.update(body_source_supported=True,mask_source_supported=True,face_region_is_observed=True,visible_eyes_open=True,reason='Actual source review; region support only, exact physical landmarks unknown.',shadow_leak_observed=False)
        for name,value in support(frame).items():item[name+'_source_supported']=value
    save(ROOT/'plan.json',plan)
    print('GRAYWOMAN128_SOURCE12_PLUS4_FROZEN',sha(ROOT/'plan.json'),fid(23))


def publish():
    path=ROOT/'plan.json';p=load(path);digest=sha(path)
    assert len(p['training_frames'])==12 and len(p['heldout_frames'])==4 and p['reference_frame']==fid(23)
    source=textwrap.dedent(inspect.getsource(publisher.main))
    replacements={
        "assert plan_sha=='96fae64878e672c42e6b8145f7850e319994bb2ed93327d6db6eb406871c6ded'":"assert plan_sha==EXPECTED_PLAN_SHA",
        'record019-source-copy-receipt127.json':'record015-source-copy-receipt128.json',
        'assert len(selected)==32':'assert len(selected)==16',
        "box=crops[frame]['paths']['body']['crop_original_xyxy']":"box=row['source_body_crop_xyxy']",
        '办公室新人127':'灰衣新人128',
        'files_copied=64':'files_copied=32',
        'RECORD019_SOURCE_IMAGES64_COPIED':'RECORD015_SOURCE_IMAGES32_COPIED',
    }
    for old,new in replacements.items():
        expected=2 if old=='办公室新人127' else 1
        assert source.count(old)==expected,(old,source.count(old))
        source=source.replace(old,new)
    namespace=dict(publisher.__dict__,ROOT=ROOT,RECORD=RECORD,EXPECTED_PLAN_SHA=digest)
    exec(compile(source,str(__file__)+'::frozen_image_only_publisher','exec'),namespace)
    namespace['main']()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','publish']);args=p.parse_args()
    if args.mode=='freeze':freeze()
    else:publish()
