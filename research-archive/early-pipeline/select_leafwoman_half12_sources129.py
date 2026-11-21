"""129 CPU source shortlist only: fit24 candidates, held4, later GS about12.

Reuse frozen128 source ranks/bins and127 native feature extraction. No detector,
fitter, GPU, source editing or final roster admission. Prepare is explicit only.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import textwrap

import select_graywoman_half12_sources128 as previous
import select_officewoman_circle24_sources127 as extractor

ROOT = previous.ROOT
SOURCE = Path('/opt/gs-slam-lab/shuolee-cross-subject-129/leafwoman')
OUT = ROOT/'leafwoman-half12-input129'
SOURCE_REVIEW = ROOT/'leafwoman-native129/source-pose-review'
VIDEO_SHA = 'c5aac80e574dba892137d59186ff17713ab0403ec1a969d788b054a364f68cd0'
PREVIOUS_SHA = 'c9655e65bd9cb8ca0b40a362e89b3ad11eb35b8114aab1b362e0384abab734dd'
EXTRACTOR_SHA = 'a2a23bd1661c66160bce3dc594d5663653f56a29331657ee04805708f872b41d'
N_BINS, N_HOLD, PER_SLOT, FIT_POOL, GS_TARGET = 12, 4, 2, 24, 12
LAST_EARLY_FRAME = 40

PROTOCOL = dict(
    eligible_source_count=40, full_native_source_count=57,
    fit_candidate_count=FIT_POOL, gaussian_target_count=GS_TARGET,
    training_time_bins=N_BINS, heldout_count=N_HOLD, heldout_reserve_count=8,
    source_episode='Only original frame1..40, chest-before-raise source episode. Frames41/42 excluded for source-reviewed motion onset before any129 fitting;43..57 other motion/return episodes excluded. Broad gesture is not exact static63D pose certification.',
    heldout='First reserve nearest2 real-PTS sources to each of4 quarter centers within early1..40; ties greater native face sharpness, greater raw COCO0:11 score, then stable ID. All8 reserves remain excluded from fit and GS, including4 unused backups. Root later admits4 heldout after source review.',
    fit_pool='Reuse128 12equal-real-PTS bins, per-bin .6rank(-native face Laplacian variance)+.4rank(-raw COCO0:11 score), ties nearest bin center then frameID. Take up to2 per bin. No new129 fit residual is read.',
    fit_pool_fallback='If top2/bin yields fewer24, fill ONLY already source-reviewed early1..40 nonreserved sources, descending minimum real-PTS distance to chosen pool; ties global .6face+.4upper-body source rank over nonreserved early sources, then frameID. Update distance each addition. No motion-onset, held reserve, unknown source or later episode refill.',
    source_score_change='raw_body_score=sum RTMW COCO scores indices0:11 (head, shoulders, elbows, wrists), excluding hips/knees/feet outside long skirt/native raster. Original.6/.4 coefficients and rank formula unchanged. Combined new-person, episode/subset, supported-region and24fit-to12GS input change, not isolated speed/frame-count ablation.',
    postfit='Only the single fresh official24-frame fit archive: in the SAME12 fixed early real-PTS bins rank supported_body_rms_px ascending; ties supported_face_rms_px (missing sorts last, never0), nearest bin center, frameID. Unknown/nonfinite body RMS is ineligible, not0. Select at most1 per bin (about12); empty bins stay empty. No heldout rank/statistics, no new fit, no unknown pool refill.',
    postfit_measurement='supported_body_rms_px must be explicitly computed from this archive native projection and existing confidence weights on source-supported BODY25 observations only. Report support IDs/count/weight sum; empty support=>null. supported_face_rms_px likewise only supported face; source detector agreement is not physical ground truth. Current128 visual reviewer does NOT yet emit these RMS values.',
    side_review='Actual native and side mesh reviewer may reject/mark uncertain bad candidates using documented evidence; only existing fitted24 can be considered. Ranking never overrides source ownership/visibility or rejects by heldout results. This helper never grants Gaussian admission.',
    reference='Fit reference is a source-only output identity label, never a pose/beta initializer. Final GS reference must belong to selected GS about12 and have source-confirmed observed face/open eyes; rank by same frozen fit quality (supported body RMS, then supported face RMS), then greater source face sharpness, then frameID. No automatic first-frame preference or mandatory inclusion of fit reference. If none is eligible, report gap. Do not refit to change the output reference.',
    confidence='Existing q90 formula/parameters estimated ONLY on official fit24 after source visibility score-copy gating. Held4/backups excluded. GS subset12 reuses same single fit24 geometry and frozen confidence provenance, no second q90 fit, no selected-pose/beta reoptimization. All24 fit candidates still influence the shared shape and pose solution even if not selected for GS; disclose this boundary.',
    visibility='Use actual source57 face/eye and8group source region flags. Unknown becomes unsupported, rawXY/NPZ/masks unchanged. Leaves overlap fingers and masks inconsistently include prop; fingers are not materialGT. Hidden/cropped knees/feet are unknown, not negative supervision.',
    coverage='RealPTS bins provide temporal dispersion only. This early gesture segment does not prove180degrees, equal angles, reliable unknown-view geometry or whole-video static motion.',
    source_review='All57 sources were already source-reviewed. Reuse SHA-bound contacts/independent reviews; do not extract more frames, generate contacts or inspect fitting residuals during source selection.',
    evidence_scope='Held4/backups are source data excluded from fitting, fit statistics and GS selection. They have no fitted held pose in the current plan and are not geometric validation, unseen independent video, or physical material/pose truth. Old validation results do not certify this new split.',
    budget='Exactly40 eligible source frames; at most24 official fit candidates and8 heldout reserves, followed by at most12 GS frames. No full57 fitting, no duplicate fit, no old-person geometry.')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def frame_number(frame): return int(frame.rsplit('_',1)[1])


def shortlist(features):
    """Pure subset/reserve/bin selection. Returns a draft, never fit admission."""
    assert sha(previous.__file__)==PREVIOUS_SHA
    early=sorted((dict(r) for r in features if 1<=frame_number(r['frame'])<=LAST_EARLY_FRAME),
                 key=lambda r:(r['time_seconds'],r['frame']))
    assert len(early)==LAST_EARLY_FRAME and len({r['frame'] for r in early})==len(early)
    assert all(math.isfinite(r[k]) for r in early for k in ('time_seconds','face_sharpness','raw_body_score'))
    result=previous.shortlist(early) # same12-bin/4x2 reserve/.6/.4/top2 code
    reserved={f for r in result['holdout_reservation'] for f in r['candidate_order']}
    by_id={r['frame']:r for r in early}
    chosen=[f for s in result['training_candidate_slots'] for f in s['candidate_order']]
    assert len(chosen)==len(set(chosen)) and not set(chosen)&reserved
    available=[r for r in early if r['frame'] not in reserved]
    sr=extractor.rank([-r['face_sharpness'] for r in available])
    br=extractor.rank([-r['raw_body_score'] for r in available])
    global_rank={r['frame']:.6*a+.4*b for r,a,b in zip(available,sr,br)}
    fallback=[]
    while len(chosen)<FIT_POOL:
        remaining=[r for r in available if r['frame'] not in chosen]
        assert remaining,'Insufficient finite early nonreserved source pool; do not lower source gates.'
        def key(r):
            d=min(abs(r['time_seconds']-by_id[f]['time_seconds']) for f in chosen)
            return (-d,global_rank[r['frame']],r['frame'])
        ordered=sorted(remaining,key=key);r=ordered[0]
        fallback.append(dict(frame=r['frame'],nearest_selected_pts_distance_seconds=-key(r)[0],
                             source_quality_rank=global_rank[r['frame']],candidate_order=[x['frame'] for x in ordered]))
        chosen.append(r['frame'])
    assert len(chosen)==FIT_POOL and not set(chosen)&reserved
    result.update(fit_candidate_frames=sorted(chosen,key=lambda f:(by_id[f]['time_seconds'],f)),
        fit_candidate_fallback=fallback,fit_candidate_count=FIT_POOL,gaussian_target_count=GS_TARGET,
        eligible_source_frames=[r['frame'] for r in early],
        fixed_time_range_seconds=[early[0]['time_seconds'],early[-1]['time_seconds']],
        source_motion_onset_excluded=['frame_000041','frame_000042'],
        fit_candidate_roster_admitted=False,postfit_gaussian_roster_created=False)
    return result


def postfit_shortlist(records,fit_candidate_frames,holdout_reserves,fixed_time_range):
    """Pure pre-registered fitted24->about12 ranking; no automatic admission."""
    assert len(fit_candidate_frames)==len(set(fit_candidate_frames))==FIT_POOL
    assert not set(fit_candidate_frames)&set(holdout_reserves)
    assert {r['frame'] for r in records}==set(fit_candidate_frames) and len(records)==FIT_POOL
    assert len({r['source_fit_sha256'] for r in records})==1
    start,end=fixed_time_range;assert end>start
    out=[]
    for slot in range(N_BINS):
        left=start+(end-start)*slot/N_BINS;right=start+(end-start)*(slot+1)/N_BINS;center=(left+right)/2
        rows=[r for r in records if left<=r['time_seconds'] and (r['time_seconds']<right or slot==N_BINS-1)]
        def valid(r):
            x=r.get('supported_body_rms_px')
            return (isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>=0
                    and r.get('supported_body_count',0)>0 and r.get('supported_body_weight_sum',0)>0
                    and r.get('side_review_status','unreviewed') not in ('rejected','uncertain'))
        def key(r):
            face=r.get('supported_face_rms_px')
            face=face if isinstance(face,(int,float)) and not isinstance(face,bool) and math.isfinite(face) and face>=0 else math.inf
            return (r['supported_body_rms_px'],face,abs(r['time_seconds']-center),r['frame'])
        ranked=sorted((r for r in rows if valid(r)),key=key)
        out.append(dict(slot=slot,time_interval=[left,right],candidate_order=[r['frame'] for r in ranked],
                        frame=ranked[0]['frame'] if ranked else None))
    return dict(selected_candidate_frames=[r['frame'] for r in out if r['frame']],coverage=out,
        missing_bins=[r['slot'] for r in out if not r['frame']],gaussian_admitted=False,
        actual_pose_review_still_required=True,heldout_used_for_ranking=False,new_fit_started=False)


def adapted_prepare():
    assert sha(extractor.__file__)==EXTRACTOR_SHA
    code=textwrap.dedent(inspect.getsource(extractor.prepare))
    edits=[("for i,row in enumerate(rows):","for i,row in enumerate(rows):\n        if not 1<=frame_number(row['frame'])<=LAST_EARLY_FRAME:continue"),
           ('[1520,2704]','[2704,1520]'),("obs['scores'][i,5:17].sum()","obs['scores'][i,:11].sum()"),
           ('raw_body12_score','raw_body_score'),("subject='officewoman',record_id='019'","subject='leafwoman',record_id='022'"),
           ("ids=sorted({f for s in selected['training_candidate_slots']+selected['holdout_reservation'] for f in s['candidate_order']})",
            "ids=sorted(set(selected['fit_candidate_frames']) | {f for s in selected['holdout_reservation'] for f in s['candidate_order']})")]
    for old,new in edits:
        assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
    ast.parse(code)
    ns=dict(extractor.__dict__,SOURCE=SOURCE,OUT=OUT,VIDEO_SHA=VIDEO_SHA,PROTOCOL=PROTOCOL,
            shortlist=shortlist,frame_number=frame_number,LAST_EARLY_FRAME=LAST_EARLY_FRAME,__file__=__file__)
    exec(compile(code,str(__file__)+'::same_native_feature_extractor','exec'),ns)
    return ns['prepare'],code


def audit():
    _,code=adapted_prepare();assert "obs['scores'][i,:11].sum()" in code
    assert 'import torch' not in code and "subject='leafwoman',record_id='022'" in code
    features=[dict(frame=f'frame_{i+1:06d}',time_seconds=i/3,face_sharpness=float(i%7),
                   raw_body_score=float(i%11),ignored_fit_error=999-i) for i in range(57)]
    before=json.dumps(features,sort_keys=True);s=shortlist(features);assert before==json.dumps(features,sort_keys=True)
    reserves=[f for x in s['holdout_reservation'] for f in x['candidate_order']]
    assert len(reserves)==8 and len(s['fit_candidate_frames'])==24
    assert max(map(frame_number,s['eligible_source_frames']))==40
    assert not set(s['fit_candidate_frames'])&set(reserves)
    assert s==shortlist(list(reversed(features)))
    changed=[dict(r,ignored_fit_error=-r['ignored_fit_error']) for r in features]
    assert s['fit_candidate_frames']==shortlist(changed)['fit_candidate_frames']
    fitted=[dict(frame=f,time_seconds=next(r['time_seconds'] for r in features if r['frame']==f),
                 source_fit_sha256='single_fresh_fit',supported_body_rms_px=1.,supported_body_count=8,
                 supported_body_weight_sum=1.,supported_face_rms_px=None) for f in s['fit_candidate_frames']]
    t=postfit_shortlist(fitted,s['fit_candidate_frames'],reserves,s['fixed_time_range_seconds'])
    assert len(t['selected_candidate_frames'])<=12 and not t['gaussian_admitted']
    none=[dict(r,supported_body_rms_px=None) for r in fitted]
    assert postfit_shortlist(none,s['fit_candidate_frames'],reserves,s['fixed_time_range_seconds'])['selected_candidate_frames']==[]
    return dict(CPU_static_audit=True,helper_sha256=sha(__file__),previous_helper_sha256=PREVIOUS_SHA,
        extractor_helper_sha256=EXTRACTOR_SHA,prepared_function_sha256=hashlib.sha256(code.encode()).hexdigest(),
        synthetic_source_count=57,synthetic_eligible_count=40,synthetic_fit_pool_count=24,synthetic_holdout_reserve_count=8,
        synthetic_postfit_target=12,unknown_body_RMS_not_zero=True,fit_errors_unused_in_source_selection=True,
        GPU_started=False,prepare_executed=False,experiment_created=False,quality_accepted=False)


def prepare():
    receipt=audit()
    reviews=[SOURCE_REVIEW/n for n in ['source-contacts.json','independent-first28-review129.json','independent-last29-review129.json']]
    assert all(p.is_file() for p in reviews),'Reuse actual source reviews; do not regenerate/extract.'
    fn,_=adapted_prepare();fn()
    p=OUT/'plan-before-source-review.json';plan=load(p)
    assert plan['source_frames']==57 and len(plan['features'])==40 and len(plan['fit_candidate_frames'])==24
    assert plan['training_frames']==[] and plan['heldout_frames']==[] and plan['reference_frame'] is None
    plan.update(state='source_only_24fit_pool_draft_root_freeze_required',source_wh=[2704,1520],
        preparation_audit=receipt,source_review_reuse=[dict(path=str(p),sha256=sha(p)) for p in reviews],
        no_new_contacts_generated=True,source_only_handwritten_review_already_exists=True,
        frozen_training_and_heldout_roster_created=False,conditional_K=True,formal_calibration_accepted=False,
        source_pose_episode='early_chest_before_raise_1_to_40',source_motion_onset_frames=[41,42],
        official_fit_target_count=24,gaussian_target_count=12,source_score_indices=list(range(11)),
        source_masks_modified=False,source_pixels_modified=False)
    p.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LEAFWOMAN129_SOURCE_DRAFT_ONLY',sha(p))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['protocol','audit','prepare']);a=parser.parse_args()
    if a.mode=='protocol':print(json.dumps(PROTOCOL,ensure_ascii=False,indent=2))
    elif a.mode=='audit':print(json.dumps(audit(),ensure_ascii=False,indent=2))
    else:prepare()
