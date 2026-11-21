"""CPU source-only shortlist; never fits, detects, edits observations, or admits a roster.

The existing football selector was frozen in its source plan, not a reusable
Python module. Retain its rank/bin/reserve/top2/fallback framework. For a full
orbit, body ranking must not depend on whether a face is visible: use native
mask-interior body sharpness and COCO12 shoulders-to-ankles score, reserving
face sharpness exclusively for the later source-reviewed reference choice.
"""
import argparse
import hashlib
import json
from pathlib import Path

N_TRAIN, N_HOLD, PER_SLOT = 24, 8, 2
SOURCE = Path('/opt/gs-slam-lab/shuolee-cross-subject-127/officewoman')
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT = ROOT/'officewoman-circle24-input127'
VIDEO_SHA = '71d993f2f1e2c1e2ad1f5e3b58c4f351cd375e172f92c3526818dfb4ab380edb'
PROTOCOL = dict(
    training_count=N_TRAIN, heldout_count=N_HOLD, heldout_reserve_count=N_HOLD*PER_SLOT,
    heldout='Eight equal-real-PTS segment centers; reserve nearest2 per center, ties raw body12 score then stable frameID. All16 reserved IDs excluded from training, even unused backups.',
    training='24equal-real-PTS bins. Exclude all16 reserves. Rank within each bin: .6rank(-native mask-interior body Laplacian variance)+.4rank(-raw COCO12 shoulder-to-ankle score). Ties nearest bin center then stable frameID. Inspect only top2 per bin.',
    rank_ties='Equal metric values share dense rank=(number of strictly better values)/(n-1), with all-zero ranks when n=1; final ID tie is stable.',
    fallback='After source review: first supported candidate per bin. Keep missing-bin gap explicit. Fill only from already reviewed finite training pool, descending nearest actual-PTS distance to selected set, ties frozen body-quality rank then frameID. Recompute distances after every addition. No reserved holdout or proved body-ownership conflict may enter training.',
    body_quality_change='New generic full-orbit input ranking before fitting: former face-sharpness ROI becomes source-mask body ROI; former COCO17 sum becomes COCO12 indices5:17 so rear face absence/hallucination cannot affect body selection. Coefficients .6/.4 unchanged, no dataset-specific threshold or fit residual.',
    face_visibility='Each candidate requires source-only face_region_is_observed=true/false/unknown plus reason. It is not physical landmark truth. Unknown is not auto-true. A rear frame may pass body review with face=false; never reject it merely for no visible face. No detector/weight modification is performed here.',
    reference='After review, among selected train24 with source-confirmed observed face and visible open eyes: .6rank(-native face sharpness)+.4rank(-raw body12 score), ties frameID. No preference for first frame, no optimized pose/beta or fitting errors.',
    source_review='Source body ownership/anatomy visibility and mask must be actually inspected on finite shortlist. Hidden fingers are unknown, not wrong ground truth. Uniform original hand2D-data weights remain zero downstream; body wrists and priors remain.',
    confidence='Only frozen train24 scores estimate existing q90 formula; heldout scores/coords never enter training statistics. Heldout evaluation uses frozen train24 q90.',
    evidence_scope='Time dispersion is not equal-angle calibration. Source-only source review may inform entrance; heldouts are excluded from this new fit, not an unseen independent video. Six intake samples support a qualitative full orbit, not exact360deg or perfect static motion.',
    source_review_budget='At most48 training and16 heldout candidate sources. No requirement that every source frame fits well, no full-sequence fitting, no expanded scan after a failed finite pool.')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rank(values):
    return [sum(other < value for other in values)/max(1,len(values)-1) for value in values]


def shortlist(features):
    """Pure deterministic source metadata selection; face values never enter body ranks."""
    fs=sorted(features,key=lambda r:(r['time_seconds'],r['frame']))
    assert len(fs)>=N_TRAIN+N_HOLD*PER_SLOT and len({r['frame'] for r in fs})==len(fs)
    start,end=fs[0]['time_seconds'],fs[-1]['time_seconds']; assert end>start
    held=[];reserved=set()
    for slot in range(N_HOLD):
        center=start+(end-start)*(slot+.5)/N_HOLD
        ordered=sorted((r for r in fs if r['frame'] not in reserved),key=lambda r:(abs(r['time_seconds']-center),-r['raw_body12_score'],r['frame']))[:PER_SLOT]
        reserved.update(r['frame'] for r in ordered)
        held.append(dict(slot=slot,center_seconds=center,candidate_order=[r['frame'] for r in ordered]))
    train=[]
    for slot in range(N_TRAIN):
        left=start+(end-start)*slot/N_TRAIN;right=start+(end-start)*(slot+1)/N_TRAIN;center=(left+right)/2
        available=[dict(r) for r in fs if r['frame'] not in reserved and left<=r['time_seconds'] and (r['time_seconds']<right or slot==N_TRAIN-1)]
        sharp=rank([-r['body_sharpness'] for r in available]);score=rank([-r['raw_body12_score'] for r in available])
        for r,s,q in zip(available,sharp,score):r['source_quality_rank']=.6*s+.4*q
        ordered=sorted(available,key=lambda r:(r['source_quality_rank'],abs(r['time_seconds']-center),r['frame']))
        train.append(dict(slot=slot,time_interval=[left,right],candidate_order=[r['frame'] for r in ordered[:PER_SLOT]],all_ordered_candidates=ordered))
    return dict(training_candidate_slots=train,holdout_reservation=held)


def prepare():
    """Called explicitly only after native inputs exist; writes one NEW audit draft."""
    import cv2
    import numpy as np
    from PIL import Image
    cv2.setNumThreads(1)
    inputs_path=SOURCE/'data/inputs.json';obs_path=SOURCE/'data/observations.npz'
    assert inputs_path.is_file() and obs_path.is_file(),'New source observations not ready; no detector or extraction is started.'
    assert not OUT.exists(),'Existing shortlist: inspect it instead of overwriting/reselecting.'
    inputs=load(inputs_path);rows=inputs['records'];assert inputs['video_sha256']==VIDEO_SHA
    with np.load(obs_path,allow_pickle=False) as f:obs=dict(f)
    assert obs['frame_ids'].astype(str).tolist()==[r['frame'] for r in rows]
    hashes={str(inputs_path):sha(inputs_path),str(obs_path):sha(obs_path),str(Path(__file__)):sha(__file__)}
    features=[]
    for i,row in enumerate(rows):
        image,mask_path=Path(row['image']),Path(row['mask'])
        assert sha(image)==row['source_sha256'] and sha(mask_path)==row['mask_sha256']
        hashes[str(image)]=row['source_sha256'];hashes[str(mask_path)]=row['mask_sha256']
        rgb=np.asarray(Image.open(image).convert('RGB'));mask=np.asarray(Image.open(mask_path).convert('L'))>0
        assert rgb.shape[:2]==mask.shape and list(rgb.shape[1::-1])==[1520,2704] and mask.any()
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);lap=cv2.Laplacian(gray,cv2.CV_64F)
        face=0.;box=row.get('face_box_xyxy')
        if box:
            x0,y0=np.floor(box[:2]).astype(int);x1,y1=np.ceil(box[2:]).astype(int)
            patch=gray[max(0,y0):min(len(gray),y1),max(0,x0):min(gray.shape[1],x1)]
            if min(patch.shape,default=0)>1:face=float(cv2.Laplacian(patch,cv2.CV_64F).var())
        features.append(dict(frame=row['frame'],time_seconds=float(row['time_seconds']),source_frame_index=row['source_frame_index'],body_sharpness=float(lap[mask].var()),face_sharpness=face,raw_body12_score=float(obs['scores'][i,5:17].sum()),finite_source_observations=bool(np.isfinite(obs['scores'][i]).all() and np.isfinite(obs['keypoints'][i]).all())))
    selected=shortlist(features)
    ids=sorted({f for s in selected['training_candidate_slots']+selected['holdout_reservation'] for f in s['candidate_order']})
    plan=dict(subject='officewoman',record_id='019',source_root=str(SOURCE),source_inputs=str(inputs_path),source_observations=str(obs_path),video_sha256=VIDEO_SHA,source_hashes=hashes,source_frames=len(rows),sample_fps=inputs['sample_fps'],algorithm=PROTOCOL,features=features,**selected,
        source_rows=rows,training_frames=[],frames=[],heldout_frames=[],review_only_frames=[],rows=[],reference_frame=None,source_rejected_frames=[],
        source_review_template={f:dict(body_source_supported=None,mask_source_supported=None,face_region_is_observed=None,visible_eyes_open=None,reason='',physical_joint_ground_truth=False) for f in ids},
        training_roster_ready=False,actual_source_visual_review=False,source_mask_actual_visual_review=False,static_gesture_plausible=False,old_optimized_pose_or_beta_used=False,old_fit_residuals_used_for_selection=False,quality_accepted=False)
    OUT.mkdir(parents=True)
    (OUT/'plan-before-source-review.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('SOURCE_SHORTLIST_ONLY_REVIEW_REQUIRED',OUT, len(ids))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['protocol','prepare']);args=p.parse_args()
    if args.mode=='protocol':print(json.dumps(PROTOCOL,ensure_ascii=False,indent=2))
    else:prepare()
