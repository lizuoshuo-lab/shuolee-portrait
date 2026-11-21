"""Fresh shared-shape winterhat fit using the existing selected-view solver."""
import copy, inspect, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import cv2
import run_sofa_selected_views124 as prior
from run_official_gaussians123 import read, save, sha, now
from review_full_sequence123 import crop_box, joined

original=prior.original
SRC=original.ROOT/'winterhat-input'
OUT=original.ROOT/'winterhat-half8-pose124'
AUDIT=original.ROOT/'winterhat-half8-input-audit124'
REFERENCE='frame_000071'
scope=prior.shared.scope.copy();scope.update(SRC=SRC,OUT=OUT)
data_source=inspect.getsource(original.data).replace("read(SRC/'selection.json')['training_frames']", "read(OUT/'selection.json')['training_frames']")
exec(compile(data_source,'<same-eight-winterhat-input-mapping>','exec'),scope)
runner=prior.runner
old="        with np.load(SRC/'reference-fit/winner-camera-space.npz') as z:beta=z['betas'].copy()"
assert runner.count(old)==1
runner=runner.replace(old,"        beta=np.zeros((1,10),dtype=np.float32)  # Official neutral identity; no previous individual shape.")
exec(compile(runner,'<same-shared-fit-official-neutral-seed>','exec'),scope)
HAND_CONTROL=original.H/'proposals/demo-fullbody-hand-data/run_hand_data_control.py'
hand_control=original.entry.module(HAND_CONTROL,'_winterhat_existing_hand_data_control124')
config_reads=[]


def hand_config(root):
    baseline=original.entry.load_official_config(root)
    candidate=hand_control.candidate_config(baseline)
    assert candidate['use_hands'] is True
    config_reads.append(dict(baseline=baseline['hand_joints_weights'],candidate=candidate['hand_joints_weights']))
    return candidate


def make_entry(n,roots):
    method,delta=original.adapter.make_entry(n,roots)
    method.__globals__['load_official_config']=hand_config
    return method,delta


scope['adapter']=SimpleNamespace(**dict(original.adapter.__dict__,make_entry=make_entry))


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    audit=read(AUDIT/'result.json');ap=read(AUDIT/'plan.json')
    decisions=[]
    for r in audit['records']:
        assert sha(r['body'])==r['body_sha256'] and sha(r['hands'])==r['hands_sha256']
        f=r['frame']
        reason=('Existing contradictory duplicate right wrist; preserve source rejection.' if f=='frame_000013' else
            'Both hand landmark sets fall on the same visible lowered hand; raised hidden hand has no independent ownership.' if f in ['frame_000001','frame_000002','frame_000003'] else
            'Raised hand and lowered hand are independently visible and source-consistent; first eligible candidate in the frozen order.' if f=='frame_000028' else
            'Also source-consistent, but later than the first acceptable frozen candidate; not selected.' if f=='frame_000029' else
            'Lowered far hand is hidden; its finger cluster lies on the skirt, without independent visible support.')
        decisions.append(dict(frame=f,actual_visual_review=True,selected=f=='frame_000028',reason=reason))
    save(AUDIT/'visual-review.json',dict(updated=now(),actual_visual_review=True,records=decisions,
        selected_additional_frame='frame_000028',selection_before_fit=True,
        no_confidence_or_target_edit=True,physical_landmark_truth=False,quality_accepted=False))
    ids=sorted(ap['retained_seven']+['frame_000028'])
    selection=read(SRC/'selection.json')
    assert len(ids)==8 and not set(ids)&set(selection['review_only_frames']) and 'frame_000013' not in ids
    OUT.mkdir()
    save(OUT/'selection.json',dict(updated=now(),training_frames=ids,reference_frame=REFERENCE,
        review_only_frames=selection['review_only_frames'],added_frame='frame_000028',removed_source_conflict='frame_000013',
        source_selection_rule=ap['rule'],input_audit=str(AUDIT/'visual-review.json'),quality_accepted=False))
    rows,target=scope['data']()
    assert [r['frame'] for r in rows]==ids
    guards={r['frame']:r['screen_flags'] for r in read(original.ROOT/'selected-observation-audit124/wrist-preselection/winterhat.json')['records']}
    records=[];tiles=[]
    for i,row in enumerate(rows):
        assert not guards[row['frame']] and row['wh']==[1520,2704]
        assert row['K_integer_centers']==rows[0]['K_integer_centers'] and np.array_equal(row['w2c'],np.eye(4))
        assert sha(row['image'])==row['source_sha256'] and sha(row['mask'])==row['mask_sha256']
        raw=Image.open(row['image']).convert('RGB');mask=Image.open(row['mask']).convert('L')
        box=crop_box(mask.getbbox(),raw.size,35);crop=raw.crop(box);points=np.asarray(crop).copy()
        for ix,color in [(original.entry.COCO,(0,255,0)),(range(25,46),(255,60,60)),(range(46,67),(60,100,255))]:
            for j in ix:
                p=target[i,j]
                if p[2]>0:cv2.circle(points,tuple(np.round(p[:2]-box[:2]).astype(int)),3,color,1,cv2.LINE_AA)
        path=Path('/mnt/d/00/记录/001')/f'分析_冬帽半圈8帧124_{row["frame"]}_原图与2D观测.png'
        assert not path.exists();panel=joined([crop,Image.fromarray(points)]);panel.save(path)
        thumb=panel.copy();thumb.thumbnail((720,650));tiles.append(thumb)
        hands=target[i,25:67,:2];hb=crop_box(np.r_[hands.min(0),hands.max(0)],raw.size,30)
        hpath=OUT/(row['frame']+'-source-hands.png');raw.crop(hb).save(hpath)
        records.append(dict(frame=row['frame'],body=str(path),hands=str(hpath),body_sha256=sha(path),hands_sha256=sha(hpath),actual_visual_review=False))
        m=np.asarray(mask.crop(box),dtype=np.float32)/255
        person=Image.fromarray((np.asarray(crop).astype(np.float32)*m[...,None]).astype(np.uint8))
        dest=Path('/mnt/d/00/记录/001')/f'mask_冬帽半圈8帧124_{row["frame"]}_原图_mask_人像.png'
        assert not dest.exists();joined([crop,mask.crop(box).convert('RGB'),person]).save(dest)
    pages=[]
    for first in (0,4):
        im=Image.new('RGB',(1440,1300))
        for i,tile in enumerate(tiles[first:first+4]):im.paste(tile,((i%2)*720,(i//2)*650))
        path=OUT/f'source-page-{first//4+1:02d}.png';im.save(path);pages.append(str(path))
    save(OUT/'source-review.json',dict(updated=now(),actual_visual_review=False,records=records,pages=pages,
        retained_input_frames=ids,rejected_input_frames=['frame_000013'],static_gesture_plausible=False,quality_accepted=False))
    sources=[Path(__file__),Path(prior.__file__),Path(prior.shared.__file__),Path(original.__file__),
        Path(original.adapter.__file__),Path(original.entry.__file__),original.H/'smplify_score_contract123.py',
        original.H/'smplify_observation_validity124.py',SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'selection.json',
        OUT/'selection.json',AUDIT/'visual-review.json',original.ASSETS/'smplx/SMPLX_NEUTRAL.npz']
    sources += [original.OFFICIAL/f for f in original.entry.PINNED]
    for row in rows:sources.extend([Path(row['image']),Path(row['mask'])])
    save(OUT/'plan.json',dict(updated=now(),state='registered_source_review_pending',subject='winterhat',frames=ids,rows=rows,
        reference_for_root_seed=REFERENCE,review_only_excluded=selection['review_only_frames'],source_rejected_frames=['frame_000013'],
        hypothesis='Apply the same selected shared-shape and ROI2x/paired-background/18k pipeline to another approximately static person.',
        user_basis='Half-circle8/full-circle16; reproducible quality-selected inputs, original repositories and minimum necessary adaptation.',
        algorithm='Existing shared10D beta and32D VPoser; per-view body root/placement/face/fingers; original5stages/max30/LBFGS and priors.',
        initialization='Official neutral zero10D beta, zero VPoser, standard front body roots; reference71 only names the source/reference pose. No previous individual fit or Gaussian initialization.',
        identity_camera_warning='All w2c are identity placeholders; body roots are view estimates, not physical scene calibration.',
        causal_scope='Cross-person pipeline comparison; roster, fresh neutral-seed shape fit and current Gaussian profile are combined.',
        shared_shape_can_update=True,old_optimized_pose_used=False,old_individual_shape_used=False,
        original_inputs_K_and_priors_preserved=True,static_gesture_assumption=True,
        budget=dict(candidates=1,selected_views=8,GPU_seconds=600,new_gaussian_steps=0),
        stop='One fit and actual review, no pose/shape/threshold/roster sweeps. All frame IDs remain frozen after this source review.',
        quality_accepted=False,gaussian_admitted=False,source_hashes={str(p):sha(p) for p in sources}))
    (OUT/'executed-runner.py.txt').write_text(runner)
    print('WINTERHAT_HALF8_FRESH_FIT_REGISTERED',ids,flush=True)


def preflight():
    assert read(OUT/'source-review.json')['actual_visual_review']
    assert read(OUT/'source-review.json')['static_gesture_plausible']
    proxy=SimpleNamespace(**dict(original.__dict__,SRC=SRC))
    code=inspect.getsource(prior.shared.preflight)
    old="    with np.load(original.SRC/'reference-fit/winner-camera-space.npz') as z:seed=z['betas'].copy()"
    assert code.count(old)==1
    code=code.replace(old,"    seed=np.zeros((1,10),dtype=np.float32)")
    namespace=dict(prior.shared.__dict__,OUT=OUT,original=proxy)
    exec(compile(code,'<same-shared-shape-neutral-preflight>','exec'),namespace)
    namespace['preflight']()
    hand_option_preflight()


def hand_option_preflight():
    baseline=original.entry.load_official_config(original.OFFICIAL)
    candidate=hand_control.candidate_config(baseline)
    assert candidate['use_hands'] is True
    assert [k for k in baseline if baseline[k]!=candidate[k]]==['hand_joints_weights']
    stage_source=(original.OFFICIAL/'smplifyx/fit_single_frame.py').read_text()
    assert 'joint_weights[:, 25:67] = curr_weights[\'hand_weight\']' in stage_source
    result=read(OUT/'preflight.json')
    result.update(original_hand_weight_option_only=True,hand_data_schedule=candidate['hand_joints_weights'],
        body_wrist_indices_unchanged=[4,7],mapping_length_unchanged=118,hand_pose_prior_retained=True,
        configuration_verified_no_additional_optimizer_updates=True)
    save(OUT/'preflight.json',result)


def finish_preflight():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'progress.json').exists()
    result=read(OUT/'preflight.json');assert all(result['checks'].values())
    p=read(OUT/'plan.json');old=p['source_hashes'][str(Path(__file__))]
    for path,digest in p['source_hashes'].items():
        if path!=str(Path(__file__)):assert sha(path)==digest,path
    save(OUT/'preflight-path-correction.json',dict(updated=now(),error='Supplemental config check used official/fit_single_frame.py instead of official/smplifyx/fit_single_frame.py.',
        original_gpu_checks_already_passed=True,optimizer_updates=0,gpu_checks_repeated=False,
        previous_wrapper_sha256=old,current_wrapper_sha256=sha(__file__)))
    p['source_hashes'][str(Path(__file__))]=sha(__file__);save(OUT/'plan.json',p)
    hand_option_preflight()
    print('WINTERHAT_PREFLIGHT_COMPLETE_WITHOUT_REPEAT_GPU',flush=True)


def run():
    scope['run']()
    assert len(config_reads)==1 and config_reads[0]['candidate']==[0.,0.,0.,0.,0.]
    result=read(OUT/'result.json')
    with np.load(OUT/'selected-fit.npz') as z:beta=z['betas'].copy()
    result.update(beta_before=np.zeros((1,10)).tolist(),beta_after=beta.tolist(),beta_delta_l2=float(np.linalg.norm(beta)),
        shared_beta_parameter_count=10,shared_shape_updated=True,freeze_betas=False,
        old_individual_shape_used=False,old_optimized_pose_used=False,source_camera_identity_placeholder=True,
        hand_data_control=dict(config_reads[0],same_for_all_frames=True,body_wrists_and_hand_pose_prior_retained=True,
            existing_control=str(HAND_CONTROL),source_sha256=sha(HAND_CONTROL)),
        actual_root_seed='Standard front rotation in all views; roots remain freely fitted body-view estimates.',
        quality_accepted=False,gaussian_admitted=False)
    save(OUT/'result.json',result)


def revise_prelaunch():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not (OUT/'preflight.json').exists() and not (OUT/'progress.json').exists()
    p=read(OUT/'plan.json');assert not (OUT/'prelaunch-input-revision.json').exists()
    save(OUT/'plan-before-hand-observation-review.json',p)
    baseline=original.entry.load_official_config(original.OFFICIAL)
    candidate=hand_control.candidate_config(baseline)
    save(OUT/'prelaunch-input-revision.json',dict(updated=now(),optimizer_updates=0,
        reason='Actual source review finds far lowered hand occluded by body/bag in71/96/109. Its detected finger positions are not supported material observations.',
        parameter='hand_joints_weights',baseline=baseline['hand_joints_weights'],candidate=candidate['hand_joints_weights'],
        existing_original_option=True,uniform_all_eight_and_both_hands=True,
        body25_wrists_retained=True,hand_pose_priors_retained=True,targets_and_confidence_unchanged=True,
        user_basis='User permits hand fitting abstention/postprocessing and previously approved uniform original hand2D-data-off control; later delegated decisions.',
        no_training_or_fitting_had_started=True,source=str(HAND_CONTROL),sha256=sha(HAND_CONTROL)))
    p.update(state='registered_source_reviewed',hand_observation_policy='Uniform original hand2D-data weights zero for all8; body25 wrists and original hand pose priors retained.',
        algorithm='Existing shared10D beta and32D VPoser; per-view root/placement/face/fingers; original5stages/max30/LBFGS/priors, uniform hand2D-data abstention.',
        causal_scope='Combined cross-person pipeline: source roster, fresh neutral shared shape fit, documented uniform hand-data abstention and current Gaussian profile. Not isolated view-count evidence.')
    p['source_hashes'][str(Path(__file__))]=sha(__file__)
    p['source_hashes'][str(HAND_CONTROL)]=sha(HAND_CONTROL)
    p['source_hashes'][str(OUT/'prelaunch-input-revision.json')]=sha(OUT/'prelaunch-input-revision.json')
    save(OUT/'plan.json',p)
    review=read(OUT/'source-review.json')
    for row in review['records']:
        assert sha(row['body'])==row['body_sha256'] and sha(row['hands'])==row['hands_sha256']
        row.update(actual_visual_review=True,far_fingers_unknown=True)
    review.update(updated=now(),actual_visual_review=True,static_gesture_plausible=True,
        finding='All8 native source/2D pages inspected. Raised hand holds strap, opposite arm lowered, visible body/boots consistent. Far fingers hidden on some views are not physical2D truth; uniform original hand-data abstention before the first fit.',
        physical_landmark_truth=False,quality_accepted=False)
    save(OUT/'source-review.json',review)
    print('WINTERHAT_SOURCE_REVIEW_AND_EXISTING_HAND_ABSTENTION_REGISTERED',flush=True)


def launch():
    assert read(OUT/'source-review.json')['actual_visual_review']
    source=inspect.getsource(prior.launch)
    namespace=dict(prior.__dict__,OUT=OUT,__file__=__file__)
    exec(compile(source,'<existing-single-process-launch>','exec'),namespace)
    namespace['launch']()


if __name__=='__main__':{'prepare':prepare,'revise-prelaunch':revise_prelaunch,'preflight':preflight,'finish-preflight':finish_preflight,'run':run,'launch':launch}[sys.argv[1]]()
