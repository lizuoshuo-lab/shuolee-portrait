"""Original-stage entry control: same full chronology, same temporal loss."""
import argparse
import copy
import os
from pathlib import Path
import run_core_temporal124 as core
from run_official_gaussians123 import read,save,sha,now

PREVIOUS=core.OUT
OUT=PREVIOUS.with_name('sofa-temporal-warmstart')
FIELDS=['body_pose_prior_weights','shape_weights','expr_weights','hand_pose_prior_weights',
        'jaw_pose_prior_weights','hand_joints_weights','face_joints_weights','coll_loss_weights','data_weights']

def stage_config(config,anchored):
    c=copy.deepcopy(config)
    if not anchored:
        if c.get('data_weights') is None:c['data_weights']=[1]*5
        for field in FIELDS:
            assert len(c[field])==5,field
            c[field]=c[field][3:]
    return c

def prepare():
    assert not OUT.exists() and read(PREVIOUS/'temporal-progress.json')['pid'] is None
    old=read(PREVIOUS/'plan.json')
    diagnostic=read(PREVIOUS/'stage33-diagnostic-replay/result-completed.json')
    assert diagnostic['endpoint_parity_passed'] and diagnostic['actual_visual_review']
    assert not read(PREVIOUS/'training-admission.json')['gaussian_admitted']
    original=core.entry.load_official_config(core.temporal.REC/'smplifyx-reference-control/official')
    anchor=stage_config(original,True);tail=stage_config(original,False)
    assert anchor==original
    assert {k for k in original if tail[k]!=original[k]}<=set(FIELDS)
    for k in FIELDS:assert tail[k]==(original.get(k) or [1]*5)[3:]
    source=(core.temporal.REC/'smplifyx-reference-control/official/smplifyx/fit_single_frame.py').read_text()
    assert 'if data_weights is None:\n        data_weights = [1, ] * 5' in source
    p=copy.deepcopy(old)
    p.update(updated=now(),state='registered',hypothesis='The exact33 replay shows original stages0-2 erase the same-run seated initializer despite an active temporal term. Test original stages3/4 for every seeded frame, retaining five stages for the26 anchor.',
        single_policy_change='Stage entry only. Original retained-stage values, temporal weights30, observations, camera initialization, shared beta and original optimizer remain unchanged.',
        method='Reference26 original five stages. Every other frame original stages3/4 plus the identical body/root temporal loss and same-run adjacent initialization.',
        previous_control=str(PREVIOUS),previous_full_roster_completed=False,
        historical_limit='Demo59 warmtail with cached fallback already failed protection. This is a different full-chronology test with explicit temporal energy and no cached fallback; no general success is assumed.',
        original_stage_indices=dict(anchor=list(range(5)),all_seeded=[3,4]),
        no_numeric_weight_tuning=True,no_cached_pose_fallback=True,no_posthoc_frame_replacement=True,
        initialization='Original zero/mean26 anchor, original reviewed shared beta; subsequent body/root seeds only from this new run. Previous stopped poses are comparison evidence only.',
        budget=dict(candidates=1,frames=56,max_wall_seconds=3600,review_after_frames=13,review_wait_seconds=900),
        stopping='Review all first13 actual fits before continuation; inspect completed28 at the known failure interval and all56. Stop on severe depth/anatomy or native-projection regression, source mismatch/nonfinite state/budget. No automatic schedule/weight sweep. No Gaussian before full pose and mask admission.',
        risks=['A wrong neighbour can propagate without the cold-start prior prefix.','Face, hands or feet can regress even if leg continuity improves.','No proof of physical camera calibration or missing/occluded geometry.'],
        sources={**old['sources'],str(Path(__file__).resolve()):sha(__file__),str(Path(core.__file__).resolve()):sha(core.__file__),
            str(PREVIOUS/'stage33-diagnostic-replay/result-completed.json'):sha(PREVIOUS/'stage33-diagnostic-replay/result-completed.json')})
    OUT.mkdir();save(OUT/'plan.json',p)
    save(OUT/'interface-check.json',dict(updated=now(),anchor_exact_original=True,retained_stage_values_exact_original=True,
        other_options_unchanged=True,implicit_data_list_matches_upstream=True,original_stage_indices=p['original_stage_indices'],
        solver_replaced=False,quality_accepted=False))
    print('SOFA_WARMSTART_TEMPORAL_REGISTERED',flush=True)

def run():
    import fcntl
    lease=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=read(OUT/'plan.json');assert not (OUT/'official-fit').exists()
    originals=core.entry.load_official_config,core.entry.fit_one,core.OUT
    active={}
    def config(root):
        return stage_config(originals[0](root),active['frame']==p['reference_frame'])
    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name;active['frame']=frame
        arrays,receipt=originals[1](**kwargs)
        anchored=frame==p['reference_frame']
        receipt.update(five_stages=anchored,original_stage_indices=list(range(5)) if anchored else [3,4],
            stage_values_unchanged=True,local_curriculum_adapter=not anchored,no_cached_fallback=True)
        return arrays,receipt
    core.entry.load_official_config,core.entry.fit_one,core.OUT=config,fit,OUT
    try:core.run()
    finally:core.entry.load_official_config,core.entry.fit_one,core.OUT=originals

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);args=parser.parse_args();(prepare if args.mode=='prepare' else run)()
