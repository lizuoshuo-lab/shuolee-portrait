"""Continue same-run neighbours using existing SMPLify-X refinement stages."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import official_smplifyx_entry123 as entry
import run_grass_translation124 as placement
from run_sofa_temporal_warmstart124 import stage_config,FIELDS
from run_official_gaussians123 import read,save,sha,now

PREVIOUS=placement.OUT
OUT=PREVIOUS.with_name('grass-temporal-refinement')
ORIGINAL_CONFIG=entry.load_official_config


def prepare():
    assert not OUT.exists() and read(PREVIOUS/'temporal-progress.json')['pid'] is None
    conclusion=read(PREVIOUS/'conclusion.json')
    assert conclusion['actual_visual_review'] and not conclusion['quality_accepted']
    assert conclusion['front_arm_direction_improved_frames']==[27,28,29,30,31,32,34,37,38]
    original=ORIGINAL_CONFIG(placement.OFFICIAL)
    anchor=stage_config(original,True);tail=stage_config(original,False)
    assert anchor==original
    changes={k:dict(before=original.get(k),after=tail[k]) for k in tail if original.get(k)!=tail[k]}
    assert set(changes)<=set(FIELDS)
    for field in FIELDS:assert tail[field]==(original.get(field) or [1]*5)[3:]
    assert tail['maxiters']==original['maxiters']==30
    p=copy.deepcopy(read(PREVIOUS/'plan.json'))
    history=[placement.grass.ROOT/'sofa-temporal-warmstart-v2/quality-stop-decision.json',
        placement.grass.REC/'demo-vhap-initialization/fullbody-smplifyx/warmtail-guarded-sequence/review/closeout.json']
    sofa,demo=map(read,history)
    assert not sofa['full_candidate_quality_accepted'] and not demo['quality_accepted']
    p.update(updated=now(),state='registered',
        hypothesis='After freeing body placement the32 anchor has plausible front-arm depth. Reapplying cold-start stages0-2 can still erase an adjacent initialized branch. Test original stages3/4 for same-run neighbours, full original5 for the zero/mean32 anchor.',
        causal_change='Original stage-entry policy only; retain all existing numeric weights, parameter freedom including3 placement scalars, native observations, K/E, shared beta, original LBFGS, temporal rotation prior and real-PTS order.',
        original_stage_indices=dict(anchor=list(range(5)),all_seeded=[3,4]),
        method='Original solver and losses; full five-stage reference32, original two final refinement stages on every same-run seeded chronological neighbour. No new optimizer or sequence solver.',
        comparison='Previous free-translation13 control for matching frames; independent94 remains an immutable diagnostic baseline for the complete roster. No result-dependent frame selection.',
        previous_control=str(PREVIOUS),old_root_translation_run=str(PREVIOUS),
        history_review=dict(sofa_failed='47/56 stopped: left knee34.77->101.23px although right-leg flips improved.',
            demo_failed='59/59 reviewed: elbow improvement did not pass whole-body/face/feet protection; cached pose selection was used there and is forbidden here.'),
        risks=['A wrong adjacent pose can propagate without cold-start stabilization.',
            'Final face/hand gradients can still worsen body or feet; earlier failures remain relevant.',
            'Projection or time smoothness alone does not establish3D correctness.'],
        stopping='One candidate. Review13 native body/face/feet, common sides and every transition; stop for any persistent wrong raised-arm depth or material native regression. Only a passed13 may continue to27 (including44/45) and all94. Stop on nonfinite state, changed input or wall budget. No sweep, old-pose initialization, pose stitching or GS without all94 review.',
        no_cached_pose_fallback=True,no_numeric_weight_tuning=True,reference_new_fit_from_zero=True,
        gaussian_admitted=False,quality_accepted=False)
    p['budget']=dict(candidates=1,frames=94,max_wall_seconds=5400,review_after_frames=13,
        additional_review_after_frames=27,review_wait_seconds=900)
    added=[Path(__file__),Path(placement.__file__),Path(sys.modules[stage_config.__module__].__file__),
        PREVIOUS/'plan.json',PREVIOUS/'conclusion.json',PREVIOUS/'visual-review.json',*history]
    p['sources'].update({str(path):sha(path) for path in added})
    save(OUT/'plan.json',p)
    save(OUT/'configuration-diff.json',dict(anchor_unchanged=True,neighbour_changes=changes))
    save(OUT/'interface-check.json',dict(updated=now(),transport=placement.grass.check_transport(),
        original_anchor_exact_config=True,retained_stage_values_exact_original=True,unchanged_other_parameters=True,
        stage_indices=p['original_stage_indices'],solver_replaced=False,optimizer_steps=0,quality_accepted=False))
    print('GRASS_REFINEMENT_REGISTERED',json.dumps(changes),flush=True)


def run():
    assert not (OUT/'official-fit').exists()
    p=read(OUT/'plan.json');active={}
    for path,digest in p['sources'].items():assert sha(path)==digest,path
    original_fit=entry.fit_one
    def config(root):
        return stage_config(ORIGINAL_CONFIG(root),active['frame']==p['reference_frame'])
    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name;active['frame']=frame
        arrays,report=original_fit(**kwargs)
        anchored=frame==p['reference_frame']
        if anchored:
            # Compare only; the old pose never initializes the new fitting.
            with np.load(PREVIOUS/'official-fit/frames'/frame/'winner-camera-space.npz',allow_pickle=False) as z:
                errors={k:float(np.max(np.abs(arrays[k]-z[k]))) for k in
                    ('vertices_camera','joints_projected_native','pose_embedding','global_orient','camera_translation','betas')}
            assert errors['joints_projected_native']<.002 and max(v for k,v in errors.items() if k!='joints_projected_native')<1e-5,errors
            save(OUT/'anchor-parity.json',dict(updated=now(),fresh_anchor=True,old_pose_initialization=False,max_abs=errors,passed=True))
        report.update(five_stages=anchored,original_stage_indices=list(range(5)) if anchored else [3,4],
            local_stage_entry_adapter=not anchored,retained_stage_values_unchanged=True,no_cached_pose_fallback=True)
        return arrays,report
    entry.load_official_config=config;entry.fit_one=fit;placement.OUT=OUT
    try:placement.run()
    finally:entry.load_official_config=ORIGINAL_CONFIG;entry.fit_one=original_fit


def launch():
    # Original sequence entry owns the sole GPU lock.
    assert not (OUT/'console.log').exists()
    cmd=[sys.executable,'-B',str(Path(__file__)),'run']
    with (OUT/'console.log').open('x') as stream:
        worker=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=os.environ.copy())
        save(OUT/'launch.json',dict(updated=now(),pid=worker.pid,command=cmd,training_started=False))
        print('GRASS_REFINEMENT_STARTED',worker.pid,flush=True)
        rc=worker.wait()
    save(OUT/'process-exit.json',dict(updated=now(),pid=worker.pid,exit_code=rc))
    raise SystemExit(rc)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','launch']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'launch':launch}[args.mode]()
