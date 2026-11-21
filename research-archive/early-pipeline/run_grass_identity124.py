"""One confidence-only causal control on the already verified temporal interface."""
import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import run_grass_refinement124 as baseline
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import read,save,sha,now

PREVIOUS=baseline.OUT
ROOT=PREVIOUS.parent
GUARD=ROOT/'hand-identity-guard124'
OUT=ROOT/'grass-temporal-handidentity'


def prepare():
    assert not OUT.exists()
    assert read(PREVIOUS/'temporal-progress.json')['state']=='completed'
    assert read(PREVIOUS/'process-exit.json')['exit_code']==0
    assert not read(PREVIOUS/'conclusion.json')['quality_accepted']
    assert read(GUARD/'result.json')['passed']
    assert read(GUARD/'visual-review.json')['grass_control_proposal_reviewable']
    plan=copy.deepcopy(read(PREVIOUS/'plan.json'))
    extra=[Path(__file__),GUARD/'plan.json',GUARD/'result.json',GUARD/'visual-review.json',
           GUARD/'grass-input-preview.npz',PREVIOUS/'plan.json',PREVIOUS/'conclusion.json',
           Path(__file__).with_name('smplify_hand_identity124.py')]
    plan['sources'].update({str(p):sha(p) for p in extra})
    plan.update(updated=now(),state='registered_not_started',previous_control=str(PREVIOUS),
        hypothesis='The74-94left hand identity collapse is a distinguishable input error driving the late far-arm flip. Withhold that ambiguous wrist/21hand-chain evidence using the frozen subject-independent identity rule while retaining original coordinates and temporal/pose priors.',
        causal_change='Input confidence only. The completed all-person guard preview is applied at the entry; frame74-94left wrist+21hand entries zero, everything else equal to the rejected94 temporal refinement. No fitter/loss/weight/schedule/camera/shape change.',
        input_change='Frozen hand-identity guard preview, retaining all raw observations for evaluation. No hand coordinates replaced; other shoulders/elbows/body/face/feet unchanged.',
        guard_state='Experimental partial input guard, not a visibility classifier. Winterhat and indoor have known false negatives; indoor33false recovery. No default adoption or automatic cross-person fitting.',
        method='Same original reference32 five-stage initialization and neighbouring stages3/4, shared beta, fresh same-run poses, originalLBFGS/VPoser/bending and temporal30 with realPTS. New run starts from zero/mean, no old per-frame pose initialization.',
        comparison='Closed grass-temporal-refinement94 retained. Input-only causal variable includes temporal propagation of withholding. Raw observations and previous metric definitions remain available; do not improve metrics by hiding removed points.',
        risks=['Withheld hand targets reduce constraints and may leave wrong depth or pull arm toward a prior.',
               'Visible right arm, face or feet can regress through shared shape/pose geometry even when their input stays unchanged.',
               'Improvement on grass does not validate the classifier or other subjects.'],
        budget=dict(candidates=1,frames=94,max_wall_seconds=5400,review_after_frames=13,
                    additional_review_after_frames=27,review_wait_seconds=900),
        stopping='Single fresh94 candidate. Review13/27against the already reviewed identical-input baseline; continue only with unchanged or accepted native body/face/feet. All94actual pose review required, especially73/74,79/80andthe21withheldframes. Stop on new anatomy/depth/face/feet regressions, nonfinite state, source change or wall budget. No threshold sweep, stitching, old-pose fallback, default adoption or GS without quality acceptance.',
        gaussian_admitted=False,quality_accepted=False)
    save(OUT/'plan.json',plan)
    check=copy.deepcopy(read(PREVIOUS/'interface-check.json'))
    check.update(updated=now(),reused_interface_checks=str(PREVIOUS/'interface-check.json'),
                 input_guard_preflight=str(GUARD/'result.json'),quality_accepted=False)
    save(OUT/'interface-check.json',check)
    print('GRASS_HANDIDENTITY_REGISTERED94',flush=True)


def run():
    assert not (OUT/'official-fit').exists()
    plan=read(OUT/'plan.json')
    for p,h in plan['sources'].items():assert sha(p)==h,p
    with np.load(GUARD/'grass-input-preview.npz',allow_pickle=False) as z:
        ids=list(z['frame_ids'].astype(str));raw=z['original_raster_checked'];modified=z['guarded_preview'];flags=z['ambiguous']
    lookup={name:i for i,name in enumerate(ids)}
    original=entry.fit_one
    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name;i=lookup[frame]
        incoming=kwargs['keypoints'].copy()
        assert np.array_equal(incoming[0],raw[i]), 'Frozen input mapping/raster contract differs.'
        kwargs['keypoints']=modified[i:i+1].copy()
        arrays,report=original(**kwargs)
        report['hand_identity_observation_guard']=dict(
            ambiguous_left_right=flags[i].tolist(),
            zeroed_confidence_indices=np.flatnonzero(raw[i,:,2]!=modified[i,:,2]).tolist(),
            coordinates_unchanged=True,raw_targets_retained_for_evaluation=True,
            physical_visibility_proven=False,input_preview_sha256=sha(GUARD/'grass-input-preview.npz'))
        return arrays,report
    entry.fit_one=fit;baseline.OUT=OUT
    try:baseline.run()
    finally:entry.fit_one=original


def launch():
    assert not (OUT/'console.log').exists()
    command=[sys.executable,'-B',str(Path(__file__)),'run']
    with (OUT/'console.log').open('x') as log:
        worker=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        save(OUT/'launch.json',dict(updated=now(),pid=worker.pid,command=command,training_started=False))
        print('GRASS_HANDIDENTITY_STARTED',worker.pid,flush=True)
        code=worker.wait()
    save(OUT/'process-exit.json',dict(updated=now(),pid=worker.pid,exit_code=code))
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','launch']);args=parser.parse_args()
    {'prepare':prepare,'run':run,'launch':launch}[args.mode]()
