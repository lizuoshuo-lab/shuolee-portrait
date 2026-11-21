"""One bounded observation-confidence control; not a deliverable pose sequence."""
import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS
import official_smplifyx_entry123 as entry
import run_official_sequence123_topology as upstream
from smplify_temporal124 import install_loss_factory

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')


def coherent_hand_keypoints(keypoints):
    """Never increase a hand's2D confidence beyond its wrist evidence.

    Body7/hand25 and body4/hand46 are the exact same SMPL-X wrist joints.
    Keep every observation and2D coordinate; only apply a continuous chain
    weight. These relative scores are not visibility probabilities.
    """
    a=np.asarray(keypoints)
    if a.ndim!=3 or a.shape[1:]!=(118,3) or not np.isfinite(a).all():
        raise ValueError('Expected finite mapped118 native keypoints.')
    if np.any(a[...,2]<0) or np.any(a[...,2]>1):
        raise ValueError('Explicit normalized confidence is required.')
    result=a.copy();gates={}
    for side,parent,start in [('left',7,25),('right',4,46)]:
        gate=np.minimum(a[:,parent,2],a[:,start,2])
        result[:,start:start+21,2] *= gate[:,None]
        gates[side]=gate.tolist()
    assert np.array_equal(result[...,:2],a[...,:2])
    assert np.array_equal(result[:,:25],a[:,:25]) and np.array_equal(result[:,67:],a[:,67:])
    return result,gates


def verify():
    p=np.zeros((3,118,3),np.float32);p[...,:2]=np.arange(236).reshape(1,118,2);p[...,2]=1
    p[1,7,2]=0;p[2,46,2]=.25
    q,g=coherent_hand_keypoints(p)
    checks=dict(reliable_input_unchanged=np.array_equal(q[0],p[0]),
        unreliable_parent_suppresses_whole_hand=bool(np.all(q[1,25:46,2]==0)),
        weaker_hand_root_applies_continuous_gate=bool(np.all(q[2,47:67,2]==.25) and q[2,46,2]==.0625),
        no_coordinate_changes=np.array_equal(q[...,:2],p[...,:2]),
        body_face_feet_unchanged=np.array_equal(q[:,:25],p[:,:25]) and np.array_equal(q[:,67:],p[:,67:]),
        confidence_never_increased=bool(np.all(q[...,2]<=p[...,2])))
    assert all(checks.values()),checks
    return checks


def run():
    baseline=ROOT/'winterhat-temporal';report=ROOT/'hand-observation-coherence';src=ROOT/'winterhat-input'
    plan=read(report/'control-plan.json');progress=read(baseline/'temporal-progress.json')
    assert progress['state']=='completed' and progress['frames']==112, 'Finish and inspect the actual baseline first.'
    assert read(baseline/'final-review/result.json')['actual_visual_review'], 'Full baseline visual review is required.'
    out=report/'control-fit';assert not out.exists(), 'One registered control only; never overwrite/resume.'
    records={r['frame']:r for r in progress['records']}
    rows={r['frame']:r for r in read(src/'data/inputs.json')['records']}
    target_paths={r['neighbour']:baseline/'official-fit/frames'/r['neighbour']/'winner-camera-space.npz'
                  for name,r in records.items() if name in plan['preregistered_control_frames'] and r['neighbour'] is not None}
    source_paths=[Path(__file__),Path(entry.__file__),Path(__file__).with_name('smplify_temporal124.py'),
                  report/'control-plan.json',baseline/'plan.json',baseline/'official-fit/fit/fitted.npz',
                  src/'data/inputs.json',src/'data/observations.npz',src/'reference-fit/winner-camera-space.npz',*target_paths.values()]
    save(report/'control-bound-inputs.json',dict(updated=now(),sources={str(p):sha(p) for p in source_paths},
         checks=verify(),old_neighbours_are_only_frozen_control_inputs=True,old_pose_initialization=False,
         gaussian_admitted=False))
    original=entry.fit_one;loader=entry.load_official;started=time.monotonic();receipts=[]
    def fit_one(**kwargs):
        if time.monotonic()-started>plan['budget']['max_wall_seconds']:
            raise RuntimeError('Registered control runtime budget exhausted.')
        frame=Path(kwargs['result_path']).parent.name
        baseline_record=records[frame];neighbour=baseline_record['neighbour'];target=None
        if neighbour is not None:
            with np.load(target_paths[neighbour],allow_pickle=False) as z:
                target=np.zeros((1,165),np.float32);target[:,:3]=z['global_orient'];target[:,3:66]=z['body_pose_axis_angle']
        kwargs['keypoints'],gates=coherent_hand_keypoints(kwargs['keypoints'])
        record=dict(frame=frame,neighbour=neighbour,chain_gates=gates,neighbour_initialization=False,
                    scope='isolated_same-neighbour_observation-weight_control')
        def load(root):
            u,p,f,o=loader(root)
            install_loss_factory(f,target,baseline_record['dt_seconds'],read(baseline/'plan.json')['nominal_dt_seconds'],record)
            return u,p,f,o
        entry.load_official=load
        try:
            arrays,receipt=original(**kwargs)
        finally:
            entry.load_official=loader
        for key in ('_last_original','_last_temporal'):
            if key in record:record[key[1:]]=float(record.pop(key).cpu())
        receipt['temporal_and_hand_control']=record
        receipt['upstream_loss_monitor_optimizer_unchanged']=False
        receipt['upstream_solver_and_baseline_terms_unchanged']=True
        receipts.append(record)
        save(report/'control-progress.json',dict(updated=now(),state='fitting',pid=os.getpid(),frames=len(receipts),total=12,records=receipts))
        return arrays,receipt
    entry.fit_one=fit_one
    sys.argv=['run_official_sequence123_topology.py','--official-root',str(ROOT.parent/'smplifyx-reference-control/official'),
        '--data-root',str(src),'--runtime-root',str(PROJECT),'--assets-root',str(ASSETS),'--project-root',str(PROJECT),
        '--length-contract-json',str(src/'length-contract.json'),'--reference-frame-id',read(src/'selection.json')['reference_frame'],
        '--shared-betas-npz',str(src/'reference-fit/winner-camera-space.npz'),'--output-dir',str(out),'--device','cuda',
        '--frame-ids',*plan['preregistered_control_frames']]
    try:
        upstream.main()
    finally:
        entry.fit_one=original;entry.load_official=loader
    save(report/'control-progress.json',dict(updated=now(),state='completed',pid=None,frames=len(receipts),total=12,records=receipts,
         quality_accepted=False,gaussian_admitted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify','run']);args=parser.parse_args()
    if args.mode=='verify':print(verify())
    else:run()
