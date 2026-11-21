"""Compose the existing upstream seed hook with the registered temporal loss.

No new optimizer, whole-sequence objective, pose filtering, or old-pose cache.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import control_smplifyx_root_seed123 as seed
import run_smplify_temporal124 as temporal
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import read, save, sha, now

ROOT=temporal.ROOT
OUT=ROOT/'winterhat-continuation'
SRC=ROOT/'winterhat-input'
OLD=ROOT/'winterhat-temporal'
HERE=Path(__file__).resolve().parent


def verify_hook():
    root=ROOT.parent/'smplifyx-reference-control/official'
    original=entry.adapted_entry_source
    entry._root_seed_original_source=original
    before,_=original(root);after,_=seed.adapted_source(root)
    checks=dict(
        same_existing_body_root_hook=True,
        one_body_seed=after.count("pose_embedding.copy_(torch.as_tensor(kwargs['_seed_pose_embedding']")==1,
        one_root_seed=after.count("body_model.global_orient.copy_(torch.as_tensor(kwargs['_seed_global_orient']")==1,
        original_camera_initialization_prefix=before.split('        if try_both_orient:\n')[0]==after.split('        _seed_trace = dict(')[0],
        original_five_stage_loop='for opt_idx, curr_weights in enumerate(tqdm(opt_weights' in after,
        original_shoulder_gate=after.count('        if try_both_orient:\n')==1,
        temporal_loss_source_unchanged=sha(HERE/'smplify_temporal124.py')==read(OLD/'plan.json')['sources'][str(HERE/'smplify_temporal124.py')])
    assert all(checks.values()),checks
    return checks


def prepare():
    assert not OUT.exists(), 'One candidate only; preserve any existing output.'
    old=read(OLD/'plan.json')
    assert read(OLD/'final-review/result.json')['actual_visual_review']
    assert not read(OLD/'training-admission.json')['quality_accepted']
    assert read(ROOT/'hand-observation-coherence/control-verdict.json')['actual_visual_review']
    rows=read(SRC/'data/inputs.json')['records'];ids=[r['frame'] for r in rows]
    assert len(ids)==112 and set(ids)==set(old['ordered_fit_frames'])
    for row in rows:
        assert np.array_equal(row['w2c'],np.eye(4)), 'Root-seed scope requires a common camera-relative gauge.'
        assert np.array_equal(row['K_integer_centers'],rows[0]['K_integer_centers'])
    p=dict(old)
    p.update(updated=now(),state='registered',
        hypothesis='In the temporal-loss baseline every body solve still starts at zero. Test a coherent body/root initial pose from the newly fitted adjacent frame while retaining the same temporal objective.',
        method='Reuse control_smplifyx_root_seed123.adapted_source and run_smplify_temporal124.run. Only body/root initialization changes after original camera initialization. No new solver.',
        changed='Coherent body VPoser latent and global_orient initial state for all five original body stages.',
        preserved=['original camera initialization and frozen camera translation','all native observations and confidence','shared reference71 beta','fixed full K','same112 processing order and real timestamps','body/root temporal weights30','five stages,LBFGS,maxiters30','original orientation branch','zero face/hand initialization'],
        old_pose_initialization=False,seed_source='Only completed adjacent poses from this new run; original zero/mean anchor71.',
        historical_negative_control=str(ROOT.parent/'demo-vhap-initialization/fullbody-smplifyx/temporal-root-sequence/review/closeout.json'),
        historical_limit='Body/root propagation without the new temporal loss failed on demo59. This combination is unproven; do not rerun that old experiment or claim the zero start is an implementation bug.',
        comparison='Against complete112 temporal baseline: initial coherent pose changes. Hand-confidence gating from the rejected12-frame control is NOT applied.',
        risks=['Adjacent wrong poses can propagate.','Camera-relative root continuity is approximate under a moving camera.','A smooth wrong limb is still rejected.'],
        criteria=['Anchor71 must match the original temporal baseline within1e-6 model meters and1e-3 native pixels.',
                  'Every runtime seed must equal a newly completed adjacent result; camera initialization remains original.',
                  'Review all completed body/face/feet and side-depth evidence; no manual frame substitution.',
                  'Previously wrong13/15/16/43/96/99 must improve without new visible pose regression before Gaussian admission.'],
        budget=dict(candidates=1,frames=112,max_wall_seconds=4200,review_after_frames=13,review_wait_seconds=900,second_review_after_frames=64),
        stopping='Review after13 and64 frames; stop the same candidate on severe new pose failure or exhausted wall budget. No weight sweep, retry or stitched output.',
        sources={str(p):sha(p) for p in [Path(__file__),Path(seed.__file__),Path(temporal.__file__),Path(entry.__file__),HERE/'smplify_temporal124.py',HERE/'run_official_sequence123.py',HERE/'run_official_sequence123_topology.py',SRC/'data/inputs.json',SRC/'data/observations.npz',SRC/'length-contract.json',SRC/'reference-fit/winner-camera-space.npz',OLD/'plan.json',OLD/'official-fit/fit/fitted.npz']})
    checks=verify_hook()
    OUT.mkdir();save(OUT/'plan.json',p)
    save(OUT/'interface-check.json',dict(updated=now(),checks=checks,real_quality_proven=False))
    print('CONTINUATION_REGISTERED',len(ids),flush=True)


def run():
    p=read(OUT/'plan.json')
    assert not (OUT/'official-fit').exists(), 'Never resume or overwrite a candidate.'
    for path,digest in p['sources'].items():assert sha(path)==digest,path
    verify_hook()
    rows={r['frame']:r for r in read(SRC/'data/inputs.json')['records']}
    previous_context,previous_save=temporal.context,temporal.save
    old_source,old_config,old_fit=entry.adapted_entry_source,entry.load_official_config,entry.fit_one
    entry._root_seed_original_source=old_source
    bank={};records={};active={};started=time.monotonic()

    def config(root):
        c=old_config(root)
        c.update(_seed_pose_embedding=active['latent'],_seed_global_orient=active['root'],_seed_root_enabled=active['neighbour'] is not None)
        return c

    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name
        assert frame==p['ordered_fit_frames'][len(bank)]
        neighbour=min(bank,key=lambda n:(abs(rows[n]['time_seconds']-rows[frame]['time_seconds']),n)) if bank else None
        active.update(neighbour=neighbour,latent=np.zeros((1,32),np.float32),root=np.zeros((1,3),np.float32))
        if neighbour:
            active.update(latent=bank[neighbour]['pose_embedding'].copy(),root=bank[neighbour]['global_orient'].copy())
            assert abs(int(frame[-6:])-int(neighbour[-6:]))==1
        arrays,receipt=old_fit(**kwargs)
        trace={k:v.tolist() for k,v in entry._root_seed_trace.items()}
        assert np.array_equal(arrays['camera_translation'],np.asarray(trace['camera_translation']))
        if neighbour:assert np.array_equal(trace['body_fit_initial_root'],active['root'])
        parity=None
        if neighbour is None:
            with np.load(OLD/'official-fit/frames'/frame/'winner-camera-space.npz',allow_pickle=False) as before:
                parity={k:float(np.max(np.abs(before[k]-arrays[k]))) for k in ['vertices_camera','joints_projected_native','camera_translation','global_orient','pose_embedding','betas']}
            assert parity['vertices_camera']<=1e-6 and parity['joints_projected_native']<=1e-3,parity
        record=dict(frame=frame,neighbour=neighbour,neighbour_used_as_initialization=neighbour is not None,
                    initial_state=trace,initial_body_latent=active['latent'].tolist(),anchor_parity=parity)
        records[frame]=record;bank[frame]=arrays
        receipt.update(initialization='Original zero anchor, then same-run adjacent body/root initial state after original camera initialization',continuation=record)
        if len(bank)==p['budget']['second_review_after_frames']:
            np.savez_compressed(OUT/'review-pause-pose-064.npz',**arrays)
            save(OUT/'review-request-064.json',dict(updated=now(),state='awaiting_review',frames=list(bank),last_frame=frame,pid=os.getpid()))
            decision=OUT/'review-decision-064.json';wait=time.monotonic()
            while not decision.exists():
                if time.monotonic()-wait>p['budget']['review_wait_seconds'] or time.monotonic()-started>p['budget']['max_wall_seconds']:
                    raise RuntimeError('Registered review or wall deadline reached.')
                time.sleep(2)
            if not read(decision).get('continue_same_candidate',False):raise RuntimeError('Stopped by actual64-frame review.')
        return arrays,receipt

    def save_progress(path,value):
        if Path(path).name=='temporal-progress.json':
            for r in value['records']:r.update(records[r['frame']])
            value['initialization_source']='same-run adjacent body/root; no historical pose initialization'
        previous_save(path,value)

    temporal.context=lambda subject:(SRC,OUT,read(SRC/'selection.json'))
    temporal.save=save_progress
    entry.adapted_entry_source,entry.load_official_config,entry.fit_one=seed.adapted_source,config,fit
    try:
        temporal.run('winterhat')
        for path,digest in p['sources'].items():assert sha(path)==digest,path
        # Correct only the inherited description; all fitted arrays are untouched.
        f=OUT/'official-fit/input-provenance.json';q=read(f);q['initialization']='Original anchor then same-run adjacent body/root after original camera initialization';save(f,q)
    except BaseException as e:
        save(OUT/'continuation-failure.json',dict(updated=now(),error=repr(e),completed_frames=len(bank)))
        path=OUT/'temporal-progress.json'
        if path.exists():
            q=read(path);q.update(state='stopped_or_failed',pid=None,error=repr(e));save(path,q)
        raise
    finally:
        temporal.context,temporal.save=previous_context,previous_save
        entry.adapted_entry_source,entry.load_official_config,entry.fit_one=old_source,old_config,old_fit


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);args=ap.parse_args()
    (prepare if args.mode=='prepare' else run)()
