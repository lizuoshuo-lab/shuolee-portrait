"""Register/run one all-frame temporal candidate using the original fitter."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS
import official_smplifyx_entry123 as entry
import run_official_sequence123_topology as upstream
from smplify_temporal124 import install_loss_factory, verify

HERE=Path(__file__).resolve().parent
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
ROOT=REC/'cross-subject-sh0-124'

def context(subject):
    if subject!='winterhat':raise ValueError('Register each additional subject explicitly before running.')
    source=ROOT/'winterhat-input'
    return source,ROOT/'winterhat-temporal',read(source/'selection.json')

def prepare(subject):
    src,out,sel=context(subject)
    assert not out.exists(), 'Keep the prior candidate; do not rerun.'
    rows=read(src/'data/inputs.json')['records']
    ids=[r['frame'] for r in rows];ts=np.array([r['time_seconds'] for r in rows])
    assert len(ids)==len(set(ids)) and np.all(np.diff(ts)>0)
    ref=ids.index(sel['reference_frame'])
    # Start at the already preregistered good reference, then expand to actual
    # adjacent frames on both sides. No result-dependent ordering or selection.
    order=sorted(range(len(rows)),key=lambda i:(abs(ts[i]-ts[ref]),ts[i],ids[i]))
    out.mkdir()
    plan=dict(updated=now(),subject=subject,state='registered',user='SMPLify这个也加上连续帧约束',
        hypothesis='Independent monocular poses can switch depth branches; a same-run adjacent rotation prior may reduce jumps without harming native image alignment.',
        method='Original SMPLify-X five stages,30 maxiter,LBFGS,fullK,VPoser,bending and shape policy; additive body/root rotation temporal loss only. No post-hoc pose filtering or neighbour pose initialization.',
        source_root=str(src),reference_frame=sel['reference_frame'],ordered_fit_frames=[ids[i] for i in order],
        source_roster=len(rows),training_frames=sel['training_frames'],review_only_frames=sel['review_only_frames'],
        timestamp_seconds=ts.tolist(),nominal_dt_seconds=float(np.median(np.diff(ts))),
        temporal=dict(body_weight=30.,root_weight=30.,distance='.5 squared Frobenius distance between joint rotation matrices',
            time_weight='(median adjacent source dt / observed neighbour dt)^2',neighbour='Nearest already fitted same-run chronological neighbour, detached',
            jaw_face_hand_rotation_temporal=False,camera_translation_temporal=False,from_frame_one=False),
        fitting_input_frames='All112 existing preprocessed frames; eight is Gaussian RGB budget, not pose trajectory length.',
        comparison='Compare the same12 earlier registered baseline frames; adding intermediate frame poses and temporal coupling is a combined sequence treatment, not a claim of temporal-weight-only causality.',
        shape='Frozen original reference71 SMPLify-X betas for this controlled comparison; not a cold-start end-to-end claim.',
        interpenetration=False,collision='Retain baseline off state for this temporal comparison; separately downloaded collision source not enabled.',
        budget=dict(candidates=1,frames=len(rows),max_wall_seconds=4200,review_after_frames=13,review_wait_seconds=900),
        stopping='Pause after13 actual fits for native/body/depth review; no Gaussian training unless full pose review passes. Stop on nonfinite loss, input mismatch, runtime budget or severe new anatomy failure. No automatic weight sweep.',
        quality_accepted=False,gaussian_training_started=False,
        sources={str(p):sha(p) for p in [Path(__file__),HERE/'smplify_temporal124.py',Path(entry.__file__),
             src/'data/inputs.json',src/'data/observations.npz',src/'length-contract.json',src/'reference-fit/winner-camera-space.npz']})
    save(out/'plan.json',plan)
    save(out/'interface-check.json',dict(updated=now(),checks=verify(),real_quality_proven=False))
    print('TEMPORAL_REGISTERED',len(rows),plan['ordered_fit_frames'][:13],flush=True)

def run(subject):
    src,out,sel=context(subject);p=read(out/'plan.json')
    assert not (out/'official-fit').exists(), 'Never resume another candidate.'
    for path,digest in p['sources'].items():assert sha(path)==digest,path
    rows={r['frame']:r for r in read(src/'data/inputs.json')['records']}
    order=p['ordered_fit_frames'];bank={};receipts=[];start=time.monotonic()
    original=entry.fit_one;loader=entry.load_official
    def fit_one(**kwargs):
        if time.monotonic()-start>p['budget']['max_wall_seconds']:raise RuntimeError('Temporal candidate wall budget exhausted.')
        frame=Path(kwargs['result_path']).parent.name;t=rows[frame]['time_seconds']
        neighbour=min(bank,key=lambda x:(abs(rows[x]['time_seconds']-t),x)) if bank else None
        dt=abs(t-rows[neighbour]['time_seconds']) if neighbour else p['nominal_dt_seconds']
        record=dict(frame=frame,neighbour=neighbour,dt_seconds=dt,prior_is_not_ground_truth=True,
                    neighbour_used_as_initialization=False)
        def load(root):
            u,pr,f,o=loader(root)
            install_loss_factory(f,bank.get(neighbour),dt,p['nominal_dt_seconds'],record)
            return u,pr,f,o
        entry.load_official=load
        try:arrays,receipt=original(**kwargs)
        finally:entry.load_official=loader
        full=np.zeros((1,165),np.float32)
        full[:,:3]=arrays['global_orient'];full[:,3:66]=arrays['body_pose_axis_angle']
        bank[frame]=full
        for key in ('_last_original','_last_temporal'):
            if key in record:record[key[1:]]=float(record.pop(key).cpu())
        receipt['temporal_extension']=record
        receipt['upstream_loss_monitor_optimizer_unchanged']=False
        receipt['upstream_solver_and_baseline_terms_unchanged']=True
        receipts.append(record);save(out/'temporal-progress.json',dict(updated=now(),state='fitting',pid=os.getpid(),frames=len(receipts),total=len(order),records=receipts))
        if len(receipts)==p['budget']['review_after_frames']:
            # The completed arrays are saved by the outer runner after return;
            # stage a read-only checkpoint here for the explicit review pause.
            np.savez_compressed(out/'review-pause-pose.npz',**arrays)
            save(out/'review-request.json',dict(updated=now(),state='awaiting_review',frames=list(bank),last_frame=frame,pid=os.getpid(),deadline_seconds=p['budget']['review_wait_seconds']))
            wait=time.monotonic()
            while not (out/'review-decision.json').exists():
                if time.monotonic()-wait>p['budget']['review_wait_seconds']:raise RuntimeError('Review deadline; no automatic continuation.')
                time.sleep(2)
            if not read(out/'review-decision.json').get('continue_same_candidate',False):raise RuntimeError('Stopped by actual pose review.')
        return arrays,receipt
    entry.fit_one=fit_one
    sys.argv=['run_official_sequence123_topology.py','--official-root',str(REC/'smplifyx-reference-control/official'),
        '--data-root',str(src),'--runtime-root',str(PROJECT),'--assets-root',str(ASSETS),'--project-root',str(PROJECT),
        '--length-contract-json',str(src/'length-contract.json'),'--reference-frame-id',sel['reference_frame'],
        '--shared-betas-npz',str(src/'reference-fit/winner-camera-space.npz'),
        '--output-dir',str(out/'official-fit'),'--device','cuda','--frame-ids',*order]
    try:upstream.main()
    finally:entry.fit_one=original;entry.load_official=loader
    save(out/'temporal-progress.json',dict(updated=now(),state='completed',pid=None,frames=len(receipts),total=len(order),records=receipts,quality_accepted=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','run']);ap.add_argument('subject');a=ap.parse_args()
    (prepare if a.mode=='prepare' else run)(a.subject)
