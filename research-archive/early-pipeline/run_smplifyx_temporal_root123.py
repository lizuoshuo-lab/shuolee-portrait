"""Original per-frame SMPLify-X with explicit adjacent body/root initialization."""
import argparse
import inspect
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np
import control_smplifyx_root_seed123 as probe
import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence
from run_official_sequence123_topology import cache_static_topology

BODY,BASE=probe.BODY,probe.BASE
OUT=BODY/'temporal-root-sequence'
PLAN=BODY/'temporal-root-sequence-plan.json'
HANDOFF=BODY/'temporal-root-sequence-gpu-handoff.json'
read,save,sha,now=probe.read,probe.save,probe.sha,probe.now


def arguments(frames,inspect_only=False):
    args=probe.common.arguments(inspect_only)
    args.frame_ids=frames;args.output_dir=str(OUT)
    return args


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    reviewed=read(probe.OUT/'review-closeout.json')
    assert reviewed['numerical_criteria_passed'] and reviewed['actual_visual_review']
    args=arguments(None,True)
    _,rows,_,_,_,bound,provenance=sequence.load_inputs(args)
    ordered=sorted(rows,key=lambda r:(int(r['source_frame_index']),r['frame']))
    ids=[r['frame'] for r in ordered];anchor=args.reference_frame_id
    assert len(set(ids))==len(ids)==59
    index=ids.index(anchor)
    # Anchor was fixed before these controls; never rank frames by this run's fit.
    order=[anchor]+ids[index+1:]+ids[:index][::-1]
    schedule=[dict(frame=anchor,seed=None,direction='anchor')]
    for start,group,direction in [(anchor,ids[index+1:],'forward'),(anchor,ids[:index][::-1],'backward')]:
        previous=start
        for frame in group:
            schedule.append(dict(frame=frame,seed=previous,direction=direction));previous=frame
    assert [r['frame'] for r in schedule]==order
    firstK=np.asarray(ordered[0]['K_integer_centers'])
    for r in ordered:
        sequence.checked_image(r)
        assert np.array_equal(r['w2c'],np.eye(4)) and np.array_equal(r['K_integer_centers'],firstK)
    anchor_path=BASE/'frames'/anchor/'winner-camera-space.npz'
    for p in [Path(args.shared_betas_npz),anchor_path,probe.OUT/'review-closeout.json']:
        bound[str(p)]=sha(p)
    sources=[Path(__file__),Path(probe.__file__),Path(probe.common.__file__),Path(sequence.__file__),Path(entry.__file__),
        Path(__file__).with_name('run_official_sequence123_topology.py')]
    sources += [Path(args.official_root)/p for p in entry.PINNED]
    p=dict(updated=now(),state='prepared',anchor=anchor,temporal_roster=ids,processing_order=order,schedule=schedule,
        hypothesis='Use the complete fitted body and whole orientation as a coherent initialization from each newly fitted temporal neighbor, while retaining the original independent camera initialization and per-frame optimizer.',
        difference_from_probe='Within-run sequential seeds in both temporal directions from the already fixed anchor33; no frozen old neighbor pose is loaded.',
        anchor_policy='Existing fixed reference33, unchanged; original zero/mean fit with shared frozen beta, not picked from this run results.',
        preserved=['all59 observations and confidence','fixed complete K','existing shared beta','original camera initialization',
            'original shoulder-gated orientations','five original stages/losses/weights','LBFGS/maxiters30','zero hands/expression start','fresh per-stage optimizer'],
        original_solver_source_unchanged=True,no_frame_selection=True,no_result_based_retry=True,no_best_result_stitching=True,
        criteria=['Anchor must reproduce the original fixed33 result within1e-3 native pixel and1e-6 meter.',
            'Every original frame must complete exactly once and preserve camera/shape/topology contracts.',
            'Report all59 face/body/feet residuals, every individual elbow, actual SO3 root and joint continuity, with no residual-based exclusion.',
            'Require no new visible reversal/collapse in all59 native overlays and full timeline; no Gaussian admission from summary medians alone.',
            'Report unobserved hands/shoes and surface limitations. This fit is not quality acceptance or a15min end-to-end result.'],
        risks=['A predecessor error can propagate across a branch.','Camera-relative roots approximate short-time continuity under moving camera.',
            'Independent camera-init translation can still be inconsistent.'],
        runtime_budget_seconds=4500,expected_seconds=2300,random_seed=read(probe.PLAN)['random_seed'],
        automatic_gaussian_training=False,quality_accepted=False,goal_complete=False,
        input_check=provenance,input_hashes=bound,source_hashes={str(p.resolve()):sha(p) for p in sources})
    save(PLAN,p)
    print('PREPARED_FULL59',anchor,'forward',len(ids[index+1:]),'backward',len(ids[:index]),flush=True)


def verify_plan():
    p=read(PLAN)
    assert len(p['temporal_roster'])==59 and set(p['processing_order'])==set(p['temporal_roster'])
    for path,digest in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(path)==digest,path
    return p


def worker():
    probe.common.verify_handoff();p=verify_plan();assert not OUT.exists()
    old_source,old_config,old_fit=entry.adapted_entry_source,entry.load_official_config,entry.fit_one
    entry._root_seed_original_source=old_source
    active={};completed=[]
    def config(root):
        c=old_config(root);assert c['maxiters']==30 and c.get('side_view_thsh',25)==25
        c.update(_seed_pose_embedding=active['latent'],_seed_global_orient=active['root'],_seed_root_enabled=active['seed'] is not None)
        return c
    def fit(**kwargs):
        probe.common.verify_handoff();verify_plan()
        frame=Path(kwargs['result_path']).parent.name
        item=next(r for r in p['schedule'] if r['frame']==frame)
        assert frame not in completed and frame==p['processing_order'][len(completed)]
        active.update(item,latent=np.zeros((1,32),np.float32),root=np.zeros((1,3),np.float32))
        seed_path=None
        if item['seed']:
            assert item['seed'] in completed
            seed_path=OUT/'frames'/item['seed']/'winner-camera-space.npz'
            with np.load(seed_path,allow_pickle=False) as s:
                assert str(s['frame_id'].item())==item['seed']
                assert np.array_equal(s['betas'],np.asarray(kwargs['shared_betas']).reshape(1,10))
                assert np.array_equal(s['K_integer_centers'],kwargs['K_integer_centers'])
                active.update(latent=s['pose_embedding'].copy(),root=s['global_orient'].copy())
        topology=cache_static_topology(kwargs['body_model'])
        arrays,report=old_fit(**kwargs)
        trace={k:v.tolist() for k,v in entry._root_seed_trace.items()}
        if seed_path:assert np.array_equal(trace['body_fit_initial_root'],active['root'])
        assert np.array_equal(arrays['camera_translation'],trace['camera_translation'])
        anchor_parity=None
        if seed_path is None:
            with np.load(BASE/'frames'/frame/'winner-camera-space.npz',allow_pickle=False) as before:
                anchor_parity={k:dict(exact=bool(np.array_equal(before[k],arrays[k])),max_abs=float(np.max(np.abs(before[k]-arrays[k]))))
                    for k in ['vertices_camera','joints_projected_native','global_orient','pose_embedding','camera_translation','betas']}
            assert anchor_parity['vertices_camera']['max_abs']<=1e-6 and anchor_parity['joints_projected_native']['max_abs']<=1e-3,anchor_parity
        report.update(initialization='Original anchor zero/mean, then within-run adjacent body/root start after unchanged camera initialization',
            temporal_seed=item,seed_path=str(seed_path) if seed_path else None,seed_sha256=sha(seed_path) if seed_path else None,
            initial_state=trace,anchor_parity=anchor_parity,runtime_optimization=topology,original_loss_optimizer_unchanged=True)
        completed.append(frame)
        return arrays,report
    runner=inspect.getsource(sequence.run)
    lock="    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')\n    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)"
    assert runner.count(lock)==1
    runner=runner.replace(lock,'    verify_handoff()')
    default="initialization='Independent official zero/mean pose each frame; explicit shared shape only'"
    assert runner.count(default)==1
    runner=runner.replace(default,"initialization='Original anchor, then explicit within-run adjacent body/root start; shared shape fixed'")
    namespace=dict(sequence.run.__globals__,verify_handoff=probe.common.verify_handoff)
    exec(compile(runner,'<original-sequence-explicit-temporal-initialization>','exec'),namespace)
    entry.adapted_entry_source,entry.load_official_config,entry.fit_one=probe.adapted_source,config,fit
    started=time.monotonic()
    try:
        import torch
        random.seed(p['random_seed']);np.random.seed(p['random_seed']);torch.manual_seed(p['random_seed'])
        namespace['run'](arguments(p['processing_order']))
        assert completed==p['processing_order']
        # Restore temporal file order for downstream named-frame joins; no fit changes.
        fpath=OUT/'fit/fitted.npz';pre_sort=sha(fpath)
        with np.load(fpath,allow_pickle=False) as f:arrays={k:f[k].copy() for k in f.files}
        ids=arrays['frame_ids'].astype(str).tolist();order=[ids.index(v) for v in p['temporal_roster']]
        for key,value in arrays.items():
            if value.ndim>0 and value.shape[0]==59:arrays[key]=value[order]
        sequence.save_npz(fpath,**arrays)
        ipath=OUT/'data/training-inputs.json';inputs=read(ipath);rows={r['frame']:r for r in inputs['records']}
        inputs['records']=[rows[x] for x in p['temporal_roster']];inputs['initialization_plan_sha256']=sha(PLAN);save(ipath,inputs)
        receipt=read(OUT/'fit/official-fit-result.json')
        receipt.update(fit_sha256=sha(fpath),training_inputs_sha256=sha(ipath),processing_order=p['processing_order'],
            exported_temporal_order=p['temporal_roster'],processing_order_fit_sha256=pre_sort,initialization_plan_sha256=sha(PLAN))
        for field in ['frame_results','official_frame_results']:
            by_id={r['frame_id']:r for r in receipt[field]};receipt[field]=[by_id[x] for x in p['temporal_roster']]
        save(OUT/'fit/official-fit-result.json',receipt)
        verify_plan()
        save(OUT/'result.json',dict(state='complete_pending_all59_review',updated=now(),frames=59,elapsed_seconds=time.monotonic()-started,
            processing_order=p['processing_order'],temporal_order=p['temporal_roster'],anchor=p['anchor'],
            plan_sha256=sha(PLAN),fit_sha256=sha(fpath),quality_accepted=False,goal_complete=False,Gaussian_training_started=False))
    finally:entry.adapted_entry_source,entry.load_official_config,entry.fit_one=old_source,old_config,old_fit


def run(pid):
    import fcntl
    p=verify_plan();assert not OUT.exists()
    lease=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=probe.common.process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=now(),supervisor_pid=os.getpid(),baseline_pid=pid,identity=identity,command=cmd,state='suspending')
    save(HANDOFF,receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if probe.common.process(pid)[1]=='T':break
            time.sleep(.05)
        assert probe.common.process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],env=env,stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=p['runtime_budget_seconds'])
        receipt['control_finished']=True
    finally:
        try:
            if probe.common.process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT);receipt.update(state='resumed',baseline_state=probe.common.process(pid)[1])
            else:receipt['state']='baseline_identity_changed_not_signaled'
        except (FileNotFoundError,ProcessLookupError):receipt['state']='baseline_exited_before_resume'
        receipt['finished']=now();save(HANDOFF,receipt);print('TEMPORAL_ROOT_BASELINE',receipt['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker()
