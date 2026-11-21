"""One equivalent-root initialization control around the existing VHAP runner."""
import argparse
import difflib
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
import traceback
import fcntl
import numpy as np
from scipy.spatial.transform import Rotation

import fit_vhap_pose_prior_sum123 as base
import fit_smplify_vhap_surface123 as bridge
from root_axis_angle_continuity123 import unwrap_root_axis_angles, checks
from run_official_gaussians123 import read,save,sha,now

ROOT=base.joint.ROOT
OUT=ROOT/'smplify-vhap-root-continuity'
PLAN=ROOT/'smplify-vhap-root-continuity-plan.json'
ORIGINAL_INITIALIZE=bridge.initialize


def process(pid):
    p=Path('/proc')/str(pid)
    state=(p/'stat').read_text().split(') ',1)[1].split()
    return (p/'cmdline').read_bytes().replace(b'\0',b' ').decode(),state[0],state[19]


def suspended_baseline():
    cmd,state,identity=process(int(os.environ['SHUOLEE_CONTROL_BASELINE_PID']))
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state=='T'
    assert identity==os.environ['SHUOLEE_CONTROL_BASELINE_IDENTITY']


def checked_initialize(t,rows):
    import torch
    receipts=ORIGINAL_INITIALIZE(t,rows)
    plan=read(PLAN);before={k:bridge.digest_tensor(getattr(t,k)) for k in bridge.PARAMS}
    original=t.global_orient.detach().cpu().numpy().copy()
    anchor=next(i for i,r in enumerate(rows) if r['frame']==plan['initialization_frame'])
    changed=unwrap_root_axis_angles(original,anchor).astype(np.float32)
    assert np.array_equal(changed[anchor],original[anchor])
    with torch.no_grad():t.global_orient.copy_(torch.as_tensor(changed,device='cuda'))
    matrix_error=float(np.max(np.abs(Rotation.from_rotvec(original).as_matrix()-Rotation.from_rotvec(changed).as_matrix())))
    assert matrix_error<1e-6
    parity=[]
    for i,row in enumerate(rows):
        t.clear_cache()
        with np.load(bridge.SOURCE/'frames'/row['frame']/'winner-camera-space.npz',allow_pickle=False) as z:
            expected_vertices=z['vertices_camera'];expected_uv=z['joints_projected_native']
        with torch.no_grad():
            v,_,j,_,_=t.forward_smplx(torch.tensor([i]))
            K=torch.as_tensor(row['K_integer_centers'],device='cuda',dtype=j.dtype)
            q=j@K.T;uv=q[...,:2]/q[...,2:]
            ve=float((v-torch.as_tensor(expected_vertices,device='cuda')).abs().max())
            pe=float((uv-torch.as_tensor(expected_uv,device='cuda')).abs().max())
        assert ve<5e-6 and pe<.002,(row['frame'],ve,pe)
        parity.append(dict(frame=row['frame'],vertices_max_m=ve,projection_max_px=pe))
    assert all(bridge.digest_tensor(getattr(t,k))==h for k,h in before.items() if k!='global_orient')
    assert t.global_step==0
    save(OUT/'root-branch-interface.json',dict(updated=now(),state='passed',frames=59,
        anchor=rows[anchor]['frame'],anchor_vector_unchanged=True,rotation_matrix_max_abs=matrix_error,
        original_root=original.tolist(),equivalent_root=changed.tolist(),parity=parity,
        max_original_delta_deg=float(np.degrees(np.linalg.norm(np.diff(original,axis=0),axis=1)).max()),
        max_equivalent_delta_deg=float(np.degrees(np.linalg.norm(np.diff(changed,axis=0),axis=1)).max()),
        non_root_parameters_unchanged=True,optimizer_updates=0,no_pose_interpolation=True,
        quality_accepted=False))
    return receipts


def runner_source():
    original=textwrap.dedent(inspect.getsource(base.joint.main))
    old="    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)"
    assert original.count(old)==1
    candidate=original.replace(old,'    suspended_baseline()  # Supervisor owns serialized GPU lease.',1)
    return original,candidate


def prepare():
    assert not OUT.exists() and not PLAN.exists(),'Read existing control; do not repeat.'
    p=read(ROOT/'smplify-vhap-pose-prior-sum-plan.json')
    for path,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(path)==h,path
    anchor=p['roster'].index(p['initialization_frame'])
    values=[]
    for frame in p['roster']:
        with np.load(bridge.SOURCE/'frames'/frame/'winner-camera-space.npz',allow_pickle=False) as z:values.append(z['global_orient'][0])
    values=np.asarray(values);out=unwrap_root_axis_angles(values,anchor)
    original,candidate=runner_source()
    p.update(created=now(),state='registered_before_execution',route='Original short VHAP control with equivalent root-angle initialization',
        hypothesis='Raw-axis-angle temporal subtraction penalizes a representation branch as a near-full turn. A deterministic equivalent initialization removes that artificial jump without changing physical pose.',
        changes=['Only choose equivalent global_orient branches around existing reference33 before the unchanged original fit.'],
        matched_control=str(ROOT/'smplify-vhap-pose-prior-sum'),optimizer_updates_budget=2270,
        runtime_budget_seconds=1200,initialization='Fresh identical official SMPLify-X per-frame values; no old VHAP or Gaussian checkpoint and no frame copying.',
        risk='Root parameter conditioning and the raw-vector temporal penalty change. Actual elbow ambiguity may remain; body/face/hands/feet may regress. No automatic Gaussian admission.',
        authorization='User delegated minimal fitting decisions; same full59 pipeline, one bounded causal initialization test.',
        cpu_contract=checks(),anchor_index=anchor,
        initial_matrix_max_abs=float(np.max(np.abs(Rotation.from_rotvec(values).as_matrix()-Rotation.from_rotvec(out).as_matrix()))),
        before_adjacent_max_deg=float(np.degrees(np.linalg.norm(np.diff(values,axis=0),axis=1)).max()),
        after_adjacent_max_deg=float(np.degrees(np.linalg.norm(np.diff(out,axis=0),axis=1)).max()),
        runner_patch=''.join(difflib.unified_diff(original.splitlines(True),candidate.splitlines(True),fromfile='existing runner',tofile='serialized GPU runner')),
        stop_conditions=['Any initial mesh/projection/identity mismatch stops before optimization.','One full59 500+1770 run; no threshold or weight search.','Review all body/face/hand/foot frames against matched short control before admission.'],
        quality_accepted=False,gaussian_training_started=False)
    for f in [Path(__file__).resolve(),Path(__file__).with_name('root_axis_angle_continuity123.py'),Path(__file__).with_name('fit_vhap_pose_prior_sum123.py')]:p['source_hashes'][str(f)]=sha(f)
    save(PLAN,p)
    print('ROOT_CONTINUITY_REGISTERED',p['before_adjacent_max_deg'],p['after_adjacent_max_deg'],flush=True)


def train():
    suspended_baseline()
    p=read(PLAN)
    assert not OUT.exists()
    base.joint.OUT=OUT;base.joint.PLAN=PLAN;base.joint.make_tracker=base.make_tracker
    bridge.initialize=checked_initialize
    original,candidate=runner_source()
    namespace=dict(base.joint.main.__globals__);namespace['suspended_baseline']=suspended_baseline
    exec(compile(candidate,'<existing-vhap-runner-serialized-gpu>','exec'),namespace)
    namespace['main']()
    (OUT/'runner-source.py').write_text(candidate)
    control=read(ROOT/'smplify-vhap-pose-prior-sum/config.json')
    assert read(OUT/'config.json')==control,'Configuration must match the short control exactly'
    for name in ('result.json','progress.json'):
        r=read(OUT/name);r.update(controlled_change=p['changes'][0],matched_control=p['matched_control'],
            original_optimizer_loss_scheduler=False,original_optimizer_and_scheduler=True,
            same_losses_as='smplify-vhap-pose-prior-sum',root_rotations_initially_identical=True,
            fresh_initialization=True,optimizer_state_reused=False,quality_accepted=False)
        save(OUT/name,r)


def run(pid):
    p=read(PLAN);assert not OUT.exists()
    for path,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(path)==h,path
    lock=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=now(),pid=pid,command=cmd,identity=identity,state='suspending')
    handoff=ROOT/'smplify-vhap-root-continuity-gpu-handoff.json';save(handoff,receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if process(pid)[1]=='T':break
            time.sleep(.05)
        assert process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with open(Path(__file__).with_name('vhap-root-continuity-train.log'),'x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',__file__,'train'],env=env,stdout=log,stderr=subprocess.STDOUT,
                check=True,timeout=p['runtime_budget_seconds'])
    finally:
        try:
            if process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT);receipt.update(state='resumed',trainer_state=process(pid)[1])
            else:receipt.update(state='baseline_identity_changed_not_signaled')
        except ProcessLookupError:
            receipt.update(state='baseline_exited_before_resume')
        except FileNotFoundError:
            receipt.update(state='baseline_exited_before_resume')
        receipt['finished']=now();save(handoff,receipt);print('ROOT_CONTROL_BASELINE',receipt['state'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','train']);p.add_argument('--pid',type=int)
    args=p.parse_args()
    try:
        prepare() if args.mode=='prepare' else run(args.pid) if args.mode=='run' else train()
    except BaseException as e:
        save(ROOT/('smplify-vhap-root-continuity-'+args.mode+'-failure.json'),dict(updated=now(),error=repr(e),traceback=traceback.format_exc(),auto_retry=False))
        raise
