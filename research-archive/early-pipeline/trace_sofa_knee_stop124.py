"""One exact replay; observe existing monitor decisions without changing them."""
import argparse
import inspect
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

import audit_sofa_knee_objective124 as audit
import trace_smplifyx_stage19_123 as trace
from run_sofa_temporal_warmstart124 import stage_config
from smplify_temporal124 import install_loss_factory
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster

entry = trace.entry
read, save, sha, now = trace.read, trace.save, trace.sha, trace.now
OUT = audit.RUN / 'knee-stopping-replay'


def target():
    row = read(audit.SRC / 'data/inputs.json')['records'][15]
    obs = audit.load_npz(audit.SRC / 'data/observations.npz')
    confidence, _ = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    x = np.zeros((1, 118, 3), np.float32)
    dst, src = np.r_[entry.COCO, 19:25, 25:67, 67:118], np.r_[0:23, 91:133, 40:91]
    x[0,dst,:2], x[0,dst,2] = obs['keypoints'][15,src], confidence[15,src]
    return row, mask_outside_raster(x, row['wh'])[0]


def verify():
    p = read(OUT / 'plan.json')
    for path, digest in p['frozen'].items():
        assert sha(path) == digest, path
    return p


def prepare():
    assert not OUT.exists()
    assert read(audit.RUN / 'quality-stop-decision.json')['pid'] is None
    result = read(audit.OUT / 'result.json')
    assert result['replay_final_loss_accepted']
    row, _ = target()
    files = [Path(__file__), Path(audit.__file__), Path(trace.__file__), Path(entry.__file__), Path(trace.probe.__file__),
             Path(inspect.getsourcefile(stage_config)), Path(__file__).with_name('smplify_temporal124.py'),
             Path(__file__).with_name('run_official_sequence123_topology.py'),
             audit.OUT / 'result.json', audit.RUN / 'plan.json', audit.SRC / 'data/inputs.json',
             audit.SRC / 'data/observations.npz', Path(row['image'])]
    files += [audit.frame(audit.RUN, i, name) for i in (16,17) for name in ('result.json','winner-camera-space.npz')]
    files += [audit.OFFICIAL / name for name in entry.PINNED]
    OUT.mkdir()
    save(OUT / 'plan.json', dict(updated=now(), frame='frame_000016',
        hypothesis='CPU fixed-state audit found a lower-energy saved16 state under the identical objective. Observe the original solver stopping and knee trajectory to distinguish iteration exhaustion, early stopping and an incorrect seed basin.',
        diagnostic_only=True, new_training_candidate=False, fitter_calls=1, max_seconds=300,
        original_stage_indices=[3,4], original_solver_loss_weights_stop_tests_unchanged=True,
        previous_states_used_only_for_exact_diagnostic_replay=True, pose_substitution=False,
        instrumentation='Reuse the observer of original optimizer.step and monitor. Serialize before post-fit analysis. No solver tensors or gradients changed during fitting.',
        parity_tolerances=dict(vertices_camera=2e-6,joints_projected_native=.002,pose_embedding=2e-6,
                               global_orient=2e-6,camera_translation=1e-7,betas=0),
        stopping='One replay only. Stop and preserve on endpoint mismatch; do not relax tolerances, launch a sequence or select its pose.',
        frozen={str(p):sha(p) for p in files}))
    print('KNEE_STOP_REPLAY_REGISTERED', flush=True)


def worker():
    p = verify()
    assert not (OUT / 'progress.json').exists()
    save(OUT / 'progress.json', dict(updated=now(), state='diagnostic_replay', pid=os.getpid()))
    originals = entry.adapted_entry_source, entry.load_official_config, entry.load_official
    start = time.monotonic()
    try:
        import torch
        import smplx
        torch.set_num_threads(4)
        row, targets = target()
        previous, expected = audit.load_npz(audit.frame(audit.RUN,17)), audit.load_npz(audit.frame(audit.RUN,16))
        utils, _, _, _ = originals[2](audit.OFFICIAL)
        mapping = utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        loader = entry.module(audit.PROJECT/'src/avatar_core/shuolee_vposer.py','_knee_stop_vposer124')
        loader.MODEL, loader.CODE = audit.ASSETS/'vposer/V02_05', audit.PROJECT/'third_party/human_body_prior'
        vposer, _, _ = loader.load_vposer('cuda')
        body = smplx.SMPLX(str(audit.ASSETS/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,
            create_transl=False,joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        trace.probe.cache_static_topology(body)
        stages, states, receipt = [], [], {}
        neighbour = np.zeros((1,165),np.float32)
        neighbour[:,:3], neighbour[:,3:66] = previous['global_orient'], previous['body_pose_axis_angle']
        source = inspect.getsource(trace.install_monitor_trace)
        old = "type(loss).__name__ == 'SMPLifyLoss'"
        assert source.count(old) == 1
        scope = dict(trace.install_monitor_trace.__globals__)
        exec(compile(source.replace(old,'isinstance(loss, fitting.SMPLifyLoss)'),'<temporal-observer-recognition>','exec'),scope)
        observer = scope['install_monitor_trace']
        entry._root_seed_original_source = originals[0]

        def config(root):
            c = stage_config(originals[1](root), False)
            assert c['maxiters'] == 30
            c.update(_seed_pose_embedding=previous['pose_embedding'],_seed_global_orient=previous['global_orient'],_seed_root_enabled=True)
            return c

        def load(root):
            values = originals[2](root)
            dt = abs(row['time_seconds']-read(audit.SRC/'data/inputs.json')['records'][16]['time_seconds'])
            install_loss_factory(values[2],neighbour,dt,read(audit.RUN/'plan.json')['nominal_dt_seconds'],receipt)
            observer(values[2],stages,states)
            return values

        entry.adapted_entry_source, entry.load_official_config, entry.load_official = trace.probe.adapted_source, config, load
        arrays, report = entry.fit_one(official_root=audit.OFFICIAL,image=trace.sequence.checked_image(row),keypoints=targets,
            K_integer_centers=np.array(row['K_integer_centers']),body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',
            shared_betas=previous['betas'],freeze_betas=True,device='cuda',diff_path=OUT/'adapter.diff')
        fit_seconds = time.monotonic()-start
        serial = [{k:({n:v.tolist() for n,v in value.items()} if k=='weights' else value) for k,value in s.items() if not k.startswith('_')} for s in stages]
        save(OUT/'monitor-stages.json',dict(stages=serial,updated=now()))
        trace.sequence.save_npz(OUT/'traced-winner.npz',**arrays)
        parity = {k:float(np.max(np.abs(arrays[k]-expected[k]))) for k in p['parity_tolerances']}
        save(OUT/'endpoint-parity.json',dict(differences=parity,tolerances=p['parity_tolerances']))
        assert all(parity[k]<=limit for k,limit in p['parity_tolerances'].items()), parity
        assert report['orientation_count']==1 and len(stages)==3
        trace.OUT = OUT
        evaluations = trace.analyse_states(body,stages,states,torch.as_tensor(targets,device='cuda'),arrays)
        knees = []
        with torch.no_grad():
            for s in stages:
                if s['kind']!='body': continue
                c = s['_context']
                for index in [s['before_state']]+[r['state'] for r in s['steps']]+[s['after_state']]:
                    saved = states[index]
                    body.reset_params(**saved['parameters'])
                    c['camera'].translation.copy_(torch.as_tensor(saved['translation'],device='cuda'))
                    aa = c['vposer'].decode(torch.as_tensor(saved['latent'],device='cuda'),output_type='aa').reshape(1,63)
                    output = body(body_pose=aa,return_verts=True,return_full_pose=True)
                    distance = (c['camera'](output.joints)-torch.as_tensor(targets[:,:,:2],device='cuda')).norm(dim=-1)
                    knees.append(dict(state=index,stage=s['stage']+3,label=saved['label'],left_knee_px=float(distance[0,13])))
        verify()
        save(OUT/'result.json',dict(updated=now(),state='replay_complete',endpoint_parity_passed=True,endpoint_parity=parity,
            fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-start,stages=serial,evaluations=evaluations,
            left_knee_trajectory=knees,original_report=report,quality_accepted=False,gaussian_admitted=False,
            diagnostic_only=True,no_new_training_candidate=True,causal_conclusion_reviewed=False))
        save(OUT/'progress.json',dict(updated=now(),state='complete',pid=None))
        print('KNEE_REPLAY_ENDPOINT_PARITY_PASSED',parity,flush=True)
    except BaseException as exc:
        save(OUT/'failure.json',dict(updated=now(),error=repr(exc)))
        save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None))
        raise
    finally:
        entry.adapted_entry_source,entry.load_official_config,entry.load_official = originals


def run():
    import fcntl
    p = verify()
    assert not (OUT/'progress.json').exists()
    lease = open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with (OUT/'replay.log').open('x') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],stdout=log,stderr=subprocess.STDOUT,
                       timeout=p['max_seconds'],check=True)
    print('KNEE_REPLAY_DONE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker'])
    args=parser.parse_args();dict(prepare=prepare,run=run,worker=worker)[args.mode]()
