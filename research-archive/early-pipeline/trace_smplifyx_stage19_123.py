"""Observe one unchanged original SMPLify-X solve; never replace its solver."""
import argparse
import inspect
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import control_smplifyx_root_seed123 as probe

common, entry, sequence = probe.common, probe.entry, probe.sequence
BODY = common.BODY
PREVIOUS = BODY/'forward-guarded-sequence'
OUT = BODY/'stage19-original-trajectory'
PLAN = BODY/'stage19-original-trajectory-plan.json'
HANDOFF = BODY/'stage19-original-trajectory-gpu-handoff.json'
FRAME = 'frame_000019'
read, save, sha, now = sequence.read, sequence.save, sequence.sha, sequence.now


def arguments(inspect_only=False):
    args = common.arguments(inspect_only)
    args.frame_ids = [FRAME]
    args.output_dir = str(OUT)
    return args


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def prepare():
    assert not PLAN.exists() and not OUT.exists()
    args = arguments(True)
    _, rows, targets, K, _, bound, provenance = sequence.load_inputs(args)
    sequence.checked_image(rows[0])
    receipt_path = PREVIOUS/'frames'/FRAME/'result.json'
    receipt = read(receipt_path)
    seed_path = Path(receipt['seed_path'])
    candidate = PREVIOUS/'frames'/FRAME/'candidate-00-temporal.npz'
    seed, expected = load_npz(seed_path), load_npz(candidate)
    assert str(seed['frame_id'].item()) == 'frame_000018'
    assert np.array_equal(seed['global_orient'], receipt['initial_state']['body_fit_initial_root'])
    assert np.array_equal(K[0], expected['K_integer_centers'])
    assert np.array_equal(seed['betas'], expected['betas'])
    assert read(PREVIOUS/'frames'/FRAME/'candidate-00-report.json')['orientation_count'] == 1
    entry._root_seed_original_source = entry.adapted_entry_source
    source, delta = probe.adapted_source(args.official_root)
    compile(source, '<existing-original-neighbor-seed-adapter>', 'exec')
    files = [Path(__file__), Path(probe.__file__), Path(common.__file__), Path(entry.__file__),
        Path(sequence.__file__), Path(probe.cache_static_topology.__code__.co_filename)]
    files += [Path(args.official_root)/p for p in entry.PINNED]
    inputs = [receipt_path, seed_path, candidate, PREVIOUS/'frames'/FRAME/'candidate-00-report.json',
        PREVIOUS/'review/stage19-replay-result.json', Path(args.shared_betas_npz),
        Path(args.assets_root)/'smplx/SMPLX_NEUTRAL.npz', Path(rows[0]['image']), Path(rows[0]['mask'])]
    bound.update({str(p):sha(p) for p in inputs})
    save(PLAN, dict(updated=now(), state='prepared', frame=FRAME, seed_path=str(seed_path), candidate_path=str(candidate),
        hypothesis='Locate the actual original stage/step that moves the valid seeded arm away, separating stage-weight tradeoffs from early stopping. Observational wrappers only.',
        fitter_calls=1, orientation_count=1, body_stages=5, maxiters=30, runtime_budget_seconds=300,
        random_seed=read(BODY/'forward-guarded-sequence-plan.json')['random_seed'],
        unchanged=['original loss/weights/curriculum', 'LBFGS and stopping tests', 'shared frozen beta',
            'independent camera initialization', 'native 2D/confidence/full K', 'actual frame18 body/root seed', 'zero hands/expression'],
        instrumentation='Wrap original monitor calls and optimizer.step only to copy parameters, returned losses, closure counts and gradients. Never alter a tensor, gradient, loss, optimizer or stopping decision during fitting.',
        endpoint_parity=dict(vertices_camera=2e-6, joints_projected_native=.002, pose_embedding=2e-6,
            global_orient=2e-6, camera_translation=1e-7, betas=0),
        interpretation_gate='Do not interpret trajectory as the previous solve unless final candidate parity passes.',
        stop_conditions=['One frame and one solve only.', 'No followup full59 or loss change from this script.',
            'Save failures and resume the exact suspended Gaussian process in finally.'],
        fit_observations_are_independent_truth=False, no_quality_acceptance=True, no_Gaussian_training=True,
        stage_adapter_delta=delta, source_hashes={str(p.resolve()):sha(p) for p in files},
        input_hashes=bound, input_check=provenance, native_wh=rows[0]['wh']))
    print('PREPARED_ORIGINAL_STAGE19_TRAJECTORY', flush=True)


def verify_plan():
    p = read(PLAN)
    for path, digest in {**p['source_hashes'], **p['input_hashes']}.items():
        assert sha(path) == digest, path
    return p


def install_monitor_trace(fitting, stages, states):
    import torch
    cls = fitting.FittingMonitor
    original_create, original_run = cls.create_fitting_closure, cls.run_fitting
    create_signature, run_signature = inspect.signature(original_create), inspect.signature(original_run)

    def create(self, *args, **kwargs):
        context = create_signature.bind(self, *args, **kwargs)
        context.apply_defaults()
        closure = original_create(self, *args, **kwargs)
        closure._stage19_context = context.arguments
        return closure

    def run(self, *args, **kwargs):
        bound = run_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        optimizer, closure, body = values['optimizer'], values['closure'], values['body_model']
        context = closure._stage19_context
        camera, latent, loss = context['camera'], context['pose_embedding'], context['loss']
        is_body = type(loss).__name__ == 'SMPLifyLoss'
        index = sum(s['kind']=='body' for s in stages) if is_body else -1
        stage = dict(kind='body' if is_body else 'camera', stage=index, steps=[], maxiters=self.maxiters,
            ftol=self.ftol, gtol=self.gtol, loss_class=type(loss).__name__)
        stage['_context'] = context
        stage['weights'] = {k:v.detach().cpu().numpy().copy() for k,v in loss.named_buffers() if k.endswith('weight')}
        if is_body: stage['_joint_weights'] = context['joint_weights'].detach().clone()
        stages.append(stage)

        def snapshot(label):
            item = dict(label=label, stage=index, kind=stage['kind'],
                parameters={n:v.detach().cpu().numpy().copy() for n,v in body.named_parameters()},
                latent=latent.detach().cpu().numpy().copy(), translation=camera.translation.detach().cpu().numpy().copy())
            states.append(item)
            return len(states)-1

        stage['before_state'] = snapshot('before')
        original_step = optimizer.step
        def step(*step_args, **step_kwargs):
            closures_before = self.steps
            result = original_step(*step_args, **step_kwargs)
            gradients = [v.grad for v in values['params'] if v.grad is not None]
            original_gmax = max([float(torch.abs(g.view(-1).max())) for g in gradients], default=0.)
            true_gmax = max([float(g.abs().max()) for g in gradients], default=0.)
            stage['steps'].append(dict(step=len(stage['steps']), returned_loss=float(result.detach()),
                closure_calls=self.steps-closures_before, original_gradient_max=original_gmax,
                absolute_gradient_max=true_gmax, state=snapshot('after_step')))
            return result
        optimizer.step = step
        try:
            result = original_run(self, *args, **kwargs)
        finally:
            optimizer.step = original_step
        stage['after_state'] = snapshot('after_stage')
        stage['returned_monitor_loss'] = result
        last = stage['steps'][-1]
        reason = 'iteration_limit'
        if not np.isfinite(last['returned_loss']): reason = 'nonfinite'
        elif len(stage['steps']) > 1 and self.ftol > 0:
            previous = stage['steps'][-2]['returned_loss']; current = last['returned_loss']
            change = (previous-current)/max(abs(previous),abs(current),1.)
            stage['last_relative_loss_change'] = change
            if change <= self.ftol: reason = 'original_ftol'
        if reason=='iteration_limit' and last['original_gradient_max'] < self.gtol: reason='original_gtol'
        stage['stop_reason_from_original_tests'] = reason
        return result

    cls.create_fitting_closure, cls.run_fitting = create, run


def analyse_states(body, stages, states, target, output):
    import torch
    rows = []; exports = {}
    with torch.no_grad():
        for s in stages:
            c = s['_context']; loss, camera = c['loss'], c['camera']
            if s['kind']=='body': loss.reset_loss_weights(s['weights'])
            indices = [s['before_state']] + [r['state'] for r in s['steps']] + [s['after_state']]
            for index in indices:
                saved = states[index]
                body.reset_params(**saved['parameters'])
                camera.translation.copy_(torch.as_tensor(saved['translation'], device=target.device))
                latent = torch.as_tensor(saved['latent'], device=target.device)
                posed = c['vposer'].decode(latent, output_type='aa').reshape(1,63)
                geometry = body(body_pose=posed, return_verts=True, return_full_pose=True)
                projected = camera(geometry.joints)
                residual = projected-target[...,:2]
                distance = torch.linalg.vector_norm(residual, dim=-1)[0]
                row = dict(state=index, kind=s['kind'], stage=s['stage'], label=saved['label'],
                    left_elbow_px=float(distance[6]), right_elbow_px=float(distance[3]),
                    left_wrist_px=float(distance[7]), right_wrist_px=float(distance[4]))
                if s['kind']=='body':
                    weights=s['_joint_weights']
                    total=loss(geometry, camera=camera, gt_joints=target[...,:2], joints_conf=target[...,2],
                        body_model_faces=body.faces_tensor.view(-1), joint_weights=weights, pose_embedding=latent, use_vposer=True)
                    terms=((weights*target[...,2]).square()[...,None]*loss.data_weight.square()*loss.robustifier(residual)).sum(-1)[0]
                    row.update(total=float(total), body_data=float(terms[:19].sum()), feet_data=float(terms[19:25].sum()),
                        hands_data=float(terms[25:67].sum()), face_data=float(terms[67:].sum()), prior_total=float(total-terms.sum()))
                rows.append(row)
                if index in [s['before_state'],s['after_state']]:
                    prefix=f'state_{index:03d}'
                    exports[prefix+'_vertices_camera']=(geometry.vertices+camera.translation[:,None]).cpu().numpy()
                    exports[prefix+'_joints_projected_native']=projected.cpu().numpy()
        for i,item in enumerate(states):
            for name,value in item['parameters'].items():exports[f'state_{i:03d}_parameter_{name}']=value
            exports[f'state_{i:03d}_pose_embedding']=item['latent']
            exports[f'state_{i:03d}_camera_translation']=item['translation']
        exports.update(faces=body.faces, K_integer_centers=output['K_integer_centers'], observed_native_keypoints=target.cpu().numpy())
        sequence.save_npz(OUT/'trajectory.npz', **exports)
    return rows


def worker():
    common.verify_handoff(); p=verify_plan(); assert not OUT.exists(); OUT.mkdir()
    start=time.monotonic()
    save(OUT/'progress.json', dict(state='tracing_original_fitter', pid=os.getpid(), updated=now()))
    originals = entry.adapted_entry_source, entry.load_official_config, entry.load_official
    try:
        import torch
        import smplx
        random.seed(p['random_seed']); np.random.seed(p['random_seed']); torch.manual_seed(p['random_seed'])
        args=arguments(); _, rows, targets, K, _, _, _=sequence.load_inputs(args)
        utils, _, _, _=originals[2](args.official_root)
        mapping=utils.smpl_to_openpose(model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
        runtime,assets=Path(args.runtime_root),Path(args.assets_root)
        loader=entry.module(runtime/'src/avatar_core/shuolee_vposer.py','_stage19_vposer')
        loader.MODEL,loader.CODE=assets/'vposer/V02_05',runtime/'third_party/human_body_prior'
        vposer,_,vposer_meta=loader.load_vposer('cuda')
        body=smplx.SMPLX(str(assets/'smplx/SMPLX_NEUTRAL.npz'),batch_size=1,num_betas=10,num_expression_coeffs=10,
            use_pca=True,num_pca_comps=12,flat_hand_mean=False,use_face_contour=False,create_body_pose=False,create_transl=False,
            joint_mapper=utils.JointMapper(mapping),dtype=torch.float32).to('cuda')
        topology=probe.cache_static_topology(body)
        seed=load_npz(p['seed_path']); expected=load_npz(p['candidate_path'])
        stages,states=[],[]
        entry._root_seed_original_source=originals[0]
        def config(root):
            c=originals[1](root)
            assert c['maxiters']==30 and c['side_view_thsh']==25
            c.update(_seed_pose_embedding=seed['pose_embedding'],_seed_global_orient=seed['global_orient'],_seed_root_enabled=True)
            return c
        def load(root):
            values=originals[2](root); install_monitor_trace(values[2],stages,states); return values
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=probe.adapted_source,config,load
        fit_start=time.monotonic()
        arrays,report=entry.fit_one(official_root=args.official_root,image=sequence.checked_image(rows[0]),keypoints=targets,
            K_integer_centers=K[0],body_model=body,vposer2=vposer,result_path=OUT/'official-result.pkl',
            shared_betas=seed['betas'],freeze_betas=True,device='cuda',diff_path=OUT/'adapter.diff')
        fit_seconds=time.monotonic()-fit_start
        assert report['orientation_count']==1 and len(stages)==6
        parity={k:float(np.max(np.abs(arrays[k]-expected[k]))) for k in p['endpoint_parity']}
        save(OUT/'endpoint-parity.json',dict(differences=parity,tolerances=p['endpoint_parity']))
        sequence.save_npz(OUT/'traced-winner.npz',**arrays)
        assert all(parity[k]<=limit for k,limit in p['endpoint_parity'].items()),parity
        body_stages=[s for s in stages if s['kind']=='body']
        first=states[body_stages[0]['before_state']]
        assert np.array_equal(first['latent'],seed['pose_embedding'])
        assert np.array_equal(first['parameters']['global_orient'],seed['global_orient'])
        assert all(np.count_nonzero(first['parameters'][k])==0 for k in ['left_hand_pose','right_hand_pose','jaw_pose','expression','leye_pose','reye_pose'])
        evaluations=analyse_states(body,stages,states,torch.as_tensor(targets,device='cuda'),arrays)
        serial=[]
        for s in stages:
            serial.append({k:({n:v.tolist() for n,v in value.items()} if k=='weights' else value)
                for k,value in s.items() if not k.startswith('_')})
        verify_plan(); common.verify_handoff()
        save(OUT/'result.json',dict(state='original_trajectory_complete_pending_visual_review',updated=now(),frame=FRAME,
            source_hashes_unchanged=True,plan_sha256=sha(PLAN),endpoint_parity=parity,endpoint_parity_passed=True,
            fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-start,stages=serial,evaluations=evaluations,
            initial_body_root_exact=True,hands_face_initialized_zero=True,original_report=report,topology_cache=topology,
            vposer=vposer_meta,quality_accepted=False,goal_complete=False,no_new_training_candidate=True,
            limits=['One failing frame diagnostic; not an all59 improvement.', 'Detector residuals are not independent physical truth.']))
        save(OUT/'progress.json',dict(state='complete_pending_visual_review',pid=None,updated=now()))
        for s in serial:
            first=next(r for r in evaluations if r['state']==s['before_state'])
            last=next(r for r in evaluations if r['state']==s['after_state'])
            print(s['kind'],s['stage'],len(s['steps']),s['stop_reason_from_original_tests'],first['left_elbow_px'],last['left_elbow_px'],flush=True)
    except BaseException as error:
        save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        save(OUT/'progress.json',dict(state='failed',pid=None,updated=now()))
        raise
    finally:
        entry.adapted_entry_source,entry.load_official_config,entry.load_official=originals


def run(pid):
    import fcntl
    p=verify_plan();assert not OUT.exists()
    lease=open('/tmp/shuolee-body-export123.lock','a+');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cmd,state,identity=common.process(pid)
    assert 'train_demo_body_upstream_offset123.py train' in cmd and state not in ('T','Z')
    receipt=dict(started=now(),supervisor_pid=os.getpid(),baseline_pid=pid,identity=identity,command=cmd,state='suspending')
    save(HANDOFF,receipt)
    try:
        os.kill(pid,signal.SIGSTOP)
        for _ in range(100):
            if common.process(pid)[1]=='T':break
            time.sleep(.05)
        assert common.process(pid)[1]=='T'
        env=dict(os.environ,SHUOLEE_CONTROL_BASELINE_PID=str(pid),SHUOLEE_CONTROL_BASELINE_IDENTITY=identity)
        with Path(__file__).with_suffix('.log').open('x',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'worker'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['runtime_budget_seconds'])
        receipt['control_finished']=True
    finally:
        try:
            if common.process(pid)[2]==identity:
                os.kill(pid,signal.SIGCONT);receipt.update(state='resumed',baseline_state=common.process(pid)[1])
            else:receipt['state']='baseline_identity_changed_not_signaled'
        except (FileNotFoundError,ProcessLookupError):receipt['state']='baseline_exited_before_resume'
        receipt['finished']=now();save(HANDOFF,receipt);print('STAGE19_BASELINE',receipt['state'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','run']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='worker':worker()
    else:run(args.pid)
