"""Human-approved fresh 1000-step static-offset learning-rate control."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json
import os
import shutil
import run_official_gaussians123 as runner


def prepare():
    assert os.name == 'nt'
    base = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
    cycle = base/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
    project = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
    source = cycle/'shared-vertex-offset-1000'
    target = cycle/'shared-vertex-offset-lr1e-5-1000'
    plan_path = cycle/'shared-offset-lr1e-5-plan.json'
    assert not target.exists() and not plan_path.exists()
    old = runner.read(cycle/'shared-vertex-offset-long-plan.json')
    config = dict(old['config'], shared_static_offset_lr=1.e-5)
    # Execute the actual new group-selection block with a small fake optimizer.
    import numpy as np
    code = (project/'src/shuolee_portrait/training.py').read_text(encoding='utf-8')
    ast.parse(code)
    begin = code.index("    if cfg.get('shared_static_offset', False):\n        offset_lr")
    end = code.index('    pipe = ', begin)
    import textwrap
    fragment = compile(textwrap.dedent(code[begin:end]), '<actual-lr-interface>', 'exec')
    checks = []
    for supplied in (None, 1.e-5, 0., float('nan')):
        groups = [dict(name='xyz', lr=.0123), dict(name='static_offset', lr=1.e-6)]
        env = dict(np=np, cfg=dict(shared_static_offset=True),
            model=SimpleNamespace(optimizer=SimpleNamespace(param_groups=groups)),
            init={'shared_static_offset': {}})
        if supplied is not None: env['cfg']['shared_static_offset_lr'] = supplied
        try:
            exec(fragment, env)
            assert supplied is None or supplied == 1.e-5
            assert groups[0]['lr'] == .0123
            assert groups[1]['lr'] == (1.e-6 if supplied is None else supplied)
            assert env['init']['shared_static_offset']['optimizer_lr'] == groups[1]['lr']
            checks.append(dict(input='default' if supplied is None else supplied, passed=True))
        except ValueError:
            assert supplied == 0. or np.isnan(supplied)
            checks.append(dict(input='zero' if supplied == 0. else 'NaN', rejected=True))
    linux = lambda p: '/mnt/'+p.as_posix()[0].lower()+p.as_posix()[2:]
    plan = dict(created=runner.now(), phase='shared_offset_learning_rate_control',
        human_answer='进行1,000步学习率对照', human_question_id='call_77c83a40504f44459740e6c5ff819065',
        user_authorized=True, changed_variable='static_offset Adam lr: 1e-6 -> 1e-5',
        source_root=linux(source), target_root=linux(target), config=config,
        actual_step_budget=1000, schedule_steps=18000, no_automatic_extension=True,
        initial_offset='zeros', optimizer_initial_state_entries=0, training_from_scratch=True,
        fit_reused_not_refitted=True, end_to_end_benchmark=False, reference_frame_id='frame_000051',
        baseline=str(source/'gaussian-run/step-001000'),
        unchanged=['94 poses/shared beta/full K/extrinsics/images/masks', 'native2704x1520/cache94',
            '200000 initial Gaussians/seed6060', 'losses/binding/densification/other optimizer rates'],
        runtime_only_difference=dict(change='immutable LBS parent integers cached on host',
            evidence=linux(cycle/'shared-vertex-offset-18000/offset-diagnostic/host-topology-adoption/result.json'),
            all94_vertices_and_fixed_adjoint_gradients_exact=True,
            native_rasterizer_backward_is_not_bitwise_deterministic=True),
        review=['seven same-camera native crops: face/body/feet', 'shared offset and triangle normal reversals',
                '36 continuous unseen views', 'world PLY covariance/SH native readback and camera package UI'],
        stop_conditions=['nonfinite gradients/loss', 'native PLY parity failure', '1000 steps: always stop for comparison'],
        input_hashes={name:runner.sha(source/name) for name in old['input_hashes']},
        source_hashes={name:runner.sha(project/name) for name in old['source_hashes']},
        existing_launcher_sha256=runner.sha(runner.__file__),
        control_launcher_sha256=runner.sha(__file__), lr_interface_checks=checks,
        upstream_complete_reproduction=False, quality_accepted=False)
    assert plan['input_hashes'] == old['input_hashes']
    runner.save(plan_path, plan)
    proposal_path = cycle/'shared-offset-rate-proposal.json'
    proposal = runner.read(proposal_path)
    proposal.update(state='approved_prepared', user_decision_received=True, human_answer=plan['human_answer'],
        approved_at=plan['created'], implementation_changed=True, training_started=False,
        plan=str(plan_path), authority='Direct human answer to the recorded risk question.')
    runner.save(proposal_path, proposal)
    for path in (base/'shuolee/current/robust-portrait-122/queue-state.json',
            base/'shuolee/current/local-clothed-surface-123/queue-state.json',
            Path(__file__).parent.parent/'shuolee-coordination/local-research-status.json'):
        state=runner.read(path)
        state.update(state='shared_offset_lr1e5_approved_prepared', updated_at=plan['created'],
            next='One fresh1000step LR1e-5 comparison; no automatic extension.', training_running=False,
            latest_shared_offset_lr=dict(plan=str(plan_path), root=str(target), user_authorized=True,
                pending_user_decision=False, steps=0, budget=1000, quality_accepted=False))
        if 'latest_shared_offset_diagnostic' in state:
            state['latest_shared_offset_diagnostic']['user_decision_pending']=False
        runner.save(path,state)
    print(json.dumps(dict(plan=str(plan_path), checks=checks, source_hashes=plan['source_hashes'])))


def run(plan_path):
    assert os.name == 'posix'
    plan=runner.read(plan_path)
    assert plan['user_authorized'] and plan['actual_step_budget']==1000
    assert plan['config']['shared_static_offset_lr']==1.e-5
    assert runner.sha(__file__)==plan['control_launcher_sha256']
    assert runner.sha(runner.__file__)==plan['existing_launcher_sha256']
    source,target=Path(plan['source_root']),Path(plan['target_root'])
    for name,digest in plan['input_hashes'].items(): assert runner.sha(source/name)==digest,name
    for name,digest in plan['source_hashes'].items(): assert runner.sha(runner.PROJECT/name)==digest,name
    assert not target.exists(), 'Fresh trial only.'
    target.mkdir()
    for name in plan['input_hashes']:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    runner.save(target/'control-input.json',dict(plan=plan,plan_sha256=runner.sha(plan_path),
        fit_reused_not_refitted=True,Gaussian_from_scratch=True,end_to_end_benchmark=False))
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    runner.SOURCES={name:plan['source_hashes']['src/shuolee_portrait/'+name] for name in ('training.py','export.py')}
    runner.CFG=plan['config']
    original=runner.review_decision
    def stop_only(path,request):
        result=original(path,request)
        if result['decision']!='stop': raise ValueError('Human-approved budget is1000; extension is not authorized.')
        return result
    runner.review_decision=stop_only
    runner.run(SimpleNamespace(root=str(target),project_root=str(runner.PROJECT),assets_root=str(runner.ASSETS)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    prepare() if args.prepare else run(Path(args.plan))
