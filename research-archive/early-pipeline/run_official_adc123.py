"""Run the human-approved fresh1000step ADC timing comparison, then stop."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json
import os
import shutil
import textwrap
import audit_gaussianavatars_params123 as audit
import run_official_gaussians123 as runner


def prepare():
    assert os.name == 'nt'
    a, o, p = audit.A, audit.O, audit.P
    proposal = audit.read(a/'candidate-proposal.json')
    assert proposal['state'] == 'prepared_awaiting_risk_approval'
    config = proposal['config']
    source = o/'shared-vertex-offset-1000'
    target = o/'shared-vertex-offset-adc-1000'
    plan_path = o/'shared-offset-adc-plan.json'
    assert not target.exists() and not plan_path.exists()
    code = (p/'src/shuolee_portrait/training.py').read_text(encoding='utf-8')
    ast.parse(code)
    begin = code.index('    options.densify_until_iter = schedule_steps\n')
    end = code.index('    if cfg.get(\'shared_static_offset\', False):\n        offset_lr', begin)
    fragment = compile(textwrap.dedent(code[begin:end]), '<actual-optional-ADC-interface>', 'exec')
    original = audit.defaults(p/'src/avatar_core/arguments/__init__.py')
    checks = []
    for timing in ({}, dict(densify_from_iter=300, densification_interval=60)):
        options = SimpleNamespace(**original)
        options.iterations = options.position_lr_max_steps = 18000
        records = []
        env = dict(cfg=timing, options=options, schedule_steps=18000, out=Path('not-written'),
            rt=SimpleNamespace(save=lambda path, data: records.append(data)),
            model=SimpleNamespace(training_setup=lambda args: None))
        exec(fragment, env)
        expected = dict(original, iterations=18000, position_lr_max_steps=18000, densify_until_iter=18000, **timing)
        assert vars(options) == expected and records == [expected]
        checks.append(dict(config=timing, effective_matches=True))
    for timing in (dict(densification_interval=0), dict(densify_from_iter=-1),
                   dict(densification_interval=True), dict(densify_from_iter=18000)):
        env['cfg'] = timing;env['options'] = SimpleNamespace(**original)
        try:
            exec(fragment, env)
        except ValueError:
            checks.append(dict(config=timing, rejected=True))
        else:
            raise AssertionError('Invalid timing accepted')
    assert config['shared_static_offset_lr'] == 1.e-6
    for name, digest in proposal['input_hashes'].items():
        assert audit.sha(source/name) == digest
    old_hashes = proposal['source_hashes']
    for name, digest in old_hashes.items():
        if name != 'src/shuolee_portrait/training.py':
            assert audit.sha(p/name) == digest
    assert audit.sha(a/'training-before-adc.py') == old_hashes['src/shuolee_portrait/training.py']
    linux = lambda path: '/mnt/'+path.as_posix()[0].lower()+path.as_posix()[2:]
    plan = dict(created=runner.now(), phase='official_ADC_timing_control',
        user_authorized=True, human_answer='进行1000步增密时序对照',
        human_question_id='call_bade889615f84b48aeabd41a11acb46a',
        source_root=linux(source), target_root=linux(target), config=config,
        actual_step_budget=1000, no_automatic_extension=True, fit_reused_not_refitted=True,
        new_gaussians_zero_offset_empty_optimizer=True, end_to_end_benchmark=False,
        input_hashes=proposal['input_hashes'], source_hashes={name:audit.sha(p/name) for name in old_hashes},
        original_source_hashes=old_hashes, existing_launcher_sha256=audit.sha(runner.__file__),
        control_launcher_sha256=audit.sha(__file__), expected_ADC_steps=proposal['expected_ADC_events_first1000'],
        timing_interface_checks=checks, baseline=str(source),
        reference_frame_id='frame_000051', training_started=False, quality_accepted=False)
    audit.save(plan_path, plan)
    proposal.update(state='approved_prepared', user_authorized_to_execute=True,
        human_answer=plan['human_answer'], human_question_id=plan['human_question_id'],
        source_modified=True, plan=str(plan_path))
    audit.save(a/'candidate-proposal.json', proposal)
    for path in (audit.R/'queue-state.json',audit.REC/'queue-state.json',audit.C/'local-research-status.json'):
        state=audit.read(path)
        state.update(state='official_ADC_timing_approved_prepared',updated_at=plan['created'],
            next='One fresh1000step ADC timing comparison; stop and review.',training_running=False,
            latest_ADC_control=dict(plan=str(plan_path), root=str(target), human_authorized=True,
                pending_user_decision=False, steps=0, budget=1000, quality_accepted=False))
        state['latest_parameter_audit']['approval_pending']=False
        audit.save(path,state)
    print(json.dumps(dict(plan=str(plan_path),checks=checks,training_sha256=plan['source_hashes']['src/shuolee_portrait/training.py'])))


def run(plan_path):
    assert os.name == 'posix'
    plan = runner.read(plan_path)
    assert plan['user_authorized'] and plan['actual_step_budget'] == 1000
    assert plan['config']['densify_from_iter'] == 300 and plan['config']['densification_interval'] == 60
    assert plan['config']['shared_static_offset_lr'] == 1.e-6
    assert runner.sha(__file__) == plan['control_launcher_sha256']
    assert runner.sha(runner.__file__) == plan['existing_launcher_sha256']
    source, target = Path(plan['source_root']), Path(plan['target_root'])
    for name, digest in plan['input_hashes'].items(): assert runner.sha(source/name) == digest
    for name, digest in plan['source_hashes'].items(): assert runner.sha(runner.PROJECT/name) == digest
    assert not target.exists(), 'A fresh controlled run is required.'
    target.mkdir()
    for name in plan['input_hashes']:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/name,dest)
    runner.save(target/'control-input.json',dict(plan=plan,plan_sha256=runner.sha(plan_path),
        fitting_reused=True,Gaussian_from_scratch=True,end_to_end_benchmark=False))
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    runner.SOURCES={name:plan['source_hashes']['src/shuolee_portrait/'+name] for name in ('training.py','export.py')}
    runner.CFG=plan['config']
    original=runner.review_decision
    def stop_only(path, request):
        decision=original(path, request)
        if decision['decision'] != 'stop': raise ValueError('Approved actual budget is1000; extension is not authorized.')
        return decision
    runner.review_decision=stop_only
    runner.run(SimpleNamespace(root=str(target),project_root=str(runner.PROJECT),assets_root=str(runner.ASSETS)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    prepare() if args.prepare else run(Path(args.plan))
