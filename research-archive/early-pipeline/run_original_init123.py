"""Reuse the current official-fit launcher for one original-initialization control."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json
import os
import shutil
import run_official_gaussians123 as runner


def prepare():
    import audit_gaussianavatars_params123 as a
    source = a.O/'shared-vertex-offset-adc-1000'
    target = a.O/'shared-vertex-offset-original-init-10000'
    path = a.A/'bound-original-init-10000-plan.json'
    assert not path.exists() and not target.exists()
    gradient = a.read(a.A/'offset-drive-audit/result.json')
    assert gradient['state'] == 'complete_no_updates' and gradient['optimizer_updates'] == 0
    cfg = a.read(source/'gaussian-run/config.json')
    keys = ['initial_points','iterations','training_height','seed','image_cache_frames',
        'export_frame_mode','shared_static_offset','shared_static_offset_lr',
        'densify_from_iter','densification_interval']
    config = {key:cfg[key] for key in keys}
    assert config['initial_points'] == 200000 and config['shared_static_offset_lr'] == 1.e-6
    config.update(initial_points=20908, executed_step_budget=10000)
    training = a.P/'src/shuolee_portrait/training.py'
    tree = ast.parse(training.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='train_exploratory_official_smplx')
    env = dict(Path=Path, _train_static=lambda *args, **kwargs:kwargs)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(training),'exec'),env)
    checks = []
    for budget in (None,1000,10000,18000):
        c = dict(iterations=18000)
        if budget is not None: c['executed_step_budget']=budget
        assert env[node.name]('dummy',c)['step_limit']==budget
        checks.append(dict(budget=budget,valid=True))
    for budget in (True,0,-1,18001,1000.5):
        try: env[node.name]('dummy',dict(iterations=18000,executed_step_budget=budget))
        except ValueError: checks.append(dict(budget=budget,rejected=True))
        else: raise AssertionError('Invalid budget accepted')
    files=['fit/fitted.npz','data/training-inputs.json','fit/official-fit-result.json']
    source_files=['src/shuolee_portrait/training.py','src/shuolee_portrait/export.py',
        'src/avatar_core/scene/smplx_gaussian_model.py','src/avatar_core/scene/gaussian_model.py']
    linux=lambda p:'/mnt/'+p.as_posix()[0].lower()+p.as_posix()[2:]
    plan=dict(updated=runner.now(),state='prepared_awaiting_specific_risk_approval',
        user_goal='Retain SMPL binding; improve outward surface placement around10000 steps',
        user_authorized_to_execute=False,source_root=linux(source),target_root=linux(target),
        config=config,actual_step_budget=10000,review_wait_step=1000,
        scientific_change='Original one-Gaussian-per-triangle centroid, unit local covariance initialization instead of200k area samples',
        unchanged=['94 actual official SMPLify-X poses','cameras','mask','native2704x1520',
            'shared canonical offset1e-6','triangle binding','seed6060','ADC300/60','18k position-LR schedule'],
        schedule_length_18000_preserved_for_isolation=True,hard_stop_step=10000,
        offset_target_is_not_guaranteed=True,uniform_inflation=False,new_solver=False,
        review_requirements=['Original/current trained SMPL-X/Gaussian at the same seven cameras',
            'Offset outward/tangent magnitude and canonical triangle normal reversals',
            'Head/hands/feet protection and profile clothing floating artifacts',
            'Stop at1000 if new folding or gross coverage collapse; continue the same live optimizer only after actual review'],
        risks=['20908 initial points may reduce early detail or coverage',
            'Continued bound optimization can stretch local triangles or reveal wrong pose alignment',
            'No promise of continuous skirt geometry from a fixed body topology or unseen surfaces'],
        extra_experiment_or_threshold_sweep=False, no_cross_experiment_resume=True,
        learned_parameters_fresh=True,fit_reused_not_refitted=True,end_to_end_benchmark=False,
        no_training_started=True,quality_accepted=False,
        source_hashes={p:a.sha(a.P/p) for p in source_files},
        input_hashes={p:a.sha(source/p) for p in files},
        launcher_sha256=a.sha(__file__),reused_launcher_sha256=a.sha(runner.__file__),
        budget_interface_checks=checks,gradient_audit_sha256=a.sha(a.A/'offset-drive-audit/result.json'))
    a.save(path,plan)
    print(json.dumps(dict(plan=str(path),checks=len(checks),config=config),ensure_ascii=False))


def run(path):
    assert os.name=='posix'
    plan=runner.read(path)
    assert plan['user_authorized_to_execute'] is True, 'Human risk approval is required for this prepared run.'
    assert plan['actual_step_budget']==plan['config']['executed_step_budget']
    assert 1000<=plan['actual_step_budget']<=18000 and plan['config']['initial_points']==20908
    assert runner.sha(__file__)==plan['launcher_sha256']
    assert runner.sha(runner.__file__)==plan['reused_launcher_sha256']
    for p,h in plan['source_hashes'].items(): assert runner.sha(runner.PROJECT/p)==h,p
    source,target=Path(plan['source_root']),Path(plan['target_root'])
    for p,h in plan['input_hashes'].items(): assert runner.sha(source/p)==h,p
    target.mkdir(exist_ok=False)
    for p in plan['input_hashes']:
        dest=target/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/p,dest)
    runner.save(target/'control-input.json',dict(plan=plan,plan_sha256=runner.sha(path)))
    runner.CFG=plan['config']
    runner.SOURCES={name:plan['source_hashes']['src/shuolee_portrait/'+name] for name in ('training.py','export.py')}
    runner.run(SimpleNamespace(root=str(target),project_root=str(runner.PROJECT),assets_root=str(runner.ASSETS)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--plan')
    args=p.parse_args()
    prepare() if args.prepare else run(Path(args.plan))
