"""Cover the depth-reinitialization stage with frozen transparent-loss settings."""
from pathlib import Path
import importlib.util,ast,shutil,argparse,sys

BASE=Path('/opt/gs-slam-lab/msv2-demo-transparent-bce-20261104')
RUN=Path('/opt/gs-slam-lab/msv2-demo-transparent-depthcheck-20261104')

def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def create():
    ex=load('prior_bce',BASE/'experiment.py')
    evidence=ex.read('/opt/gs-slam-lab/msv2-pollution-diagnostic-20261104/stages.json')
    assert evidence['optimizer_updates']==0 and evidence['models_unchanged']
    assert not ex.read(BASE/'gate1000/gate.json')['passed']
    assert not RUN.exists();RUN.mkdir()
    shutil.copytree(BASE/'source',RUN/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for n in ['config.json','initial-points.npz','training-inputs.json','native-inputs.json','superresolution.json','source-manifest.json']:
        shutil.copy2(BASE/n,RUN/n)
    code=(BASE/'experiment.py').read_text(encoding='utf-8').replace(str(BASE),str(RUN))
    code=code.replace('        if iteration==1000:', '        if iteration in (1000,3000,8000,18000):')
    code=code.replace('rgba.gate(sys.modules[__name__],self.gaussians)', 'rgba.gate(sys.modules[__name__],self.gaussians,iteration,enforce=iteration>=3000)')
    ast.parse(code);(RUN/'experiment.py').write_text(code,encoding='utf-8')
    hooks=(BASE/'transparent_hooks.py').read_text(encoding='utf-8')
    hooks=hooks.replace('def gate(ex, model):','def gate(ex, model, iteration=1000, enforce=True):')
    hooks=hooks.replace("baseline = Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104/model/point_cloud/iteration_1000/point_cloud.ply')",
        "baseline = Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104/model/point_cloud') / f'iteration_{iteration}/point_cloud.ply'")
    hooks=hooks.replace('old = GaussianModel(0); old.load_ply(str(baseline)); old.active_sh_degree = 0',
        'old = GaussianModel(model.max_sh_degree); old.load_ply(str(baseline)); old.active_sh_degree = model.active_sh_degree')
    hooks=hooks.replace("out = ex.RUN / 'gate1000'", "out = ex.RUN / f'gate{iteration}'")
    hooks=hooks.replace("t['outside_alpha_mean'] <= max(.5 * b['outside_alpha_mean'], .01)", "t['outside_alpha_mean'] <= .01")
    hooks=hooks.replace('dict(passed=passed, criteria=criteria, means=means, checks=checks,',
        'dict(passed=passed, enforced=enforce, iteration=iteration, criteria=criteria, means=means, checks=checks,')
    hooks=hooks.replace("print('TRANSPARENT_1000_GATE',", "print('TRANSPARENT_STAGE_GATE', iteration,")
    hooks=hooks.replace('    if not passed:', '    if enforce and not passed:')
    hooks=hooks.replace("raise RuntimeError('Transparent 1000-step protection gate failed; inspect gate1000/gate.json before additional training')",
        "raise RuntimeError(f'Transparent {iteration}-step gate failed; inspect gate{iteration}/gate.json')")
    ast.parse(hooks);(RUN/'transparent_hooks.py').write_text(hooks,encoding='utf-8')
    plan=ex.read(BASE/'plan.json')
    plan.update(created=ex.now(),method='SwinIR4x transparent MSV2 with post-depth stage validation',
        single_main_variable='No loss or training parameter changes relative to the prior BCE trial; extend observation past the previously unexecuted depth reconstruction.',
        changes=['Additional stage coverage and checkpoint diagnostics only'],
        pretraining_evidence=str('/opt/gs-slam-lab/msv2-pollution-diagnostic-20261104/stages.json'),
        protocol_correction='Both prior runs remain failed at 1000. That point preceded the depth-support fix at 2000. The new run records 1000 as a diagnostic and enforces the same absolute one-percent outside-alpha target at 3000,8000,18000.',
        early_gate=dict(iteration=3000,outside_opacity_mean_max=.01,inside_opacity_drop_max=.03,foreground_l1_ratio_max=1.2),
        stop_policy='Stop at 3000 if absolute outside-alpha exceeds 0.01 or subject protection fails; repeat at 8000 and18000. Do not change weights or acceptance thresholds after viewing results.',
        learned_gaussians_reused=False,optimizer_reinitialized=True,
        prior_trial=str(BASE),prior_trial_accepted=False)
    ex.save(RUN/'plan.json',plan)
    ex.save(RUN/'adapter-manifest.json',{n:ex.sha(RUN/n) for n in ['experiment.py','transparent_hooks.py','source/msv2/train.py']})
    shutil.copy2('/opt/gs-slam-lab/msv2-demo-transparent-20261104/frozen-baseline-diagnostic.json',RUN/'frozen-baseline-diagnostic.json')
    print('POST_DEPTH_VALIDATION_CREATED',RUN,flush=True)

def probe():
    sys.path.insert(0,str(RUN))
    helper=load('transparent_probe',Path(__file__).with_name('transparent_msv2_demo.py'));helper.RUN=RUN;helper.probe()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe']);args=p.parse_args();globals()[args.action]()
