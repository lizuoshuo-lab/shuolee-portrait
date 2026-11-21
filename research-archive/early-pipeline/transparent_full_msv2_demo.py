"""Run the user-selected transparent configuration for the full MSV2 schedule."""
from pathlib import Path
import argparse
import ast
import importlib.util
import shutil
import sys

BASE = Path('/opt/gs-slam-lab/msv2-demo-transparent-depthcheck-20261104')
RUN = Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
RECORD = Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo_18000')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def create():
    ex = load('selected_transparent', BASE / 'experiment.py')
    assert ex.read(BASE / 'rgba-probe.json')['passed']
    assert ex.read(BASE / 'gate3000/gate.json')['iteration'] == 3000
    assert not RUN.exists() and not RECORD.exists()
    RUN.mkdir()
    shutil.copytree(BASE / 'source', RUN / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    files = ['config.json', 'initial-points.npz', 'training-inputs.json',
             'native-inputs.json', 'superresolution.json', 'source-manifest.json',
             'transparent_hooks.py', 'frozen-baseline-diagnostic.json']
    identical = {}
    for name in files:
        shutil.copy2(BASE / name, RUN / name)
        assert ex.sha(BASE / name) == ex.sha(RUN / name)
        identical[name] = ex.sha(RUN / name)
    code = (BASE / 'experiment.py').read_text(encoding='utf-8')
    code = code.replace(str(BASE), str(RUN)).replace(str(ex.RECORD), str(RECORD))
    old = 'rgba.gate(sys.modules[__name__],self.gaussians,iteration,enforce=iteration>=3000)'
    assert code.count(old) == 1
    code = code.replace(old, 'rgba.gate(sys.modules[__name__],self.gaussians,iteration,enforce=False)')
    ast.parse(code)
    (RUN / 'experiment.py').write_text(code, encoding='utf-8')
    plan = ex.read(BASE / 'plan.json')
    plan.update(created=ex.now(), method='User-selected SwinIR4x transparent MSV2, full 18000-step schedule',
        selected_configuration=str(BASE), comparator='/opt/gs-slam-lab/msv2-demo-swinir4x-20261104',
        human_direction='手部是因为手里拿着东西，就用这版训练',
        single_main_variable='Complete the selected configuration to 18000 labels; numeric opacity diagnostics no longer abort training, as explicitly requested by the user.',
        changes=['Run to the full schedule; stage opacity statistics are recorded without early rejection'],
        comparability='Same transparent loss, source, seed, untrained hull, input RGB/masks, full K, extrinsics and MSV2 schedule as the selected 3000-step candidate; fresh initialization.',
        cached_unoptimized_hull=True, old_trained_gaussians_loaded=False,
        learned_gaussians_reused=False, optimizer_reinitialized=True,
        protocol_correction='The prior 3000-step stopped result and its failed numerical gate remain unchanged. User selected its visible pollution reduction and requested full training with the held object retained.',
        early_gate=dict(enforced=False, iterations=[1000, 3000, 8000, 18000], outside_opacity_mean_reference=.01),
        stop_policy='Finish at 18000 labels. Nonfinite loss, CUDA failure or insufficient memory still stop execution. Opacity or held-object appearance metrics are diagnostic only.',
        final_gate='Review all 59 source views and 12 orbit views; record black pollution, retained person and held object, and remaining unseen geometry separately. Do not alter source masks or fabricate missing surfaces.',
        prior_trial=str(BASE), prior_trial_accepted=False, user_selected_visible_improvement=True,
        pretraining_evidence=str(BASE / 'rgba-probe.json'),
        known_baseline_limitations='Held-object image/mask ambiguity retained at user request; missing back observations remain. No change to masks, cameras, loss coefficients or source images.')
    ex.save(RUN / 'plan.json', plan)
    ex.save(RUN / 'configuration-parity.json', dict(passed=True, selected_run=str(BASE),
        unchanged_files=identical, optimizer_updates=0, trained_state_copied=False,
        adapter_changes=['Run/output paths', 'enforce=False for existing diagnostic gate']))
    ex.save(RUN / 'adapter-manifest.json', {name: ex.sha(RUN / name)
        for name in ['experiment.py', 'transparent_hooks.py', 'source/msv2/train.py']})
    print('FULL_TRANSPARENT_RUN_CREATED', RUN, flush=True)


def probe():
    sys.path.insert(0, str(RUN))
    helper = load('transparent_probe', Path(__file__).with_name('transparent_msv2_demo.py'))
    helper.RUN = RUN
    helper.probe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['create', 'probe'])
    globals()[parser.parse_args().action]()
