"""One controlled fresh comparison after diagnosing saturated opacity leakage."""
from pathlib import Path
import argparse, ast, importlib.util, shutil, sys

BASE=Path('/opt/gs-slam-lab/msv2-demo-transparent-20261104')
RUN=Path('/opt/gs-slam-lab/msv2-demo-transparent-bce-20261104')
OUT=Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo')

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def create():
    old=load('stopped_transparent',BASE/'experiment.py')
    diagnostic=old.read(BASE/'gate-diagnostic.json')
    assert not old.read(BASE/'gate1000/gate.json')['passed']
    assert diagnostic['optimizer_updates']==0 and diagnostic['model_unchanged']
    assert not RUN.exists()
    RUN.mkdir()
    shutil.copytree(BASE/'source',RUN/'source',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['config.json','initial-points.npz','training-inputs.json','native-inputs.json','superresolution.json','source-manifest.json']:
        shutil.copy2(BASE/name,RUN/name)
    code=(BASE/'experiment.py').read_text(encoding='utf-8').replace(str(BASE),str(RUN))
    ast.parse(code);(RUN/'experiment.py').write_text(code,encoding='utf-8')
    hooks=(BASE/'transparent_hooks.py').read_text(encoding='utf-8')
    previous='''    error = (alpha - mask).abs()
    silhouette = .5 * ((error * mask).sum() / mask.sum().clamp_min(1) +
                       (error * (1 - mask)).sum() / (1 - mask).sum().clamp_min(1))'''
    proposed='''    # Class-balanced Bernoulli log loss supervises alpha itself, independently
    # of RGB. Unlike L1 it retains pressure on saturated false-positive alpha.
    silhouette = -.5 * ((mask * alpha.clamp_min(1e-6).log()).sum() / mask.sum().clamp_min(1) +
                        ((1 - mask) * (1 - alpha).clamp_min(1e-6).log()).sum() / (1 - mask).sum().clamp_min(1))'''
    assert hooks.count(previous)==1
    hooks=hooks.replace(previous,proposed)
    ast.parse(hooks);(RUN/'transparent_hooks.py').write_text(hooks,encoding='utf-8')
    plan=old.read(BASE/'plan.json')
    plan.update(created=old.now(),method='SwinIR4x + MSV2 foreground RGB and balanced alpha cross entropy',
        comparator=str(BASE),single_main_variable='Silhouette loss: balanced L1 replaced by balanced Bernoulli cross entropy; coefficient remains 0.2',
        loss='0.8 foreground L1 + 0.2 foreground-window SSIM on straight RGB + 0.2 balanced alpha cross entropy',
        hypothesis='L1 gives weak gradients to saturated false-positive opacity. Cross entropy should reduce opaque leakage while keeping foreground coverage.',
        changes=['Balanced alpha L1 replaced by balanced alpha cross entropy'],
        fixed=['59 SwinIR4x RGB and masks','full K and extrinsics','foreground photo and SSIM support','foreground-only depth sampling','seed 6060','untrained initial hull','MSV2 schedule'],
        diagnosis=str(BASE/'gate-diagnostic.json'),failed_previous_gate=str(BASE/'gate1000/gate.json'),
        stop_policy='Same 1000-step gate against frozen original SwinIR4x; no threshold relaxation or parameter sweep.',
        pretraining_evidence='Frozen local opacity-logit gradient L1 norm increased about 6-14x with finite gradients at the same coefficient; not yet a training result.')
    old.save(RUN/'plan.json',plan)
    old.save(RUN/'adapter-manifest.json',{n:old.sha(RUN/n) for n in ['experiment.py','transparent_hooks.py','source/msv2/train.py']})
    print('FRESH_ALPHA_CE_EXPERIMENT_CREATED',RUN,flush=True)

def probe():
    sys.path.insert(0,str(RUN))
    helper=load('transparent_probe',Path(__file__).with_name('transparent_msv2_demo.py'))
    helper.RUN=RUN;helper.probe()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe']);args=p.parse_args()
    globals()[args.action]()
