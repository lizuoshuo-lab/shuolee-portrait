"""Keep the MSV2 model fixed; add only an exterior-boundary mask penalty."""
from pathlib import Path
import argparse, ast, importlib.util, shutil, sys

BASE=Path('/opt/gs-slam-lab/msv2-demo-transparent-full-20261104')
RUN=Path('/opt/gs-slam-lab/msv2-demo-boundary-mask-20261104')
RECORD=Path('/mnt/d/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比/mask_penalty4x')
DIAGNOSTIC=RECORD.parent/'face-noise-analysis/mask-penalty-diagnostic.json'


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def create():
    ex=load('baseline_mask',BASE/'experiment.py')
    evidence=ex.read(DIAGNOSTIC)
    assert evidence['source_model_unchanged'] and evidence['no_model_parameter_edits']
    assert all(c['gradients']['existing']['finite'] for c in evidence['checks'])
    assert not RUN.exists() and not RECORD.exists()
    RUN.mkdir()
    shutil.copytree(BASE/'source',RUN/'source',ignore=shutil.ignore_patterns('__pycache__'))
    unchanged={}
    for name in ['config.json','initial-points.npz','training-inputs.json','native-inputs.json',
                 'source-manifest.json','superresolution.json','frozen-baseline-diagnostic.json']:
        shutil.copy2(BASE/name,RUN/name);unchanged[name]=ex.sha(RUN/name)
        assert unchanged[name]==ex.sha(BASE/name)
    adapter=(BASE/'experiment.py').read_text(encoding='utf-8').replace(str(BASE),str(RUN)).replace(str(ex.RECORD),str(RECORD))
    ast.parse(adapter);(RUN/'experiment.py').write_text(adapter,encoding='utf-8')
    hook=(BASE/'transparent_hooks.py').read_text(encoding='utf-8')
    addition='''
BOUNDARIES = OrderedDict()


def boundary_background(camera, mask):
    """Same source mask; a band only selects where to emphasize negative BCE."""
    import torch
    from scipy.ndimage import maximum_filter
    key = id(camera)
    if key not in BOUNDARIES:
        while len(BOUNDARIES) >= 2:
            BOUNDARIES.popitem(last=False)
        if not hasattr(camera, '_exterior_band_numpy'):
            scale = camera.row.get('upsampling_scale', camera.row.get('input_linear_scale', 1))
            radius = int(round(8 * scale))
            camera._exterior_band_numpy = maximum_filter(camera.mask > 2, size=2*radius+1) & (camera.mask < 255)
        BOUNDARIES[key] = torch.as_tensor(camera._exterior_band_numpy, device='cuda')[None]
    BOUNDARIES.move_to_end(key)
    return (1 - mask) * BOUNDARIES[key]

'''
    hook=hook.replace('\ndef rgba_loss(',addition+'\ndef rgba_loss(',1)
    old='    loss = photo + .2 * silhouette'
    assert hook.count(old)==1
    hook=hook.replace(old,'''    # Keep every existing mask term. Add a local background term so that
    # the person boundary is not diluted by millions of empty pixels.
    boundary_weight = boundary_background(camera, mask)
    boundary_loss = -(boundary_weight * (1 - alpha).clamp_min(1e-6).log()).sum() / boundary_weight.sum().clamp_min(1)
    loss = photo + .2 * silhouette + .1 * boundary_loss''')
    old='foreground_ssim=ssim.detach(), alpha_loss=silhouette.detach())'
    assert hook.count(old)==1
    hook=hook.replace(old,'foreground_ssim=ssim.detach(), alpha_loss=silhouette.detach(), boundary_mask_loss=boundary_loss.detach())')
    ast.parse(hook);(RUN/'transparent_hooks.py').write_text(hook,encoding='utf-8')
    plan=ex.read(BASE/'plan.json')
    for key in ['diagnosis','failed_previous_gate','prior_trial','prior_trial_accepted','protocol_correction']:
        plan.pop(key,None)
    plan.update(created=ex.now(),method='SwinIR4x MSV2 with exterior-boundary mask penalty',
        human_direction='不要在模型上修，我觉得是mask没有起作用，应该有mask惩罚；训练模型不要改',
        single_main_variable='Only mask supervision: retain balanced alpha BCE and add 0.1 separately normalized exterior-boundary negative BCE',
        model_architecture_changed=False,gaussian_model_source_changed=False,msv2_schedule_changed=False,
        postprocessing_pruning=False,source_masks_changed=False,pretraining_evidence=str(DIAGNOSTIC),
        loss='0.8 foreground L1 + 0.2 foreground-window SSIM + 0.2 balanced alpha BCE + 0.1 exterior-boundary negative BCE',
        boundary_band='8 pixels at original resolution (32 at 4x); outside source mask only; mask labels are unchanged',
        hypothesis='The existing alpha mask penalty is active, but local exterior artifacts are diluted by its global-background normalization. A separately normalized boundary term should suppress face-adjacent opacity during training.',
        changes=['Add one separately normalized exterior-boundary mask penalty'],
        fixed=['MSV2 model and optimizer source','all59 SwinIR4x RGB and original soft masks','K and w2c',
            'untrained200000-point initialization','seed6060','foreground photometric loss','MSV2 schedule','18000 labels'],
        comparator=str(BASE),comparison_scope='Same full4x inputs and from-scratch training. No edits to or pruning of a trained model.',
        comparability='Same transparent4x baseline inputs, model, seed, initial points and MSV2 schedule; only the extra exterior-boundary mask penalty differs.',
        known_baseline_limitations='Held-object image/mask ambiguity and missing back observations retained. Masks, cameras, source images and existing loss coefficients are unchanged; one extra mask term is added.',
        user_selected_visible_improvement=False,user_selected_baseline_visible_improvement=True,
        learned_gaussians_reused=False,old_trained_gaussians_loaded=False,optimizer_reinitialized=True,
        final_gate='All59 same-camera face exterior alpha, person coverage, face/person RGB error and12 orbit views. Seek >=50% face-exterior-alpha reduction with <=0.01 inside-coverage drop and no visible loss of face, clothing, feet or held object. No mask-based render clipping.',
        selected_configuration=str(BASE),quality_accepted=False)
    ex.save(RUN/'plan.json',plan)
    ex.save(RUN/'configuration-parity.json',dict(passed=True,unchanged_files=unchanged,
        upstream_source_hashes_unchanged=True,model_source_unchanged=True,MSV2_schedule_unchanged=True,
        source_masks_unchanged=True,trained_state_copied=False,postprocess_pruning=False,
        only_training_change='transparent_hooks.py mask loss adds an exterior-boundary term'))
    ex.save(RUN/'adapter-manifest.json',{n:ex.sha(RUN/n) for n in ['experiment.py','transparent_hooks.py','source/msv2/train.py']})
    print('MASK_ONLY_RUN_CREATED',RUN,flush=True)


def probe():
    m=load('mask_probe',Path(__file__).with_name('transparent_msv2_demo.py'));m.RUN=RUN;m.probe()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe']);args=p.parse_args();globals()[args.action]()
