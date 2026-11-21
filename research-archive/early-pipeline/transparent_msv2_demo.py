"""Create and verify an isolated transparent-training comparison on SwinIR inputs."""
from pathlib import Path
import argparse, ast, importlib.util, json, shutil
import numpy as np

BASE = Path('/opt/gs-slam-lab/msv2-demo-swinir4x-20261104')
RUN = Path('/opt/gs-slam-lab/msv2-demo-transparent-20261104')
OUT = Path('/mnt/d/00/记录/2026-11-04_透明背景_MSV2/demo')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def create():
    ex = load('swinir_base', BASE / 'experiment.py')
    assert not RUN.exists()
    RUN.mkdir()
    shutil.copytree(BASE / 'source', RUN / 'source', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['config.json', 'initial-points.npz', 'training-inputs.json', 'native-inputs.json', 'superresolution.json']:
        shutil.copy2(BASE / name, RUN / name)
    shutil.copy2(Path(__file__).with_name('transparent_msv2_hooks.py'), RUN / 'transparent_hooks.py')
    text = (BASE / 'experiment.py').read_text(encoding='utf-8')
    text = text.replace(str(BASE), str(RUN)).replace(str(ex.RECORD), str(OUT))
    text = text.replace('existing 3x demo pixel/camera contract', 'SwinIR 4x foreground RGB plus alpha contract')
    text = text.replace('The upstream training loop, loss, CUDA rasterizers and densification are frozen.',
                        'The upstream schedule is retained; the RGB/alpha loss and depth sample support are explicit adaptations.')
    assert text.count('Camera.cache[key]=rgb*a') == 1
    text = text.replace('Camera.cache[key]=rgb*a', 'Camera.cache[key]=rgb')
    text = text.replace("    return result\n", "    mask=torch.as_tensor(camera.mask.copy(),device='cuda')>252\n    result['accum_alpha']=torch.where(mask,result['accum_alpha'].squeeze(),torch.ones_like(mask,dtype=torch.float32)).reshape_as(result['accum_alpha'])\n    return result\n")
    text = text.replace("    module.Scene=Scene;module.network_gui.try_connect=lambda:None", "    import transparent_hooks as rgba\n    module.rgba_loss=rgba.rgba_loss\n    module.Scene=Scene;module.network_gui.try_connect=lambda:None")
    text = text.replace("        if iteration in (1000,18000):", "        if iteration==1000:\n            import transparent_hooks as rgba\n            import sys\n            rgba.gate(sys.modules[__name__],self.gaussians)\n        if iteration in (1000,18000):")
    text = text.replace("            save(RUN/'progress.json',value);", "            value.update({k:float(v) for k,v in rgba.LAST.items()})\n            save(RUN/'progress.json',value);")
    text = text.replace("torch=runtime();verify_source();assert read(RUN/'preflight.json')['passed']", "torch=runtime();verify_source();assert read(RUN/'preflight.json')['passed'];assert read(RUN/'rgba-probe.json')['passed']")
    ast.parse(text); (RUN / 'experiment.py').write_text(text, encoding='utf-8')
    loop = RUN / 'source/msv2/train.py'; code = loop.read_text(encoding='utf-8')
    old = '''        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)'''
    assert code.count(old) == 1
    code = code.replace(old, '        loss, Ll1 = rgba_loss(viewpoint_cam, gaussians, pipe, render_pkg)')
    ast.parse(code); loop.write_text(code, encoding='utf-8')
    manifest = {p.relative_to(RUN / 'source').as_posix(): ex.sha(p) for p in (RUN / 'source').rglob('*') if p.is_file()}
    ex.save(RUN / 'source-manifest.json', manifest)
    plan = ex.read(BASE / 'plan.json')
    plan.update(created=ex.now(), method='SwinIR4x + MSV2 foreground RGB and explicit alpha',
                comparator=str(BASE), single_main_variable='Training input contract: foreground RGB + opacity instead of flattened RGB',
                inputs_reused=True, learned_gaussians_reused=False, optimizer_reinitialized=True,
                loss='0.8 foreground L1 + 0.2 foreground-window SSIM on straight RGB + 0.2 balanced silhouette L1',
                depth_sampling='Reliable foreground mask >252/255 only; raw uncomposited RGB for point colors',
                alpha_representation='RGB and corresponding soft mask kept separately, equivalent to straight RGBA; no matte RGB target',
                background_supervision=False, transparent_export=True,
                evaluation_backgrounds=['black', 'white', 'gray', 'checkerboard'],
                hypothesis='Flattened background RGB and background depth resampling create opaque dark/bright floaters; foreground color and explicit alpha should reduce leakage without erasing the person.',
                early_gate=dict(iteration=1000, outside_opacity_mean_ratio_max=.5, outside_opacity_absolute_floor=.01, inside_opacity_drop_max=.03, foreground_l1_ratio_max=1.2),
                final_gate='Review all 59 source views plus 12 orbit views on multiple backgrounds; preserve head, hands, device, clothing and feet. Format checks do not imply geometry accepted.',
                changes=['Separate RGB and alpha supervision', 'Exclude every SSIM window touching invalid mask pixels', 'Exclude background pixels from depth reinitialization'],
                possible_remaining_causes=['Mask boundary errors and SwinIR context bleeding', 'Camera residuals', 'Nonstatic subject and missing views'])
    ex.save(RUN / 'plan.json', plan)
    ex.save(RUN / 'adapter-manifest.json', {n: ex.sha(RUN / n) for n in ['experiment.py', 'transparent_hooks.py', 'source/msv2/train.py']})
    print('TRANSPARENT_EXPERIMENT_CREATED', RUN, flush=True)


def probe():
    import sys
    sys.path.insert(0, str(RUN))
    ex = load('transparent_experiment', RUN / 'experiment.py')
    torch = ex.runtime()
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(12 * 1024 ** 3 / total)
    import transparent_hooks as rgba
    from scene import GaussianModel
    from gaussian_renderer import render_imp
    torch.manual_seed(6060)
    args, data, opt, pipe = ex.options()
    model = GaussianModel(0); scene = ex.Scene(data, model); model.training_setup(opt); model.init_culling(59)
    cams = {c.uid:c for c in scene.cameras}; checks=[]
    for i in [0,29,58]:
        cam = cams[i]; pkg = render_imp(cam, model, pipe, torch.zeros(3,device='cuda'), culling=model._culling[:,i])
        loss, _ = rgba.rgba_loss(cam, model, pipe, pkg); loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(model._xyz.grad).all() and torch.isfinite(model._opacity.grad).all()
        checks.append(dict(frame=cam.image_name, loss=float(loss), finite_position_and_opacity_gradients=True))
        model.optimizer.zero_grad(set_to_none=True)
        del pkg, loss
    # Change arbitrary RGB under transparent pixels. Values and gradients must be unchanged.
    cam = cams[0]; mask, valid, support = rgba.targets(cam)
    source = cam.original_image.detach()
    prediction = (source * .91 + .04).clone().requires_grad_()
    results=[]; gradients=[]
    for matte in [0.,1.,.37]:
        target = torch.where(valid, source, torch.full_like(source,matte))
        loss,_,_ = rgba.foreground_photo(prediction,target,valid,support)
        results.append(float(loss)); gradients.append(torch.autograd.grad(loss,prediction)[0])
    maximum = max(float((g-gradients[0]).abs().max()) for g in gradients)
    outside = max(float(g[:,~valid[0]].abs().max()) for g in gradients)
    inside_gradient = float(gradients[0][:, valid[0]].abs().max())
    assert min(results)>0 and inside_gradient>0
    assert max(results)-min(results)<1e-7 and maximum<1e-7 and outside==0
    with torch.no_grad():
        depth=ex.fullk_depth(cam,model,pipe,torch.zeros(3,device='cuda'),culling=model._culling[:,0])
        probability=(1-depth['accum_alpha'].squeeze()).clamp_min(0)
        outside_mass=float(probability[mask[0] <= 252/255].sum()/probability.sum())
        assert outside_mass==0
    ex.save(RUN/'rgba-probe.json',dict(passed=True,created=ex.now(),optimizer_updates=0,checks=checks,
       transparent_target_matte_losses=results,matte_gradient_max_difference=maximum,
       outside_target_rgb_gradient_max=outside,inside_rgb_gradient_max=inside_gradient,depth_background_probability=outside_mass,
       peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,quality_accepted=False))
    print('RGBA_PROBE_PASSED',flush=True)


def diagnose():
    ex = load('frozen_swinir', BASE / 'experiment.py')
    torch = ex.runtime()
    from scene import GaussianModel
    from gaussian_renderer import render
    from types import SimpleNamespace
    from PIL import Image
    final = BASE / 'model/point_cloud/iteration_18000/point_cloud.ply'
    before = ex.sha(final)
    model = GaussianModel(3); model.load_ply(str(final)); model.active_sh_degree = 3
    pipe = SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    rows = ex.read(BASE / 'native-inputs.json')['records']; checks=[]
    with torch.no_grad():
        for i in [0,29,58]:
            c=ex.Camera(rows[i],i);mask=torch.as_tensor(c.mask.copy(),device='cuda',dtype=torch.float32)/255
            black=render(c,model,pipe,torch.zeros(3,device='cuda'))['render']
            alpha=render(c,model,pipe,torch.zeros(3,device='cuda'),override_color=torch.ones_like(model.get_xyz))['render'][0]
            depth=ex.fullk_depth(c,model,pipe,torch.zeros(3,device='cuda'))
            probability=(1-depth['accum_alpha'].squeeze()).clamp_min(0)
            checks.append(dict(frame=c.image_name,outside_alpha_mean=float(alpha[mask<.01].mean()),
                outside_alpha_gt_01_fraction=float((alpha[mask<.01]>.1).float().mean()),
                outside_rgb_mean=float(black[:,mask<.01].mean()),
                depth_resample_probability_outside_reliable_foreground=float(probability[mask<=252/255].sum()/probability.sum()),
                reliable_foreground_fraction=float((mask>252/255).float().mean())))
    assert ex.sha(final)==before
    ex.save(RUN/'frozen-baseline-diagnostic.json',dict(created=ex.now(),optimizer_updates=0,model_unchanged=True,
       source_model_sha256=before,checks=checks,
       conclusion='Dark low-RGB pixels can still carry opacity; the original depth-resampling distribution assigns mass to masked background. This is frozen-state evidence, not a complete causal history.'))
    print('FROZEN_BASELINE_DIAGNOSTIC',json.dumps(checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','probe','diagnose']);args=p.parse_args()
    globals()[args.action]()
