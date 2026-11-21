"""One bounded SSIM-only cache parity/performance control; no model updates."""
import ast
import gc
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import fcntl
import numpy as np
from PIL import Image
import torch
import cross_subject_sh0124 as gs
from cache_training_targets124 import adapt
from native_target_cache124 import NativeTargetCache
from fixed_target_ssim124 import FixedTargetSSIM

OUT = gs.ROOT / 'fixed-target-ssim-control124'
SUBJECTS = ['grass-selected-sh0-pipeline124', 'winterhat-selected-sh0-pipeline124',
            'sofa-selected-sh0-pipeline124', 'indoor-selected-sh0-pipeline124']
STANDARD = gs.PROJECT / 'src/avatar_core/utils/loss_utils.py'


def prepare():
    assert not OUT.exists(), 'Do not overwrite a registered candidate'
    previous = gs.read(gs.ROOT / 'cached-native-step-control124/result.json')
    assert previous['all28_original_raster_backward_checked']
    hashes = dict(gs.read(gs.ROOT / 'target-cache-control124/plan.json')['source_hashes'])
    hashes.update(gs.read(gs.ROOT / 'cached-native-step-control124/plan.json')['source_hashes'])
    for path in [Path(__file__), gs.HERE/'fixed_target_ssim124.py', STANDARD]:
        hashes[str(path)] = gs.sha(path)
    for path, digest in hashes.items():
        assert gs.sha(path) == digest, path
    OUT.mkdir()
    gs.save(OUT/'plan.json', dict(updated=gs.now(), subjects=SUBJECTS, source_hashes=hashes,
        hypothesis='Cache the original fixed target SSIM means, squared means, variance, window and indoor known-neighborhood support; retain the original expression and prediction operations.',
        changed_variable='Only lifetime of target-only SSIM terms. Both modes already use the verified identical GPU RGB target cache.',
        frozen='All28 native frames, source masks/validity, conditional K, fitted inputs, original loss formula/weights, fresh20908 Gaussians and zero shared offset. No fitting or optimizer updates.',
        parity=dict(pixel_loss_abs_max=1e-7, pixel_gradient_abs_max=1e-10,
                    native_loss_abs_max=1e-7, render_abs_max=0,
                    parameter_gradient_atol=1e-7, parameter_gradient_rtol=2e-4),
        tests=['CPU original scalar and gradient parity including unknown NaN/Inf and image padding',
               'Wrong frame, copied target, mutated target/validity, differentiable target and empty support rejection',
               'All28 native final-render pixel loss/gradient parity',
               'All28 original native raster and full enabled loss backward at fresh initialization'],
        benchmark=dict(warmup_cycles=1, paired_blocks=3, cycles_per_block=2,
                       scope='Original mesh/raster/all losses/backward with synchronized timings; target-cache baseline versus target+SSIM-cache.',
                       excludes='Optimizer update, densification, video decoding, detectors, fitting, export, real quality or cold15min.'),
        budget_seconds=300, peak_allocated_limit_bytes=12*2**30,
        optimizer_updates=0, production_adopted=False, quality_accepted=False,
        end_to_end_15min_accepted=False))
    gs.save(OUT/'progress.json', dict(updated=gs.now(),state='registered',pid=None))
    print('SSIM_CACHE_REGISTERED', flush=True)


def cpu_controls():
    standard = gs.module(STANDARD, 'ssim_cache_standard_control')
    masked = gs.module(gs.ROOT/SUBJECTS[-1]/'loss_utils_validity.py', 'ssim_cache_masked_control')
    torch.manual_seed(17)
    checks = []
    for label, module, partial, nonfinite in [
        ('unmasked', standard, False, False), ('all_known', masked, False, False),
        ('partial', masked, True, False), ('partial_nonfinite', masked, True, True)]:
        target = torch.rand(3, 33, 40)
        prediction = torch.rand(3, 33, 40)
        valid = torch.ones(1, 33, 40, dtype=torch.bool) if module is masked else None
        if partial:
            valid[:, 22:, 25:] = False
        if nonfinite:
            target[:, 23:, 26:] = float('nan')
            prediction[:, 23:, 26:] = float('inf')
        prediction.requires_grad_(True)
        cache = FixedTargetSSIM(module, [target], ['a'], [valid])
        kw = {'valid_mask': valid} if valid is not None else {}
        original = module.ssim(prediction, target, **kw)
        cached = cache.ssim(prediction, target, frame_id='a')
        grad1 = torch.autograd.grad(original, prediction)[0]
        grad2 = torch.autograd.grad(cached, prediction)[0]
        assert torch.isfinite(cached) and torch.equal(original, cached)
        assert torch.equal(grad1, grad2)
        if partial:
            assert torch.count_nonzero(grad2[:, ~valid[0]]) == 0
        checks.append(dict(name=label, scalar_bitwise=True, gradient_bitwise=True))
    rejected = []
    target = torch.rand(3, 33, 40)
    valid = torch.ones(1, 33, 40, dtype=torch.bool)
    cache = FixedTargetSSIM(masked, [target], ['a'], [valid])
    def expect(label, call):
        try:
            call()
        except (ValueError, RuntimeError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted invalid cache: ' + label)
    expect('wrong_frame', lambda: cache.ssim(target, target, frame_id='b'))
    expect('target_copy', lambda: cache.ssim(target, target.clone(), frame_id='a'))
    target.add_(0.01)
    expect('mutated_target', lambda: cache.ssim(target, target, frame_id='a'))
    cache = FixedTargetSSIM(masked, [target], ['a'], [valid])
    valid[0, 0, 0] = False
    expect('mutated_validity', lambda: cache.ssim(target, target, frame_id='a'))
    expect('differentiable_target', lambda: FixedTargetSSIM(standard, [target.clone().requires_grad_()], ['a']))
    expect('duplicate_frames', lambda: FixedTargetSSIM(standard, [target, target], ['a','a']))
    tiny_support = torch.zeros_like(valid); tiny_support[0,16,20] = True
    expect('empty_neighborhood_support', lambda: FixedTargetSSIM(masked, [target], ['a'], [tiny_support]))
    return dict(checks=checks, rejected=rejected, passed=True)


def setup(subject):
    root = gs.ROOT/subject
    source = (root/'adapter.py').read_text()
    function = ast.parse(source).body[0]
    cut = next(i for i,n in enumerate(function.body) if isinstance(n,ast.FunctionDef) and n.name=='status')
    body = function.body[:cut]
    assignments = [n for n in body if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='run' for t in n.targets)]
    assert len(assignments)==1
    assignments[0].value = ast.parse('BENCH_OUT/SUBJECT',mode='eval').body
    function.body = [*body, ast.parse('return locals()').body[0]]
    namespace = dict(gs.__dict__, OUT=root, BENCH_OUT=OUT, SUBJECT=subject, suspended_baseline=lambda:None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])), str(root/'adapter.py')+':setup-only','exec'), namespace)
    context = namespace['train']()
    model = context['model']; model.create_from_pcd(None,1.); model.training_setup(context['opt'])
    assert len(model._xyz)==20908 and not model.optimizer.state
    target_cache = NativeTargetCache(context['pixels'],context['masks'],[r['frame'] for r in context['rows']])
    patched = ast.parse(adapt(source)).body[0]
    images = next(n for n in patched.body if isinstance(n,ast.ClassDef) and n.name=='Images')
    images_namespace = dict(context, target_cache=target_cache)
    exec(compile(ast.Module(body=[images],type_ignores=[]),str(root/'adapter.py')+':target-cache','exec'),images_namespace)
    original = ast.parse((gs.UP/'train.py').read_text())
    train = next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='training')
    loop = next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
    begin = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and any(
        isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='select_mesh_by_timestep' for c in ast.walk(n)))
    end = next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
               and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward')
    body = [*ast.parse('gaussians.optimizer.zero_grad(set_to_none=True)').body,
            *loop.body[begin:end+1], *ast.parse('return image, losses').body]
    step = ast.FunctionDef(name='native_step',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='viewpoint_cam')],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    step_namespace = dict(context, gaussians=model, iteration=1, debug_from=-1)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[step],type_ignores=[])),str(gs.UP/'train.py')+':native-ssim-test','exec'),step_namespace)
    return context, target_cache, images_namespace['Images'](), step_namespace


def subject_control(subject, plan, budget):
    ctx, targets, dataset, namespace = setup(subject)
    model = ctx['model']; rows = ctx['rows']
    module = ctx.get('validity_loss') or sys.modules[ctx['ssim'].__module__]
    validities = [getattr(cam,'supervision_validity',None) for cam in ctx['cams']]
    torch.cuda.synchronize(); start=time.monotonic()
    cache = FixedTargetSSIM(module, targets.targets, [r['frame'] for r in rows], validities)
    torch.cuda.synchronize(); prepare_seconds=time.monotonic()-start
    (OUT/(subject+'-derived-ssim.py')).write_text(cache.derived_source+'\n')
    params = [(g['name'],p) for g in model.optimizer.param_groups for p in g['params']]
    initial = [p.detach().clone() for _,p in params]
    weight = ctx['opt'].lambda_dssim
    controls = []
    torch.cuda.reset_peak_memory_stats()
    def native(mode,index):
        frame = rows[index]['frame']
        namespace['ssim'] = ctx['ssim'] if mode=='original' else lambda prediction,target: cache.ssim(prediction,target,frame_id=frame)
        return namespace['native_step'](dataset[index])
    for index,row in enumerate(rows):
        frame = row['frame']; target=targets.get(index,frame); valid=validities[index]
        kw = {'valid_mask':valid} if valid is not None else {}
        prediction = torch.from_numpy(np.array(Image.open(gs.ROOT/subject/'gaussian-run/step-018000'/f'{frame}_gaussian.png').convert('RGB'))).permute(2,0,1).float().div(255).cuda().requires_grad_(True)
        l1 = module.l1_loss(prediction,target,**kw)
        original = (1-weight)*l1+weight*(1-module.ssim(prediction,target,**kw))
        cached = (1-weight)*module.l1_loss(prediction,target,**kw)+weight*(1-cache.ssim(prediction,target,frame_id=frame))
        pg1 = torch.autograd.grad(original,prediction)[0]
        pg2 = torch.autograd.grad(cached,prediction)[0]
        pixel_loss_error=abs(float(original)-float(cached));pixel_gradient_error=float((pg1-pg2).abs().max())
        assert pixel_loss_error<=plan['parity']['pixel_loss_abs_max']
        assert pixel_gradient_error<=plan['parity']['pixel_gradient_abs_max']
        rendered,losses=native('original',index)
        render1=rendered.detach().clone();values1={k:float(v) for k,v in losses.items()}
        gradients1=[p.grad.detach().clone() for _,p in params]
        rendered,losses=native('cached',index)
        render_error=float((render1-rendered.detach()).abs().max())
        loss_errors={k:abs(values1[k]-float(v)) for k,v in losses.items()};gradient_errors={}
        assert render_error==0 and max(loss_errors.values())<=plan['parity']['native_loss_abs_max']
        for (name,param),gradient in zip(params,gradients1):
            error=float((gradient-param.grad).abs().max()) if gradient.numel() else 0.
            assert torch.allclose(gradient,param.grad,rtol=plan['parity']['parameter_gradient_rtol'],atol=plan['parity']['parameter_gradient_atol']),(subject,frame,name,error)
            gradient_errors[name]=error
        controls.append(dict(frame=frame,pixel_loss_error=pixel_loss_error,pixel_gradient_max_abs=pixel_gradient_error,
            render_max_abs=render_error,loss_errors=loss_errors,parameter_gradient_max_abs=gradient_errors))
        del prediction,original,cached,l1,pg1,pg2,rendered,losses,render1,gradients1
        budget()
    def block(mode,cycles):
        torch.cuda.synchronize();start=time.monotonic()
        for _ in range(cycles):
            for index in range(len(dataset)):
                rendered,losses=native(mode,index);float(losses['total'].detach())
        torch.cuda.synchronize()
        return 1000*(time.monotonic()-start)/(cycles*len(dataset))
    for mode in ('original','cached'):block(mode,1)
    pairs=[]
    for order in [['original','cached'],['cached','original'],['original','cached']]:
        pair=dict(order=order)
        for mode in order:pair[mode+'_ms']=block(mode,2);budget()
        pairs.append(pair)
    assert not model.optimizer.state
    assert all(torch.equal(p.detach(),v) for (_,p),v in zip(params,initial))
    old=statistics.median(p['original_ms'] for p in pairs);new=statistics.median(p['cached_ms'] for p in pairs)
    return dict(subject=subject,controls=controls,paired_blocks=pairs,median_original_ms=old,median_cached_ms=new,
        speedup=old/new,ssim_additional_bytes=cache.additional_bytes,target_cache_bytes=targets.bytes,
        cache_prepare_seconds=prepare_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        points=len(model._xyz),parameter_values_bitwise_unchanged=True,optimizer_updates=0)


def run():
    assert not (OUT/'launch.json').exists(),'Inspect existing candidate; do not rerun'
    plan=gs.read(OUT/'plan.json')
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gs.save(OUT/'launch.json',dict(updated=gs.now(),pid=os.getpid(),command=sys.argv,
        proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    started=time.monotonic();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cpu=cpu_controls();gs.save(OUT/'cpu-controls.json',cpu)
    def budget():
        assert time.monotonic()-started<plan['budget_seconds']
        assert torch.cuda.max_memory_allocated()<plan['peak_allocated_limit_bytes']
    results=[]
    for subject in SUBJECTS:
        gs.save(OUT/'progress.json',dict(updated=gs.now(),state='native_ssim_cache_control',pid=os.getpid(),subject=subject,completed=len(results)))
        result=subject_control(subject,plan,budget);results.append(result)
        gs.save(OUT/(subject+'-result.json'),result)
        print('SSIM_CACHE_SUBJECT',subject,result['median_original_ms'],result['median_cached_ms'],flush=True)
        gc.collect();torch.cuda.empty_cache()
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    gs.save(OUT/'result.json',dict(updated=gs.now(),state='bounded_ssim_cache_control_complete',results=results,
        seconds=time.monotonic()-started,all28_native_frames_checked=True,cpu_controls_passed=cpu['passed'],
        all_source_hashes_unchanged=True,optimizer_updates=0,production_adopted=False,
        quality_accepted=False,end_to_end_15min_accepted=False,
        scope=plan['benchmark']['scope'],excludes=plan['benchmark']['excludes']))
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state='completed',pid=None))
    print('SSIM_CACHE_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/'failure.json',dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
            gs.save(OUT/'progress.json',dict(updated=gs.now(),state='failed',pid=None))
        raise
