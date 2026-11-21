"""Actual original native raster/loss backward, zero optimizer updates.

Reuse the frozen model setup and original loop fragment through backward.
Only the target fetch path alternates; historical experiment files stay frozen.
"""
import ast
import difflib
import gc
import inspect
import os
from pathlib import Path
import statistics
import textwrap
import time
import traceback
import fcntl
import sys
import torch
import cross_subject_sh0124 as gs
from cache_training_targets124 import adapt
from native_target_cache124 import NativeTargetCache

OUT=gs.ROOT/'cached-native-step-control124'
SUBJECTS=['grass-selected-sh0-pipeline124','winterhat-selected-sh0-pipeline124','sofa-selected-sh0-pipeline124','indoor-selected-sh0-pipeline124']


def prepare():
    assert not OUT.exists()
    prior=gs.read(gs.ROOT/'target-cache-control124/result.json')
    assert prior['all28_native_targets_and_gradients_verified']
    hashes={str(p):gs.sha(p) for p in [Path(__file__),gs.HERE/'cache_training_targets124.py',gs.HERE/'native_target_cache124.py',gs.UP/'train.py',gs.ROOT/'target-cache-control124/result.json']}
    diffs={}
    for subject in SUBJECTS:
        root=gs.ROOT/subject;p=gs.read(root/'plan.json')
        for path,digest in p['source_hashes'].items():assert gs.sha(path)==digest,path
        hashes.update(p['source_hashes'])
        source=(root/'adapter.py').read_text();new=adapt(source)
        diffs[subject]=''.join(difflib.unified_diff(source.splitlines(True),new.splitlines(True),fromfile='frozen-input',tofile='cache-only-input'))
        hashes[str(root/'adapter.py')]=gs.sha(root/'adapter.py')
    OUT.mkdir()
    for subject,diff in diffs.items():(OUT/(subject+'.diff')).write_text(diff)
    gs.save(OUT/'plan.json',dict(updated=gs.now(),source_hashes=hashes,subjects=SUBJECTS,
        hypothesis='The exact GPU target cache reduces full native render+regularizer+backward wall time without changing pixels, gradients or model parameters.',
        original_fragment='Pinned original train.py: select mesh, render, native colour and geometry losses, backward. No optimizer.step/densify.',
        setup='Frozen original subject adapter prefix, redirect artifact directory only. Fresh one-Gaussian-per-triangle and zero shared offset. No checkpoint load.',
        parity=dict(render_abs_max=0,loss_abs_max=1e-7,parameter_gradient_atol=1e-7,parameter_gradient_rtol=2e-4),
        gradient_tolerance_reason='Native raster CUDA backward uses atomic accumulation; target and pixel-gradient parity is separately bitwise verified.',
        warmup_cycles_per_mode=1,paired_blocks=3,cycles_per_block=2,budget_seconds=300,peak_allocated_limit_bytes=12*2**30,
        optimizer_updates=0,production_adopted=False,quality_accepted=False,end_to_end_15min_accepted=False))
    print('NATIVE_CACHE_STEP_REGISTERED',flush=True)


def run():
    assert not (OUT/'launch.json').exists(),'Never rerun a started candidate'
    plan=gs.read(OUT/'plan.json')
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gs.save(OUT/'launch.json',dict(updated=gs.now(),pid=os.getpid(),command=sys.argv))
    started=time.monotonic(); results=[]
    def budget():
        assert time.monotonic()-started<plan['budget_seconds']
        assert torch.cuda.max_memory_allocated()<plan['peak_allocated_limit_bytes']
    for subject in SUBJECTS:
        gs.save(OUT/'progress.json',dict(updated=gs.now(),state='native_benchmark',pid=os.getpid(),subject=subject,completed=len(results)))
        root=gs.ROOT/subject;source=(root/'adapter.py').read_text()
        fn=ast.parse(source).body[0]
        cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.FunctionDef) and n.name=='status')
        body=fn.body[:cut]
        assignments=[n for n in body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='run' for t in n.targets)]
        assert len(assignments)==1
        assignments[0].value=ast.parse('BENCH_OUT / SUBJECT',mode='eval').body
        body.append(ast.parse('return locals()').body[0]);fn.body=body
        ns=dict(gs.__dict__,OUT=root,BENCH_OUT=OUT,SUBJECT=subject,suspended_baseline=lambda:None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(root/'adapter.py')+':setup-only','exec'),ns)
        ctx=ns['train']()
        model=ctx['model'];model.create_from_pcd(None,1.);model.training_setup(ctx['opt'])
        assert len(model._xyz)==20908 and not model.optimizer.state
        params=[(group['name'],param) for group in model.optimizer.param_groups for param in group['params']]
        initial=[p.detach().clone() for _,p in params]
        legacy_dataset=ctx['Images']()
        # Execute the new input adapter's actual Images class, not a mock fetch.
        patched=ast.parse(adapt(source)).body[0]
        images_class=next(n for n in patched.body if isinstance(n,ast.ClassDef) and n.name=='Images')
        cache=NativeTargetCache(ctx['pixels'],ctx['masks'],[r['frame'] for r in ctx['rows']])
        cached_ns=dict(ctx,target_cache=cache)
        exec(compile(ast.Module(body=[images_class],type_ignores=[]),str(OUT/(subject+'.diff')),'exec'),cached_ns)
        cached_dataset=cached_ns['Images']()
        original=ast.parse((gs.UP/'train.py').read_text())
        train=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='training')
        loop=next(n for n in train.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration')
        begin=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.If) and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='select_mesh_by_timestep' for c in ast.walk(n)))
        end=next(i for i,n in enumerate(loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='backward')
        fragment=loop.body[begin:end+1]
        body=[*ast.parse('gaussians.optimizer.zero_grad(set_to_none=True)').body,*fragment,*ast.parse('return image, losses').body]
        step_fn=ast.FunctionDef(name='native_step',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='viewpoint_cam')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
        step_ns=dict(ctx,gaussians=model,iteration=1,debug_from=-1)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[step_fn],type_ignores=[])),str(gs.UP/'train.py')+':native-step-only','exec'),step_ns)
        native=step_ns['native_step'];controls=[]
        torch.cuda.reset_peak_memory_stats()
        for index,row in enumerate(ctx['rows']):
            legacy_cam=legacy_dataset[index];legacy_target=legacy_cam.original_image.cuda()
            rendered,losses=native(legacy_cam)
            image_a=rendered.detach().clone();value_a=float(losses['total']);ga=[p.grad.detach().clone() for _,p in params]
            cached_cam=cached_dataset[index];assert torch.equal(legacy_target,cached_cam.original_image)
            rendered,losses=native(cached_cam)
            ve=abs(value_a-float(losses['total'])); re=float((image_a-rendered.detach()).abs().max());grad_errors={}
            for (name,p),g in zip(params,ga):
                error=float((g-p.grad).abs().max()) if g.numel() else 0.
                assert torch.allclose(g,p.grad,rtol=plan['parity']['parameter_gradient_rtol'],atol=plan['parity']['parameter_gradient_atol']),(subject,row['frame'],name,error)
                grad_errors[name]=error
            assert re==0 and ve<=plan['parity']['loss_abs_max']
            controls.append(dict(frame=row['frame'],render_max_abs=re,loss_abs=ve,gradient_max_abs=grad_errors))
            del rendered,losses,image_a,ga,legacy_target
            budget()
        def block(dataset,cycles):
            torch.cuda.synchronize();t=time.monotonic()
            for _ in range(cycles):
                for index in range(len(dataset)):
                    rendered,losses=native(dataset[index]);float(losses['total'].detach())
            torch.cuda.synchronize()
            return 1000*(time.monotonic()-t)/(cycles*len(dataset))
        datasets={'legacy':legacy_dataset,'cache':cached_dataset}
        for dataset in datasets.values():block(dataset,1)
        pairs=[]
        for order in [['legacy','cache'],['cache','legacy'],['legacy','cache']]:
            row=dict(order=order)
            for mode in order:row[mode+'_ms']=block(datasets[mode],2);budget()
            pairs.append(row)
        assert not model.optimizer.state
        assert all(torch.equal(p.detach(),v) for (_,p),v in zip(params,initial))
        old=statistics.median(r['legacy_ms'] for r in pairs);new=statistics.median(r['cache_ms'] for r in pairs)
        result=dict(subject=subject,controls=controls,paired_blocks=pairs,median_legacy_ms=old,median_cached_ms=new,speedup=old/new,points=len(model._xyz),cache_bytes=cache.bytes,peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameter_values_bitwise_unchanged=True,optimizer_updates=0)
        results.append(result);gs.save(OUT/(subject+'-result.json'),result)
        print('NATIVE_CACHE_STEP_SUBJECT',subject,old,new,flush=True)
        del ctx,model,params,initial,legacy_dataset,cached_dataset,cached_ns,step_ns,native,ns,datasets,cache,legacy_cam,cached_cam,dataset,g,p
        gc.collect();torch.cuda.empty_cache()
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    gs.save(OUT/'result.json',dict(updated=gs.now(),state='native_zero_update_benchmark_complete',results=results,seconds=time.monotonic()-started,all_source_hashes_unchanged=True,all28_original_raster_backward_checked=True,optimizer_updates=0,production_adopted=False,quality_accepted=False,end_to_end_15min_accepted=False,scope='Original native render, all enabled losses and backward at fresh20908 initialization. No optimizer update, densification or video cold start timing.'))
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state='completed',pid=None));print('NATIVE_CACHE_STEP_COMPLETE',flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as error:
        if OUT.exists():gs.save(OUT/'failure.json',dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise
