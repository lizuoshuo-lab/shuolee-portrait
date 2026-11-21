"""Check the actual optional adapter on every original native camera, no updates."""
import difflib
import gc
import inspect
import os
from pathlib import Path
import sys
import time
import traceback
import fcntl
import torch
import cross_subject_sh0124 as gs
import probe_ssim_cache124 as control
from cache_training_ssim124 import adapt

OUT = gs.ROOT/'fixed-target-ssim-interface124'


def setup_function(cached):
    source = inspect.getsource(control.setup)
    if cached:
        changes = {
            "source = (root/'adapter.py').read_text()": "source = adapt_ssim((root/'adapter.py').read_text())",
            "target_cache = NativeTargetCache(context['pixels'],context['masks'],[r['frame'] for r in context['rows']])": "target_cache = context['target_cache']",
            "patched = ast.parse(adapt(source)).body[0]": "patched = ast.parse(source).body[0]"}
        for old,new in changes.items():
            assert source.count(old)==1
            source=source.replace(old,new)
    namespace=dict(control.__dict__,OUT=OUT/('cached' if cached else 'original'),adapt_ssim=adapt)
    exec(compile(source,str(OUT/'actual-setup.py'),'exec'),namespace)
    return namespace['setup']


def prepare():
    assert not OUT.exists()
    result=gs.read(control.OUT/'result.json')
    assert result['all28_native_frames_checked'] and result['all_source_hashes_unchanged']
    hashes=dict(gs.read(control.OUT/'plan.json')['source_hashes'])
    for path in [Path(__file__),gs.HERE/'cache_training_ssim124.py',control.OUT/'result.json']:
        hashes[str(path)]=gs.sha(path)
    for path,digest in hashes.items():assert gs.sha(path)==digest,path
    OUT.mkdir();(OUT/'original').mkdir();(OUT/'cached').mkdir()
    for subject in control.SUBJECTS:
        old=(gs.ROOT/subject/'adapter.py').read_text();new=adapt(old)
        (OUT/(subject+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original-frozen',tofile='optional-target-ssim-cache')))
    gs.save(OUT/'plan.json',dict(updated=gs.now(),source_hashes=hashes,subjects=control.SUBJECTS,
        hypothesis='Real adapter binds fixed SSIM statistics to the camera actually rendered; identical to original full native loss/backward on all28 frames.',
        parity=gs.read(control.OUT/'plan.json')['parity'],budget_seconds=180,peak_allocated_limit_bytes=12*2**30,
        optimizer_updates=0,baseline='Verified target cache plus original SSIM',candidate='Target cache plus camera-bound target-only SSIM cache',
        production_files_modified=False,quality_accepted=False,end_to_end_15min_accepted=False))
    print('SSIM_INTERFACE_REGISTERED',flush=True)


def subject_check(subject,plan,budget):
    left=setup_function(False)(subject);right=setup_function(True)(subject)
    ctx1,targets1,dataset1,ns1=left;ctx2,targets2,dataset2,ns2=right
    model1=ctx1['model'];model2=ctx2['model']
    params1=[(g['name'],p) for g in model1.optimizer.param_groups for p in g['params']]
    params2=[(g['name'],p) for g in model2.optimizer.param_groups for p in g['params']]
    assert [n for n,p in params1]==[n for n,p in params2]
    initial=[p.detach().clone() for n,p in params1]
    assert all(torch.equal(p1.detach(),p2.detach()) for (_,p1),(_,p2) in zip(params1,params2))
    controls=[]
    for i,row in enumerate(ctx1['rows']):
        image1,loss1=ns1['native_step'](dataset1[i])
        image2,loss2=ns2['native_step'](dataset2[i])
        render_error=float((image1-image2).abs().max())
        loss_errors={k:abs(float(loss1[k])-float(v)) for k,v in loss2.items()}
        assert render_error==0 and max(loss_errors.values())<=plan['parity']['native_loss_abs_max']
        errors={}
        for (name,p1),(_,p2) in zip(params1,params2):
            error=float((p1.grad-p2.grad).abs().max()) if p1.numel() else 0.
            assert torch.allclose(p1.grad,p2.grad,rtol=plan['parity']['parameter_gradient_rtol'],atol=plan['parity']['parameter_gradient_atol'])
            errors[name]=error
        assert ctx2['ssim_camera_cache'].current_frame==row['frame']
        controls.append(dict(frame=row['frame'],render_max_abs=render_error,loss_errors=loss_errors,
                             parameter_gradient_max_abs=errors,actual_camera_binding=True))
        del image1,image2,loss1,loss2
        budget()
    adapter=ctx2['ssim_camera_cache'];wrong_camera_rejected=False;stale_frame_rejected=False
    try:adapter.use_camera(object())
    except ValueError:wrong_camera_rejected=True
    try:adapter.ssim(targets2.targets[0],targets2.targets[0])
    except RuntimeError:stale_frame_rejected=True
    assert wrong_camera_rejected and stale_frame_rejected
    assert not model1.optimizer.state and not model2.optimizer.state
    assert all(torch.equal(p.detach(),v) for params in (params1,params2) for (_,p),v in zip(params,initial))
    return dict(subject=subject,controls=controls,unregistered_camera_rejected=wrong_camera_rejected,
        stale_frame_after_camera_error_rejected=stale_frame_rejected,parameter_values_bitwise_unchanged=True,
        optimizer_updates=0,ssim_additional_bytes=adapter.cache.additional_bytes)


def run():
    assert not (OUT/'launch.json').exists()
    plan=gs.read(OUT/'plan.json')
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gs.save(OUT/'launch.json',dict(updated=gs.now(),pid=os.getpid(),command=sys.argv))
    started=time.monotonic();results=[]
    def budget():
        assert time.monotonic()-started<plan['budget_seconds']
        assert torch.cuda.max_memory_allocated()<plan['peak_allocated_limit_bytes']
    for subject in control.SUBJECTS:
        gs.save(OUT/'progress.json',dict(updated=gs.now(),state='checking_actual_interface',pid=os.getpid(),subject=subject))
        result=subject_check(subject,plan,budget);results.append(result)
        gs.save(OUT/(subject+'-result.json'),result)
        print('SSIM_INTERFACE_SUBJECT_PASS',subject,flush=True)
        gc.collect();torch.cuda.empty_cache()
    for path,digest in plan['source_hashes'].items():assert gs.sha(path)==digest,path
    gs.save(OUT/'result.json',dict(updated=gs.now(),state='actual_camera_interface_verified',results=results,
        seconds=time.monotonic()-started,all28_actual_adapter_frames_checked=True,all_source_hashes_unchanged=True,
        optimizer_updates=0,production_files_modified=False,optional_adapter_available=True,
        quality_accepted=False,end_to_end_15min_accepted=False))
    gs.save(OUT/'progress.json',dict(updated=gs.now(),state='completed',pid=None))
    print('SSIM_INTERFACE_COMPLETE',flush=True)


if __name__=='__main__':
    try:globals()[sys.argv[1]]()
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/'failure.json',dict(updated=gs.now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
            gs.save(OUT/'progress.json',dict(updated=gs.now(),state='failed',pid=None))
        raise
