"""One registered, zero-update native target-cache CUDA experiment.

Reads all four frozen rosters. The timed region is target preparation plus
original colour-loss forward/backward, NOT the full GS pipeline.
"""
import argparse
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from native_target_cache124 import NativeTargetCache
from run_official_gaussians123 import read, save, sha, now

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
OUT = ROOT/'target-cache-control124'
SUBJECTS = ['grass-selected-sh0-pipeline124','winterhat-selected-sh0-pipeline124',
            'sofa-selected-sh0-pipeline124','indoor-selected-sh0-pipeline124']
STANDARD = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core/utils/loss_utils.py')


def load_module(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    assert not OUT.exists(), 'Do not overwrite or repeat a registered control'
    hashes={str(p):sha(p) for p in [Path(__file__),HERE/'native_target_cache124.py',STANDARD]}
    subjects=[]
    for subject in SUBJECTS:
        root=ROOT/subject
        run=read(root/'gaussian-run/result.json')
        assert run['state']=='bounded_control_completed'
        rows=read(root/'data/training-inputs.json')['records']
        assert [r['frame'] for r in rows]==read(root/'plan.json')['selected_frames']
        paths=[root/'data/training-inputs.json',root/'plan.json',root/'adapter.py',root/'gaussian-run/result.json',root/'gaussian-run/loss.jsonl']
        for row in rows:
            paths.extend([Path(row['image']),Path(row['mask']),root/'gaussian-run/step-018000'/f'{row["frame"]}_gaussian.png'])
            if 'validity' in row: paths.append(Path(row['validity']))
        if subject.startswith('indoor'): paths.append(root/'loss_utils_validity.py')
        hashes.update({str(p):sha(p) for p in paths})
        subjects.append(dict(subject=subject,frames=[r['frame'] for r in rows],dimensions=[r['wh'] for r in rows]))
    OUT.mkdir()
    save(OUT/'plan.json',dict(updated=now(),hypothesis='Repeated CPU float/mask composition and H2D transfer can be eliminated by one exact target cache; unchanged loss and pixel gradients.',
        subjects=subjects,source_hashes=hashes,changed_variable='Target storage lifetime and device only; CPU composition preserved bitwise',
        frozen='Native pixels/masks, all roster frames, conditional K, all loss code and weights, no optimizer or renderer modifications',
        parity=dict(target_bitwise=True,loss_abs_max=1e-7,pixel_gradient_abs_max=1e-10),
        benchmark=dict(warmup_cycles_per_mode=2,paired_blocks=3,cycles_per_block=2,mode_order='legacy/cache then cache/legacy then legacy/cache',include='target fetch and unchanged L1/SSIM forward/backward plus synchronization',exclude='raster, mesh, optimizer, densification, decoding, fit, export, physical quality'),
        budget_seconds=300,peak_allocated_limit_bytes=12*2**30,one_run_per_subject=True,training_started=False,
        no_model_updates=True,production_adapter_unchanged=True,quality_accepted=False,end_to_end_15min_accepted=False))
    save(OUT/'progress.json',dict(updated=now(),state='registered',pid=None))
    print('TARGET_CACHE_REGISTERED',len(subjects),sum(len(s['frames']) for s in subjects),flush=True)


def run():
    assert not (OUT/'launch.json').exists(), 'Existing run must be inspected, not repeated'
    plan=read(OUT/'plan.json')
    for path,digest in plan['source_hashes'].items(): assert sha(path)==digest,path
    lock=open('/tmp/shuolee-native122-gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(OUT/'launch.json',dict(updated=now(),pid=os.getpid(),command=sys.argv,proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]))
    started=time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    std=load_module(STANDARD,'original_colour_loss124')
    results=[]
    def budget():
        assert time.monotonic()-started < plan['budget_seconds'],'Bounded benchmark time exceeded'
        assert torch.cuda.max_memory_allocated() < plan['peak_allocated_limit_bytes'],'Memory budget exceeded'
    for subject in SUBJECTS:
        budget()
        root=ROOT/subject; rows=read(root/'data/training-inputs.json')['records']
        save(OUT/'progress.json',dict(updated=now(),state='benchmarking',pid=os.getpid(),subject=subject,completed=len(results),training_started=False))
        pixels=[torch.from_numpy(np.array(Image.open(r['image']).convert('RGB'))).permute(2,0,1) for r in rows]
        masks=[torch.from_numpy(np.array(Image.open(r['mask']).convert('L')))[None] for r in rows]
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        t=time.monotonic()
        cache=NativeTargetCache(pixels,masks,[r['frame'] for r in rows]);torch.cuda.synchronize()
        cache_seconds=time.monotonic()-t
        predictions=[torch.from_numpy(np.array(Image.open(root/'gaussian-run/step-018000'/f'{r["frame"]}_gaussian.png').convert('RGB'))).permute(2,0,1).float().div(255).cuda().requires_grad_(True) for r in rows]
        validity=[torch.from_numpy(np.array(Image.open(r['validity']))==255)[None].cuda() if 'validity' in r else None for r in rows]
        lm=load_module(root/'loss_utils_validity.py','original_indoor_loss124') if subject.startswith('indoor') else std
        options=read(root/'gaussian-run/effective-optimization.json');weight=options['lambda_dssim']
        def fetch(mode,i):
            if mode=='cache': return cache.get(i,rows[i]['frame'])
            return (pixels[i].float().div(255)*masks[i].float().div(255)).cuda()
        def objective(pred,gt,i):
            kw={'valid_mask':validity[i]} if validity[i] is not None else {}
            return (1-weight)*lm.l1_loss(pred,gt,**kw)+weight*(1-lm.ssim(pred,gt,**kw))
        controls=[]
        for i,row in enumerate(rows):
            gt=fetch('legacy',i);cached=fetch('cache',i)
            assert torch.equal(gt,cached)
            values=[]; gradients=[]
            for target in [gt,cached]:
                loss=objective(predictions[i],target,i)
                gradients.append(torch.autograd.grad(loss,predictions[i])[0])
                values.append(float(loss))
            ve=abs(values[0]-values[1]); ge=float((gradients[0]-gradients[1]).abs().max())
            assert ve<=plan['parity']['loss_abs_max'] and ge<=plan['parity']['pixel_gradient_abs_max'],(row['frame'],ve,ge)
            controls.append(dict(frame=row['frame'],target_bitwise=True,loss_abs_error=ve,pixel_gradient_max_abs=ge,width=row['wh'][0],height=row['wh'][1]))
            del gt,cached,loss,gradients
            budget()
        try: cache.get(0,'wrong_frame_identity')
        except ValueError: mismatch_rejected=True
        else: raise AssertionError('Identity mismatch silently accepted')
        def block(mode,cycles):
            torch.cuda.synchronize(); start=time.monotonic()
            for _ in range(cycles):
                for i,pred in enumerate(predictions):
                    pred.grad=None
                    gt=fetch(mode,i); loss=objective(pred,gt,i);loss.backward()
                    float(loss.detach())  # original training also synchronizes .item each step
                    pred.grad=None
            torch.cuda.synchronize()
            return 1000*(time.monotonic()-start)/(cycles*len(rows))
        for mode in ['legacy','cache']: block(mode,2)
        blocks=[]
        for index,order in enumerate([['legacy','cache'],['cache','legacy'],['legacy','cache']]):
            item=dict(index=index,order=order)
            for mode in order:item[mode+'_ms']=block(mode,2);budget()
            blocks.append(item)
        legacy=statistics.median(b['legacy_ms'] for b in blocks);cached=statistics.median(b['cache_ms'] for b in blocks)
        result=dict(subject=subject,controls=controls,frame_mismatch_rejected=mismatch_rejected,cache_bytes=cache.bytes,cache_prepare_seconds=cache_seconds,paired_blocks=blocks,median_legacy_ms=legacy,median_cached_ms=cached,speedup=legacy/cached,peak_allocated_bytes=torch.cuda.max_memory_allocated(),parity_passed=True,optimizer_updates=0)
        results.append(result);save(OUT/(subject+'-result.json'),result)
        print('TARGET_CACHE_SUBJECT_DONE',subject,json.dumps({k:result[k] for k in ['median_legacy_ms','median_cached_ms','speedup','cache_bytes']}),flush=True)
        del cache,predictions,validity,pixels,masks
        gc.collect();torch.cuda.empty_cache()
    for path,digest in plan['source_hashes'].items(): assert sha(path)==digest,path
    save(OUT/'result.json',dict(updated=now(),state='bounded_zero_update_microbenchmark_complete',results=results,seconds=time.monotonic()-started,all28_native_targets_and_gradients_verified=True,all_source_hashes_unchanged=True,training_started=False,production_adopted=False,quality_accepted=False,end_to_end_15min_accepted=False,limitation='Target fetch plus colour-loss microbenchmark on frozen rendered PNG pixel values. No raster/mesh/optimizer or cold video-to-model timing is measured.'))
    save(OUT/'progress.json',dict(updated=now(),state='completed',pid=None,training_started=False))
    print('TARGET_CACHE_CONTROL_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);args=p.parse_args()
    try: globals()[args.mode]()
    except BaseException as error:
        if OUT.exists():save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc(),auto_retry=False))
        raise
