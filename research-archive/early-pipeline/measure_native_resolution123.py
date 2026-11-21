"""Bounded GPU cost check on an unchanged saved model; no optimizer updates."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    project = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
    sys.path.insert(0, str(project/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.training import camera, foreground_loss
    plan = rt.read(root/'reviews/native-resolution-cost-plan.json')
    assert plan['heights'] == [899, 1520] and plan['optimizer_updates'] == 0
    assert plan['frame_id'] == 'frame_000032' and plan['measured_repeats'] == 5
    for filename, digest in plan['sources'].items():
        assert rt.sha(filename) == digest, filename
    out = root/'reviews/native-resolution-cost'
    out.mkdir(exist_ok=False)
    rt.configure(Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets'), out)
    import torch
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    with open('/tmp/shuolee-native122-gpu.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = GaussianModel(3)
        model.load_ply(str(root/'gaussian-run/step-018000/world.ply'))
        fields = ['_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity']
        def digest():
            h = hashlib.sha256()
            for name in fields:
                h.update(getattr(model, name).detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        before = digest()
        row = next(v for v in rt.read(root/'data/training-inputs.json')['records'] if v['frame'] == plan['frame_id'])
        assert rt.sha(row['image']) == row['source_sha256']
        assert rt.sha(row['mask']) == row['mask_sha256']
        with Image.open(row['image']) as source: rgb = source.convert('RGB')
        with Image.open(row['mask']) as source: mask = source.convert('L')
        pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        bg = torch.zeros(3, device='cuda')
        results = []
        for height in plan['heights']:
            cam, record = camera(row, 31, height)
            gt = torch.from_numpy(np.array(rgb.resize(tuple(record['wh']), Image.Resampling.LANCZOS))).permute(2,0,1).cuda().float()/255
            alpha = torch.from_numpy(np.array(mask.resize(tuple(record['wh']), Image.Resampling.BILINEAR)))[None].cuda().float()/255
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            wall, cuda = [], []
            for repeat in range(1+plan['measured_repeats']):
                if time.monotonic()-started > plan['max_seconds']:
                    raise RuntimeError('Bounded resolution cost check exceeded its budget.')
                for name in fields: getattr(model, name).grad = None
                t0 = time.monotonic()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                image = render(cam, model, pipe, bg)['render']
                photo, _, _ = foreground_loss(image, gt, alpha)
                rendered_alpha = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
                loss = photo + .1*(rendered_alpha-alpha).abs().mean()
                loss.backward()
                end.record(); torch.cuda.synchronize()
                if repeat:
                    wall.append(time.monotonic()-t0)
                    cuda.append(begin.elapsed_time(end)/1000)
                assert all(getattr(model,name).grad is not None and torch.isfinite(getattr(model,name).grad).all() for name in fields)
                del image, photo, rendered_alpha, loss
            results.append(dict(height=height, wh=record['wh'], projection_max_error_px=record['projection_error_px'],
                measured_repeats=len(wall), wall_seconds=wall, cuda_seconds=cuda,
                median_wall_seconds=float(np.median(wall)), median_cuda_seconds=float(np.median(cuda)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                image_loading_in_timed_region=False, optimizer_updates=0))
            del gt, alpha
            for name in fields: getattr(model,name).grad = None
            torch.cuda.empty_cache()
        after = digest()
        assert before == after
        report = dict(state='completed', updated=rt.now(), plan_sha256=rt.sha(root/'reviews/native-resolution-cost-plan.json'),
            model_points=len(model.get_xyz), frame_id=row['frame'], records=results,
            model_parameter_hash_before=before, model_parameter_hash_after=after, optimizer_updates=0,
            elapsed_seconds=time.monotonic()-started, complete_training_benchmark=False,
            scope='Saved 18k static reference model. Same two raster calls and image losses/backward as training; no mesh refresh, optimizer step, densification, I/O or snapshots in timed region.',
            native_cache_bytes=2704*1520*4*94, model_quality_changed=False)
        rt.save(out/'result.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
