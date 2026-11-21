"""Check and time cached SH0 target loss on the 24 exact delivered 006 rasters."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
from statistics import mean, median
import sys
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[1]
BASE_TRAINING = Path('/mnt/d/Projects/Shuolee-smpl-sixworker-opacity/src/shuolee_portrait/training.py')
BASE_SHA = 'fe73a6f19838ed55f723df5bffe0d480a513cf3ebacf3d91b3affc15e9fd665a'
RUN = Path('/opt/gs-slam-lab/shuolee-smpl-sixworker-opacity-006/run')
PINS = {
    'result.json': '4b228f050170a172997a12fa7e4c2d06361877098c03232fa8011202e0f25683',
    'fit/official-fit-result.json': '86a83a1e04cf669784c1f5037a9878bbae5563a7af257c0bbf1e1d10887f3e76',
    'data/training-inputs.json': 'e753246da0365a49258d8e17ba7eb1fd28728fdbb7a01b3d5c1e803204d2d2e2',
    'fit/selection.json': '7760e10a8d1b2c17f731d0773f80b5605ecc2fdcef26e5d6fd40453912cf8c32',
    'gaussian-run/config.json': '68e6af7e75b73b81def26a4647ff43721209235b8a7a63babb136b54a15bc867',
}
TIMED = (2, 6, 10, 14, 18, 22)
REPEATS = 12
sys.path.insert(0, str(PROJECT/'src'))
from shuolee_portrait import runtime as rt, training


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(output):
    require(not output.exists() and not output.is_relative_to(RUN.parent) and not output.is_relative_to(PROJECT),
            'Use a new benchmark output directory.')
    require(rt.sha(BASE_TRAINING) == BASE_SHA, 'The original six-worker training source changed.')
    for name, digest in PINS.items():
        require(rt.sha(RUN/name) == digest, f'Changed 006 source: {name}')
    rows = rt.read(RUN/'data/training-inputs.json')['records']
    ids = [row['frame'] for row in rows]
    selection = rt.read(RUN/'fit/selection.json')
    require(len(ids) == len(set(ids)) == 24 and set(ids) == set(selection['selected_frames'])
            and not set(ids) & set(selection['retained_holdouts']), 'The frozen 006 roster differs.')
    for row in rows:
        require(row['wh'] == [2704, 1520] and rt.sha(row['image']) == row['source_sha256']
                and rt.sha(row['mask']) == row['mask_sha256'], f'Changed native source: {row["frame"]}')
    plan = dict(state='fixed_target_inputs_verified', frame_ids=ids, timed_frames=[ids[i] for i in TIMED],
        protected_holdouts=selection['retained_holdouts'], resolution=[2704, 1520],
        source_run=str(RUN), source_sha256=PINS, baseline_training_sha256=BASE_SHA,
        candidate_training_sha256=rt.sha(PROJECT/'src/shuolee_portrait/training.py'),
        script_sha256=rt.sha(Path(__file__)), timed_repeats_per_frame=REPEATS,
        real_targets=True, synthetic_live_predictions=True, optimizer_updates=0,
        gaussian_training_started=False, cold_to_export_measured=False)
    return rows, plan


def byte_equal(a, b):
    import torch
    return (a.shape == b.shape and a.dtype == b.dtype and
            torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),
                        b.detach().contiguous().reshape(-1).view(torch.uint8)))


def tensor_bytes(values):
    return sum(value.numel()*value.element_size() for value in values)


def benchmark(rows, plan, output):
    import torch
    require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA device for this loss-only test.')
    spec = importlib.util.spec_from_file_location('shuolee_portrait._original_training', BASE_TRAINING)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.backends.cuda.matmul.allow_tf32 = False  # Same as the actual SH0 path.
    output.mkdir(parents=True, exist_ok=False)
    rt.save(output/'preregistration.json', plan)
    targets = []
    for row in rows:
        rgb, alpha, box = training._native_foreground_target(row)
        targets.append((rgb.cuda(), alpha.cuda(), box))
    torch.cuda.synchronize()
    target_resident = torch.cuda.memory_allocated()
    expected_resident = tensor_bytes(value for rgb, alpha, _ in targets for value in (rgb, alpha))
    require(expected_resident == rt.read(RUN/'gaussian-run/config.json')['cached_target_bytes'],
            'Baseline cropped-target allocation differs from the delivered run.')

    def predicted(i):
        _, mask, _ = targets[i]
        generator = torch.Generator(device='cuda').manual_seed(713+i)
        alpha = (.65*mask+.1*torch.rand(mask.shape, generator=generator, device='cuda')).requires_grad_()
        rgb = (torch.rand((3, *mask.shape[-2:]), generator=generator, device='cuda') * alpha.detach()).requires_grad_()
        return rgb, alpha

    def step(i, fixture, fixed=None):
        loss = (baseline._sh0_foreground_alpha_loss(*fixture, *targets[i]) if fixed is None else
                training._sh0_foreground_alpha_loss(*fixture, *targets[i], fixed_target=fixed))
        grads = torch.autograd.grad(loss[0]+.1*loss[3], fixture)
        return (*loss, *grads)

    base_peak = 0
    for i in TIMED:
        fixture = predicted(i)
        step(i, fixture)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step(i, fixture)
        torch.cuda.synchronize()
        base_peak = max(base_peak, torch.cuda.max_memory_allocated())
        del fixture

    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    fixed = [training._fixed_sh0_target(*target) for target in targets]
    torch.cuda.synchronize()
    build_seconds = perf_counter()-start
    build_peak = torch.cuda.max_memory_allocated()
    fixed_resident = torch.cuda.memory_allocated()-target_resident
    fixed_tensor_bytes = tensor_bytes(value for (terms, area) in fixed for value in (*terms, area))

    parity = {}
    for i, row in enumerate(rows):
        fixture = predicted(i)
        original, cached = step(i, fixture), step(i, fixture, fixed[i])
        keys = ('photo', 'l1', 'ssim', 'alpha', 'rgb_gradient', 'alpha_gradient')
        parity[row['frame']] = {name: byte_equal(a, b) for name, a, b in zip(keys, original, cached, strict=True)}
        del fixture, original, cached
    exact = all(all(entry.values()) for entry in parity.values())
    if not exact:
        rt.save(output/'result.json', dict(state='parity_failed', parity=parity, optimizer_updates=0))
        raise RuntimeError('Loss/gradient bytes differ. See the preserved parity receipt.')

    def measure(i, fixture, cache):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = perf_counter()
        begin.record()
        step(i, fixture, cache)
        end.record()
        end.synchronize()
        return begin.elapsed_time(end), (perf_counter()-start)*1000.

    times = {}
    cache_peak = 0
    for i in TIMED:
        fixture = predicted(i)
        for _ in range(3):
            step(i, fixture)
            step(i, fixture, fixed[i])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step(i, fixture, fixed[i])
        torch.cuda.synchronize()
        cache_peak = max(cache_peak, torch.cuda.max_memory_allocated())
        samples = {'original': [], 'cached': []}
        for repeat in range(REPEATS):
            order = ('original', 'cached') if repeat % 2 == 0 else ('cached', 'original')
            for name in order:
                samples[name].append(measure(i, fixture, fixed[i] if name == 'cached' else None))
        times[rows[i]['frame']] = {name: dict(event_ms=median(t[0] for t in values),
                                             wall_ms=median(t[1] for t in values), samples_ms=values)
                                  for name, values in samples.items()}
        del fixture
    saved_wall = mean(row['original']['wall_ms']-row['cached']['wall_ms'] for row in times.values())
    result = dict(state='completed_loss_only', preregistration_sha256=rt.sha(output/'preregistration.json'),
        exact_forward_gradient_parity=True, parity=parity, timing=times, cache_build_seconds=build_seconds,
        baseline_target_bytes=expected_resident, fixed_terms_tensor_bytes=fixed_tensor_bytes,
        cuda_fixed_terms_resident_bytes=fixed_resident, cuda_cache_build_peak_allocated_bytes=build_peak,
        cuda_original_loss_peak_allocated_bytes=base_peak, cuda_cached_loss_peak_allocated_bytes=cache_peak,
        sampled_mean_saved_wall_ms=saved_wall, projection_10000_loss_only_seconds=saved_wall*10-build_seconds,
        optimizer_updates=0, gaussian_training_started=False, cold_to_export_measured=False,
        gpu=torch.cuda.get_device_name())
    rt.save(output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key not in ('parity', 'timing')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUN.parent.with_name('shuolee-sh0-target-cache006'))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    rows, plan = prepare(output)
    if args.preflight:
        print(json.dumps(dict(plan, gpu_calls=0)))
        return
    with Path('/tmp/shuolee-native122-gpu.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        benchmark(rows, plan, output)


if __name__ == '__main__':
    main()
