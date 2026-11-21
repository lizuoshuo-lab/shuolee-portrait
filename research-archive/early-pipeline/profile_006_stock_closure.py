"""Sample eight original CPU objective closures on two immutable 006 fits."""

import argparse
from collections import defaultdict
from contextlib import ExitStack
import fcntl
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

sys.dont_write_bytecode = True
import observe_006_official_fit_stages as observer


stock, seq, need = observer.stock, observer.stock.seq, observer.stock.need
OUTPUT = stock.LAB / 'shuolee-smpl-fit-closure-profile006'
REFERENCE = stock.LAB / 'shuolee-smpl-fit-stages006-methodhook'
FRAMES = ('frame_000007', 'frame_000021')
STAGES = (3, 4)
CALLS = (32, 256)
PINS = {
    REFERENCE / 'preregistration.json': 'c0353892df1539c9e3bca670148a1e87d70630463d3ba6a532a2d0e276e92823',
    REFERENCE / 'result.json': 'bd5e91d743e94c61c9f96770ce86774ce40d461e6cd608ab8d32ae98e4bd0a16',
    Path(observer.__file__): '713f65c251a508084b6f01fafb38f14afa2e0a6b7f960e7347a94565b6ce08ab',
}
PREFIX = 'audit_006/'


def prepare(output):
    seq.check_bound(PINS)
    prepared = observer.prepare(output)
    plan, inputs, rows, beta, latent, official, assets, archived = prepared
    reference = seq.rt.read(REFERENCE / 'result.json')
    need(reference['state'] == 'completed' and reference['exact_stock_parity']
         and reference['preregistration_sha256'] == PINS[REFERENCE / 'preregistration.json']
         and tuple(row['frame'] for row in rows[:2]) == FRAMES, 'Previous stock observer or frozen cohort differs.')
    expected = {}
    for frame in FRAMES:
        receipt = reference['records'][frame]
        need(receipt['exact_archive_parity'], 'Selected profiling frame lacked exact prior parity.')
        phases = receipt['phases']
        need([p['stage'] for p in phases] == ['camera_init', 0, 1, 2, 3, 4]
             and all(phases[1+s]['monitor_closure_calls'] >= max(CALLS) for s in STAGES),
             'Expected one orientation and both sample ordinals in each late stage.')
        expected[frame] = [p['monitor_closure_calls'] for p in phases]
    plan.update(scope='CPU operator timing only. Exactly two existing stock fits and eight original calls.',
                frame_ids=list(FRAMES), cohort_indices=plan['cohort_indices'][:2],
                source_frame_indices=[row['source_frame_index'] for row in rows[:2]],
                selected_stock_closure_ordinals=list(CALLS), selected_body_stages=list(STAGES),
                expected_original_phase_counts=expected, optimizer_rule_changed=False, fit_calls=2,
                sampled_closures=8, training_steps=0, gpu_calls=0,
                profiler_overhead_included=True, cold_to_export_measured=False)
    plan['source_pins'].update({str(path): digest for path, digest in PINS.items()})
    plan['source_pins'][str(Path(__file__).resolve())] = seq.rt.sha(__file__)
    return plan, inputs, rows[:2], beta, latent, official, assets, archived


def tagged(fn, name, torch):
    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(PREFIX + name):
            return fn(*args, **kwargs)
    return wrapped


def sample(closure, args, kwargs, body, vposer, loss):
    import torch

    body_module = import_module('smplx.body_models')
    lbs_module = import_module('smplx.lbs')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                record_shapes=False, profile_memory=False, with_stack=False) as profile:
        with ExitStack() as patches:
            for obj, attr, tag in ((body, 'forward', 'body_forward'),
                                   (vposer, 'decode', 'vposer_forward'),
                                   (loss, 'forward', 'loss_forward'),
                                   (body_module, 'lbs', 'lbs_forward'),
                                   (lbs_module, 'batch_rigid_transform', 'rigid_forward'),
                                   (torch.Tensor, 'backward', 'backward')):
                patches.enter_context(patch.object(obj, attr, new=tagged(getattr(obj, attr), tag, torch)))
            value = tagged(closure, 'closure', torch)(*args, **kwargs)
    averages = profile.key_averages()
    spans = {e.key.removeprefix(PREFIX): dict(inclusive_us=e.cpu_time_total, self_us=e.self_cpu_time_total,
                                             calls=e.count) for e in averages if e.key.startswith(PREFIX)}
    need(len(spans) == 7 and all(span['calls'] == 1 for span in spans.values()), 'Missing or repeated sampled span.')
    outer = sum(spans[name]['inclusive_us'] for name in ('body_forward', 'vposer_forward', 'loss_forward', 'backward'))
    need(outer <= spans['closure']['inclusive_us']
         and spans['rigid_forward']['inclusive_us'] <= spans['lbs_forward']['inclusive_us']
         <= spans['body_forward']['inclusive_us'], 'Unexpected overlapping profile spans.')
    nodes = defaultdict(lambda: [0., 0., 0])
    prefix = 'autograd::engine::evaluate_function: '
    for event in profile.events():
        if not event.name.startswith(prefix):
            continue
        parent = event.cpu_parent
        while parent is not None and not parent.name.startswith(prefix):
            parent = parent.cpu_parent
        if parent is None:
            bucket = nodes[event.name.removeprefix(prefix)]
            bucket[0] += event.cpu_time_total
            bucket[1] += event.self_cpu_time_total
            bucket[2] += 1
    node_total = sum(v[0] for v in nodes.values())
    need(0 < node_total <= spans['backward']['inclusive_us'], 'CPU autograd roots escaped the backward span.')
    ranked = sorted(nodes.items(), key=lambda e: -e[1][0])
    return value, dict(spans=spans, closure_other_us=spans['closure']['inclusive_us'] - outer,
        root_autograd_inclusive_us=node_total, root_autograd_self_us=sum(v[1] for v in nodes.values()),
        root_autograd_count=sum(v[2] for v in nodes.values()),
        top_autograd_roots=[dict(name=name, inclusive_us=v[0], self_us=v[1], calls=v[2]) for name, v in ranked[:12]])


def observe_frame(job, expected_counts):
    phases, samples = [], []
    loader = seq.official.load_official

    def observed_load(path):
        utils, prior, fitting, optimizer = loader(path)
        monitor = fitting.FittingMonitor
        create = monitor.create_fitting_closure

        def observed_create(self, *args, **kwargs):
            closure = create(self, *args, **kwargs)
            stage = len(phases) - 1
            counts = dict(stage='camera_init' if stage == -1 else stage, calls=0)
            phases.append(counts)

            def observed_closure(*call_args, **call_kwargs):
                counts['calls'] += 1
                ordinal = counts['calls']
                if stage in STAGES and ordinal in CALLS:
                    value, data = sample(closure, call_args, call_kwargs, args[1], kwargs['vposer'], kwargs['loss'])
                    samples.append(dict(stage=stage, ordinal=ordinal, **data))
                    return value
                return closure(*call_args, **call_kwargs)
            return observed_closure

        monitor.create_fitting_closure = observed_create
        return utils, prior, fitting, optimizer

    with patch.object(seq.official, 'load_official', new=observed_load):
        arrays, receipt = seq._fit_cpu_job(job)
    need([phase['calls'] for phase in phases] == expected_counts
         and [(row['stage'], row['ordinal']) for row in samples] == [(s, c) for s in STAGES for c in CALLS],
         'Closure calls or fixed profiling slots differ from the prior exact stock observer.')
    return arrays, receipt, phases, samples


def run(prepared, output):
    plan, inputs, rows, beta, latent, official, assets, archived = prepared
    seq.rt.configure(assets)
    seq._init_cpu_fit(official, assets)
    _, _, vposer, _ = seq._CPU_FIT
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    need(sha256(pose.tobytes()).hexdigest() == plan['shared_body_target_sha256'], 'Frozen shared target changed.')
    selection = seq.rt.read(stock.RUN / 'fit/selection.json')['reasons']
    records = {}
    for row in rows:
        frame = row['frame']
        slot = inputs['ids'].index(frame)
        target, K = inputs['targets'][slot:slot+1], inputs['K'][slot]
        job = (row, output / frame, target, K, beta, True, pose)
        arrays, receipt, phases, samples = observe_frame(job, plan['expected_original_phase_counts'][frame])
        report = seq.rt.read(receipt['path'])
        seq.check_result(report, row)
        old_arrays, old_report, old_flags = archived[frame]
        scores, flags, _, _ = stock.stable_selection._candidate(row, arrays, target[0], K, beta, None, report, True)
        need(arrays.keys() == old_arrays.keys() and all(stock.exact(v, old_arrays[key]) for key, v in arrays.items())
             and report == old_report and scores == selection[frame]['scores'] and flags == old_flags == selection[frame]['screen_flags'],
             f'Profile changed archived stock arrays, fit receipt or screening on {frame}.')
        records[frame] = dict(source_frame_index=row['source_frame_index'], exact_stock=True,
            screen_flags=flags, phases=phases, samples=samples)
        seq.rt.save(output / 'progress.json', dict(completed=list(records), exact_stock=True, optimizer_training_steps=0))
    seq.check_bound(plan['source_pins'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'profile'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(output.parent == stock.LAB and not output.exists(), 'Use a new isolated profile output.')
    prepared = prepare(output)
    plan = prepared[0]
    if args.mode == 'check':
        print(json.dumps(dict(state='006_closure_profile_inputs_verified', frames=list(FRAMES),
                              pins=len(plan['source_pins']), fit_calls=0, gpu_calls=0)))
        return
    with observer.LEASE.open('a+b') as lease, stock.LOCK.open('rb') as fit_lock:
        for lock in (lease, fit_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stock.guard_machine()
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        started = perf_counter()
        try:
            records = run(prepared, output)
            result = dict(state='stock_cpu_closure_profile_complete', records=records, fit_calls=2, gpu_calls=0,
                sampled_closures=8, exact_stock_arrays_and_phase_counts=True, gaussian_training_steps=0,
                profiler_overhead_included=True, cold_to_export_measured=False, elapsed_seconds=perf_counter()-started,
                preregistration_sha256=seq.rt.sha(output / 'preregistration.json'))
            seq.rt.save(output / 'result.json', result)
            print('006_CLOSURE_PROFILE_COMPLETE', list(records), result['elapsed_seconds'], flush=True)
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=repr(exc), elapsed_seconds=perf_counter()-started))
            raise


if __name__ == '__main__':
    main()
