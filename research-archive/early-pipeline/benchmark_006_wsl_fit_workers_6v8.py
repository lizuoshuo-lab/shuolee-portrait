"""Matched six/eight-worker CPU timings on 24 frozen original 006 fits."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
from hashlib import sha256
from itertools import islice
import json
from multiprocessing import get_context
import os
from pathlib import Path
from statistics import median
import subprocess
from time import perf_counter

import benchmark_006_wsl_fit_workers as stock
from review_bound_alpha_ab import pin


LEASE = Path('/tmp/shuolee-native122-gpu.lock')
WIN_PS = Path('/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe')
ORDER = (6, 8, 8, 6)
GIB = 1024**3


def prepare(output):
    stock.need(not output.exists() and not output.is_relative_to(stock.RUN.parent)
               and not output.is_relative_to(stock.PROJECT), 'Use a fresh external worker benchmark output.')
    seq = stock.seq
    plan, inputs, _, beta, latent, official, assets, _ = stock.prepare()
    eligible = inputs['scope']['fit_frame_ids']
    offsets = [(2*i+1)*len(eligible)//48 for i in range(24)]
    frames = [eligible[index] for index in offsets]
    rows = [inputs['rows'][inputs['ids'].index(frame)] for frame in frames]
    stock.need(len(set(frames)) == 24 and not set(frames) & set(plan['protected_holdouts']),
               'Expected 24 deterministic and non-held 006 candidates.')
    selection = seq.rt.read(stock.RUN/'fit/selection.json')
    saved = {}
    pins = plan['source_pins']
    for row in rows:
        frame = row['frame']
        folder = stock.RUN/'fit/frames'/frame
        array_path, report_path = folder/'winner-camera-space.npz', folder/'result.json'
        arrays, report = stock.load_arrays(array_path), seq.rt.read(report_path)
        seq.check_result(report, row)
        slot = inputs['ids'].index(frame)
        observed, K = inputs['targets'][slot], inputs['K'][slot]
        scores, flags, _, _ = stock.stable_selection._candidate(row, arrays, observed, K, beta, None, report, True)
        stock.need(stock.exact(arrays['betas'], beta) and stock.np.array_equal(arrays['K_integer_centers'], K)
                   and report['shared_body_target_sha256'] == plan['shared_body_target_sha256']
                   and scores == selection['reasons'][frame]['scores'] and flags == selection['reasons'][frame]['screen_flags'],
                   f'Archived stock outputs or pose screen differ: {frame}')
        saved[frame] = (arrays, report, scores, flags)
        pins.update({str(path): seq.rt.sha(path) for path in (array_path, report_path)})
        pins.update({row['image']: row['source_sha256'], row['mask']: row['mask_sha256']})
    seq.check_bound(pins)
    plan.update(scope='Only 24 deterministic frozen-shape candidates. No new shared shape, sequence, or Gaussian training.',
        worker_counts=[6, 8], timing_order=list(ORDER), timed_arm_count=len(ORDER), frame_ids=frames,
        source_frame_indices=[row['source_frame_index'] for row in rows], candidate_order_indices=offsets,
        archived_orientation_counts={frame: saved[frame][1]['orientation_count'] for frame in frames},
        start_method='spawn', script_sha256=seq.rt.sha(Path(__file__)), windows_commit_headroom_min=8*GIB,
        wsl_preflight_memory_min=8*GIB, wsl_active_memory_min=4*GIB, wsl_active_commit_headroom_min=2*GIB,
        cold_to_export_measured=False)
    return plan, inputs, rows, beta, latent, official, assets, saved


def memory_status():
    mem = dict((line.split()[0].rstrip(':'), int(line.split()[1])*1024)
               for line in Path('/proc/meminfo').read_text().splitlines()
               if line.split()[0] in ('MemAvailable:', 'CommitLimit:', 'Committed_AS:'))
    return dict(available=mem['MemAvailable'], commit_limit=mem['CommitLimit'], committed=mem['Committed_AS'])


def child_rss():
    """Sum simultaneous WSL RSS of only the benchmark's direct worker processes."""
    parent = str(os.getpid())
    workers = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            status = dict(line.split(':', 1) for line in (entry/'status').read_text().splitlines() if ':' in line)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if status.get('PPid', '').strip() == parent and 'VmRSS' in status:
            workers[entry.name] = int(status['VmRSS'].split()[0])*1024
    return workers


def host_guard():
    command = ('$m=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory; '
               '$m | Select-Object AvailableMBytes,CommittedBytes,CommitLimit | ConvertTo-Json -Compress')
    raw = subprocess.check_output([str(WIN_PS), '-NoProfile', '-NonInteractive', '-Command', command], text=True)
    host = json.loads(raw)
    stock.need(host['AvailableMBytes'] >= 8192 and host['CommitLimit']-host['CommittedBytes'] >= 8*GIB,
               'Windows available memory or commit headroom below 8 GiB.')
    return host


def run_arm(jobs, workers, official, assets):
    start = perf_counter()
    results = {}
    peaks = dict(worker_sum_rss_bytes=0, one_worker_rss_bytes=0, wsl_min_available_bytes=None,
                 wsl_min_commit_headroom_bytes=None)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                             initializer=stock.seq._init_cpu_fit, initargs=(official, assets)) as pool:
        queue = iter(jobs)
        pending = {pool.submit(stock.seq._fit_cpu_job, job): job[0]['frame'] for job in islice(queue, workers)}
        while pending:
            done, _ = wait(pending, timeout=.4, return_when=FIRST_COMPLETED)
            rss, mem = child_rss(), memory_status()
            commit = mem['commit_limit']-mem['committed']
            peaks['worker_sum_rss_bytes'] = max(peaks['worker_sum_rss_bytes'], sum(rss.values()))
            peaks['one_worker_rss_bytes'] = max(peaks['one_worker_rss_bytes'], max(rss.values(), default=0))
            for name, value in (('wsl_min_available_bytes', mem['available']), ('wsl_min_commit_headroom_bytes', commit)):
                peaks[name] = value if peaks[name] is None else min(peaks[name], value)
            stock.need(mem['available'] >= 4*GIB and commit >= 2*GIB,
                       'WSL memory or commit guard stopped queueing new work; running jobs are preserved.')
            for future in done:
                frame = pending.pop(future)
                results[frame] = future.result()
                following = next(queue, None)
                if following is not None:
                    pending[pool.submit(stock.seq._fit_cpu_job, following)] = following[0]['frame']
    return results, perf_counter()-start, peaks


def benchmark(prepared, output):
    import torch
    seq = stock.seq
    plan, inputs, rows, beta, latent, official, assets, archived = prepared
    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _, vposer, _ = seq.official.create_models(official_root=official, assets=assets, device='cpu', static_topology=False)
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    stock.need(sha256(pose.tobytes()).hexdigest() == plan['shared_body_target_sha256'], 'Shared target changed.')
    del vposer
    output.mkdir(parents=True, exist_ok=False)
    seq.rt.save(output/'preregistration.json', plan)
    slots = {row['frame']: inputs['ids'].index(row['frame']) for row in rows}
    arms = []
    for arm, workers in enumerate(ORDER, 1):
        host = host_guard()
        machine = memory_status()
        stock.need(machine['available'] >= 8*GIB and machine['commit_limit']-machine['committed'] >= 8*GIB,
                   'WSL available memory or commit headroom below 8 GiB.')
        arm_dir = output/f'arm-{arm}-workers-{workers}'
        arm_dir.mkdir()
        jobs = [(row, arm_dir/row['frame'], inputs['targets'][slots[row['frame']]:slots[row['frame']]+1],
                 inputs['K'][slots[row['frame']]], beta, True, pose) for row in rows]
        try:
            completed, elapsed, peaks = run_arm(jobs, workers, official, assets)
            comparisons = {}
            for row in rows:
                frame = row['frame']
                arrays, receipt = completed[frame]
                pin(Path(receipt['path']), receipt['sha256'])
                report = seq.rt.read(receipt['path'])
                seq.check_result(report, row)
                slot = slots[frame]
                scores, flags, _, _ = stock.stable_selection._candidate(row, arrays, inputs['targets'][slot],
                    inputs['K'][slot], beta, None, report, True)
                a, r, s, f = archived[frame]
                delta = sorted(name for name in arrays.keys() | a.keys() if name not in arrays or name not in a
                               or not stock.exact(arrays[name], a[name]))
                comparisons[frame] = dict(exact=not delta and report == r and scores == s and flags == f,
                    differing_arrays=delta, report_exact=report == r, scores_exact=scores == s, flags_exact=flags == f)
            exact = all(item['exact'] for item in comparisons.values())
            arms.append(dict(arm=arm, workers=workers, seconds=elapsed, exact_archive_parity=exact,
                             host_before=host, wsl_before=machine, peaks=peaks, comparisons=comparisons))
            seq.rt.save(output/'progress.json', dict(state='running' if exact else 'parity_failed', arms=arms))
            stock.need(exact, 'Worker count or observer altered exact stock outputs.')
            print(json.dumps(dict(arm=arm, workers=workers, seconds=elapsed, peaks=peaks)), flush=True)
        except BaseException as error:
            seq.rt.save(output/'result.json', dict(state='failed', failed_arm=arm, workers=workers,
                        error_type=type(error).__name__, error=str(error), completed_arms=arms,
                        cold_to_export_measured=False, gaussian_training_started=False))
            raise
    seq.check_bound(plan['source_pins'])
    medians = {n: median(a['seconds'] for a in arms if a['workers'] == n) for n in (6, 8)}
    result = dict(state='completed_exact_candidate_probe', exact_archive_parity=True, arms=arms,
                  median_seconds_six=medians[6], median_seconds_eight=medians[8],
                  speedup=medians[6]/medians[8], full_shape_pass_measured=False, cold_to_export_measured=False,
                  gaussian_training_started=False, quality_accepted=False,
                  preregistration_sha256=seq.rt.sha(output/'preregistration.json'))
    seq.rt.save(output/'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'arms'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=stock.LAB/'shuolee-smpl-fit-workers006-6v8')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    prepared = prepare(output)
    if args.preflight:
        plan = prepared[0]
        print(json.dumps(dict(state='frozen_worker_probe_inputs_verified', frame_ids=plan['frame_ids'],
            protected_holdouts=plan['protected_holdouts'], pins=len(plan['source_pins']), order=list(ORDER), fits_started=0)))
        return
    with LEASE.open('a+') as lease, stock.LOCK.open('rb') as local:
        for lock in (lease, local):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        stock.guard_machine()
        stock.need(len(os.sched_getaffinity(0)) >= 8, 'Eight WSL vCPUs are required.')
        benchmark(prepared, output)


if __name__ == '__main__':
    main()
