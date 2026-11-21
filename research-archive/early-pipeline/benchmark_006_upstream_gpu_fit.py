"""Bounded native CUDA replay of the frozen 006 SMPLify-X CPU fit cohort."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

import benchmark_006_wsl_fit_workers as cpu
import numpy as np


CPU_RUN = cpu.LAB / 'shuolee-smpl-fit-workers006'
OUTPUT = cpu.LAB / 'shuolee-smpl-upstream-gpu-fit006'
GPU_LOCK = Path('/tmp/shuolee-native122-gpu.lock')
CPU_PINS = {
    CPU_RUN / 'result.json': 'a2f08b91319104c7891652df738a7e725caf542891847f15d87aba561d68c7f4',
    CPU_RUN / 'preregistration.json': '1d8e9476103af9dc1ccedb78ac71d81352bf3b76063787ee488598fcfe4bec85',
}
seq = cpu.seq
need = cpu.need
screen = cpu.stable_selection


def prepare():
    """Prove exact identity with both timed CPU arms, then decode the target on CPU."""
    seq.check_bound(CPU_PINS)
    result, reg = (seq.rt.read(path) for path in CPU_PINS)
    plan, inputs, rows, beta, latent, official, assets, archived = cpu.prepare()
    need(result['state'] == 'complete' and result['exact_parity']
         and result['preregistration_sha256'] == CPU_PINS[CPU_RUN / 'preregistration.json']
         and all(plan[key] == reg[key] for key in
                 ('source_commit', 'frame_ids', 'source_frame_indices', 'protected_holdouts',
                  'shared_body_target_sha256', 'source_pins', 'archived_orientation_counts')),
         'The native CUDA arm must match the completed exact CPU benchmark.')
    import torch

    seq.rt.configure(assets)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _, vposer, _ = seq.official.create_models(
        official_root=official, assets=assets, device='cpu', static_topology=False)
    pose = seq.official.decode_shared_body_pose(vposer, latent, device='cpu')
    need(sha256(pose.tobytes()).hexdigest() == reg['shared_body_target_sha256'],
         'The CPU-decoded shared target no longer matches the archived fitter input.')
    plan.update(
        scope='Serial native CUDA replay. Per-frame solver API and inputs unchanged. '
              'CPU operational reference uses worker pools; timing includes this concurrency difference.',
        static_topology=False, floating_point='float32', cuda_tf32=False,
        cuda_workers=1, gpu_device='cuda:0', gpu_seconds_ceiling=result['seconds_six'],
        gpu_timer='One reusable model pair setup, all frame fits and artifact I/O, '
                  'CUDA synchronization, model cleanup. CPU target decode, availability probe '
                  'and post-fit validation outside timer.',
        cpu_reference_seconds={'workers_4': result['seconds_four'], 'workers_6': result['seconds_six']},
        cpu_runtime=reg['runtime'], cpu_reference_pins={str(path): value for path, value in CPU_PINS.items()},
        source_pins={**plan['source_pins'], str(Path(__file__).resolve()): seq.rt.sha(__file__)},
        formal_holdouts_used=False, quality_accepted=False)
    return plan, inputs, rows, beta, pose, official, assets, archived


def fit(row, slot, inputs, beta, pose, body, vposer, official, destination):
    frame = row['frame']
    destination.mkdir()
    with (destination / 'fit.log').open('x', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            arrays, report = seq.official.fit_frame(
                official_root=official, image=seq.checked_image(row),
                keypoints=inputs['targets'][slot:slot + 1], K_integer_centers=inputs['K'][slot],
                body_model=body, vposer2=vposer, result_path=destination / 'official-result.pkl',
                shared_betas=beta, freeze_betas=True, device='cuda:0',
                diff_path=destination / 'input-output-adapter.diff', static_topology=False,
                shared_body_pose=pose)
    arrays['frame_id'] = np.asarray(frame)
    np.savez_compressed(destination / 'winner-camera-space.npz', **arrays)
    report.update(state='complete', frame_id=frame, source_frame_index=row['source_frame_index'],
                  source_image_sha256=row['source_sha256'], quality_accepted=False)
    seq.rt.save(destination / 'result.json', report)
    return arrays, report


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def delta(a, b):
    d = np.linalg.norm(a.astype(np.float64) - b.astype(np.float64), axis=-1)
    return dict(max=float(d.max()), rms=rms(d))


def body_error(observed, fitted, height):
    body = screen.BODY[observed[screen.BODY, 2] ** 2 > 0]
    d = np.linalg.norm(fitted['joints_projected_native'][0, body].astype(np.float64)
                       - observed[body, :2].astype(np.float64), axis=-1)
    return dict(points=len(body), max_px=float(d.max()), rms_px=rms(d),
                max_over_height=float(d.max()) / height)


def root_angle_deg(a, b):
    def quat(vector):
        axis_angle = vector.astype(np.float64).reshape(3)
        angle = np.linalg.norm(axis_angle)
        return np.r_[np.cos(angle / 2), .5 * np.sinc(angle / (2 * np.pi)) * axis_angle]

    cosine = np.clip(abs(float(quat(a) @ quat(b))), 0., 1.)
    return float(np.rad2deg(2 * np.arccos(cosine)))


def inspect(plan, row, slot, inputs, beta, archived, fit_result):
    arrays, report = fit_result
    seq.check_result(report, row)
    frame = row['frame']
    old, old_report, old_flags = archived[frame]
    target = inputs['targets'][slot]
    scores, flags, _, _ = screen._candidate(
        row, arrays, target, inputs['K'][slot], beta,
        (old['vertices_camera'].shape, old['faces']), report, True)
    height = row['initial_box_xyxy'][3] - row['initial_box_xyxy'][1]
    need(report['runtime_optimization'] is None
         and report['shared_body_target_sha256'] == plan['shared_body_target_sha256']
         and all(cpu.exact(arrays[key], old[key]) for key in
                 ('betas', 'K_integer_centers', 'K_compute_float32', 'faces')),
         'CUDA fit changed a frozen shape, camera, pose target or topology.')
    return dict(
        frame=frame, source_frame_index=row['source_frame_index'], source_sha256=row['source_sha256'],
        native_body_box_height_px=height, exact_betas_and_complete_K=True,
        complete_native_K=arrays['K_integer_centers'].tolist(),
        source_camera_w2c=row['w2c'], old_screen_flags=old_flags, cuda_screen_flags=flags,
        same_screen=flags == old_flags, old_scores=plan['old_scores'][frame], cuda_scores=scores,
        old_supported_body=body_error(target, old, height), cuda_supported_body=body_error(target, arrays, height),
        orientation_counts={'cpu': old_report['orientation_count'], 'cuda': report['orientation_count']},
        orientation_choice={name: 'only_branch_0' if result['orientation_count'] == 1 else 'winner_index_not_exported'
                            for name, result in [('cpu', old_report), ('cuda', report)]},
        root_delta_deg=root_angle_deg(old['global_orient'], arrays['global_orient']),
        camera_translation_delta_m=delta(old['camera_translation'], arrays['camera_translation'])['max'],
        joints_camera_delta_m=delta(old['joints_camera'], arrays['joints_camera']),
        joints_projection_delta_px=delta(old['joints_projected_native'], arrays['joints_projected_native']),
        supported_body_projection_delta_px=delta(
            old['joints_projected_native'][0, screen.BODY[target[screen.BODY, 2] ** 2 > 0]],
            arrays['joints_projected_native'][0, screen.BODY[target[screen.BODY, 2] ** 2 > 0]]),
        differing_arrays=sorted(key for key in old.keys() | arrays.keys()
                                if key not in old or key not in arrays or not cpu.exact(old[key], arrays[key])),
        same_orientation_count=old_report['orientation_count'] == report['orientation_count'])


class TimingBound(TimeoutError):
    pass


def deadline(_signal, _frame):
    raise TimingBound('The native CUDA arm reached the measured six-worker CPU time.')


def benchmark(prepared, output):
    import torch

    plan, inputs, rows, beta, pose, official, assets, archived = prepared
    original_reasons = seq.rt.read(cpu.RUN / 'fit/selection.json')['reasons']
    plan['old_scores'] = {row['frame']: original_reasons[row['frame']]['scores'] for row in rows}
    need(torch.cuda.is_available() and torch.cuda.device_count() == 1,
         'The benchmark requires one visible native CUDA device.')
    plan['runtime'] = dict(python=sys.version, numpy=np.__version__, torch=torch.__version__,
                           cuda=torch.version.cuda, device_index=0,
                           wsl_vcpus=len(os.sched_getaffinity(0)), environment_prefix=sys.prefix)
    output.mkdir()
    seq.rt.save(output / 'preregistration.json', plan)
    slots = {frame: i for i, frame in enumerate(inputs['ids'])}
    completed = []
    active = None
    setup_seconds = None
    device_name = None
    state = 'complete'
    error = None
    begin = perf_counter()
    previous_handler = signal.signal(signal.SIGALRM, deadline)
    try:
        signal.setitimer(signal.ITIMER_REAL, plan['gpu_seconds_ceiling'])
        body, vposer, _ = seq.official.create_models(
            official_root=official, assets=assets, device='cuda:0', static_topology=False)
        body.eval()
        device_name = torch.cuda.get_device_name(0)
        setup_seconds = perf_counter() - begin
        for row in rows:
            active = row['frame']
            started = perf_counter()
            slot = slots[active]
            result = fit(row, slot, inputs, beta, pose, body, vposer, official, output / active)
            torch.cuda.synchronize()
            completed.append((row, slot, result, perf_counter() - started))
            print(json.dumps(dict(frame=active, completed=len(completed),
                                  total_seconds=perf_counter() - begin)), flush=True)
        del body, vposer
        torch.cuda.synchronize()
    except TimingBound as exc:
        state, error = 'timing_bound_reached', str(exc)
    except Exception as exc:
        state, error = 'failed', f'{type(exc).__name__}: {exc}'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous_handler)
    elapsed = perf_counter() - begin
    audits = []
    try:
        for row, slot, result, seconds in completed:
            audits.append(dict(seconds=seconds, **inspect(plan, row, slot, inputs, beta, archived, result)))
        seq.check_bound(plan['source_pins'])
        seq.check_bound(CPU_PINS)
    except Exception as exc:
        state, error = 'parity_failed', f'{type(exc).__name__}: {exc}'
    done = [record['frame'] for record in audits]
    all_done = state == 'complete' and done == plan['frame_ids']
    receipt = dict(
        state=state, error=error, last_active_frame=active, seconds=elapsed,
        setup_seconds=setup_seconds, cuda_device_name=device_name, completed_frames=done,
        frames_with_valid_fit_artifacts=len(completed), expected_frames=plan['frame_ids'],
        parity=dict(complete=all_done, exact_all_outputs=all_done and all(not r['differing_arrays'] for r in audits),
                    unchanged_all_screens=all_done and all(r['same_screen'] for r in audits)),
        gpu_to_cpu6_speedup=plan['cpu_reference_seconds']['workers_6'] / elapsed if all_done else None,
        records=audits, preregistration_sha256=seq.rt.sha(output / 'preregistration.json'),
        cold_to_export_measured=False, gaussian_training_started=False, quality_accepted=False)
    seq.rt.save(output / 'result.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'records'}), flush=True)
    need(state in ('complete', 'timing_bound_reached'), error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'benchmark'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    need(not output.exists() and output.parent == cpu.LAB
         and not output.is_relative_to(cpu.PROJECT),
         'Use a fresh benchmark directory directly under the local lab.')
    need('torch' not in sys.modules, 'Set device visibility before torch is imported.')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.mode == 'benchmark' else ''
    prepared = prepare()
    plan = prepared[0]
    if args.mode == 'check':
        print(json.dumps(dict(state='frozen_inputs_verified', frames=plan['frame_ids'],
                              protected_holdouts=plan['protected_holdouts'], gpu_calls=0, fits_started=0,
                              pins=len(plan['source_pins']), gpu_seconds_ceiling=plan['gpu_seconds_ceiling'],
                              frozen_body_target=plan['shared_body_target_sha256'])))
        return
    from shuolee_portrait.cli import gpu_lock

    with GPU_LOCK.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            cpu.guard_machine()
            benchmark(prepared, output)


if __name__ == '__main__':
    main()
