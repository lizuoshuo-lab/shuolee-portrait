"""One fixed synthetic Torch-consumer CPU/CUDA parity probe; never train or rasterize.

Root must inspect actual GPU processes before invocation. Only a new registered
output directory is allowed. A timeout receipt is finalized by the parent after
process exit; the watchdog never writes the main result.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
import traceback

SCHEMA = 'shuolee-torch-consumer-gpu-parity/1'
FIXTURE = 'torch-consumer123-fixed-cpu-fixture-v1'
LIMITS = dict(wall_seconds=60, max_cuda_forwards=12, expected_cuda_forwards=11,
              max_cuda_backwards=1, float32_atol=3e-6, float32_rtol=2e-6,
              float64_atol=2e-12, float64_rtol=2e-12,
              gradient_atol=2e-10, gradient_rtol=1e-9, gauge_pixel_atol=2e-10)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def local(value):
    value = str(value)
    if os.name != 'nt' and len(value) > 2 and value[1] == ':':
        return Path('/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/'))
    return Path(value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         delete=False, suffix='.tmp') as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=local, required=True)
    output = parser.parse_args().output_dir.resolve()
    plan_path = output / 'plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    need(plan['schema'] == SCHEMA and plan['fixture'] == FIXTURE and plan['limits'] == LIMITS,
         'Registered fixed parity plan required.')
    for name in ('progress.json', 'result.json', 'timeout.json'):
        need(not (output / name).exists(), 'Already started; inspect prior evidence instead of repeating.')
    sources = {local(p).resolve(): digest for p, digest in plan['source_sha256'].items()}
    project = local(plan['project']).resolve()
    required = {Path(__file__).resolve(), project / 'tests/test_free_clothing_torch.py',
                project / 'src/shuolee_portrait/free_clothing_torch.py',
                project / 'src/shuolee_portrait/free_clothing_transport.py',
                project / 'src/shuolee_portrait/camera_world_units.py',
                project / 'src/avatar_core/utils/sh_utils.py'}
    need(required <= set(sources), 'Required source/fixture hash missing.')
    for path, digest in sources.items():
        need(sha(path) == digest, 'Source differs from plan: ' + str(path))
    start, start_utc = time.monotonic(), datetime.now(timezone.utc)
    result = dict(schema=SCHEMA, fixture=FIXTURE, passed=False, pid=os.getpid(), ppid=os.getppid(),
                  started=start_utc.isoformat(), executable=sys.executable, argv=sys.argv,
                  plan_sha256=sha(plan_path), source_sha256=plan['source_sha256'], checks=[],
                  cuda_forward_attempted=0, cuda_forward_completed=0,
                  cuda_backward_attempted=0, cuda_backward_completed=0, cpu_analytic_backwards=0,
                  native_raster_calls=0, body_model_forwards=0, optimizer_steps=0,
                  person_images_loaded=0, ply_exports=0, old_test_suites_run=0,
                  training_admitted=False, physical_camera_accepted=False, synthetic_only=True,
                  native_bridge_validated=False, performance_200k_validated=False)
    proc = Path('/proc/self/cmdline')
    if proc.exists():
        result['actual_process_cmdline'] = proc.read_bytes().decode(errors='replace').split('\0')[:-1]
    write(output / 'progress.json', dict(pid=os.getpid(), phase='starting', started=result['started']), True)
    lock, terminal = threading.RLock(), {'owner': None}

    def times():
        now = datetime.now(timezone.utc)
        return dict(finished=now.isoformat(), elapsed_monotonic_seconds=time.monotonic() - start,
                    elapsed_utc_seconds=(now - start_utc).total_seconds())

    def counters():
        return {k: result[k] for k in ('cuda_forward_attempted', 'cuda_forward_completed',
                                       'cuda_backward_attempted', 'cuda_backward_completed')}

    def progress(phase):
        with lock:
            if terminal['owner'] != 'timeout':
                write(output / 'progress.json', dict(pid=os.getpid(), phase=phase, **counters(), **times()))

    def timed_out():
        with lock:
            if terminal['owner'] is not None:
                return
            terminal['owner'] = 'timeout'
            try:
                write(output / 'timeout.json', dict(schema=SCHEMA, pid=os.getpid(), passed=False,
                      error='TimeoutError', configured_wall_seconds=60, final_result_owner='parent',
                      plan_sha256=result['plan_sha256'], source_sha256=plan['source_sha256'],
                      training_admitted=False, synthetic_only=True, **counters(), **times()), True)
            finally:
                os._exit(124)

    watchdog = threading.Timer(LIMITS['wall_seconds'], timed_out)
    watchdog.daemon = True
    watchdog.start()
    try:
        sys.path.insert(0, str(project / 'src'))
        import numpy as np
        import torch
        from shuolee_portrait.free_clothing_torch import FreeClothingTorch
        spec = importlib.util.spec_from_file_location('fixed_consumer_fixture', project / 'tests/test_free_clothing_torch.py')
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)  # definitions only; never invoke its unittest runner
        need(fixture.CONTROL_ID == 'torch-clothing-consumer123-v1' and
             fixture.FRAMES == ('sheared', 'orbit', 'reference', 'half_turn'), 'Unexpected CPU fixture.')
        need(sys.executable == plan['python'] and torch.__version__ == plan['torch_version'], 'Wrong Python/Torch runtime.')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Exactly the registered local GPU 0 is required.')
        need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible CUDA device required.')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        result['runtime'] = dict(torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                                 gpu_name=properties.name, capability=list(torch.cuda.get_device_capability(device)),
                                 total_memory=properties.total_memory, uuid=str(getattr(properties, 'uuid', 'unavailable')))

        def check(name, passed, **metrics):
            result['checks'].append(dict(name=name, passed=bool(passed), **metrics))
            need(passed, 'Fixed parity control failed: ' + name)

        def close(name, actual, expected, dtype, atol=None, rtol=None):
            values = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
            expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
            key = 'float32' if dtype == torch.float32 else 'float64'
            atol = LIMITS[key + '_atol'] if atol is None else atol
            rtol = LIMITS[key + '_rtol'] if rtol is None else rtol
            delta = np.abs(values.astype(np.float64) - expected.astype(np.float64))
            check(name, np.isfinite(values).all() and np.all(delta <= atol + rtol * np.abs(expected)),
                  max_abs=float(delta.max()), atol=atol, rtol=rtol)

        def forward(consumer, frame, parameters, covariance=True):
            with lock:
                need(terminal['owner'] is None and result['cuda_forward_attempted'] < LIMITS['max_cuda_forwards'], 'Forward budget/ownership violated.')
                result['cuda_forward_attempted'] += 1
                progress('forward_' + frame)
            value = consumer(frame, *parameters, gaussian_ids=fixture.IDS, return_covariance=covariance)
            torch.cuda.synchronize(device)
            with lock:
                result['cuda_forward_completed'] += 1
            return value

        source, camera_map = fixture.source(), fixture.cameras()
        cpu_covariance = fixture.covariance_independent(*fixture.params()[1:]).numpy()
        oracle = {}
        for frame in fixture.FRAMES:
            E = camera_map[frame]
            oracle[frame] = source.transport_numpy(frame, fixture.XYZ, cpu_covariance,
                                                    -E[:3, :3].T @ E[:3, 3], gaussian_ids=fixture.IDS)
        baselines, consumers, parameters = {}, {}, {}
        fields = ('centers', 'covariance', 'reference_view_directions', 'rotations', 'support_vertex_residual')
        for dtype in (torch.float32, torch.float64):
            consumer = FreeClothingTorch.from_cpu(source, camera_length_contract=fixture.length_contract(),
                                                  frame_source_w2c=camera_map, device=device, dtype=dtype)
            consumers[dtype] = consumer
            parameters[dtype] = tuple(torch.tensor(v, dtype=dtype, device=device, requires_grad=(dtype == torch.float64))
                                      for v in (fixture.XYZ, fixture.QUATS, fixture.LOGS))
            for frame in fixture.FRAMES:
                out = forward(consumer, frame, parameters[dtype])
                baselines[dtype, frame] = out
                for name in fields:
                    close(str(dtype) + '_' + frame + '_' + name, out[name], oracle[frame][name], dtype)
                check(str(dtype) + '_' + frame + '_identity_dtype', out['frame_id'] == frame and
                      out['gaussian_ids'] == fixture.IDS and all(out[k].dtype == dtype for k in fields))
                if frame in ('reference', 'half_turn', 'orbit'):
                    expected = (fixture.XYZ if frame == 'reference' else
                                fixture.XYZ @ fixture.HALF.T + [0., 0., 9.] if frame == 'half_turn' else
                                1.4 * fixture.XYZ @ fixture.Q.T + fixture.SHIFT)
                    close(str(dtype) + '_' + frame + '_analytic_centers', out['centers'], expected, dtype)

        base32 = baselines[torch.float32, 'orbit']
        no_cov = forward(consumers[torch.float32], 'orbit', parameters[torch.float32], False)
        check('no_covariance_allocation', no_cov['covariance'] is None)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            autocast = forward(consumers[torch.float32], 'orbit', parameters[torch.float32])
        for name in ('centers', 'quats', 'scales', 'log_scales', 'reference_view_directions'):
            close('no_cov_' + name, no_cov[name], base32[name], torch.float32, 0., 0.)
            close('autocast_' + name, autocast[name], base32[name], torch.float32, 0., 0.)
            check('autocast_dtype_' + name, autocast[name].dtype == torch.float32)

        scaled = FreeClothingTorch.from_cpu(fixture.source(fixture.mesh_frames() * 2.),
                   camera_length_contract=fixture.length_contract(2., 'sfm_units'), frame_source_w2c=camera_map,
                   device=device, dtype=torch.float64)
        scaled_params = tuple(torch.tensor(v, dtype=torch.float64, device=device) for v in
                              (fixture.XYZ * 2., fixture.QUATS, fixture.LOGS + np.log(2.)))
        gauge = forward(scaled, 'orbit', scaled_params)
        base64 = baselines[torch.float64, 'orbit']
        close('gauge_centers', gauge['centers'] / 2., base64['centers'], torch.float64)
        close('gauge_covariance', gauge['covariance'] / 4., base64['covariance'], torch.float64)
        close('gauge_SH_direction', gauge['reference_view_directions'], base64['reference_view_directions'], torch.float64)
        K = np.array([[810., 4., 337.2], [0., 790., 211.8], [0., 0., 1.]])
        projections = []
        for factor, value in ((1., base64), (2., gauge)):
            E = camera_map['orbit'].copy(); E[:3, 3] *= factor
            xyz = value['centers'].detach().cpu().numpy()
            image = (xyz @ E[:3, :3].T + E[:3, 3]) @ K.T
            projections.append(image[:, :2] / image[:, 2:])
        close('gauge_projection_pixels', projections[0], projections[1], torch.float64, LIMITS['gauge_pixel_atol'], 0.)

        # One CUDA backward reuses the already-counted orbit graph. The CPU
        # analytic oracle does not call either transport implementation.
        eval_sh = fixture.sh_evaluator()
        sh_np = np.random.default_rng(237).normal(0, .1, (5, 3, 16))
        sh = torch.tensor(sh_np, dtype=torch.float64, device=device, requires_grad=True)
        weights = torch.arange(1, 16, dtype=torch.float64, device=device).reshape(5, 3) / 10
        cw = torch.tensor([[.3, .12, -.08], [.12, .5, .04], [-.08, .04, .9]], dtype=torch.float64, device=device)
        loss = (base64['centers'] * weights).sum() + 7 * (base64['covariance'] * cw).sum() + (eval_sh(3, sh, base64['reference_view_directions']) * weights).sum()
        with lock:
            need(result['cuda_backward_attempted'] == 0, 'Only one CUDA backward allowed.')
            result['cuda_backward_attempted'] += 1
        loss.backward(); torch.cuda.synchronize(device)
        result['cuda_backward_completed'] += 1
        x2, q2, l2 = fixture.params(grad=True)
        sh2 = torch.tensor(sh_np, dtype=torch.float64, requires_grad=True)
        Q, shift = torch.tensor(fixture.Q), torch.tensor(fixture.SHIFT)
        direct_x = 1.4 * x2 @ Q.T + shift
        direct_cov = 1.4**2 * Q[None] @ fixture.covariance_independent(q2, l2) @ Q.T[None]
        E = camera_map['orbit']
        center_ref = ((-E[:3, :3].T @ E[:3, 3]) - fixture.SHIFT) @ fixture.Q / 1.4
        dirs = x2 - torch.tensor(center_ref); dirs = dirs / torch.linalg.vector_norm(dirs, dim=-1, keepdim=True)
        w_cpu, cw_cpu = weights.cpu(), cw.cpu()
        expected_loss = (direct_x * w_cpu).sum() + 7 * (direct_cov * cw_cpu).sum() + (eval_sh(3, sh2, dirs) * w_cpu).sum()
        expected_grads = torch.autograd.grad(expected_loss, (x2, q2, l2, sh2))
        result['cpu_analytic_backwards'] += 1
        for name, parameter, expected in zip(('XYZ', 'quaternion', 'log_scale', 'SH'), (*parameters[torch.float64], sh), expected_grads):
            need(parameter.grad is not None, 'Missing CUDA gradient: ' + name)
            check('nonzero_gradient_' + name, bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.abs().max()) > 1e-7)
            close('analytic_gradient_' + name, parameter.grad, expected, torch.float64,
                  LIMITS['gradient_atol'], LIMITS['gradient_rtol'])
        check('fixed_execution_budget', result['cuda_forward_attempted'] == result['cuda_forward_completed'] == 11 and
              result['cuda_backward_attempted'] == result['cuda_backward_completed'] == 1)
        need(not any(name == 'gsplat' or name.startswith('gsplat.') for name in sys.modules), 'Unexpected native renderer import.')
        for path, digest in sources.items():
            need(sha(path) == digest, 'Source changed during probe: ' + str(path))
        need(sha(plan_path) == result['plan_sha256'], 'Plan changed during probe.')
        result['this_fixture_cuda_parity_validated'] = True
        result['passed'] = True
    except BaseException as error:
        result.update(passed=False, error=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        watchdog.cancel()
        with lock:
            if terminal['owner'] != 'timeout':
                result.update(**times())
                if result['elapsed_monotonic_seconds'] > LIMITS['wall_seconds']:
                    result.update(passed=False, error='TimeoutError', message='Monotonic wall budget exceeded.')
                terminal['owner'] = 'main'
                write(output / 'result.json', result, True)
                progress('completed' if result['passed'] else 'failed')
                print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 124 if terminal['owner'] == 'timeout' else (0 if result['passed'] else 1)


if __name__ == '__main__':
    sys.exit(main())
