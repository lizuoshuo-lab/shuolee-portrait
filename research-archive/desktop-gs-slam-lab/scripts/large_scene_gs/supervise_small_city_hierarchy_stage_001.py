"""CPU supervisor of ONE explicitly authorized SmallCity hierarchy/pilot stage.

Default prepares a CPU plan. --check-dispatch is read-only. --execute alone
launches the selected stage after actual source/full-training/resource gates.
No restart, retry, next-stage dispatch, process termination, or budget change.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-005'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'
LOCAL = ROOT / 'scripts/large_scene_gs/run_local.py'
DRIVER = ROOT / 'scripts/large_scene_gs/run_small_city_hierarchy.py'
PILOT = ROOT / 'scripts/large_scene_gs/run_small_city_post_preflight_001.py'
PILOT_RUN = 'small-city-post00-i64-r4-001'
DRIVER_SHA = '0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84'
CONFIG_SHA = '55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'
PILOT_SHA = '11a68bc78bdf58a521934bf941f17bd0790b60a58f3354b2c458ee0c11b43a75'
LOCAL_SHA = '1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2'
OBSERVER_SHA = 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a'
JOB_HELPER_SHA = 'b7c0015374521b201a4e28845cb9ff79ca9b4c597d5c05f80bf69aa03f58378d'
PW = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
PW_ENV = os.environ.copy()
PW_ENV['PSModulePath'] = os.pathsep.join([str(PW.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def identifier(value):
    if not value.replace('-', '').isalnum() or len(value) > 100:
        raise ValueError('Invalid supervision run')
    return value


def fixed_sources():
    for path, expected in [(DRIVER, DRIVER_SHA), (LOCK / 'driver-source.py', DRIVER_SHA), (LOCK / 'configuration.json', CONFIG_SHA), (PILOT, PILOT_SHA), (LOCAL, LOCAL_SHA),
                           (ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py', OBSERVER_SHA),
                           (ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py', JOB_HELPER_SHA)]:
        if digest(path) != expected:
            raise RuntimeError('Fixed hierarchy/pilot/config/environment source changed')
    driver = load('_single_stage_fixed005_driver', LOCK / 'driver-source.py')
    driver.ROOT, driver.OUTPUT, driver.REPO = ROOT, BASE, ROOT / 'repos/hierarchical-3d-gaussians'
    driver.NATIVE_REPO = ROOT / 'repos/gaussian-hierarchy-large-gs'
    driver.BUILD, driver.EXES = BASE / 'hierarchy-build', driver.NATIVE_REPO / 'build-local/Release'
    config = read(LOCK / 'configuration.json')
    manifest_path = BASE / 'official-inputs/small-city-training-002/manifest.json'
    if digest(manifest_path) != config['input_manifest_sha256']:
        raise RuntimeError('Actual frozen training manifest changed')
    for row in config['source_records']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Pinned official source/native artifact changed')
    return driver, config, read(manifest_path)


def command_for(stage, chunk, config):
    prefix = [str(PYTHON), '-u', str(LOCAL)]
    if stage == 'post64':
        return prefix + [str(PILOT), '--stage', 'post', '--run', PILOT_RUN, '--execute']
    argv = prefix + [str(DRIVER), '--stage', stage, '--run', LOCK.name, '--input', config['input'], '--coarse-run', config['coarse_run']]
    for name, run in config['chunk_runs'].items():
        argv += ['--chunk-run', name + '=' + run]
    if chunk:
        argv += ['--chunk', chunk]
    return argv + ['--execute']


def completed_training(driver, config, manifest):
    rows = []
    for stage, chunk, run in [('coarse', None, config['coarse_run'])] + [('chunk', c, r) for c, r in config['chunk_runs'].items()]:
        saved, receipt, source = driver.completed_training(run, stage, chunk, config, manifest, None)
        if source['requested_iterations'] != 30000:
            raise RuntimeError('Single-stage workflow requires each selected original full30000 budget')
        directory = BASE / 'official-training' / run
        audit = read(directory / 'model-audit.json')
        parameter = Path(audit['parameter_path']).resolve()
        if parameter.parent != saved.resolve() or digest(parameter) != receipt['model_parameter_sha256']:
            raise RuntimeError('Completed real training parameter path/hash mismatch')
        rows.append({'run': run, 'stage': stage, 'chunk': chunk, 'outcome_sha256': digest(directory / 'outcome.json'),
                     'model_sha256': receipt['model_parameter_sha256'], 'saved_directory': str(saved)})
    return rows


def stage_target(stage, chunk):
    if stage == 'post64':
        root = BASE / 'official-post-preflight' / PILOT_RUN
        return root / 'outcome.json', root / 'creator-input/hierarchy.hier_opt', 'post64_preflight'
    directory = LOCK / 'trained_chunks' / chunk if chunk else LOCK
    if stage == 'creator':
        return directory / 'creator-outcome.json', directory / 'hierarchy.hier', 'unoptimized'
    if stage == 'post':
        return directory / 'post-outcome.json', directory / 'hierarchy.hier_opt', 'post_optimized'
    if stage == 'merge-unoptimized':
        return LOCK / 'merge-unoptimized-outcome.json', LOCK / 'merged-unoptimized.hier', 'unoptimized'
    return LOCK / 'merge-outcome.json', LOCK / 'merged.hier', 'post_optimized'


def require_stage_receipt(path, model, kind):
    receipt = read(path)
    if (receipt['status'] != 'completed' or receipt['exit_code'] != 0 or receipt['source_kind'] != kind
            or receipt['configuration_sha256'] != CONFIG_SHA or digest(model) != receipt['hierarchy_sha256']):
        raise RuntimeError('Actual matching prior/final stage receipt/model/config failed')
    return receipt


def preconditions(stage, chunk):
    receipt, output, _ = stage_target(stage, chunk)
    if receipt.exists() or output.exists():
        raise RuntimeError('Actual stage receipt/output exists; refuse overwrite or retry')
    if stage == 'creator':
        if receipt.parent.exists():
            raise RuntimeError('Creator target directory already exists')
    elif stage == 'post64':
        root = BASE / 'official-post-preflight' / PILOT_RUN
        if (root / 'creator-input').exists():
            raise RuntimeError('Pilot already copied/started; refuse retry')
        pilot = load('_single_stage_fixed_post64', PILOT)
        driver, codec, helper, training, config, manifest = pilot.modules()
        if read(root / 'configuration.json')['wrapper_source_sha256'] != PILOT_SHA:
            raise RuntimeError('Prepared pilot source configuration changed')
        pilot.source_binding(driver, codec, helper, training, config, manifest)
    elif stage == 'post':
        directory = LOCK / 'trained_chunks' / chunk
        require_stage_receipt(directory / 'creator-outcome.json', directory / 'hierarchy.hier', 'unoptimized')
        if (directory / 'post-stage').exists():
            raise RuntimeError('Formal post directory already exists; refuse retry')
    else:
        if (LOCK / (stage + '-stage')).exists():
            raise RuntimeError('Merger stage directory already exists')
        post = stage == 'merge'
        for name in read(LOCK / 'configuration.json')['chunk_runs']:
            directory = LOCK / 'trained_chunks' / name
            require_stage_receipt(directory / ('post-outcome.json' if post else 'creator-outcome.json'),
                                  directory / ('hierarchy.hier_opt' if post else 'hierarchy.hier'),
                                  'post_optimized' if post else 'unoptimized')


def process_snapshot():
    text = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='GaussianHierarchyCreator.exe' OR Name='GaussianHierarchyMerger.exe'\" | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run([str(PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', text], env=PW_ENV,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Read-only actual process observation failed')
    data = json.loads(result.stdout or '[]')
    return data if isinstance(data, list) else [data]


def busy_processes(snapshot):
    observer = load('_single_stage_exact_argv', ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py')
    scripts = {'train_coarse.py', 'train_single.py', 'train_post.py', 'evaluate_coarse.py', 'evaluate_city.py',
               'evaluate_official_hierarchy.py', 'evaluate_local_city_hierarchy.py', 'run_small_city_hierarchy.py',
               'run_small_city_post_preflight_001.py'}
    return [p for p in snapshot if p['Name'].lower() != 'python.exe' or
            {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & scripts]


def dispatch_gate(stage, chunk):
    driver, config, manifest = fixed_sources()
    training = completed_training(driver, config, manifest)
    preconditions(stage, chunk)
    free = shutil.disk_usage(ROOT).free
    if free < 80 * 1024**3:
        raise RuntimeError('System-drive reserve below80GiB')
    snapshot = process_snapshot()
    if busy_processes(snapshot):
        raise RuntimeError('Actual trainer/Creator/post/Merger/evaluator still live; exclusive dispatch refused')
    return {'status': 'actual_source_full_training_prior_stage_disk_and_noGPU_process_gates_passed',
            'observed_utc': now(), 'stage': stage, 'chunk': chunk, 'actual_complete_training': training,
            'system_drive_free_bytes': free, 'actual_processes': snapshot, 'GPU_started': False,
            'command_argv': command_for(stage, chunk, config)}


def child_birth(child):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    creation, exit_time, kernel_time, user_time = [wintypes.FILETIME() for _ in range(4)]
    if not kernel.GetProcessTimes(wintypes.HANDLE(int(child._handle)), ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel_time), ctypes.byref(user_time)):
        raise ctypes.WinError(ctypes.get_last_error())
    ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    return {'pid': child.pid, 'creation_FILETIME_100ns': ticks, 'creation_Unix_ms': (ticks - 116444736000000000) // 10000}


def creation_ms(value):
    match = re.match(r'/Date\((-?\d+)', str(value))
    if match:
        return int(match.group(1))
    normalized = re.sub(r'(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d|$)', r'\1', str(value)).replace('Z', '+00:00')
    instant = datetime.fromisoformat(normalized).astimezone(timezone.utc)
    difference = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (difference.days * 86400 + difference.seconds) * 1000 + difference.microseconds // 1000


def final_receipt(stage, chunk):
    path, model, kind = stage_target(stage, chunk)
    if stage == 'post64':
        receipt = read(path)
        if (receipt['status'] != 'completed_post64_preflight_only' or receipt['exit_code'] != 0
                or receipt['source_kind'] != kind or receipt['eligible_for_formal_merge']
                or receipt['requested_iterations'] != 64 or receipt['optimizer_update_calls'] != 63
                or digest(model) != receipt['output_hierarchy_sha256']):
            raise RuntimeError('Actual pilot terminal receipt/sourcekind/budget/modelSHA failed')
    else:
        receipt = require_stage_receipt(path, model, kind)
        if stage == 'post' and (receipt['requested_iterations'] != 15000 or receipt['optimizer_update_calls'] != 14999):
            raise RuntimeError('Actual full post budget/update receipt failed')
    driver, config, _ = fixed_sources()
    codec_row = next(r for r in config['frozen_helper_records'] if r['name'] == 'read_hierarchy.py')
    if digest(codec_row['snapshot_path']) != codec_row['sha256']:
        raise RuntimeError('Frozen CPU hierarchy decoder changed')
    codec = load('_single_stage_final_decoder', codec_row['snapshot_path'])
    h = codec.read_hierarchy(model, expected_unowned_tail=100000 if stage in ('post', 'post64') else 0)
    if h.summary() != receipt['hierarchy']:
        raise RuntimeError('Actual terminal model storage differs from official receipt')
    return {'receipt_path': str(path), 'receipt_sha256': digest(path), 'model_path': str(model),
            'model_sha256': digest(model), 'source_kind': kind, 'actual_receipt': receipt, 'decoded_summary': h.summary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['creator', 'post64', 'post', 'merge-unoptimized', 'merge'])
    parser.add_argument('--chunk', choices=['0_0', '0_1', '1_0', '1_1'])
    parser.add_argument('--supervision-run', required=True, type=identifier)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true'); mode.add_argument('--check-dispatch', action='store_true')
    args = parser.parse_args()
    if (args.stage in ('creator', 'post', 'post64')) != (args.chunk is not None) or (args.stage == 'post64' and args.chunk != '0_0'):
        raise ValueError('Exactly select the stage chunk; post64 is00-only; Merger has no chunk')
    driver, config, manifest = fixed_sources()
    root = BASE / 'official-hierarchy-stage-supervision' / args.supervision_run
    static = {'supervisor_source_sha256': digest(__file__), 'stage': args.stage, 'chunk': args.chunk,
              'command_argv': command_for(args.stage, args.chunk, config), 'fixed_driver_sha256': DRIVER_SHA,
              'fixed_pilot_sha256': PILOT_SHA, 'formal_configuration_sha256': CONFIG_SHA,
              'input_manifest_sha256': config['input_manifest_sha256'], 'run_local_sha256': LOCAL_SHA,
              'one_stage_only': True, 'automatic_retry_or_next_stage': False, 'parent_exclusive_dispatch_required': True}
    if args.check_dispatch:
        print(json.dumps(dispatch_gate(args.stage, args.chunk), ensure_ascii=False))
        return
    if not args.execute:
        if root.exists():
            raise RuntimeError('Existing CPU supervision plan; refuse overwrite')
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'configuration.json', static)
        shutil.copyfile(__file__, root / 'supervisor-source.py')
        actual = []
        for name in config['chunk_runs'].values():
            path = BASE / 'official-training' / name / 'outcome.json'
            actual.append({'run': name, 'observed_status': read(path)['status'] if path.is_file() else 'not_created'})
        write(root / 'plan.json', {'status': 'CPU_plan_only_dispatch_not_authorized_or_verified', 'GPU_started': False,
              'actual_chunk_states': actual, 'required_future_gates': 'Actual four completed full30000+coarse receipts and model hashes, prior stage real inputs, fixed source/config/manifest,80GiB and fresh no GPU stage processes beforePopen.',
              'durability_limit': 'Record Windows job membership only; no claim of surviving app/job closure, logout or reboot.'})
        print(json.dumps({'status': 'CPU_single_stage_plan_prepared_no_GPU', 'root': str(root)}))
        return
    gate = dispatch_gate(args.stage, args.chunk)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'supervisor-outcome.json').exists() or ((root / 'configuration.json').exists() and read(root / 'configuration.json') != static):
        raise RuntimeError('Existing runtime receipt or changed CPU plan; refuse overwrite/retry')
    write(root / 'configuration.json', static)
    shutil.copyfile(__file__, root / 'supervisor-source.py')
    write(root / 'dispatch-audit.json', gate)
    old_supervisor = load('_single_stage_job_membership_only', ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py')
    receipt = {'status': 'supervisor_started_child_not_started', 'started_utc': now(), 'supervisor_pid': os.getpid(),
               'supervisor_parent_pid': os.getppid(), 'stage': args.stage, 'chunk': args.chunk,
               'command_argv': static['command_argv'], 'cwd': str(ROOT), 'source_identity': static,
               'windows_job_membership': old_supervisor.current_job_membership(),
               'CUDA_initialized_by_supervisor': False, 'retry_or_next_stage_policy': 'Never automatically restart, retry, change budget, or dispatch another stage',
               'durability_limit': 'No survival promise after app/job closure, logout or reboot.'}
    path = root / 'supervisor-outcome.json'
    write(path, receipt)
    child = None
    start = time.perf_counter()
    try:
        observer = load('_single_stage_observed_argv', ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py')
        with (root / 'stage-wrapper.log').open('wb') as log:
            child = subprocess.Popen(static['command_argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            birth = child_birth(child)
            receipt.update(status='child_started', child_pid=child.pid, actual_child_birth=birth)
            write(path, receipt)
            while True:
                code = child.poll()
                observation = {'observed_utc': now(), 'child_poll_exit_code': code, 'child_pid': child.pid}
                try:
                    snapshot = process_snapshot()
                    current = next((p for p in snapshot if p['ProcessId'] == child.pid), None)
                    if current:
                        if creation_ms(current['CreationDate']) != birth['creation_Unix_ms'] or observer.windows_argv(current['CommandLine'])[1:] != static['command_argv'][1:]:
                            observation['actual_child_identity_mismatch'] = 'Actual child PID creation/argv differs from launched process handle'
                    observation.update(actual_processes=snapshot, actual_child_process=current)
                except Exception as error:
                    # No inference of terminal from timeout/missing observation.
                    observation['process_observation_error'] = f'{type(error).__name__}: {error}'
                with (root / 'process-observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
                write(root / 'latest-process-observation.json', observation)
                receipt.update(last_observed_utc=observation['observed_utc'])
                write(path, receipt)
                if observation.get('actual_child_identity_mismatch'):
                    raise RuntimeError(observation['actual_child_identity_mismatch'])
                if code is not None:
                    break
                time.sleep(15)
        receipt.update(status='child_terminal', child_exit_code=code)
        write(path, receipt)
        if code != 0:
            raise RuntimeError('Actual stage child exited nonzero; preserve without retry')
        snapshot = process_snapshot()
        if busy_processes(snapshot):
            raise RuntimeError('GPU-stage processes still live after child exit; terminal completion unproven')
        completed = final_receipt(args.stage, args.chunk)
        receipt.update(status='completed_one_actual_stage_only', actual_final_stage=completed,
                       terminal_actual_gpu_stage_processes=[], next_stage_dispatched=False)
    except Exception as error:
        receipt.update(status='failed_or_interrupted_preserved_no_retry', error_type=type(error).__name__, error=str(error),
                       actual_child_poll_exit_code=child.poll() if child is not None else None)
    finally:
        receipt.update(ended_utc=now(), elapsed_seconds=time.perf_counter()-start)
        write(path, receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt['status'] != 'completed_one_actual_stage_only':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
