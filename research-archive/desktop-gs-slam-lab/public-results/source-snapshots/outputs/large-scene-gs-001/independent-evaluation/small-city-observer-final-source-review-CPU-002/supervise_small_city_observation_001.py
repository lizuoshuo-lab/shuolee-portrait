"""CPU durable supervisor for exactly one authorized observation-only full-budget run."""
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
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'small-city-chunk11-observation-i30000-r4-001'
PLAN = BASE / 'official-training-observation-plans' / f'{RUN}-v3'
OUT = BASE / 'official-training-observation' / RUN
SUP = BASE / 'official-training-observation-supervision' / RUN
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'
LOCAL = ROOT / 'scripts/large_scene_gs/run_local.py'
STAGE = ROOT / 'scripts/large_scene_gs/run_small_city_observation_001.py'
OBSERVER = ROOT / 'scripts/large_scene_gs/observe_small_city_train_single_001.py'
PW = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
PW_ENV = os.environ.copy()
PW_ENV['PSModulePath'] = os.pathsep.join([str(PW.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])
SOURCE_HASHES = {
    'run_local.py': '1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2',
    'run_small_city_observation_001.py': 'ebe2581d8ae8bee8fa48519c383030a0e74be0677c38b81855715dee6a16570c',
    'observe_small_city_train_single_001.py': '59bc69b80fb50283cddd5a35c115fb3ecd9ba37bef675c74e6f1736cc02af384',
    'observe_small_city_process_roles.py': 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a',
    'verify_small_city_stage_binding_004.py': '8da91ef734ca6579d94d0016b2863816a2bd7eedb752eaaafba8d8f626a59e90',
    'supervise_small_city_stage.py': 'b7c0015374521b201a4e28845cb9ff79ca9b4c597d5c05f80bf69aa03f58378d',
}
GPU_SCRIPTS = {'train_coarse.py', 'train_single.py', 'train_post.py', 'evaluate_coarse.py', 'evaluate_city.py',
               'evaluate_official_hierarchy.py', 'evaluate_local_city_hierarchy.py', 'run_small_city_hierarchy.py',
               'run_small_city_post_preflight_001.py', 'observe_small_city_train_single_001.py', 'run_small_city_observation_001.py'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load(filename):
    path = ROOT / 'scripts/large_scene_gs' / filename
    if sha(path) != SOURCE_HASHES[filename]:
        raise RuntimeError('Frozen supervisor/helper source changed')
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_gate():
    for name, expected in SOURCE_HASHES.items():
        if sha(ROOT / 'scripts/large_scene_gs' / name) != expected:
            raise RuntimeError(f'Frozen observation source changed: {name}')
    plan = read(PLAN / 'configuration.json')
    if (plan['requested_iterations'] != 30000 or plan['expected_optimizer_update_calls'] != 29999
            or plan['resolution_pixels'] != [256, 172] or plan['exposure_mode'] != 'official'
            or plan['chunk'] != '1_1' or plan['coarse_run'] != 'small-city-coarse-i30000-r4-003'
            or plan['formal_training_run'] is not False or plan['seed'] != 0
            or plan['observer_source_sha256'] != SOURCE_HASHES[OBSERVER.name]
            or plan['launcher_source_sha256'] != SOURCE_HASHES[STAGE.name]):
        raise RuntimeError('Frozen full-budget observation plan differs')
    for item in plan['official_source_files']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Pinned actual official training/Adam source changed')
    manifest = BASE / 'official-inputs/small-city-training-002/manifest.json'
    if sha(manifest) != plan['input_manifest_sha256']:
        raise RuntimeError('Frozen observation input manifest changed')
    if sha(PLAN / 'launcher-source.py') != SOURCE_HASHES[STAGE.name] or sha(PLAN / 'environment-launcher-source.py') != SOURCE_HASHES[LOCAL.name]:
        raise RuntimeError('Frozen CPU-plan source snapshots changed')
    if (sha(PLAN / 'observer-source.py') != SOURCE_HASHES[OBSERVER.name] or
            sha(PLAN / 'initial-raw-oracle.json') != '3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236' or
            sha(plan['initial_raw_oracle']) != plan['initial_raw_oracle_sha256']):
        raise RuntimeError('Frozen actual initialized-row oracle/observer snapshot changed')
    return plan


def processes():
    command = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='GaussianHierarchyCreator.exe' OR Name='GaussianHierarchyMerger.exe' OR Name='GaussianHierarchyMergerUnoptimized.exe'\" | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run([str(PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command], env=PW_ENV,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError('Actual process snapshot failed')
    data = json.loads(result.stdout or '[]')
    return data if isinstance(data, list) else [data]


def busy(snapshot):
    observer = load('observe_small_city_process_roles.py')
    return [p for p in snapshot if p['Name'].lower() != 'python.exe' or
            {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & GPU_SCRIPTS]


def birth(child):
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
    delta = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1000 + delta.microseconds // 1000


def dispatch_gate():
    plan = source_gate()
    gate = load('verify_small_city_stage_binding_004.py').completed_gate('small-city-coarse-i30000-r4-003', 'coarse', None)
    if gate['stage_outcome_sha256'] != plan['coarse_outcome_sha256']:
        raise RuntimeError('Actual complete coarse binding changed')
    if shutil.disk_usage(ROOT).free < 80 * 1024**3:
        raise RuntimeError('80GiB reserve guard rejected dispatch')
    snapshot = processes()
    if busy(snapshot):
        raise RuntimeError('An actual Python/native GPU stage exists; do not dispatch')
    if OUT.exists() or SUP.exists():
        raise RuntimeError('Existing observation run or supervision; no overwrite/retry')
    return {'status': 'CPU_full_observation_source_coarse_input_disk_and_actual_noGPU_dispatch_gates_passed',
            'observed_utc': utc(), 'actual_processes': snapshot, 'actual_complete_coarse': gate,
            'plan_sha256': sha(PLAN / 'configuration.json'), 'system_drive_free_bytes': shutil.disk_usage(ROOT).free,
            'GPU_execution_started': False, 'CUDA_initialized': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-dispatch', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        result = dispatch_gate() if args.check_dispatch else {'status': 'CPU_supervisor_contract_only', 'GPU_execution_started': False, 'CUDA_initialized': False}
        print(json.dumps(result, ensure_ascii=False))
        return
    initial = dispatch_gate()
    helper = load('supervise_small_city_stage.py')
    argv_helper = load('observe_small_city_process_roles.py')
    command = [str(PYTHON), '-u', str(LOCAL), str(STAGE), '--stage', 'chunk', '--run', RUN, '--input', 'small-city-training-002',
               '--chunk', '1_1', '--coarse-run', 'small-city-coarse-i30000-r4-003', '--iterations', '30000',
               '--resolution', '4', '--exposure', 'official', '--port', '6024', '--execute']
    SUP.mkdir(parents=True)
    for name in SOURCE_HASHES:
        shutil.copyfile(ROOT / 'scripts/large_scene_gs' / name, SUP / name)
    shutil.copyfile(Path(__file__), SUP / 'supervisor-source.py')
    receipt = {'status': 'supervisor_started_child_not_started', 'started_utc': utc(), 'supervisor_pid': os.getpid(),
               'supervisor_parent_pid': os.getppid(), 'command_argv': command, 'source_sha256': sha(Path(__file__)),
               'source_contract': SOURCE_HASHES, 'CPU_prelaunch_gates': initial, 'Windows_job_membership': helper.current_job_membership(),
               'retry_policy': 'No retry, next-stage launch, repair, source update or process termination',
               'diagnostic_observation_only': True, 'formal_completed_model_claimed': False, 'CUDA_initialized_by_supervisor': False}
    save(SUP / 'supervisor-outcome.json', receipt)
    child = None
    try:
        with (SUP / 'stage-wrapper.log').open('wb') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            actual_birth = birth(child)
            receipt.update(status='child_started', actual_child_birth=actual_birth, child_pid=child.pid)
            save(SUP / 'supervisor-outcome.json', receipt)
            while True:
                code = child.poll()
                observation = {'observed_utc': utc(), 'child_poll_exit_code': code,
                               'progress': helper.tail_progress(OUT / 'training.log')}
                source_gate()
                if sha(Path(__file__)) != receipt['source_sha256'] or sha(PLAN / 'configuration.json') != initial['plan_sha256']:
                    raise RuntimeError('Frozen supervisor source or exact prepared plan changed during observation')
                try:
                    snapshot = processes()
                    relevant = {os.getpid(), os.getppid(), child.pid}
                    while True:
                        extra = {p['ProcessId'] for p in snapshot if p['ParentProcessId'] in relevant}
                        if extra <= relevant:
                            break
                        relevant |= extra
                    observation['actual_descendant_processes'] = [p for p in snapshot if p['ProcessId'] in relevant]
                    current = next((p for p in snapshot if p['ProcessId'] == child.pid), None)
                    if current and (creation_ms(current['CreationDate']) != actual_birth['creation_Unix_ms'] or
                                    argv_helper.windows_argv(current['CommandLine'])[1:] != command[1:]):
                        observation['actual_child_identity_mismatch'] = 'Actual child PID birth/argv no longer matches launched handle'
                except Exception as error:
                    # A query failure never dispatches or retries another process.
                    observation['process_observation_error'] = f'{type(error).__name__}: {error}'
                with (SUP / 'process-observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
                save(SUP / 'latest-process-observation.json', observation)
                if observation.get('actual_child_identity_mismatch'):
                    raise RuntimeError(observation['actual_child_identity_mismatch'])
                if code is not None:
                    receipt['actual_child_poll_exit_code'] = code
                    break
                time.sleep(15)
        terminal = processes()
        if busy(terminal):
            raise RuntimeError('An actual GPU child/native stage still exists after child exit; do not report release')
        stage = read(OUT / 'outcome.json')
        observed = read(OUT / 'observer/observer-outcome.json')
        if code == 86 and stage['status'] == 'diagnostic_stopped_first_observed_nonfinite_no_repair' and observed['status'] == 'diagnostic_stopped_at_first_observed_nonfinite':
            receipt['status'] = 'completed_diagnostic_nonfinite_observation_stop_not_full_training'
        elif (code == 0 and stage['status'] == 'completed_observation_only_final_finite_not_formal'
                and observed['actual_original_exposure_Adam_step_calls'] == 29999
                and observed['actual_original_opacity_grad_condition_visits'] == 29999
                and observed['actual_original_Gaussian_Adam_step_calls'] + observed['actual_original_opacity_grad_None_skips'] == 29999
                and 30000 in observed['successful_original_save_iterations']):
            receipt['status'] = 'completed_fullbudget_observation_only_not_formal'
        else:
            raise RuntimeError('Actual diagnostic child/stage/observer terminal receipts disagree')
        receipt.update(actual_stage_outcome_sha256=sha(OUT / 'outcome.json'), actual_observer_outcome_sha256=sha(OUT / 'observer/observer-outcome.json'),
                       actual_stage_outcome=stage, actual_terminal_GPU_stage_processes=[], actual_terminal_process_snapshot=terminal,
                       GPU_training_owner_released=True, Creator_post_Merger_or_LOD_complete=False)
    except Exception as error:
        receipt.update(status='supervisor_or_diagnostic_failure_preserved_no_retry', error_type=type(error).__name__, error=str(error),
                       actual_child_poll_exit_code=child.poll() if child is not None else None)
    finally:
        receipt['completed_utc'] = utc()
        save(SUP / 'supervisor-outcome.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt['status'] not in {'completed_diagnostic_nonfinite_observation_stop_not_full_training', 'completed_fullbudget_observation_only_not_formal'}:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
