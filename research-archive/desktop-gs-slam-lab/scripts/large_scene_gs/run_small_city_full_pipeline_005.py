"""CPU-only serial controller of the authorized four unchanged full GPU stages.

Adopts the already launched 00 without restarting it, then calls the dedicated
Hidden Start-Process launcher for 01,10,11 only after real completion, absence
of active trainer processes, and independent CPU saved-model audit. Stops on
failure and never retries, lowers budgets, or advances to Creator/post/Merger.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
NAMES = {'0_0': 'small-city-chunk00-i30000-r4-003', '0_1': 'small-city-chunk01-i30000-r4-003',
         '1_0': 'small-city-chunk10-i30000-r4-003', '1_1': 'small-city-chunk11-i30000-r4-003'}
SOURCE_HASHES = {'supervise_small_city_stage.py': 'b7c0015374521b201a4e28845cb9ff79ca9b4c597d5c05f80bf69aa03f58378d',
                 'run_small_city_stage.py': 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704'}
POWERSHELL = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
CHILD_POWERSHELL_ENV = os.environ.copy()
CHILD_POWERSHELL_ENV['PSModulePath'] = os.pathsep.join([str(POWERSHELL.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])
GPU_SCRIPTS = {'train_single.py', 'train_coarse.py', 'train_post.py', 'evaluate_city.py',
               'evaluate_coarse.py', 'evaluate_official_hierarchy.py'}


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


def save(path, obj):
    target = Path(path)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, target)


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/large_scene_gs' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, errors='replace', timeout=20,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        return {'query_succeeded': False, 'exit_code': result.returncode, 'stderr': result.stderr}
    rows = list(csv.reader(result.stdout.splitlines()))
    values = []
    for row in rows:
        if len(row) != 6:
            return {'query_succeeded': False, 'unparsed_stdout': result.stdout}
        values.append({'name': row[0].strip(), 'uuid': row[1].strip(), 'driver': row[2].strip(),
                       'total_MiB': int(row[3].strip()), 'used_MiB': int(row[4].strip()), 'free_MiB': int(row[5].strip())})
    return {'query_succeeded': True, 'gpus': values,
            'scope': 'Whole-device snapshot including desktop/app/native allocations. Sampled every15s; neither torch allocated peak nor native instantaneous peak.'}


def source_contract(snapshot):
    for filename, expected in SOURCE_HASHES.items():
        if sha(ROOT / 'scripts/large_scene_gs' / filename) != expected:
            raise ValueError(f'Frozen stage/supervisor source changed: {filename}')
    for filename, expected in snapshot.items():
        if sha(ROOT / 'scripts/large_scene_gs' / filename) != expected:
            raise ValueError(f'Pipeline source dependency changed: {filename}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adopt-first-existing', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'CPU_plan_only', 'CUDA_initialized': False, 'sequence': NAMES,
                          'actual_iterations_each': 30000, 'coarse_dependency': 'small-city-coarse-i30000-r4-003'}))
        return
    if not args.adopt_first_existing:
        raise ValueError('This recovery pipeline must adopt the already launched official00')
    directory = BASE / 'official-training-pipeline/small-city-four-chunks-005'
    directory.mkdir(parents=True, exist_ok=False)
    source_files = ['run_small_city_full_pipeline_005.py', 'launch_small_city_full_chunk_003.ps1',
                    'audit_small_city_full_chunk_004.py', 'observe_small_city_process_roles.py',
                    'run_local.py', 'verify_small_city_stage_binding_004.py', *SOURCE_HASHES]
    snapshots = {filename: sha(ROOT / 'scripts/large_scene_gs' / filename) for filename in source_files}
    for filename in source_files:
        shutil.copyfile(ROOT / 'scripts/large_scene_gs' / filename, directory / filename)
    process_module = load_module('exact_windows_process_observer', 'observe_small_city_process_roles.py')
    supervisor_module = load_module('unchanged_supervisor', 'supervise_small_city_stage.py')
    binding_module = load_module('new004_provenance_bindings', 'verify_small_city_stage_binding_004.py')
    input_path = BASE / 'official-inputs/small-city-training-002/manifest.json'
    input_sha = sha(input_path)
    coarse_dir = BASE / 'official-training/small-city-coarse-i30000-r4-003'
    coarse_sha = sha(coarse_dir / 'outcome.json')
    state = {'status': 'running', 'started_utc': utc(), 'controller_pid': os.getpid(), 'controller_parent_pid': os.getppid(),
             'windows_job_membership': supervisor_module.current_job_membership(), 'sequence': NAMES,
             'requested_iterations_each': 30000, 'expected_optimizer_update_calls_each': 29999,
             'resolution': 4, 'exposure': 'official', 'input_manifest_sha256': input_sha,
             'actual_coarse_run': coarse_dir.name, 'actual_coarse_outcome_sha256': coarse_sha,
             'source_snapshots_sha256': snapshots, 'official_algorithm_modified': False, 'retry_policy': 'No automatic retry or protocol change',
             'completed00_preserved_without_restart': True, 'existing01_adopted_without_restart': True, 'completed_chunks': [], 'current_chunk': None,
             'CUDA_initialized_by_controller': False, 'Creator_post_Merger_authorized_by_controller': False}
    save(directory / 'state.json', state)
    start = time.perf_counter()
    try:
        for index, (chunk, run) in enumerate(NAMES.items()):
            source_contract(snapshots)
            if sha(input_path) != input_sha or sha(coarse_dir / 'outcome.json') != coarse_sha:
                raise ValueError('Frozen input/completed coarse changed during pipeline')
            if shutil.disk_usage(ROOT).free < 80 * 1024**3:
                raise ValueError('System-drive reserve below80GiB')
            state.update(current_chunk=chunk, current_run=run, phase='adopt_existing00' if index == 0 else 'launch_next_after_previous_full_audit')
            save(directory / 'state.json', state)
            if index == 0:
                provenance = binding_module.completed_gate(run, 'chunk', chunk)
                audit_path = BASE / 'recovery-audit' / f'{run}-completion-004' / 'completion-audit.json'
                if sha(audit_path) != 'f108f64b88e8e9a136c86e9d110fc0f274a3cfe5681b30988a42c4f9f091b183':
                    raise ValueError('Previously completed00 physical audit bytes changed')
                audit = read(audit_path)
                if audit['model_parameter_sha256'] != provenance['model_parameter_sha256'] or not all(g['float32_bitexact'] for g in audit['shared_sky_all_parameter_groups_bitexact'].values()):
                    raise ValueError('Completed00 physical/source provenance differs')
                save(directory / 'preserved-completed00-fresh005-gate.json', provenance)
                state['completed_chunks'].append({'chunk': chunk, 'run': run, 'audit': str(audit_path), 'audit_sha256': sha(audit_path),
                    'model_sha256': audit['model_parameter_sha256'], 'G': audit['actual_saved_gaussians'],
                    'logged_torch_peak_allocated_bytes': audit['logged_torch_peak_allocated_bytes'], 'existing_completed_model_preserved': True})
                save(directory / 'state.json', state)
                continue
            if index > 1:
                snapshot = process_module.observe(run)
                busy = [p for p in snapshot['actual_python_processes'] if
                        {Path(a).name.lower() for a in process_module.windows_argv(p.get('CommandLine') or '')[1:]} & GPU_SCRIPTS]
                if busy:
                    raise ValueError('Another exact-argv GPU trainer/evaluator is live; do not launch next stage')
                launch_log = directory / f'launch-{chunk}.log'
                with launch_log.open('wb') as log:
                    code = subprocess.call([str(POWERSHELL), '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                        '-File', str(ROOT / 'scripts/large_scene_gs/launch_small_city_full_chunk_003.ps1'), '-Chunk', chunk],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, env=CHILD_POWERSHELL_ENV)
                if code != 0:
                    raise RuntimeError(f'Hidden stage launcher actual exit{code}; see {launch_log}')
            launch_path = BASE / 'official-training-supervision-launch' / run / 'launch-receipt.json'
            if not launch_path.is_file():
                raise ValueError('Actual adopted/started stage launch receipt absent')
            launch = read(launch_path)
            if (launch['run'] != run or launch['chunk'] != chunk or launch['requested_iterations'] != 30000
                or launch['input_manifest_sha256'] != input_sha or launch['actual_coarse_outcome_sha256'] != coarse_sha
                or launch['supervisor_source_sha256'] != SOURCE_HASHES['supervise_small_city_stage.py']
                or launch['short_preflight_model_used_for_initialization']):
                raise ValueError('Actual hidden launch contract differs from authorized full stage')
            expected_flags = {'--stage': 'chunk', '--run': run, '--input': 'small-city-training-002',
                              '--chunk': chunk, '--coarse-run': coarse_dir.name, '--iterations': '30000',
                              '--resolution': '4', '--exposure': 'official'}
            launch_argv = launch['argument_list']
            for flag, value in expected_flags.items():
                if launch_argv.count(flag) != 1 or launch_argv[launch_argv.index(flag) + 1] != value:
                    raise ValueError(f'Hidden launch actual argv differs: {flag}')
            if index == 1:
                adopted_sup_path = BASE / 'official-training-supervision' / run / 'supervisor-outcome.json'
                adopted_sup = read(adopted_sup_path)
                adopted_snapshot = process_module.observe(run)
                adopted_config = read(BASE / 'official-training' / run / 'configuration.json')
                if adopted_sup['status'] == 'completed':
                    adoption = binding_module.completed_gate(run, 'chunk', chunk)
                    adoption['adoption_scope'] = 'Already completed01 actual final receipt; no restart'
                else:
                    adoption = binding_module.live_binding(run, launch, adopted_sup, adopted_snapshot['actual_python_processes'], process_module.windows_argv, adopted_config)
                save(directory / 'adopted01-strong-binding-005.json', adoption)
            missing_supervisor_observations = 0
            while True:
                source_contract(snapshots)
                observation = {'observed_utc': utc(), 'chunk': chunk, 'run': run,
                               'progress': supervisor_module.tail_progress(BASE / 'official-training' / run / 'training.log'),
                               'GPU_memory': gpu_snapshot(), 'system_drive_free_bytes': shutil.disk_usage(ROOT).free}
                sup_path = BASE / 'official-training-supervision' / run / 'supervisor-outcome.json'
                sup = read(sup_path) if sup_path.is_file() else None
                config_path = BASE / 'official-training' / run / 'configuration.json'
                if config_path.is_file():
                    config = read(config_path)
                    if (config['stage'] != 'chunk' or config['chunk'] != chunk or config['coarse_run'] != coarse_dir.name
                        or config['requested_iterations'] != 30000 or config['resolution_pixels'] != [256, 172]
                        or config['exposure_mode'] != 'official' or config['input_manifest_sha256'] != input_sha
                        or config['coarse_outcome_sha256'] != coarse_sha
                        or config['launcher_source_sha256'] != SOURCE_HASHES['run_small_city_stage.py']):
                        raise ValueError('Actual stage configuration/source/coarse binding differs from authorized full run')
                    observation['actual_stage_configuration_sha256'] = sha(config_path)
                try:
                    snapshot = process_module.observe(run)
                    names = lambda p: {Path(a).name.lower() for a in process_module.windows_argv(p.get('CommandLine') or '')[1:]}
                    trainer = [p for p in snapshot['actual_python_processes'] if 'train_single.py' in names(p)]
                    scope = [p for p in snapshot['actual_python_processes'] if p['ProcessId'] in
                             {launch['start_process_pid'], sup.get('supervisor_pid') if sup else None} or
                             (names(p) & GPU_SCRIPTS) or (run in process_module.windows_argv(p.get('CommandLine') or '') and
                             names(p) & {'run_small_city_stage.py', 'supervise_small_city_stage.py'})]
                    observation.update(actual_relevant_processes=scope, trainer_pids=[p['ProcessId'] for p in trainer])
                    live_sup = any(p['ProcessId'] in {launch['start_process_pid'], sup.get('supervisor_pid') if sup else None}
                                   for p in snapshot['actual_python_processes'])
                    sup = read(sup_path) if sup_path.is_file() else None
                    observation['supervisor_status'] = sup['status'] if sup else 'receipt_not_written_yet'
                    if sup and sup['status'] == 'completed' and not trainer and not snapshot['related_training_processes']:
                        observation['completed_supervisor_and_no_live_trainer'] = True
                    elif sup and sup['status'] in ('failed', 'supervisor_failed'):
                        observation['terminal_failure'] = sup
                    elif not live_sup:
                        missing_supervisor_observations += 1
                        observation['absent_supervisor_consecutive_authoritative_observations'] = missing_supervisor_observations
                    else:
                        missing_supervisor_observations = 0
                except Exception as error:
                    # An observation failure is not terminal. Do not restart or advance.
                    observation['process_observation_error'] = f'{type(error).__name__}: {error}'
                with (directory / 'observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
                save(directory / 'latest-observation.json', observation)
                state.update(last_observed_utc=observation['observed_utc'], phase='waiting_actual_stage',
                             last_logged_iterations=observation['progress'].get('last_logged_iterations'),
                             actual_trainer_pids=observation.get('trainer_pids', []))
                save(directory / 'state.json', state)
                if observation.get('terminal_failure'):
                    raise RuntimeError(f'Actual stage supervisor failed: {observation["terminal_failure"]}')
                if missing_supervisor_observations >= 2:
                    raise RuntimeError('Supervisor absent in two actual process snapshots without completed receipt; preserve interrupted state, no retry')
                if observation.get('completed_supervisor_and_no_live_trainer'):
                    break
                time.sleep(15)
            audit_log = directory / f'completed-audit-{chunk}.log'
            state.update(phase='independent_CPU_saved_model_audit')
            save(directory / 'state.json', state)
            with audit_log.open('wb') as log:
                code = subprocess.call([str(ROOT / '.venv-large-gs/Scripts/python.exe'), '-X', 'utf8',
                    str(ROOT / 'scripts/large_scene_gs/audit_small_city_full_chunk_004.py'), '--chunk', chunk],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            if code != 0:
                raise RuntimeError(f'Completed full chunk CPU audit actual exit{code}; see {audit_log}')
            audit_path = BASE / 'recovery-audit' / f'{run}-completion-004' / 'completion-audit.json'
            audit = read(audit_path)
            state['completed_chunks'].append({'chunk': chunk, 'run': run, 'audit': str(audit_path), 'audit_sha256': sha(audit_path),
                                              'model_sha256': audit['model_parameter_sha256'], 'G': audit['actual_saved_gaussians'],
                                              'logged_torch_peak_allocated_bytes': audit['logged_torch_peak_allocated_bytes']})
            save(directory / 'state.json', state)
            print(json.dumps({'event': 'actual_full_chunk_completed_audited', **state['completed_chunks'][-1]}), flush=True)
        state.update(status='completed_four_full_chunks_only', current_chunk=None, current_run=None,
                     phase='stop_before_Creator_post_Merger_until_parent_authorizes', all_four_full_training_complete=True,
                     Creator_post_Merger_or_LOD_complete=False)
    except Exception as error:
        state.update(status='failed_or_interrupted_preserved_no_retry', error_type=type(error).__name__, error=str(error))
    finally:
        state.update(completed_utc=utc(), elapsed_seconds=time.perf_counter() - start)
        save(directory / 'state.json', state)
        print(json.dumps(state, ensure_ascii=False), flush=True)
    if state['status'] != 'completed_four_full_chunks_only':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
