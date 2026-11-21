"""Real CPU positive and in-memory corruption controls for004 handoff gates."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts/large_scene_gs' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    target = BASE / 'official-training-pipeline-review/small-city-four-chunks-004'
    target.mkdir(parents=True, exist_ok=False)
    binding = load('binding004_cpu_controls', 'verify_small_city_stage_binding_004.py')
    process = load('exact_argv_cpu_controls', 'observe_small_city_process_roles.py')
    run = 'small-city-chunk00-i30000-r4-003'
    launch = binding.read(BASE / 'official-training-supervision-launch' / run / 'launch-receipt.json')
    sup = binding.read(BASE / 'official-training-supervision' / run / 'supervisor-outcome.json')
    config = binding.read(BASE / 'official-training' / run / 'configuration.json')
    snapshot = process.observe(run)['actual_python_processes']
    live = binding.live_binding(run, launch, sup, snapshot, process.windows_argv, config)
    coarse = binding.completed_gate('small-city-coarse-i30000-r4-003', 'coarse', None)
    results = [{'case': 'actual_live00_shim_supervisor_creation_and_descendant_chain', 'passed': True},
               {'case': 'actual_completed_coarse003_fixed005_gate_and_argv', 'passed': True}]
    def rejected(name, function):
        try:
            function()
        except (ValueError, RuntimeError) as error:
            results.append({'case': name, 'rejected': True, 'error_type': type(error).__name__, 'error': str(error)})
        else:
            raise AssertionError(f'Negative control accepted: {name}')
    changed_launch = copy.deepcopy(launch)
    changed_launch['start_process_creation_utc'] = '2026-10-12T04:27:25.5109594Z'
    rejected('launch_shim_creation_shifted10seconds', lambda: binding.live_binding(run, changed_launch, sup, snapshot, process.windows_argv, config))
    changed = copy.deepcopy(snapshot)
    next(p for p in changed if p['ProcessId'] == sup['supervisor_pid'])['CreationDate'] = '/Date(1791779245545)/'
    rejected('supervisor_reused_PID_wrong_creation', lambda: binding.live_binding(run, launch, sup, changed, process.windows_argv, config))
    changed = copy.deepcopy(snapshot)
    next(p for p in changed if p['ProcessId'] == 26484)['ParentProcessId'] = 999999
    rejected('trainer_shim_detached_from_supervisor_ancestry', lambda: binding.live_binding(run, launch, sup, changed, process.windows_argv, config))
    gate, gate_config, manifest = binding.import_fixed_gate()
    original_read = gate.read
    coarse_name = 'small-city-coarse-i30000-r4-003'
    coarse_directory = BASE / 'official-training' / coarse_name
    original_receipt, original_config = original_read(coarse_directory / 'outcome.json'), original_read(coarse_directory / 'configuration.json')
    cases = [('receipt_exit_nonzero', 'receipt', lambda d: d.update(exit_code=1)),
             ('receipt_budget64', 'receipt', lambda d: d.update(requested_iterations=64)),
             ('receipt_optimizer_updates_short', 'receipt', lambda d: d.update(optimizer_update_calls=29998)),
             ('receipt_image_updates_short', 'receipt', lambda d: d.update(training_image_updates=29998)),
             ('receipt_finalpath_iteration64', 'receipt', lambda d: d.update(saved_model_directory=str(coarse_directory / 'model/point_cloud/iteration_64'))),
             ('configuration_source_changed_true', 'config', lambda d: d.update(source_changed=True)),
             ('configuration_trainer_SHA_wrong', 'config', lambda d: d.update(trainer_source_sha256='0'*64)),
             ('configuration_actual_CLI_budget64', 'config', lambda d: d['effective_official_arguments'].update(iterations=64))]
    for name, kind, mutate in cases:
        receipt_data, config_data = copy.deepcopy(original_receipt), copy.deepcopy(original_config)
        mutate(receipt_data if kind == 'receipt' else config_data)
        def fake_read(path):
            resolved = Path(path).resolve()
            if resolved == (coarse_directory / 'outcome.json').resolve():
                return receipt_data
            if resolved == (coarse_directory / 'configuration.json').resolve():
                return config_data
            return original_read(path)
        gate.read = fake_read
        rejected(name, lambda: gate.completed_training(coarse_name, 'coarse', None, gate_config, manifest, None))
    gate.read = original_read
    if 'torch' in sys.modules:
        raise AssertionError('CPU controls imported torch')
    sources = {name: binding.sha(ROOT / 'scripts/large_scene_gs' / name) for name in
               ('run_small_city_full_pipeline_004.py', 'audit_small_city_full_chunk_004.py',
                'verify_small_city_stage_binding_004.py', 'launch_small_city_full_pipeline_004.ps1')}
    report = {'status': 'CPU_actual_live_binding_and_fixed005_completion_controls_passed',
              'observed_utc': datetime.now(timezone.utc).isoformat(), 'CUDA_initialized': False, 'torch_imported': False,
              'source_sha256': sources, 'live00_binding': live, 'real_coarse003_completion_binding': coarse,
              'controls': results, 'negative_scope': 'Only in-memory copies of receipt/config/process snapshots corrupted; authoritative files and live processes unchanged.',
              'GPU_restart_or_termination_by_verifier': False}
    (target / 'CPU-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(__file__, target / 'source.py')
    print(json.dumps({'report': str(target / 'CPU-verification.json'), 'controls': len(results), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
