"""CPU-only final binding and terminal summary; never starts or stops training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
NAMES = {
    '0_0': 'small-city-chunk00-i30000-r4-003',
    '0_1': 'small-city-chunk01-i30000-r4-003',
    '1_0': 'small-city-chunk10-i30000-r4-003',
    '1_1': 'small-city-chunk11-i30000-r4-003',
}
BINDING_SHA = '8da91ef734ca6579d94d0016b2863816a2bd7eedb752eaaafba8d8f626a59e90'
OBSERVER_SHA = 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a'
AUDIT_SHA = 'e9a4e197e84d1bb845349d60d5fd23f73376283ff0984cc1f2a922c6423fc37c'
PIPELINE_SHA = '7161a47c55cc659302b2da68318f006f30471cca5fed80ad6990a1644cce8a9f'
MANIFEST_SHA = '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
COARSE_SHA = 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31'
PHYSICAL_STATUS = 'completed_actual_full30000_chunk_model_sky_source_and_terminal_CPU_audit_passed'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(filename, expected):
    path = ROOT / 'scripts/large_scene_gs' / filename
    require(sha(path) == expected, f'Frozen CPU helper changed: {filename}')
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect():
    state_path = BASE / 'official-training-pipeline/small-city-four-chunks-005/state.json'
    initial_state_sha = sha(state_path)
    state = read(state_path)
    require(state['status'] == 'completed_four_full_chunks_only', 'Four-stage controller is not actually completed')
    require(state['all_four_full_training_complete'] is True, 'Four completed stages absent')
    require(state['current_chunk'] is None and state['current_run'] is None, 'Current stage still selected')
    require(state['Creator_post_Merger_or_LOD_complete'] is False, 'Scope changed beyond chunk training')
    require(state['sequence'] == NAMES and len(state['completed_chunks']) == 4, 'Formal sequence changed')
    require(state['input_manifest_sha256'] == MANIFEST_SHA, 'Input protocol changed')
    require(state['actual_coarse_run'] == 'small-city-coarse-i30000-r4-003', 'Coarse binding changed')
    pipeline_path = ROOT / 'scripts/large_scene_gs/run_small_city_full_pipeline_005.py'
    require(sha(pipeline_path) == PIPELINE_SHA, 'Controller source changed')
    for filename, expected in state['source_snapshots_sha256'].items():
        require(sha(ROOT / 'scripts/large_scene_gs' / filename) == expected, f'Current frozen source changed: {filename}')
        require(sha(state_path.parent / filename) == expected, f'Archived frozen source changed: {filename}')
    binding = load('verify_small_city_stage_binding_004.py', BINDING_SHA)
    observer = load('observe_small_city_process_roles.py', OBSERVER_SHA)
    snapshot = observer.observe(NAMES['1_1'])
    active_scripts = {'train_coarse.py', 'train_single.py', 'train_post.py', 'run_small_city_stage.py',
                      'supervise_small_city_stage.py', 'run_small_city_full_pipeline_005.py'}
    active = [p for p in snapshot['actual_python_processes'] if
              {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & active_scripts]
    require(not active, f'Actual trainer, wrapper, supervisor or controller still exists: {active}')
    all_pids = {p['ProcessId'] for p in snapshot['actual_python_processes']}
    require(state['controller_pid'] not in all_pids and state['controller_parent_pid'] not in all_pids,
            'Original controller or venv shim still exists')
    rows = []
    terminal_ids = []
    for (chunk, run), recorded in zip(NAMES.items(), state['completed_chunks']):
        require(recorded['chunk'] == chunk and recorded['run'] == run, 'Completed stage order changed')
        audit_path = BASE / 'recovery-audit' / f'{run}-completion-004/completion-audit.json'
        require(str(audit_path) == recorded['audit'] and sha(audit_path) == recorded['audit_sha256'],
                'Immutable per-stage physical audit changed')
        audit = read(audit_path)
        require(audit['status'] == PHYSICAL_STATUS and audit['script_sha256'] == AUDIT_SHA, 'Physical audit is not passed')
        require(audit['input_manifest_sha256'] == MANIFEST_SHA and audit['coarse_model_sha256'] == COARSE_SHA,
                'Physical audit does not share actual input/coarse')
        require(audit['requested_iterations'] == 30000 and audit['optimizer_update_calls'] == 29999, 'Formal budget not met')
        require(all(audit['all_raw_arrays_finite'].values()), 'Raw saved arrays are not finite')
        require(audit['activated_scale_min_max_scene_units'][0] > 0 and
                audit['float32_quaternion_norm_min_max'][0] > 0, 'Invalid scale/quaternion')
        require(len(audit['shared_sky_all_parameter_groups_bitexact']) == 6 and
                all(group['float32_bitexact'] and group['actual_sha256'] == group['expected_sha256']
                    for group in audit['shared_sky_all_parameter_groups_bitexact'].values()), 'Shared sky changed')
        require(audit['actual_SH2_nonzero_scalar_coefficients'] > 0 and
                audit['actual_SH3_nonzero_scalar_coefficients'] > 0, 'Actual higher-degree SH learning evidence absent')
        require(audit['frozen_test_exposure_names'] == 0 and
                audit['actual_initial_gaussians'] == audit['predicted_initial_gaussians'], 'Initialization/exposure protocol changed')
        fresh_gate = binding.completed_gate(run, 'chunk', chunk)
        require(fresh_gate['model_parameter_sha256'] == recorded['model_sha256'] == audit['model_parameter_sha256'],
                'Current final parameter bytes differ from completion audit')
        require(fresh_gate['stage_outcome_sha256'] == audit['stage_outcome_sha256'] and
                fresh_gate['source_config_sha256'] == audit['stage_configuration_sha256'], 'Stage/config receipt changed')
        sup_path = BASE / 'official-training-supervision' / run / 'supervisor-outcome.json'
        sup = read(sup_path)
        require(sup['status'] == 'completed' and sup['child_exit_code'] == 0 and
                sup['stage_outcome_status'] == 'completed', 'Supervisor did not reach real completed exit0')
        require(sup['stage_outcome_sha256'] == audit['stage_outcome_sha256'], 'Supervisor child receipt differs')
        ids = [sup['supervisor_pid'], sup['supervisor_parent_pid'], sup['child_pid']]
        require(not (set(ids) & all_pids), 'Actual old supervisor/shim/child PID still exists')
        terminal_ids.extend(ids)
        rows.append({
            'chunk': chunk, 'run': run, 'requested_iterations': 30000, 'derived_optimizer_updates': 29999,
            'actual_initial_gaussians': audit['actual_initial_gaussians'],
            'actual_saved_gaussians': audit['actual_saved_gaussians'],
            'elapsed_seconds': audit['elapsed_seconds'],
            'logged_torch_peak_allocated_bytes': audit['logged_torch_peak_allocated_bytes'],
            'memory_scope': audit['memory_scope'], 'model_parameter_sha256': audit['model_parameter_sha256'],
            'physical_audit': str(audit_path), 'physical_audit_sha256': sha(audit_path),
            'stage_outcome_sha256': audit['stage_outcome_sha256'],
            'supervisor_outcome': str(sup_path), 'supervisor_outcome_sha256': sha(sup_path),
            'shared_sky_all_six_parameter_groups_bitexact': True,
            'actual_SH2_nonzero_scalar_coefficients': audit['actual_SH2_nonzero_scalar_coefficients'],
            'actual_SH3_nonzero_scalar_coefficients': audit['actual_SH3_nonzero_scalar_coefficients'],
            'fresh_fixed005_gate': fresh_gate,
        })
    # Recheck after hashing/model/source gates. A snapshot before CPU work alone is insufficient for handoff.
    terminal_snapshot = observer.observe(NAMES['1_1'])
    active_after = [p for p in terminal_snapshot['actual_python_processes'] if
                   {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & active_scripts]
    require(not active_after, 'A trainer/controller appeared during final CPU verification')
    require(not (set(terminal_ids + [state['controller_pid'], state['controller_parent_pid']]) &
                 {p['ProcessId'] for p in terminal_snapshot['actual_python_processes']}), 'Original pipeline PID exists in final snapshot')
    require(sha(state_path) == initial_state_sha, 'Controller final state changed during CPU verification')
    return {
        'status': 'four_formal_30000_chunk_models_sources_shared_sky_and_actual_terminal_verified_CPU',
        'observed_utc': utc(), 'source_sha256': sha(Path(__file__)), 'CPU_auditor_pid': os.getpid(),
        'scope': 'Four full official chunk training stages only. No merged unique scene count or hierarchy/LOD rendering claim.',
        'input_manifest_sha256': MANIFEST_SHA, 'actual_coarse_run': 'small-city-coarse-i30000-r4-003',
        'actual_coarse_model_sha256': COARSE_SHA,
        'pipeline_state': str(state_path), 'pipeline_state_sha256': sha(state_path),
        'pipeline_source_sha256': PIPELINE_SHA, 'chunks': rows,
        'total_derived_optimizer_updates': sum(r['derived_optimizer_updates'] for r in rows),
        'derived_count_scope': 'Derived from unchanged official loop/control flow and actual exit0/final model; not instrumented.',
        'sum_of_actual_training_elapsed_seconds': sum(r['elapsed_seconds'] for r in rows),
        'gaussian_count_scope': 'Per-chunk counts overlap and each includes a shared 100000 sky prefix; summation is not a unique merged count.',
        'actual_active_training_or_controller_processes': active_after,
        'actual_final_python_process_snapshot': terminal_snapshot['actual_python_processes'],
        'actual_old_supervisor_child_controller_pids_absent': sorted(set(terminal_ids + [state['controller_pid'], state['controller_parent_pid']])),
        'CUDA_initialized_by_this_audit': False, 'torch_imported_by_this_audit': False,
        'all_four_full_training_complete': True, 'GPU_training_ownership_released': True,
        'Creator_post_Merger_or_LOD_complete': False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'status': 'CPU_summary_not_executed', 'scope': 'Requires all four final receipts and actual terminal processes',
                          'CUDA_initialized': False}))
        return
    result = collect()
    directory = BASE / 'recovery-audit/small-city-four-full-chunks-005-completion-001'
    require(not directory.exists(), 'Preserve existing final summary; choose a new namespace instead of overwrite')
    directory.mkdir(parents=True)
    (directory / 'completion-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (directory / 'summary-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(directory / 'completion-summary.json'),
                      'sha256': sha(directory / 'completion-summary.json'), 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
