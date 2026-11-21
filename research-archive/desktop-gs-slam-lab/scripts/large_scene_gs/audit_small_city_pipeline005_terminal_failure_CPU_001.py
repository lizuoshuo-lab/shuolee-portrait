"""Actual terminal handoff: three valid full chunks and one numerically failed final model."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
TARGET = BASE / 'recovery-audit/small-city-four-chunks-005-terminal-numerical-failure-001'
NAMES = {'0_0': 'small-city-chunk00-i30000-r4-003', '0_1': 'small-city-chunk01-i30000-r4-003',
         '1_0': 'small-city-chunk10-i30000-r4-003', '1_1': 'small-city-chunk11-i30000-r4-003'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(name, expected):
    path = ROOT / 'scripts/large_scene_gs' / name
    require(sha(path) == expected, 'Frozen CPU gate/observer helper changed')
    spec = importlib.util.spec_from_file_location(name.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    require(not TARGET.exists(), 'Preserve existing terminal failure evidence')
    state_path = BASE / 'official-training-pipeline/small-city-four-chunks-005/state.json'
    state_sha = sha(state_path)
    state = read(state_path)
    require(state['status'] == 'failed_or_interrupted_preserved_no_retry' and state['current_chunk'] == '1_1', 'Actual terminal failure status changed')
    require(state['sequence'] == NAMES and len(state['completed_chunks']) == 3, 'Actual formal chunk completion scope changed')
    for filename, expected in state['source_snapshots_sha256'].items():
        require(sha(ROOT / 'scripts/large_scene_gs' / filename) == expected and sha(state_path.parent / filename) == expected,
                'Actual controller frozen source/snapshot changed')
    binding = load('verify_small_city_stage_binding_004.py', '8da91ef734ca6579d94d0016b2863816a2bd7eedb752eaaafba8d8f626a59e90')
    observer = load('observe_small_city_process_roles.py', 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a')
    rows = []
    for (chunk, run), saved in zip(NAMES.items(), state['completed_chunks']):
        require(saved['chunk'] == chunk and saved['run'] == run, 'Completed chunk namespace order changed')
        audit_path = Path(saved['audit'])
        require(sha(audit_path) == saved['audit_sha256'], 'Immutable completed physical audit bytes changed')
        audit = read(audit_path)
        gate = binding.completed_gate(run, 'chunk', chunk)
        require(gate['model_parameter_sha256'] == saved['model_sha256'] == audit['model_parameter_sha256'], 'Actual completed model bytes changed')
        require(all(x['float32_bitexact'] for x in audit['shared_sky_all_parameter_groups_bitexact'].values()), 'Shared sky gate failed')
        rows.append({'chunk': chunk, 'run': run, 'status': 'completed_valid_full30000',
                     'final_G': audit['actual_saved_gaussians'], 'elapsed_seconds': audit['elapsed_seconds'],
                     'logged_torch_peak_allocated_bytes': audit['logged_torch_peak_allocated_bytes'],
                     'model_parameter_sha256': gate['model_parameter_sha256'], 'physical_audit': str(audit_path),
                     'physical_audit_sha256': sha(audit_path), 'fresh_fixed005_completed_training_gate': gate})
    run = NAMES['1_1']
    out = BASE / 'official-training' / run
    stage = read(out / 'outcome.json')
    sup_path = BASE / 'official-training-supervision' / run / 'supervisor-outcome.json'
    sup = read(sup_path)
    require(stage['status'] == 'failed' and stage['exit_code'] == 0 and stage['error'] == 'Saved model has nonfinite parameters', 'Stage numeric failure receipt differs')
    require(sup['status'] == 'failed' and sup['child_exit_code'] == 1 and sup['stage_outcome_sha256'] == sha(out / 'outcome.json'), 'Actual wrapper child return evidence differs')
    diagnosis_path = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-CPU-001/diagnosis.json'
    lineage_path = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-lineage-CPU-002/lineage-activation.json'
    require(sha(diagnosis_path) == '338423bcf489386fbf7dd5a47c06281ad9ea01794d00eed2bfe8e347e56c13c7', 'Failure raw CPU audit bytes changed')
    require(sha(lineage_path) == '68bc09e49af01f0a2afe46c3ff0a41e25365b848dd25142da189aafb6dd6d8df', 'Failure lineage CPU audit bytes changed')
    diagnosis = read(diagnosis_path)
    require(sha(diagnosis['saved_parameter_path']) == diagnosis['saved_parameter_sha256'], 'Failed saved model bytes changed')
    for item in diagnosis['original_failed_run_files_preserved']:
        require(Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256'], 'Original failed run artifact changed')
    snapshot = observer.observe(run)
    relevant_scripts = {'train_single.py', 'train_coarse.py', 'run_small_city_stage.py', 'supervise_small_city_stage.py', 'run_small_city_full_pipeline_005.py'}
    active = [p for p in snapshot['actual_python_processes'] if
              {Path(a).name.lower() for a in observer.windows_argv(p.get('CommandLine') or '')[1:]} & relevant_scripts]
    require(not active, 'Actual trainer/controller/supervisor still active; no GPU handoff')
    actual_pids = {p['ProcessId'] for p in snapshot['actual_python_processes']}
    old_pids = {state['controller_pid'], state['controller_parent_pid']}
    for run_name in NAMES.values():
        completed_sup = read(BASE / 'official-training-supervision' / run_name / 'supervisor-outcome.json')
        old_pids.update(completed_sup[key] for key in ('supervisor_pid', 'supervisor_parent_pid', 'child_pid'))
    require(not (old_pids & actual_pids), 'Original pipeline/supervisor/child PID still exists')
    require(sha(state_path) == state_sha, 'Pipeline state changed during actual final audit')
    result = {
        'status': 'three_valid_full30000_chunks_one_numerically_failed30k_chunk_actual_terminal_CPU_audited_no_retry',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)),
        'pipeline_state': str(state_path), 'pipeline_state_sha256': state_sha,
        'requested_budget_each': 30000, 'resolution': 4, 'exposure': 'official',
        'official_algorithm_modified': False, 'completed_valid_full_chunks': rows,
        'completed_valid_chunk_count': 3, 'total_formal_chunks': 4,
        'failed_chunk': {
            'run': run, 'chunk': '1_1', 'official_trainer_exit_code': 0, 'stage_wrapper_child_exit_code': 1,
            'stage_status': stage['status'], 'supervisor_status': sup['status'],
            'stage_outcome_sha256': sha(out / 'outcome.json'), 'supervisor_outcome_sha256': sha(sup_path),
            'failed_saved_G': diagnosis['actual_saved_G'], 'failed_parameter_sha256': diagnosis['saved_parameter_sha256'],
            'invalid_gaussian_rows': diagnosis['invalid_gaussian_rows_count'], 'invalid_shared_sky_rows': diagnosis['invalid_shared_sky_rows_count'],
            'raw_failure_diagnosis': str(diagnosis_path), 'raw_failure_diagnosis_sha256': sha(diagnosis_path),
            'lineage_activation_diagnosis': str(lineage_path), 'lineage_activation_diagnosis_sha256': sha(lineage_path),
            'logged_torch_peak_allocated_bytes': diagnosis['logged_torch_peak_allocated_bytes'],
            'cause': None, 'cause_scope': 'Saved single scaffold Gaussian geometry/scale/rotation/opacity NaN reproduced. First iteration and computational cause unobserved.',
        },
        'actual_original_pipeline_controller_supervisor_trainer_processes': active,
        'actual_old_controller_supervisor_child_pids_absent': sorted(old_pids),
        'actual_final_python_process_snapshot': snapshot['actual_python_processes'],
        'GPU_training_owner_released': True,
        'release_scope': 'Original Python training owner terminal. Future native/evaluation stage must independently enforce global GPU exclusivity.',
        'CUDA_initialized_by_this_audit': False, 'torch_imported_by_this_audit': False,
        'automatic_retry_performed': False, 'raw_finite_gate_relaxed': False, 'failed_parameters_repaired_or_rows_deleted': False,
        'all_four_full_training_complete': False, 'Creator_post_Merger_or_LOD_complete': False,
    }
    TARGET.mkdir(parents=True)
    path = TARGET / 'terminal-audit.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (TARGET / 'audit-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path),
                      'valid_chunk_count': 3, 'GPU_training_owner_released': True, 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
