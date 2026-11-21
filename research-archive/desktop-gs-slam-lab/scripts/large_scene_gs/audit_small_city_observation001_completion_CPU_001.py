"""Read-only CPU terminal/physical audit of the observation-only model; no formal promotion."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'small-city-chunk11-observation-i30000-r4-001'
OUT = BASE / 'official-training-observation' / RUN
TARGET = BASE / 'recovery-audit/small-city-observation001-completion-CPU-001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def load(filename, expected):
    path = ROOT / 'scripts/large_scene_gs' / filename
    require(sha(path) == expected, 'Frozen pure CPU helper changed')
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    require(not TARGET.exists(), 'Preserve existing observation completion evidence')
    sup_module = load('supervise_small_city_observation_001.py', 'bb0a8cd8f7efd2666c56022a397ecdb2b969b02c20806538c25f90c10a7ec370')
    plan = sup_module.source_gate()
    stage = read(OUT / 'outcome.json')
    cfg = read(OUT / 'configuration.json')
    observed = read(OUT / 'observer/observer-outcome.json')
    sup_path = BASE / 'official-training-observation-supervision' / RUN / 'supervisor-outcome.json'
    sup = read(sup_path)
    require(stage['status'] == 'completed_observation_only_final_finite_not_formal' and stage['exit_code'] == 0 and stage['source_changed'] is False, 'Actual observation stage not full finite exit0')
    require(sup['status'] == 'completed_fullbudget_observation_only_not_formal' and sup['actual_child_poll_exit_code'] == 0, 'Real Popen handle did not report child exit0')
    require(sup['actual_stage_outcome_sha256'] == sha(OUT / 'outcome.json') and sup['actual_observer_outcome_sha256'] == sha(OUT / 'observer/observer-outcome.json'), 'Supervisor terminal receipt hashes changed')
    require(observed['actual_original_exposure_Adam_step_calls'] == observed['actual_original_opacity_grad_condition_visits'] == 29999 and
            observed['actual_original_Gaussian_Adam_step_calls'] + observed['actual_original_opacity_grad_None_skips'] == 29999 and
            observed['successful_original_save_iterations'] == [30000] and observed['first_nonfinite_event'] is None,
            'Actual conditional Gaussian/exposure/save-return observation counters failed')
    require(not (OUT / 'observer/first-nonfinite.json').exists(), 'Runtime detected a nonfinite event')
    for key in ('command_argv', 'original_official_command_argv', 'input_manifest_sha256', 'coarse_outcome_sha256',
                'observer_source_sha256', 'launcher_source_sha256', 'trainer_source_sha256', 'requested_iterations', 'resolution_pixels', 'exposure_mode'):
        require(cfg[key] == plan[key], 'Actual stage differs from prepared observation plan')
    require(cfg['source_changed'] is False and cfg['formal_training_run'] is False, 'Observation source/scope changed')
    require(observed['runtime_hook_contract']['official_sys_argv'] == cfg['original_official_command_argv'][2:], 'Actual runpy original argv differs')
    require(observed['runtime_hook_contract']['driver_sha256'] == cfg['observer_source_sha256'] and
            observed['runtime_hook_contract']['OurAdam_sha256'] == 'fb552b3c31f07ef541dd71a2ac2b8fef72833423a5ca7d9fdf4f17cc22e1c9ba', 'Runtime observer/Adam source differs')
    for item in cfg['official_source_files']:
        require(sha(item['path']) == item['sha256'] and sha(OUT / 'official-source' / item['relative_path']) == item['sha256'], 'Original live/snapshot official source changed')
    require(sha(OUT / 'launcher-source.py') == cfg['launcher_source_sha256'] and sha(OUT / 'observer-source.py') == cfg['observer_source_sha256'], 'Actual stage/observer snapshots changed')
    initial = read(OUT / 'observer/initial-raw6group-identity.json')
    require(initial['actual_match_passed'] is True and all(v['shape_equal'] and v['float32_bits_equal'] for v in initial['group_checks'].values()), 'Real initial GPU raw6group bit identity failed')
    parsed = load('run_small_city_stage.py', 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704')
    util = load('audit_small_city_full_chunk_004.py', 'e9a4e197e84d1bb845349d60d5fd23f73376283ff0984cc1f2a922c6423fc37c')
    model_audit = read(OUT / 'model-audit.json')
    parameter = Path(model_audit['parameter_path'])
    require(parameter == OUT / 'model/point_cloud/iteration_30000/point_cloud.ply' and sha(parameter) == stage['model_parameter_sha256'], 'Actual final I30000 model changed')
    for item in model_audit['files']:
        require(sha(item['path']) == item['sha256'], 'Actual saved model file changed')
    v = parsed.ply_vertices(parameter)
    raw = {name: bool(np.isfinite(v[name]).all()) for name in v.dtype.names}
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scales = np.exp(util.rows(v, [f'scale_{i}' for i in range(3)], len(v)))
        norms = np.linalg.norm(util.rows(v, [f'rot_{i}' for i in range(4)], len(v)), axis=1)
    require(all(raw.values()) and np.isfinite(scales).all() and (scales > 0).all() and np.isfinite(norms).all() and (norms > 0).all(), 'Saved raw/activated physical arrays failed')
    coarse_path = BASE / 'official-training/small-city-coarse-i30000-r4-003/model/point_cloud/iteration_30000/point_cloud.ply'
    require(sha(coarse_path) == 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31', 'Actual coarse model changed')
    coarse = parsed.ply_vertices(coarse_path)
    sky = {}
    for name, names in [('xyz', ['x', 'y', 'z']), ('log_scale', [f'scale_{i}' for i in range(3)]),
                        ('quaternion_wxyz', [f'rot_{i}' for i in range(4)]), ('opacity_logit', ['opacity']), ('SH_DC', [f'f_dc_{i}' for i in range(3)])]:
        sky[name] = util.bit_audit(util.rows(v, names, 100000), util.rows(coarse, names, 100000))
    actual_rest = util.rows(v, [f'f_rest_{i}' for i in range(45)], 100000).reshape(100000, 3, 15).transpose(0, 2, 1)
    expected_rest = np.zeros((100000, 15, 3), dtype=np.float32)
    expected_rest[:, :3, :] = util.rows(coarse, [f'f_rest_{i}' for i in range(9)], 100000).reshape(100000, 3, 3).transpose(0, 2, 1)
    sky['SH_rest_SH1_to_SH3_zero_padded'] = util.bit_audit(actual_rest, expected_rest)
    require(all(x['float32_bitexact'] for x in sky.values()), 'Actual shared sky raw6group bit identity failed')
    sh2 = sum(int(np.count_nonzero(v[f'f_rest_{channel*15+basis}'])) for channel in range(3) for basis in range(3, 8))
    sh3 = sum(int(np.count_nonzero(v[f'f_rest_{channel*15+basis}'])) for channel in range(3) for basis in range(8, 15))
    require(sh2 > 0 and sh3 > 0, 'No actual higher-degree SH learning evidence')
    journal = OUT / 'observer/observation-journal.jsonl'
    counts = {'Adam.step': 0, 'original_opacity_gradient_None_condition_skip': 0, 'whole_raw_parameter_finite_sample': 0}
    with journal.open(encoding='utf-8') as stream:
        for line in stream:
            entry = json.loads(line)
            phase = entry['phase']
            if phase in counts:
                counts[phase] += 1
            require(not entry.get('before_nonfinite_paths') and not entry.get('after_nonfinite_paths'), 'Journal contains suspect nonfinite values')
            if phase == 'whole_raw_parameter_finite_sample':
                require(all(a['all_finite'] for a in entry['whole'].values()), 'A full-parameter sample failed')
    require(counts['Adam.step'] == observed['actual_original_Gaussian_Adam_step_calls'] and
            counts['original_opacity_gradient_None_condition_skip'] == observed['actual_original_opacity_grad_None_skips'], 'Actual journal/counter mismatch')
    failed_path = BASE / 'recovery-audit/small-city-chunk11-i30000-r4-003-nonfinite-CPU-001/diagnosis.json'
    require(sha(failed_path) == '338423bcf489386fbf7dd5a47c06281ad9ea01794d00eed2bfe8e347e56c13c7', 'Original failed diagnosis changed')
    failed = read(failed_path)
    for item in failed['original_failed_run_files_preserved']:
        require(Path(item['path']).stat().st_size == item['bytes'] and sha(item['path']) == item['sha256'], 'An original failed003 artifact changed')
    actual = sup_module.processes()
    require(not sup_module.busy(actual), 'A real GPU process exists; global release not verified')
    old_pids = {24928, 12152, 40448, 40696, 36136, 5092, 35084, 37396}
    require(not old_pids.intersection(p['ProcessId'] for p in actual), 'An original observation owner process still exists')
    result = {
        'status': 'full30000_observation_only_saved_physical_arrays_source_sky_counters_and_actual_terminal_CPU_verified_not_formal',
        'observed_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(Path(__file__)), 'run': RUN,
        'stage_outcome_sha256': sha(OUT / 'outcome.json'), 'stage_configuration_sha256': sha(OUT / 'configuration.json'),
        'observer_outcome_sha256': sha(OUT / 'observer/observer-outcome.json'), 'supervisor_outcome_sha256': sha(sup_path),
        'saved_parameter_path': str(parameter), 'model_parameter_sha256': sha(parameter), 'saved_G': len(v),
        'all_raw_parameter_fields_finite': raw, 'activated_float32_scale_min_max_scene_units': [float(scales.min()), float(scales.max())],
        'float32_quaternion_norm_min_max': [float(norms.min()), float(norms.max())],
        'shared_sky_raw6group_bitexact': sky, 'actual_SH2_nonzero_scalar_coefficients': sh2, 'actual_SH3_nonzero_scalar_coefficients': sh3,
        'actual_Gaussian_Adam_calls': observed['actual_original_Gaussian_Adam_step_calls'],
        'actual_exposure_Adam_calls': observed['actual_original_exposure_Adam_step_calls'],
        'actual_original_opacity_gradient_None_skips': observed['actual_original_opacity_grad_None_skips'],
        'actual_original_condition_visits': observed['actual_original_opacity_grad_condition_visits'], 'actual_journal_phase_counts': counts,
        'journal_sha256': sha(journal), 'runtime_initial_suspect_raw6group_bit_identity_passed': True,
        'first_nonfinite_observation': None, 'scope': 'This observation run did not reproduce failed003 numerical invalidity. It is not evidence of a numerical fix, bitexact deterministic reproduction or formal004 completion.',
        'old_failed003_all_original_file_count_preserved': len(failed['original_failed_run_files_preserved']),
        'old_failed003_parameter_sha256_unchanged': failed['saved_parameter_sha256'], 'original_failed003_still_failed': True,
        'actual_terminal_process_snapshot': actual, 'actual_old_observation_processes_absent': sorted(old_pids),
        'GPU_training_owner_released': True, 'CUDA_initialized_by_this_audit': False, 'torch_imported_by_this_audit': False,
        'formal_completed_model_claimed': False, 'formal_original_chunks_complete': 3,
        'Creator_post_Merger_or_LOD_complete': False,
    }
    TARGET.mkdir(parents=True)
    path = TARGET / 'completion-audit.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (TARGET / 'audit-source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path), 'saved_G': len(v),
                      'all_sky6group_bitexact': True, 'actual_Gaussian_calls': observed['actual_original_Gaussian_Adam_step_calls'],
                      'actual_None_skips': observed['actual_original_opacity_grad_None_skips'], 'old_failed003_files_unchanged': len(failed['original_failed_run_files_preserved']),
                      'GPU_owner_released': True, 'CUDA_initialized': False}))


if __name__ == '__main__':
    main()
