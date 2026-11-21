"""CPU-only binding for three legacy local runs and one observed local run.

This does not execute training/native stages, initialize CUDA, reclassify an
observation receipt as formal, or update the frozen hierarchy005 workflow.
Every invocation writes a new audit directory; a missing final model is a
real rejection, not a provisional completed four-chunk receipt.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
OLD = BASE / 'official-hierarchy/small-city-hierarchy-005'
INPUT = BASE / 'official-inputs/small-city-training-002'
OBS_RUN = 'small-city-chunk11-observation-i30000-r4-001'
OBS = BASE / 'official-training-observation' / OBS_RUN
SUP = BASE / 'official-training-observation-supervision' / OBS_RUN
PLAN = BASE / 'official-training-observation-plans' / (OBS_RUN + '-v3')
COARSE_RUN = 'small-city-coarse-i30000-r4-003'
COARSE = BASE / 'official-training' / COARSE_RUN
LEGACY = {c: BASE / 'official-training' / f'small-city-chunk{c.replace("_", "")}-i30000-r4-003'
          for c in ('0_0', '0_1', '1_0')}
# These are completed bytes already on disk, not invented future model hashes.
LEGACY_PINS = {
    '0_0': ('e923e7ed8d15ada83fe7245b1759706ffc686763dd2d8cb5ed5f5ea2e740f1ab',
            '6d8ce937e4afea85c809d5c4059bf669ac8eb23cb8c83e920cc26c6b7d5124ea',
            'fc8e91ccbd9d4fdb11e35674610340ef24b4a3136580040cbd1fe024d0a5fb51'),
    '0_1': ('32e7ee3db6a0b6e716b0f04a448dfa97869d4aabaef1563826ed0f0750b1cbe6',
            '059c548af70808fb74360811cff536276f932793fdfae98296ba8b322bbcfc98',
            'd930df8ef5555dc1fb111f3e88e7350eacf2159bc4832587d0b14f51df094e43'),
    '1_0': ('81f15626a88e6738813f79f0ed31a1cbdbb3fc01b496f330316b29078f91cc14',
            '3198ff4aabb54b78837ca82aeeeb52ef7a31ccd1410109741b7ccccd6baeb3a7',
            'd81db42847302d47f39dbf771bcf7ddc42cc3a1e5d4ed080b8a4e4ab7c479a01'),
    'coarse': ('492cea0b35018f36d7eb41b5b5f866b85d73c3d3f71dba72904ce84c85b7703b',
            '8214314c1371002928ecf327b348883b93a45fdb5a8fbae180b5f3452ec6f7d2',
            'f1fdabd706eef046fdf39eb1ed452c110b90554e83b29ab0ed4bd7c8e02d9bea'),
}
PINS = {
    'manifest': '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5',
    'old_config': '55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea',
    'old_workflow': '0e51b5c51d0742a20315d02c37355aaae7e9f1d905069a1f6594da0ee06d7f84',
    'training_helper': 'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704',
    'codec': '583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538',
    'observer': '59bc69b80fb50283cddd5a35c115fb3ecd9ba37bef675c74e6f1736cc02af384',
    'stage': 'ebe2581d8ae8bee8fa48519c383030a0e74be0677c38b81855715dee6a16570c',
    'supervisor': 'bb0a8cd8f7efd2666c56022a397ecdb2b969b02c20806538c25f90c10a7ec370',
    'hidden_launcher': '42be603955a0701c32bc8739ba96d6fc6a4adeb2f190157a739f9f21f0159366',
    'plan': '8b51155d5300bd75c3ebef0c4544da8e2b5a393cd90909ed97430ca0544df826',
    'oracle': '3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236',
    'OurAdam': 'fb552b3c31f07ef541dd71a2ac2b8fef72833423a5ca7d9fdf4f17cc22e1c9ba',
    'run_local': '1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2',
    'coarse_model': 'abd4f6af0463dce35648182a66d2ae3d6893417901f3885c5be1eb98368f2e31',
    'coarse_outcome': '8214314c1371002928ecf327b348883b93a45fdb5a8fbae180b5f3452ec6f7d2',
}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(p, expected):
    require(sha(p) == expected, f'Pinned bytes changed: {p}')
    return {'path': str(p), 'sha256': expected}


def samepath(a, b):
    return Path(a).resolve() == Path(b).resolve()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sources():
    # Import only these pinned pure NumPy/stdlib helpers later. No old
    # completed_training(), main(), Scene, renderer, Torch, or native call.
    paths = {
        'manifest': INPUT / 'manifest.json', 'old_config': OLD / 'configuration.json',
        'old_workflow': OLD / 'driver-source.py',
        'training_helper': OLD / 'helper-snapshots/run_small_city_stage.py',
        'codec': OLD / 'helper-snapshots/read_hierarchy.py',
        'plan': PLAN / 'configuration.json',
        'observer': ROOT / 'scripts/large_scene_gs/observe_small_city_train_single_001.py',
        'stage': ROOT / 'scripts/large_scene_gs/run_small_city_observation_001.py',
        'supervisor': ROOT / 'scripts/large_scene_gs/supervise_small_city_observation_001.py',
        'hidden_launcher': ROOT / 'scripts/large_scene_gs/launch_small_city_observation_001.ps1',
        'OurAdam': ROOT / 'repos/hierarchical-3d-gaussians/scene/OurAdam.py',
        'run_local': ROOT / 'scripts/large_scene_gs/run_local.py',
        'oracle': BASE / 'official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json',
        'coarse_outcome': COARSE / 'outcome.json',
    }
    records = {key: pin(path, PINS[key]) for key, path in paths.items()}
    for name, directory in [('coarse', COARSE)] + list(LEGACY.items()):
        records['legacy_' + name] = [pin(directory / filename, digest) for filename, digest in
            zip(('configuration.json', 'outcome.json', 'model-audit.json'), LEGACY_PINS[name])]
    return records


def schema(records):
    return {'schema_version': 1, 'namespace': 'small-city-hierarchy-006',
            'status': 'CPU_binding_contract_prepared_not_four_chunks_completed',
            'binder_source_sha256': sha(__file__), 'source_pins': records,
            'coarse': str(COARSE), 'ordered_chunk_training_origins': [
                {'chunk': c, 'directory': str(LEGACY[c]), 'training_source_kind': 'legacy_local_official_training',
                 'actual_Gaussian_Adam_calls': None, 'derived_Gaussian_call_upper_bound': 29999}
                for c in LEGACY] + [{'chunk': '1_1', 'directory': str(OBS),
                    'training_source_kind': 'observed_local_official_training',
                    'formal_training_run': False, 'actual_final_model_sha256': None}],
            'requested_iterations_each': 30000, 'shared_sky_prefix_count': 100000,
            'test_exposure_fit': False, 'frozen_global_test_count': 117,
            'hierarchy_state_source_kind_separate': ['unoptimized', 'post_optimized'],
            'native_execution_supported_by_this_binder': False,
            'pending_versioned_connections': ['006 native workflow', '006 post64 preflight',
                '006 single-stage supervisor/Hidden launcher', '006 evaluation adapter',
                '006 posterior1404 auditor and perceptual/boundary binders'],
            'meaning': 'A future receipt may bind a mixed-origin local model lineage; it does not turn observation receipts into formal training or prove any Creator/post/Merger/render execution.'}


def observed_terminal(proofs):
    stage = read(OBS / 'outcome.json')
    proofs['observed_stage_status_seen'] = stage.get('status')
    proofs['observed_stage_outcome_at_query'] = {'path': str(OBS / 'outcome.json'), 'sha256': sha(OBS / 'outcome.json')}
    # First gate intentionally runs before any large model read/import. A
    # live, interrupted or diagnostic-only run cannot mint a model receipt.
    require(stage.get('status') == 'completed_observation_only_final_finite_not_formal',
            'Observed11 has no completed finite full-budget outcome; actual current status=' + str(stage.get('status')))
    observed = read(OBS / 'observer/observer-outcome.json')
    supervisor = read(SUP / 'supervisor-outcome.json')
    require(stage.get('exit_code') == 0 and stage.get('source_changed') is False and
            stage.get('formal_completed_model_claimed') is False, 'Observed stage terminal/source/formal flags')
    require(observed['status'] == 'official_full30000_observation_returned_no_detected_nonfinite_not_formal' and
            observed['actual_exit_code'] == 0 and observed['first_nonfinite_event'] is None and
            observed['formal_completed_model_claimed'] is False and observed['full_budget_requested'] == 30000,
            'Observed original runpy normal return/full budget/no nonfinite evidence')
    g, e, v, s = [observed[k] for k in ('actual_original_Gaussian_Adam_step_calls',
        'actual_original_exposure_Adam_step_calls', 'actual_original_opacity_grad_condition_visits',
        'actual_original_opacity_grad_None_skips')]
    require(all(type(x) is int for x in (g, e, v, s)) and 0 < g <= 29999 and s >= 0 and
            e == v == 29999 and g+s == 29999 and
            len(observed['actual_original_opacity_grad_None_skip_contexts']) == s and
            30000 in observed['successful_original_save_iterations'], 'Actual observed step/skip/save accounting')
    require(stage['optimizer_update_calls'] == g and stage['actual_exposure_optimizer_calls'] == e and
            stage['actual_Gaussian_optimizer_None_gradient_skips'] == s, 'Stage/observer measured counters differ')
    require(supervisor['status'] == 'completed_fullbudget_observation_only_not_formal' and
            supervisor['actual_child_poll_exit_code'] == 0 and supervisor['GPU_training_owner_released'] is True and
            supervisor['actual_terminal_GPU_stage_processes'] == [] and
            supervisor['formal_completed_model_claimed'] is False and
            supervisor['actual_stage_outcome_sha256'] == sha(OBS / 'outcome.json') and
            supervisor['actual_observer_outcome_sha256'] == sha(OBS / 'observer/observer-outcome.json'),
            'Observed supervisor actual terminal receipt does not match completed child/model')
    require(supervisor.get('actual_child_birth') and supervisor.get('actual_terminal_process_snapshot') is not None,
            'Missing launched child birth/terminal process evidence')
    contract = read(OBS / 'observer/runtime-hook-contract.json')
    require(contract['driver_sha256'] == PINS['observer'] and contract['OurAdam_sha256'] == PINS['OurAdam'] and
            contract['trainer_sha256'] == read(PLAN / 'configuration.json')['trainer_source_sha256'] and
            contract['exact_initial_raw_identity_oracle_sha256'] == PINS['oracle'] and
            contract['safe_state_seed'] == 0 and contract['runpy_run_name'] == '__main__' and
            contract['original_parameters_gradients_optimizer_state_hyperparameters_RNG_not_modified_by_observer'] is True,
            'Observed actual runtime source/original-call contract')
    require(observed['runtime_hook_contract'] == contract, 'Runtime hook proof changed')
    proofs['observed_terminal_accounting'] = {'actual_Gaussian_Adam_calls': g, 'actual_exposure_Adam_calls': e,
        'actual_opacity_grad_condition_visits': v, 'actual_None_gradient_skips': s,
        'successful_original_save_iterations': observed['successful_original_save_iterations'],
        'formal_training_run': False, 'source_kind': 'observed_local_official_training',
        'observation_overhead': 'Read-only CPU copies, CUDA synchronization, tracing and finite scans change timing and peak memory; not an equal timing/memory benchmark of legacy003.'}
    for p in [OBS / 'observer/observer-outcome.json', OBS / 'observer/runtime-hook-contract.json', SUP / 'supervisor-outcome.json']:
        proofs.setdefault('terminal_files', []).append({'path': str(p), 'sha256': sha(p)})
    return observed, contract


def fresh_terminal():
    # Read-only Windows process query; never launch/stop a training process or
    # query a CUDA device. This CPU binder itself is excluded from GPU owners.
    command = "Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python|GaussianHierarchy'} | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Compress"
    raw = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                         capture_output=True, text=True, check=True).stdout.strip()
    rows = json.loads(raw) if raw else []
    rows = rows if isinstance(rows, list) else [rows]
    busy = []
    for r in rows:
        cmd = str(r.get('CommandLine') or '').lower()
        name = str(r.get('Name') or '').lower()
        native = 'gaussianhierarchy' in name
        official = any('/'+f in cmd.replace('\\', '/') for f in ('train_single.py', 'train_coarse.py', 'train_post.py'))
        observer = 'observe_small_city_train_single_001.py' in cmd and '--execute' in cmd
        evaluator = ('evaluate_' in cmd and ('--gpu' in cmd or 'run_local.py' in cmd))
        wrapper = '--execute' in cmd and any(f in cmd for f in
            ('run_small_city_stage.py', 'run_small_city_observation_001.py',
             'run_small_city_hierarchy.py', 'run_small_city_post_preflight_001.py'))
        if native or official or observer or evaluator or wrapper:
            busy.append(r)
    require(not busy, 'Actual GPU-stage process still alive: ' + json.dumps(busy, ensure_ascii=False))
    return {'queried_utc': datetime.now(timezone.utc).isoformat(), 'actual_snapshot': rows,
            'actual_gpu_stage_processes': busy, 'query': command, 'CUDA_query_performed': False}


def model_numeric(saved, stage, training, codec):
    import numpy as np
    ply = saved / 'point_cloud.ply'
    if ply.is_file():
        a = training.ply_vertices(ply)
        names = a.dtype.names
        rest = 9 if stage == 'coarse' else 45
        required = ['x', 'y', 'z', 'opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)] + [f'f_dc_{i}' for i in range(3)] + [f'f_rest_{i}' for i in range(rest)]
        require(all(n in names for n in required) and len([n for n in names if n.startswith('f_rest_')]) == rest,
                'PLY raw field/SH layout differs')
        count = len(a)
        for b in range(0, count, 65536):
            z = a[b:b+65536]
            require(all(np.isfinite(z[n]).all() for n in names), 'Nonfinite saved PLY raw field')
            scales = np.stack([z[f'scale_{i}'] for i in range(3)], axis=1)
            quat = np.stack([z[f'rot_{i}'] for i in range(4)], axis=1)
            with np.errstate(over='ignore', under='ignore'):
                activated = np.exp(scales)
            require(np.isfinite(activated).all() and (activated > 0).all() and
                    (np.linalg.norm(quat.astype(np.float64), axis=1) > 0).all(), 'Exp scale/nonzero wxyz activation fails')
        path = ply
    else:
        require(stage == 'chunk', 'Coarse must use actual PLY')
        path = saved / 'point_cloud.bin'
        a = codec.read_gaussian_bin(path)
        count = len(a['positions'])
        require(count > 8000000 and path.stat().st_size == 4+236*count, 'Official SH3 binary contract')
        for b in range(0, count, 65536):
            require(all(np.isfinite(v[b:b+65536]).all() for v in a.values()), 'Nonfinite saved bin')
            with np.errstate(over='ignore', under='ignore'):
                scale = np.exp(a['log_scales'][b:b+65536])
            require(np.isfinite(scale).all() and (scale > 0).all() and
                    (np.linalg.norm(a['rotations_wxyz'][b:b+65536].astype(np.float64), axis=1) > 0).all(), 'Binary activation fails')
    return {'path': str(path), 'sha256': sha(path), 'gaussian_count': count,
            'raw_finite': True, 'exp_f32_scales_finite_positive': True, 'wxyz_norm_positive': True,
            'max_SH_degree': 1 if stage == 'coarse' else 3, 'CPU_chunk_size': 65536}


def sky_bits(scaffold, saved, training, codec):
    import numpy as np
    coarse = training.ply_vertices(scaffold / 'point_cloud.ply')
    p = 100000
    expected = {'positions': np.stack([coarse[n][:p] for n in ('x', 'y', 'z')], axis=1),
        'opacity_logits': coarse['opacity'][:p],
        'log_scales': np.stack([coarse[f'scale_{i}'][:p] for i in range(3)], axis=1),
        'rotations_wxyz': np.stack([coarse[f'rot_{i}'][:p] for i in range(4)], axis=1),
        'sh': np.zeros((p, 16, 3), dtype=np.float32)}
    expected['sh'][:, 0] = np.stack([coarse[f'f_dc_{c}'][:p] for c in range(3)], axis=1)
    for c in range(3):
        for j in range(3):
            expected['sh'][:, j+1, c] = coarse[f'f_rest_{c*3+j}'][:p]
    if (saved / 'point_cloud.ply').is_file():
        v = training.ply_vertices(saved / 'point_cloud.ply')
        actual = {'positions': np.stack([v[n][:p] for n in ('x', 'y', 'z')], axis=1),
            'opacity_logits': v['opacity'][:p],
            'log_scales': np.stack([v[f'scale_{i}'][:p] for i in range(3)], axis=1),
            'rotations_wxyz': np.stack([v[f'rot_{i}'][:p] for i in range(4)], axis=1),
            'sh': np.zeros((p, 16, 3), dtype=np.float32)}
        actual['sh'][:, 0] = np.stack([v[f'f_dc_{c}'][:p] for c in range(3)], axis=1)
        for c in range(3):
            for j in range(15):
                actual['sh'][:, j+1, c] = v[f'f_rest_{c*15+j}'][:p]
    else:
        actual = codec.read_gaussian_bin(saved / 'point_cloud.bin')
    equality = {name: bool(np.array_equal(np.ascontiguousarray(ref).view(np.uint32),
                                        np.ascontiguousarray(actual[name][:p]).view(np.uint32)))
                for name, ref in expected.items()}
    require(all(equality.values()), 'Shared sky raw float32 bit identity/SH3 padding differs')
    return equality


def bind_one(directory, stage, chunk, manifest, workflow, training, codec, observed=False):
    receipt, config, audit = [read(directory / f) for f in ('outcome.json', 'configuration.json', 'model-audit.json')]
    require(receipt['status'] == ('completed_observation_only_final_finite_not_formal' if observed else 'completed') and
            receipt['exit_code'] == 0 and receipt['requested_iterations'] == 30000 and
            (not observed or receipt.get('source_changed') is False),
            'Complete full-budget/source-preserved stage required')
    args = config['effective_official_arguments']
    require(config['official_commit'] == workflow.COMMIT and config['stage'] == stage and config.get('chunk') == chunk and
            config['requested_iterations'] == args['iterations'] == 30000 and config['source_changed'] is False and
            config['input_manifest_sha256'] == PINS['manifest'] and config['exposure_mode'] == 'official' and
            config['train_test_exp'] is False and args['eval'] is True and args['train_test_exp'] is False and
            args['resolution'] == 4 and args['data_device'] == 'cpu', 'Training configuration/reader/iteration scope differs')
    data = INPUT / 'aligned' if stage == 'coarse' else INPUT / 'chunks' / chunk
    split = manifest if stage == 'coarse' else next(c for c in manifest['chunks'] if c['name'] == chunk)
    require(config['training_names'] == split['train_names'] and config['stage_test_names'] == split['test_names'] and
            not set(split['train_names']) & set(manifest['test_names']) and
            samepath(args['source_path'], data) and samepath(config['official_test_path'], data / 'sparse/0/test.txt') and
            config['official_test_sha256'] == sha(data / 'sparse/0/test.txt') and
            config['official_reader_CPU_audit_sha256'] == sha(INPUT / 'official-reader-CPU-audit.json') and
            args['images'] == manifest['images_absolute'] and args['alpha_masks'] == manifest['masks_absolute'], 'Frozen image/split/Reader identity mismatch')
    trainer = ROOT / 'repos/hierarchical-3d-gaussians' / ('train_coarse.py' if stage == 'coarse' else 'train_single.py')
    require(config['trainer_source_sha256'] == sha(trainer), 'Official trainer source hash differs')
    for item in config['official_source_files']:
        pin(item['path'], item['sha256'])
        pin(directory / 'official-source' / item['relative_path'], item['sha256'])
    saved = directory / 'model/point_cloud/iteration_30000'
    require(samepath(receipt['saved_model_directory'], saved) and int((saved / 'pc_info.txt').read_text()) == 100000,
            'Actual final save path/common100k sky missing')
    text = (directory / 'training.log').read_text(encoding='utf-8', errors='replace')
    require('[ITER 30000] Saving Gaussians' in text and 'Training complete.' in text, 'Actual original final save/normal training log missing')
    for f in audit['files']:
        require(Path(f['path']).resolve().parent == saved.resolve() and sha(f['path']) == f['sha256'], 'Model audited bytes/path changed')
    numeric = model_numeric(saved, stage, training, codec)
    require(numeric['sha256'] == receipt['model_parameter_sha256'] and samepath(audit['parameter_path'], numeric['path']) and
            numeric['gaussian_count'] == audit['gaussian_count'] == receipt['gaussian_count'] and
            any(samepath(f['path'], numeric['path']) and f['sha256'] == numeric['sha256'] for f in audit['files']), 'Receipt/model/CPU numeric identity differs')
    exposure_path = directory / 'model/exposure.json'
    require(sha(exposure_path) == audit['exposure']['sha256'], 'Trained exposure changed')
    exposure = workflow.exposure_audit(exposure_path, split['train_names'], manifest['test_names'])
    sky = None
    if stage == 'coarse':
        require(numeric['sha256'] == PINS['coarse_model'] and audit['exact_SfM_position_freeze_passed'], 'Actual selected coarse differs')
        rechecked = training.audit_model(saved, stage, data, None)
        require(rechecked['exact_SfM_position_freeze_passed'], 'Coarse original SfM rows no longer exact')
    else:
        scaffold = COARSE / 'model/point_cloud/iteration_30000'
        require(config['coarse_run'] == COARSE_RUN and config['coarse_outcome_sha256'] == PINS['coarse_outcome'] and
                samepath(args['scaffold_file'], scaffold) and samepath(args['bounds_file'], data) and args['skybox_locked'] is True and
                args['depths'] == manifest['depths_absolute'] and args['depth_l1_weight_init'] == 1.0 and
                args['depth_l1_weight_final'] == .01, 'Own coarse/sky/depth default contract differs')
        sky = workflow.shared_sky_audit(scaffold, data, saved, training, codec)
        sky['float32_uint32_bit_identity'] = sky_bits(scaffold, saved, training, codec)
    return {'training_directory': str(directory), 'stage': stage, 'chunk': chunk,
            'training_source_kind': 'observed_local_official_training' if observed else 'legacy_local_official_training',
            'formal_training_run': False if observed else None,
            'requested_iterations': 30000, 'actual_Gaussian_Adam_calls': receipt['optimizer_update_calls'] if observed else None,
            'derived_Gaussian_call_upper_bound': 29999,
            'legacy_optimizer_update_calls_as_recorded': None if observed else receipt.get('optimizer_update_calls'),
            'legacy_count_semantics': None if observed else 'Budget-derived upper bound; actual Gaussian calls not instrumented. Do not assert actual=29999.',
            'configuration': {'path': str(directory / 'configuration.json'), 'sha256': sha(directory / 'configuration.json')},
            'training_receipt': {'path': str(directory / 'outcome.json'), 'sha256': sha(directory / 'outcome.json'), 'status': receipt['status']},
            'model_audit_sha256': sha(directory / 'model-audit.json'), 'model': numeric,
            'exposure': exposure, 'shared_sky': sky, 'training_names': split['train_names'],
            'stage_test_names': split['test_names'], 'training_log_sha256': sha(directory / 'training.log')}


def observed_sources(contract):
    config, baseline = read(OBS / 'configuration.json'), read(BASE / 'official-training/small-city-chunk11-i30000-r4-003/configuration.json')
    prepared = read(PLAN / 'configuration.json')
    require(all(config.get(k) == value for k, value in prepared.items() if k != 'system_drive_free_bytes'),
            'Actual observed configuration differs from its immutable prepared v3 plan')
    require(config['formal_training_run'] is False and config['seed'] == 0 and config['observer_source_sha256'] == PINS['observer'] and
            config['launcher_source_sha256'] == PINS['stage'] and config['OurAdam_source_sha256'] == PINS['OurAdam'] and
            config['initial_raw_oracle_sha256'] == PINS['oracle'], 'Actual observed config/oracle/source identities')
    pin(config['baseline_failed_configuration'], config['baseline_failed_configuration_sha256'])
    expected = dict(baseline['effective_official_arguments'])
    expected['model_path'] = str(OBS / 'model')
    # Windows path separator spelling is irrelevant only for the new -m value.
    actual = dict(config['effective_official_arguments'])
    require(samepath(actual.pop('model_path'), expected.pop('model_path')) and actual == expected, 'Observed official args differ beyond isolated -m')
    oldargv = baseline['command_argv'][2:]
    newargv = config['original_official_command_argv'][2:]
    normalized = list(newargv)
    normalized[normalized.index('-m')+1] = oldargv[oldargv.index('-m')+1]
    require(normalized == oldargv and contract['official_sys_argv'] == newargv, 'Original official argv differs beyond model path')
    require(config['command_argv'] == config['actual_observation_child_argv'], 'Actual observer child argv differs')
    require(config['official_source_files'] == prepared['official_source_files'], 'Observed official source set differs from pinned plan')
    for item in config['official_source_files']:
        pin(item['path'], item['sha256'])
        pin(OBS / 'official-source' / item['relative_path'], item['sha256'])
    for name, key in [('observer-source.py', 'observer'), ('launcher-source.py', 'stage'), ('environment-launcher-source.py', 'run_local')]:
        pin(OBS / name, PINS[key])
        pin(PLAN / name, PINS[key])
    pin(PLAN / 'initial-raw-oracle.json', PINS['oracle'])
    for name, key in [('observe_small_city_train_single_001.py', 'observer'), ('run_small_city_observation_001.py', 'stage'),
                      ('supervisor-source.py', 'supervisor')]:
        pin(SUP / name, PINS[key])
    identity = read(OBS / 'observer/initial-raw6group-identity.json')
    require(identity.get('actual_match_passed') is True and identity['oracle_sha256'] == PINS['oracle'] and
            identity['original_coarse_row'] == 1778538 and identity['suspect_selected_prefix_row'] == 1377084 and
            set(identity['group_checks']) == {'xyz', 'f_dc', 'f_rest', 'opacity', 'scaling', 'rotation'} and
            all(v['shape_equal'] is True and v['float32_bits_equal'] is True for v in identity['group_checks'].values()),
            'Actual initial six-group uint32 equality did not pass')
    return {'actual_original_argv_only_output_path_changed': True,
            'initial_raw6group_identity': {'path': str(OBS / 'observer/initial-raw6group-identity.json'),
                                         'sha256': sha(OBS / 'observer/initial-raw6group-identity.json')},
            'actual_configuration_sha256': sha(OBS / 'configuration.json'), 'oracle_sha256': PINS['oracle']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--bind-completed-models', action='store_true')
    args = parser.parse_args()
    require(re.fullmatch(r'[a-z0-9][a-z0-9-]{0,99}', args.audit_run) is not None, 'Unsafe audit identifier')
    out = LOCK / 'binding-audits' / args.audit_run
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'binder-source.py')
    result = {'status': 'CPU_preparation_started', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'binder_source_sha256': sha(__file__), 'CUDA_initialized': False, 'Torch_imported': False,
              'native_or_training_execution_started': False, 'four_chunks_bound': False,
              'formal_observation_reclassification': False}
    try:
        records = sources()
        contract = schema(records)
        write(out / 'binding-contract.json', contract)
        if not args.bind_completed_models:
            result['status'] = 'CPU_contract_prepared_no_models_bound'
        else:
            proofs = {}
            result['read_only_completion_proofs'] = proofs
            _, hook = observed_terminal(proofs)
            proofs['observed_source_chain'] = observed_sources(hook)
            proofs['fresh_terminal_before_models'] = fresh_terminal()
            workflow = load('lineage006_workflow_cpu_helpers', OLD / 'driver-source.py')
            training = load('lineage006_ply_cpu', OLD / 'helper-snapshots/run_small_city_stage.py')
            codec = load('lineage006_hier_cpu', OLD / 'helper-snapshots/read_hierarchy.py')
            # Snapshots are relocated. Restore path metadata for pure helpers;
            # main/execute/completed_training are never invoked.
            workflow.ROOT, workflow.OUTPUT = ROOT, BASE
            workflow.REPO = ROOT / 'repos/hierarchical-3d-gaussians'
            training.ROOT, training.BASE, training.REPO = ROOT, BASE, workflow.REPO
            manifest = read(INPUT / 'manifest.json')
            split = workflow.input_split_audit(INPUT, manifest)
            coarse = bind_one(COARSE, 'coarse', None, manifest, workflow, training, codec)
            chunks = [bind_one(LEGACY[c], 'chunk', c, manifest, workflow, training, codec) for c in LEGACY]
            chunks.append(bind_one(OBS, 'chunk', '1_1', manifest, workflow, training, codec, observed=True))
            proofs['fresh_terminal_after_models'] = fresh_terminal()
            require(sha(__file__) == result['binder_source_sha256'], 'Binder source changed during audit')
            sources()
            receipt = {'status': 'actual_complete_mixed_local_training_lineage_CPU_bound',
                       'training_lineage_kind': 'three_legacy_local_plus_one_observed_local_official_training',
                       'formal_observation_reclassification': False, 'coarse': coarse, 'ordered_chunks': chunks,
                       'observed_terminal': proofs['observed_terminal_accounting'], 'split': split,
                       'source_pins': records, 'binder_source_sha256': result['binder_source_sha256'],
                       'Creator_post_Merger_or_evaluation_complete': False,
                       'native_execution_authorized_by_receipt': False,
                       'source_kind_note': 'Training origin kind is distinct from future hierarchy unoptimized/post_optimized state.',
                       'time_memory_comparison_limit': proofs['observed_terminal_accounting']['observation_overhead']}
            write(out / 'lineage-binding.json', receipt)
            result.update(status='actual_complete_mixed_local_training_lineage_CPU_bound', four_chunks_bound=True,
                          lineage_binding_path=str(out / 'lineage-binding.json'), lineage_binding_sha256=sha(out / 'lineage-binding.json'))
    except Exception as exc:
        result.update(status='CPU_binding_rejected_actual_missing_or_failed_gate', error_type=type(exc).__name__, error=str(exc))
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), Torch_imported='torch' in sys.modules,
                  numpy_imported='numpy' in sys.modules)
    write(out / 'outcome.json', result)
    print(json.dumps({'status': result['status'], 'directory': str(out), 'four_chunks_bound': result['four_chunks_bound'],
                      'error': result.get('error'), 'CUDA_initialized': False}, ensure_ascii=False))
    if result['status'] == 'CPU_binding_rejected_actual_missing_or_failed_gate':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
