"""Independent64-iteration official SmallCity post preflight, never a formal merge input.

plan/bind-source/audit-post are CPU-only. post requires --execute and parent
exclusive GPU scheduling. Real formalCreator00 inputs are copied after hash
verification into a separate pilot directory; Creator is never rerun here.
"""
import argparse
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

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
INPUT = BASE / 'official-inputs/small-city-training-002'
DEFAULT_RUN = 'small-city-post00-i64-r4-006'
DRIVER_SHA = 'a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6'
CONFIG_SHA = '562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47'
INPUT_SHA = '2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise ValueError('Invalid independent pilot run identifier')
    return value


def modules():
    if digest(LOCK / 'driver-source.py') != DRIVER_SHA or digest(LOCK / 'configuration.json') != CONFIG_SHA or digest(INPUT / 'manifest.json') != INPUT_SHA:
        raise RuntimeError('Fixed006 workflow/config or002 input changed')
    driver = load('_post64_fixed006_driver', LOCK / 'driver-source.py')
    driver.ROOT, driver.OUTPUT, driver.REPO = ROOT, BASE, ROOT / 'repos/hierarchical-3d-gaussians'
    driver.NATIVE_REPO = ROOT / 'repos/gaussian-hierarchy-large-gs'
    driver.BUILD, driver.EXES = BASE / 'hierarchy-build', driver.NATIVE_REPO / 'build-local/Release'
    driver.ENV_LAUNCHER, driver.TRAINING_PYTHON = ROOT / 'scripts/large_scene_gs/run_local.py', PYTHON
    config, manifest = read(LOCK / 'configuration.json'), read(INPUT / 'manifest.json')
    for row in config['source_records']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Frozen official runtime source or binary changed')
    snapshots = {r['name']: r for r in config['frozen_helper_records']}
    for row in snapshots.values():
        if digest(row['snapshot_path']) != row['sha256']:
            raise RuntimeError('Frozen CPU helper changed')
    codec = load('_post64_fixed_codec', snapshots['read_hierarchy.py']['snapshot_path'])
    previous = sys.modules.get('read_hierarchy')
    sys.modules['read_hierarchy'] = codec
    try:
        helper = load('_post64_fixed_cpu_freeze', snapshots['run_official_hierarchy.py']['snapshot_path'])
    finally:
        if previous is None:
            del sys.modules['read_hierarchy']
        else:
            sys.modules['read_hierarchy'] = previous
    helper.ROOT, helper.OUTPUT, helper.REPO = ROOT, BASE, driver.REPO
    helper.BUILD, helper.EXES = driver.BUILD, driver.EXES
    training = load('_post64_fixed_cpu_PLY', snapshots['run_small_city_stage.py']['snapshot_path'])
    driver.execution_source_gate()
    driver.fixed_sources()
    return driver, codec, helper, training, config, manifest


def protection(config):
    paths = [LOCK / n for n in ('configuration.json', 'plan.json', 'driver-source.py', 'cpu-binding-verification.json')]
    paths += [Path(r['snapshot_path']) for r in config['frozen_helper_records']]
    paths += [Path(r['path']) for r in config['source_records']]
    paths += [ROOT / 'scripts/large_scene_gs/run_small_city_stage.py', ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py']
    paths += [Path(config['training_lineage_binding_path'])]
    return {str(p): digest(p) for p in sorted(set(paths), key=str)}


def require_unchanged(records):
    if any(digest(p) != value for p, value in records.items()):
        raise RuntimeError('Protected formal/source bytes changed')


def pilot_path(run):
    base = BASE / 'official-post-preflight'
    path = base / identifier(run)
    if not path.resolve().is_relative_to(base.resolve()) or path.resolve().is_relative_to(LOCK.resolve()):
        raise RuntimeError('Pilot must stay in its independent namespace')
    return path


def training_binding(driver, config, manifest, training):
    _, codec, _training, _ = driver.frozen_helpers(LOCK)
    driver.lineage_gate(config, manifest, training, codec)
    coarse, coarse_receipt, _ = driver.completed_training(config['coarse_run'], 'coarse', None, config, manifest, training)
    chunk_run = config['chunk_runs']['0_0']
    chunk, chunk_receipt, chunk_config = driver.completed_training(chunk_run, 'chunk', '0_0', config, manifest, training)
    exposure = driver.exposure_audit(BASE / 'official-training' / chunk_run / 'model/exposure.json', chunk_config['training_names'], manifest['test_names'])
    actual_exposure = read(BASE / 'official-training' / chunk_run / 'model-audit.json')['exposure']
    if exposure['sha256'] != actual_exposure['sha256']:
        raise RuntimeError('Actual trained chunk exposure changed after its stage audit')
    return coarse, chunk, coarse_receipt, chunk_receipt, exposure


def source_binding(driver, codec, helper, training, config, manifest):
    source = LOCK / 'trained_chunks/0_0'
    required = [source / name for name in ('creator-outcome.json', 'hierarchy.hier', 'anchors.bin', 'exposure.json')]
    if not all(path.is_file() for path in required):
        raise RuntimeError('Actual formal006 Creator00 receipt/hierarchy/anchors/exposure not available; no author tree or CPU plan substitute')
    coarse, chunk, coarse_receipt, chunk_receipt, trained_exposure = training_binding(driver, config, manifest, training)
    creator = driver.creator_gate('0_0', config, manifest, training, codec, helper, read(LOCK / 'plan.json')['camera_archives'])
    if (creator['status'] != 'completed' or creator['exit_code'] != 0 or creator['source_kind'] != 'unoptimized'
            or creator['configuration_sha256'] != CONFIG_SHA or creator['source_training_receipt'] != chunk_receipt
            or creator['input_model']['sha256'] != chunk_receipt['model_parameter_sha256']
            or creator['training_lineage_binding_sha256'] != driver.BINDING_SHA
            or creator['training_lineage_binding_path'] != str(driver.BINDING_PATH)):
        raise RuntimeError('Creator00 actual local training/input/config provenance failed')
    archives = read(LOCK / 'plan.json')['camera_archives']['0_0']
    if creator['creator_train_camera_archive'] != archives or archives['heldout_pose_count'] != 0:
        raise RuntimeError('Creator00 did not use the frozen train-only camera archive')
    for row in archives['files']:
        if digest(row['path']) != row['sha256']:
            raise RuntimeError('Creator train-only camera archive changed')
    expected_command = [str(driver.EXES / 'GaussianHierarchyCreator.exe'), str(chunk / 'point_cloud.ply'),
                        str(LOCK / 'creator-camera-archives/0_0'), str(source), str(coarse)]
    if creator['command_argv'] != expected_command:
        raise RuntimeError('Actual Creator argv differs from selected local input/archive/scaffold')
    files = {name: digest(source / name) for name in ('hierarchy.hier', 'anchors.bin', 'exposure.json')}
    if files['hierarchy.hier'] != creator['hierarchy_sha256'] or files['anchors.bin'] != creator['anchor_sha256']:
        raise RuntimeError('Real Creator tree or anchors changed')
    exposure = driver.exposure_audit(source / 'exposure.json', next(c['train_names'] for c in manifest['chunks'] if c['name'] == '0_0'), manifest['test_names'])
    if files['exposure.json'] != creator['copied_exposure_sha256'] or exposure['sha256'] != trained_exposure['sha256']:
        raise RuntimeError('Creator exposure differs from the actual trained chunk matrices')
    h = codec.read_hierarchy(source / 'hierarchy.hier')
    anchors = codec.read_anchors(source / 'anchors.bin', len(h.positions))
    if len(anchors) != creator['anchor_count'] or h.summary() != creator['hierarchy']:
        raise RuntimeError('Actual Creator hierarchy count/anchor decoding differs')
    numeric = helper.hierarchy_numeric_audit(h)
    shared_sky = driver.shared_sky_audit(coarse, INPUT / 'chunks/0_0', chunk, training, codec)
    if shared_sky != creator['shared_skybox']:
        raise RuntimeError('Creator common raw100k sky provenance differs')
    return {'status': 'actual_formalCreator00_train_source_tree_anchor_exposure_shared_sky_CPU_bound',
            'source_directory': str(source), 'source_creator_receipt_sha256': digest(source / 'creator-outcome.json'),
            'source_file_sha256': files, 'source_hierarchy_summary': h.summary(), 'source_anchor_count': len(anchors),
            'actual_source_creator_command': creator['command_argv'], 'actual_coarse_directory': str(coarse),
            'actual_coarse_model_sha256': coarse_receipt['model_parameter_sha256'],
            'actual_chunk_model_sha256': chunk_receipt['model_parameter_sha256'], 'trained_exposure': exposure,
            'raw_shared100k_sky': shared_sky, 'numeric_audit': numeric, 'GPU_started': False,
            'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA}


def no_live_gpu_stage():
    driver, _, _, _, _, _ = modules()
    return driver.no_conflicting_gpu_stage()


def audit_post(root, driver, codec, helper, training, config, manifest):
    prepared_config = read(root / 'configuration.json')
    if (digest(ROOT / 'scripts/large_scene_gs/run_small_city_post_preflight_006.py') != prepared_config['wrapper_source_sha256']
            or digest(root / 'wrapper-source.py') != prepared_config['wrapper_source_sha256']):
        raise RuntimeError('Independent006 post64 wrapper source changed during stage')
    binding = source_binding(driver, codec, helper, training, config, manifest)
    prepared = read(root / 'source-binding.json')
    if prepared != binding:
        raise RuntimeError('Actual formal source differs from its preflight binding')
    protected = read(root / 'protected-formal-files.json')
    require_unchanged(protected)
    directory = root / 'creator-input'
    for name, expected in binding['source_file_sha256'].items():
        if digest(directory / name) != expected:
            raise RuntimeError('Preflight changed its copied Creator/anchor/frozen exposure input')
    command = driver.post_command(directory, INPUT / 'chunks/0_0', Path(binding['actual_coarse_directory']), manifest, 64)
    receipt = read(root / 'outcome.json')
    if receipt['status'] != 'process_exited_zero' or receipt['exit_code'] != 0 or receipt['command_argv'] != command:
        raise RuntimeError('CPU post audit needs actual matching exit0 preflight receipt')
    text = (directory / 'post-stage/stage.log').read_text(encoding='utf-8', errors='replace')
    if '[ITER 64] Saving Gaussians' not in text or 'Training complete.' not in text:
        raise RuntimeError('Actual64 final-save/normal-return evidence missing')
    before = codec.read_hierarchy(directory / 'hierarchy.hier')
    after = codec.read_hierarchy(directory / 'hierarchy.hier_opt', expected_unowned_tail=100000)
    anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
    freeze = helper.post_freeze_audit(before, after, anchors, Path(binding['actual_coarse_directory']), 100000)
    numeric = helper.hierarchy_numeric_audit(after)
    no_live_gpu_stage()
    require_unchanged(protected)
    receipt.update(status='completed_post64_preflight_only', requested_iterations=64, optimizer_update_calls=63,
                   update_count_scope='Derived from unchanged official final backward/save64 then return before optimizer step64, exit0 and matching final-save log; not optimizer instrumentation',
                   source_kind='post64_preflight', eligible_for_formal_merge=False, formal15000_post_complete=False,
                   formal_source_configuration_sha256=CONFIG_SHA, copied_input_hashes=binding['source_file_sha256'],
                   training_lineage_binding_path=str(driver.BINDING_PATH), training_lineage_binding_sha256=driver.BINDING_SHA,
                   training_lineage_kind=config['training_lineage_kind'], formal_observation_reclassification=False,
                   output_hierarchy=str(after.path), output_hierarchy_sha256=digest(after.path), hierarchy=after.summary(),
                   frozen_fields=freeze, numeric_audit=numeric, frozen_trained_exposure=binding['trained_exposure'],
                   depth_loss_present=False, exposure_optimizer_updates=0, shared_sky_tail=100000,
                   CUDA_initialized_by_CPU_audit=False, protected_formal_files_unchanged=True, audited_utc=now())
    write(root / 'post-freeze-audit.json', freeze)
    write(root / 'outcome.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['plan', 'bind-source', 'post', 'audit-post'], required=True)
    parser.add_argument('--run', default=DEFAULT_RUN, type=identifier)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--revise-unexecuted-plan', action='store_true')
    args = parser.parse_args()
    if args.run != DEFAULT_RUN:
        raise RuntimeError('006 pilot namespace is fixed; independent64 cannot write another or formal namespace')
    if args.stage == 'post' and not args.execute:
        raise ValueError('GPU post requires --execute after exclusive parent authorization')
    if args.stage != 'post' and args.execute:
        raise ValueError('--execute is only valid for the actual GPU post stage')
    if args.revise_unexecuted_plan and args.stage != 'plan':
        raise ValueError('Only a CPU plan with no stage receipts may be explicitly revised')
    root = pilot_path(args.run)
    driver, codec, helper, training, formal, manifest = modules()
    protected = protection(formal)
    coarse, chunk, coarse_receipt, chunk_receipt, exposure = training_binding(driver, formal, manifest, training)
    source = LOCK / 'trained_chunks/0_0'
    static = {'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,
              'training_lineage_kind': formal['training_lineage_kind'], 'formal_observation_reclassification': False,
              'wrapper_source_sha256': digest(__file__), 'formal_source_configuration_sha256': CONFIG_SHA,
              'formal_workflow_driver_sha256': DRIVER_SHA, 'input_manifest_sha256': INPUT_SHA,
              'chunk': '0_0', 'actual_coarse_run': formal['coarse_run'], 'actual_chunk_run': formal['chunk_runs']['0_0'],
              'actual_coarse_model_sha256': coarse_receipt['model_parameter_sha256'], 'actual_chunk_model_sha256': chunk_receipt['model_parameter_sha256'],
              'actual_chunk_exposure_sha256': exposure['sha256'], 'requested_iterations': 64, 'expected_optimizer_update_calls': 63,
              'source_kind': 'post64_preflight', 'eligible_for_formal_merge': False,
              'formal15000_configuration_or_outputs_modified': False, 'fresh_Creator_executed': False,
              'actual_source_Creator_directory': str(source), 'independent_output_directory': str(root)}
    configuration_path = root / 'configuration.json'
    if args.stage == 'plan':
        if root.exists():
            allowed = {'configuration.json','plan.json','wrapper-source.py','protected-formal-files.json','CPU-plan-history'}
            if not args.revise_unexecuted_plan or set(p.name for p in root.iterdir()) - allowed:
                raise RuntimeError('Pilot namespace already exists or has stage receipts; refuse plan overwrite')
            old_sha = digest(root / 'configuration.json')
            historical = root / 'CPU-plan-history' / old_sha
            historical.mkdir(parents=True,exist_ok=False)
            for name in allowed-{'CPU-plan-history'}:
                shutil.copyfile(root/name,historical/name)
        else:
            root.mkdir(parents=True, exist_ok=False)
        command = driver.post_command(root / 'creator-input', INPUT / 'chunks/0_0', coarse, manifest, 64)
        pure = load('_post64_pure_official_args', driver.REPO / 'arguments/__init__.py')
        p = argparse.ArgumentParser()
        pure.ModelParams(p); pure.OptimizationParams(p); pure.PipelineParams(p)
        p.add_argument('--save_iterations', nargs='+', type=int); p.add_argument('--disable_viewer', action='store_true')
        effective = vars(p.parse_args(command[3:]))
        if effective['iterations'] != 64 or effective['resolution'] != 4 or not effective['eval'] or effective['train_test_exp']:
            raise RuntimeError('Official post64/r4/train-only CLI parse failed')
        write(configuration_path, static)
        shutil.copyfile(__file__, root / 'wrapper-source.py')
        write(root / 'protected-formal-files.json', protected)
        available = all((source / name).is_file() for name in ('creator-outcome.json', 'hierarchy.hier', 'anchors.bin', 'exposure.json'))
        write(root / 'plan.json', {'status': 'CPU_prepared_actual_Creator_dependency_pending' if not available else 'CPU_prepared_source_Creator_needs_actual_binding',
              'GPU_started': False, 'configuration_sha256': digest(configuration_path), 'actual_Creator_available': available,
              'actual_Creator_tree_or_anchor_sha256': None, 'actual_existing_coarse_and_chunk00_training_gates_passed': True,
              'actual_trained_exposure': exposure, 'official_post_command_preview': command, 'effective_official_arguments': effective,
              'future_input_policy': 'Copy only real verified formal006 Creator00 hierarchy.hier/anchors.bin/exposure.json and recheck exact SHA; never rerunCreator or use author/preflight tree.',
              'future_output_policy': 'Only independent pilot creator-input/hierarchy.hier_opt; source_kindpost64_preflight, not eligible for formal006Merger or15000post completion.',
              'post_protocol': 'Unchanged official photo L1+SSIM post; actual trained chunk exposure loaded/frozen/use_trained_expTrue; no depth loss/no exposure steps;100k coarse sky appendedonce, frozen anchors and half-writer sky roundtrip verified.',
              'preGPU_policy': 'Parent exclusive scheduling only after mixed-source all5 full models terminal; exact006 active ownership chain; fixed runtime,80GiB reserve, fresh Python and nativeCreator process observation before dispatch.',
              'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,
              'prior_Truck_method_reference': str(BASE / 'official-hierarchy/truck-official-hierarchy-post-preflight-001/trained_chunks/0_0/post-outcome.json')})
        require_unchanged(protected)
        print(json.dumps({'status': 'CPU_prepared_post64_plan_no_GPU', 'root': str(root), 'actual_Creator_available': available}))
        return
    if not configuration_path.is_file() or read(configuration_path) != static:
        raise RuntimeError('Prepared pilot config/wrapper/source/training lineage changed')
    require_unchanged(read(root / 'protected-formal-files.json'))
    if args.stage == 'audit-post':
        print(json.dumps(audit_post(root, driver, codec, helper, training, formal, manifest)))
        return
    binding = source_binding(driver, codec, helper, training, formal, manifest)
    if args.stage == 'bind-source':
        path = root / 'source-binding-CPU.json'
        if path.exists():
            raise RuntimeError('Existing CPU source binding; refuse overwrite')
        write(path, binding)
        require_unchanged(protected)
        print(json.dumps({'status': binding['status'], 'GPU_started': False}))
        return
    driver.require_runtime()
    driver.execution_source_gate()
    process_gate = no_live_gpu_stage()
    for chunk_name, run in formal['chunk_runs'].items():
        driver.completed_training(run, 'chunk', chunk_name, formal, manifest, training)
    directory = root / 'creator-input'
    if directory.exists() or (root / 'outcome.json').exists():
        raise RuntimeError('Pilot actual stage already started; no overwrite/retry')
    directory.mkdir(exist_ok=False)
    for name, expected in binding['source_file_sha256'].items():
        shutil.copyfile(source / name, directory / name)
        if digest(directory / name) != expected or digest(source / name) != expected:
            raise RuntimeError('Actual Creator source-to-pilot copy hash differs')
    write(root / 'source-binding.json', binding)
    write(root / 'preGPU-process-audit.json', process_gate)
    (directory / 'post-stage').mkdir(exist_ok=False)
    command = driver.post_command(directory, INPUT / 'chunks/0_0', coarse, manifest, 64)
    receipt = {'status': 'running_post64_preflight', 'started_utc': now(), 'command_argv': command,
               'source_kind': 'post64_preflight', 'eligible_for_formal_merge': False,
               'requested_iterations': 64, 'expected_optimizer_update_calls': 63, 'cwd': str(driver.REPO),
               'training_lineage_binding_path': str(driver.BINDING_PATH), 'training_lineage_binding_sha256': driver.BINDING_SHA,
               'wrapper_pid': os.getpid(), 'official_algorithm_modified': False}
    write(root / 'outcome.json', receipt)
    start = time.perf_counter()
    with (directory / 'post-stage/stage.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=driver.REPO, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        receipt['child_pid'] = child.pid
        receipt['actual_child_birth'] = driver.process_birth(child.pid, owned_handle=child._handle)
        write(root / 'outcome.json', receipt)
        code = child.wait()
    receipt.update(status='process_exited_zero' if code == 0 else 'failed_post64_preflight_preserved', exit_code=code,
                   elapsed_seconds=time.perf_counter()-start, process_ended_utc=now(), actual_child_poll_exit_code=child.poll())
    write(root / 'outcome.json', receipt)
    if code:
        raise SystemExit(code)
    result = audit_post(root, driver, codec, helper, training, formal, manifest)
    print(json.dumps({'status': result['status'], 'optimizer_update_calls': 63, 'eligible_for_formal_merge': False}))


if __name__ == '__main__':
    main()
