"""CPU supervisor for one explicitly selected hierarchy006 stage.

Default writes a CPU plan. --check-dispatch is read-only. Only --execute
starts one child. No retries, next-stage dispatch, or process termination.
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
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
SUPERVISION = BASE / 'official-hierarchy-stage-supervision-006'
ACTIVE = SUPERVISION / 'active-stage.lock.json'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'
LOCAL = ROOT / 'scripts/large_scene_gs/run_local.py'
DRIVER = ROOT / 'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
PILOT = ROOT / 'scripts/large_scene_gs/run_small_city_post_preflight_006.py'
PILOT_RUN = 'small-city-post00-i64-r4-006'
PILOT_PIN = SUPERVISION / 'source-lock/post64-source-lock.json'
BINDING = LOCK / 'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
BINDING_SHA = '1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'
DRIVER_SHA = 'a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6'
CONFIG_SHA = '562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47'
PLAN_SHA = '8a59bbfc69ebe9e7bcaf45b82403cb6474c889abd99ab2c190ccce6fb53ead5f'
LOCAL_SHA = '1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2'
OBSERVER = ROOT / 'scripts/large_scene_gs/observe_small_city_process_roles.py'
OBSERVER_SHA = 'cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a'
JOB_HELPER = ROOT / 'scripts/large_scene_gs/supervise_small_city_stage.py'
JOB_HELPER_SHA = 'b7c0015374521b201a4e28845cb9ff79ca9b4c597d5c05f80bf69aa03f58378d'
PW = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
PW_ENV = os.environ.copy()
PW_ENV['PSModulePath'] = os.pathsep.join([str(PW.parent / 'Modules'), str(Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/Modules')])
CHUNKS = ('0_0', '0_1', '1_0', '1_1')


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
    if not re.fullmatch(r'[A-Za-z0-9-]{1,100}', value):
        raise ValueError('Invalid supervision run')
    return value


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def pilot_source_pin():
    pin = read(PILOT_PIN)
    require(pin['wrapper_path'] == str(PILOT) and pin['run'] == PILOT_RUN
            and pin['native_driver_sha256'] == DRIVER_SHA and pin['formal_configuration_sha256'] == CONFIG_SHA
            and pin['training_lineage_binding_sha256'] == BINDING_SHA
            and re.fullmatch('[a-f0-9]{64}', pin['wrapper_sha256']) is not None,
            'Independent post64 source pin differs from fixed006 source/config/lineage')
    require(digest(PILOT) == pin['wrapper_sha256'], 'Actual separate post64 source changed')
    return pin


def verify_record_tree(value):
    """Rehash every actual path/hash record, without treating strings as proof."""
    count = 0
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            require(digest(value['path']) == value['sha256'], 'Actual lineage/source path SHA changed: ' + value['path'])
            count += 1
        for child in value.values():
            count += verify_record_tree(child)
    elif isinstance(value, list):
        for child in value:
            count += verify_record_tree(child)
    return count


def fixed_sources(stage):
    pairs = [(DRIVER, DRIVER_SHA), (LOCK / 'driver-source.py', DRIVER_SHA),
             (LOCK / 'configuration.json', CONFIG_SHA), (BINDING, BINDING_SHA),
             (LOCK / 'plan.json', PLAN_SHA),
             (LOCAL, LOCAL_SHA), (OBSERVER, OBSERVER_SHA), (JOB_HELPER, JOB_HELPER_SHA)]
    if stage == 'post64':
        pairs.append((PILOT, pilot_source_pin()['wrapper_sha256']))
    for path, expected in pairs:
        require(digest(path) == expected, 'Fixed006 source/config/binding changed: ' + str(path))
    driver = load('_supervisor006_fixed_driver', DRIVER)
    config = read(LOCK / 'configuration.json')
    binding = read(BINDING)
    require(binding['status'] == 'actual_complete_mixed_local_training_lineage_CPU_bound', 'Actual completed006 lineage required')
    require(not binding['formal_observation_reclassification'], 'Observed training must not be reclassified formal')
    require(tuple(r['chunk'] for r in binding['ordered_chunks']) == CHUNKS, 'Actual fixed chunk order differs')
    require(binding['ordered_chunks'][3]['training_source_kind'] == 'observed_local_official_training'
            and binding['ordered_chunks'][3]['formal_training_run'] is False, 'Observed11 origin lost')
    for row in binding['ordered_chunks'][:3]:
        require(row['actual_Gaussian_Adam_calls'] is None and row['derived_Gaussian_call_upper_bound'] == 29999,
                'Legacy actual Gaussian calls unknown;29999 is only derived upper bound')
    terminal = binding['observed_terminal']
    require(terminal['actual_Gaussian_Adam_calls'] == 29951 and terminal['actual_None_gradient_skips'] == 48
            and terminal['actual_exposure_Adam_calls'] == 29999
            and terminal['actual_opacity_grad_condition_visits'] == 29999,
            'Observed original conditional Adam accounting changed')
    require(binding['ordered_chunks'][3]['model']['sha256'] == '367bef34e584616daeda938e2e0b5826a8e4bbf30d13b4d987db32c9f72abcb4', 'Observed physical final model identity differs')
    record_count = verify_record_tree(binding)
    for row in [binding['coarse'], *binding['ordered_chunks']]:
        require(row['requested_iterations'] == 30000 and row['model']['raw_finite']
                and row['model']['exp_f32_scales_finite_positive'] and row['model']['wxyz_norm_positive'],
                'Actual full30000 finite trained input required')
        require(digest(Path(row['training_directory']) / 'model-audit.json') == row['model_audit_sha256'], 'Actual model audit changed')
        require(digest(Path(row['training_directory']) / 'training.log') == row['training_log_sha256'], 'Actual training log changed')
    for row in config['source_records']:
        require(digest(row['path']) == row['sha256'], 'Pinned official runtime source/native executable changed')
    for row in config['frozen_helper_records']:
        require(digest(row['snapshot_path']) == row['sha256'], 'Frozen CPU helper changed')
    manifest_path = BASE / 'official-inputs/small-city-training-002/manifest.json'
    require(digest(manifest_path) == config['input_manifest_sha256'], 'Actual input manifest changed')
    manifest = read(manifest_path)
    require(config['training_lineage_binding_sha256'] == BINDING_SHA
            and config['post_iterations'] == 15000 and config['expected_post_optimizer_calls'] == 14999,
            'Actual006 binding/formalpost budget changed')
    inputs = manifest_path.parent
    require(digest(inputs / 'frozen-split.json') == manifest['frozen_split_sha256'], 'Actual frozen split changed')
    for copy in read(manifest['source_copy_audit']):
        require(digest(copy['destination']) == copy['destination_sha256'], 'Actual aligned/chunk metadata changed')
    for archive in read(LOCK / 'plan.json')['camera_archives'].values():
        require(archive['heldout_pose_count'] == 0, 'Creator camera archive contains heldout poses')
        for file in archive['files']:
            require(digest(file['path']) == file['sha256'], 'Actual train-only Creator archive changed')
    driver.fixed_sources()
    return driver, config, binding, manifest, record_count


def command_for(stage, chunk, config):
    prefix = [str(PYTHON), '-u', str(LOCAL)]
    if stage == 'post64':
        return prefix + [str(PILOT), '--stage', 'post', '--run', PILOT_RUN, '--execute']
    argv = prefix + [str(DRIVER), '--stage', stage, '--run', LOCK.name,
                     '--input', config['input'], '--coarse-run', config['coarse_run']]
    for name in CHUNKS:
        argv += ['--chunk-run', name + '=' + config['chunk_runs'][name]]
    if chunk:
        argv += ['--chunk', chunk]
    return argv + ['--execute']


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
    require(receipt['status'] == 'completed' and receipt['exit_code'] == 0 and receipt['source_kind'] == kind
            and receipt['configuration_sha256'] == CONFIG_SHA and digest(model) == receipt['hierarchy_sha256'],
            'Actual prior/final stage receipt/model/config failed')
    require(receipt.get('training_lineage_binding_sha256') == BINDING_SHA, 'Prior/final stage does not bind real006 lineage')
    if kind == 'post_optimized' and Path(path).name == 'post-outcome.json':
        require(receipt['requested_iterations'] == 15000 and receipt['optimizer_update_calls'] == 14999,
                'Formal Merger inputs require post15000/source-derived14999; no64-step pilot')
    return receipt


def preconditions(stage, chunk, driver, config, binding, manifest):
    receipt, output, _ = stage_target(stage, chunk)
    require(not receipt.exists() and not output.exists(), 'Actual stage receipt/output exists; refuse overwrite/retry')
    if stage == 'creator':
        require(not receipt.parent.exists(), 'Creator target directory exists; refuse overwrite/retry')
    elif stage == 'post64':
        root = BASE / 'official-post-preflight' / PILOT_RUN
        require(not (root / 'creator-input').exists(), 'Pilot copied/started already; refuse retry')
        pilot = load('_supervisor006_post64', PILOT)
        modules = pilot.modules()
        require(read(root / 'configuration.json')['wrapper_source_sha256'] == pilot_source_pin()['wrapper_sha256'], 'Prepared pilot source changed')
        pilot.source_binding(*modules)
    elif stage == 'post':
        directory = LOCK / 'trained_chunks' / chunk
        creator = require_stage_receipt(directory / 'creator-outcome.json', directory / 'hierarchy.hier', 'unoptimized')
        require(digest(directory / 'anchors.bin') == creator['anchor_sha256']
                and digest(directory / 'exposure.json') == creator['copied_exposure_sha256'], 'Actual Creator anchors/exposure changed')
        require(not (directory / 'post-stage').exists(), 'Formal post directory exists; refuse retry')
        helper, codec, training, _ = driver.frozen_helpers(LOCK)
        archives = read(LOCK / 'plan.json')['camera_archives']
        driver.creator_gate(chunk, config, manifest, training, codec, helper, archives)
    else:
        require(not (LOCK / (stage + '-stage')).exists(), 'Merger stage directory exists; refuse retry')
        post = stage == 'merge'
        for name in CHUNKS:
            directory = LOCK / 'trained_chunks' / name
            require_stage_receipt(directory / ('post-outcome.json' if post else 'creator-outcome.json'),
                                  directory / ('hierarchy.hier_opt' if post else 'hierarchy.hier'),
                                  'post_optimized' if post else 'unoptimized')
        helper, codec, training, _ = driver.frozen_helpers(LOCK)
        archives = read(LOCK / 'plan.json')['camera_archives']
        for name in CHUNKS:
            (driver.post_gate if post else driver.creator_gate)(name, config, manifest, training, codec, helper, archives)


def process_snapshot():
    command = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe' OR Name='GaussianHierarchyCreator.exe' OR Name='GaussianHierarchyMerger.exe' OR Name='GaussianHierarchyMergerUnoptimized.exe'\" | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run([str(PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command], env=PW_ENV,
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    require(result.returncode == 0, 'Actual Get-Cim process observation failed: ' + result.stderr)
    data = json.loads(result.stdout or '[]')
    return data if isinstance(data, list) else [data]


def busy_processes(snapshot):
    observer = load('_supervisor006_exact_argv', OBSERVER)
    wrapper_scripts = {'run_small_city_stage.py', 'run_small_city_observation_001.py',
                       'run_small_city_hierarchy.py', 'run_small_city_hierarchy_006.py',
                       'run_small_city_post_preflight_001.py', 'run_small_city_post_preflight_006.py'}
    busy = []
    for process in snapshot:
        if process['Name'].lower() not in {'python.exe', 'pythonw.exe'}:
            busy.append(process)
            continue
        argv = observer.windows_argv(process.get('CommandLine') or '')
        names = {Path(a).name.lower() for a in argv[1:]}
        if (any(re.fullmatch(r'train_[a-z0-9_]+\.py', n) for n in names)
                or any(re.fullmatch(r'evaluate_(coarse|city|official_hierarchy|local_city_hierarchy(?:_006)?)\.py', n) for n in names)
                or (names & wrapper_scripts and '--execute' in argv)):
            busy.append(process)
    return busy


def dispatch_gate(stage, chunk, own_lock=None):
    if ACTIVE.exists():
        require(own_lock is not None and read(ACTIVE) == own_lock, 'An active006 stage lock exists; no restart or concurrent dispatch')
    driver, config, binding, manifest, record_count = fixed_sources(stage)
    driver.execution_source_gate()
    helper, codec, training, _ = driver.frozen_helpers(LOCK)
    lineage = driver.lineage_gate(config, manifest, training, codec)
    require(lineage['all_five_saved_model_bindings_rechecked'] and not lineage['GPU_started'],
            'Actual all5 trained input CPU gate failed')
    preconditions(stage, chunk, driver, config, binding, manifest)
    free = shutil.disk_usage(ROOT).free
    require(free >= 80 * 1024**3, 'System-drive reserve below80GiB')
    snapshot = process_snapshot()
    require(not busy_processes(snapshot), 'Actual trainer/Creator/post/Merger/evaluator still live; dispatch refused')
    return {'status': 'actual006_source_lineage_input_prior_stage_disk_noGPU_gates_passed',
            'observed_utc': now(), 'stage': stage, 'chunk': chunk,
            'lineage_binding_path': str(BINDING), 'lineage_binding_sha256': BINDING_SHA,
            'actual_rehashed_lineage_records': record_count, 'actual_all5_model_binding_recheck': {
                'all_five_saved_model_bindings_rechecked': lineage['all_five_saved_model_bindings_rechecked'],
                'training_lineage_binding_sha256': lineage['training_lineage_binding_sha256'], 'GPU_started': False},
            'actual_training_models': [
                {'chunk': r.get('chunk'), 'training_source_kind': r['training_source_kind'], 'model': r['model'],
                 'actual_Gaussian_Adam_calls': r['actual_Gaussian_Adam_calls']} for r in [binding['coarse'], *binding['ordered_chunks']]],
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
    return {'pid': child.pid, 'creation_FILETIME_100ns': ticks, 'creation_Unix_ms': (ticks - 116444736000000000) // 10000,
            'identity_source': 'real subprocess.Popen process handle plus Windows GetProcessTimes'}


def current_process_identity():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetCommandLineW.restype = wintypes.LPWSTR
    observer = load('_supervisor006_self_exact_argv', OBSERVER)
    class OwnHandle:
        pid = os.getpid()
        _handle = kernel.GetCurrentProcess()
    return child_birth(OwnHandle()), observer.windows_argv(kernel.GetCommandLineW())


def creation_ms(value):
    match = re.match(r'/Date\((-?\d+)', str(value))
    if match:
        return int(match.group(1))
    normalized = re.sub(r'(\.\d{6})\d+(?=Z|[+-]\d\d:\d\d|$)', r'\1', str(value)).replace('Z', '+00:00')
    instant = datetime.fromisoformat(normalized).astimezone(timezone.utc)
    difference = instant - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (difference.days * 86400 + difference.seconds) * 1000 + difference.microseconds // 1000


def exact_child_argv(actual, expected):
    return (len(actual) == len(expected) and os.path.normcase(str(Path(actual[0]).resolve())) == os.path.normcase(str(PYTHON.resolve()))
            and actual[1:] == expected[1:])


def observed_descendants(snapshot, child_pid):
    pids = {child_pid}
    previous = -1
    while previous != len(pids):
        previous = len(pids)
        pids.update(p['ProcessId'] for p in snapshot if p['ParentProcessId'] in pids)
    return [p for p in snapshot if p['ProcessId'] in pids]


def live_seen_descendants(snapshot, seen):
    return [p for p in snapshot if (p['ProcessId'], creation_ms(p['CreationDate'])) in seen]


def final_receipt(stage, chunk):
    path, model, kind = stage_target(stage, chunk)
    driver, config, binding, manifest, _ = fixed_sources(stage)
    driver.no_conflicting_gpu_stage()
    if stage == 'post64':
        receipt = read(path)
        require(receipt['status'] == 'completed_post64_preflight_only' and receipt['exit_code'] == 0
                and receipt['source_kind'] == kind and receipt['eligible_for_formal_merge'] is False
                and receipt['requested_iterations'] == 64 and receipt['optimizer_update_calls'] == 63
                and digest(model) == receipt['output_hierarchy_sha256'], 'Actual pilot terminal receipt/budget/modelSHA failed')
        require(receipt.get('training_lineage_binding_sha256') == BINDING_SHA, 'Pilot terminal lineage failed')
    else:
        receipt = require_stage_receipt(path, model, kind)
        if stage == 'creator':
            ref = next(r for r in binding['ordered_chunks'] if r['chunk'] == chunk)
            require(receipt['input_model']['sha256'] == ref['model']['sha256']
                    and receipt['source_training_receipt'] == read(ref['training_receipt']['path']), 'Creator input actual training provenance failed')
            require(receipt['source_exposure']['sha256'] == ref['exposure']['sha256']
                    and receipt['copied_exposure_sha256'] == ref['exposure']['sha256']
                    and digest(model.parent / 'exposure.json') == ref['exposure']['sha256']
                    and digest(model.parent / 'anchors.bin') == receipt['anchor_sha256'], 'Creator actual exposure/anchor provenance failed')
            expected_creator = read(LOCK / 'plan.json')['creator_command_previews'][chunk]
            require(receipt['command_argv'] == expected_creator, 'Creator actual five-argument command differs')
            helper, codec, training, _ = driver.frozen_helpers(LOCK)
            driver.creator_gate(chunk, config, manifest, training, codec, helper, read(LOCK / 'plan.json')['camera_archives'])
        if stage == 'post':
            require(receipt['requested_iterations'] == 15000 and receipt['optimizer_update_calls'] == 14999,
                    'Actual formal post15000/source-derived14999 receipt failed')
            helper, codec, training, _ = driver.frozen_helpers(LOCK)
            driver.post_gate(chunk, config, manifest, training, codec, helper, read(LOCK / 'plan.json')['camera_archives'])
    codec_row = next(r for r in config['frozen_helper_records'] if r['name'] == 'read_hierarchy.py')
    codec = load('_supervisor006_final_decoder', codec_row['snapshot_path'])
    h = codec.read_hierarchy(model, expected_unowned_tail=100000 if stage in ('post', 'post64') else 0)
    require(h.summary() == receipt['hierarchy'], 'Actual terminal model storage differs from receipt')
    return {'receipt_path': str(path), 'receipt_sha256': digest(path), 'model_path': str(model),
            'model_sha256': digest(model), 'source_kind': kind, 'actual_receipt': receipt, 'decoded_summary': h.summary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['creator', 'post64', 'post', 'merge-unoptimized', 'merge'])
    parser.add_argument('--chunk', choices=CHUNKS)
    parser.add_argument('--supervision-run', required=True, type=identifier)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--check-dispatch', action='store_true')
    args = parser.parse_args()
    require((args.stage in ('creator', 'post', 'post64')) == (args.chunk is not None)
            and (args.stage != 'post64' or args.chunk == '0_0'), 'Exactly select one stage chunk;post64 is00-only;Merger has no chunk')
    driver, config, binding, manifest, _ = fixed_sources(args.stage)
    root = SUPERVISION / args.supervision_run
    static = {'supervisor_source_sha256': digest(__file__), 'stage': args.stage, 'chunk': args.chunk,
              'command_argv': command_for(args.stage, args.chunk, config), 'fixed_driver_sha256': DRIVER_SHA,
              'fixed_pilot_sha256': pilot_source_pin()['wrapper_sha256'] if args.stage == 'post64' else None,
              'pilot_source_pin_sha256': digest(PILOT_PIN) if args.stage == 'post64' else None,
              'formal_configuration_sha256': CONFIG_SHA,
              'formal_CPU_plan_sha256': PLAN_SHA,
              'lineage_binding_path': str(BINDING), 'lineage_binding_sha256': BINDING_SHA,
              'input_manifest_sha256': config['input_manifest_sha256'], 'run_local_sha256': LOCAL_SHA,
              'one_stage_only': True, 'automatic_retry_or_next_stage': False, 'parent_exclusive_dispatch_required': True}
    if args.check_dispatch:
        print(json.dumps(dispatch_gate(args.stage, args.chunk), ensure_ascii=False))
        return
    if not args.execute:
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'configuration.json', static)
        shutil.copyfile(__file__, root / 'supervisor-source.py')
        write(root / 'plan.json', {'status': 'CPU_single_stage_plan_only_GPU_not_started', 'GPU_started': False,
              'actual_completed_lineage_sha256': BINDING_SHA,
              'required_fresh_dispatch_gates': 'actual sources/config/input/model SHA, prior-stage receipts,80GiB,no live GPU stages and exclusive006 atomic lock',
              'durability_limit': 'No survival claim after app/job closure,logout or reboot.'})
        print(json.dumps({'status': 'CPU_single_stage_plan_prepared_no_GPU', 'root': str(root)}))
        return
    if root.exists():
        require(set(p.name for p in root.iterdir()) <= {'configuration.json', 'supervisor-source.py', 'plan.json'}
                and read(root / 'configuration.json') == static and digest(root / 'supervisor-source.py') == digest(__file__),
                'Existing runtime files or changed CPU plan; refuse overwrite/retry')
    else:
        root.mkdir(parents=True, exist_ok=False)
        write(root / 'configuration.json', static)
        shutil.copyfile(__file__, root / 'supervisor-source.py')
    SUPERVISION.mkdir(parents=True, exist_ok=True)
    supervisor_birth, supervisor_argv = current_process_identity()
    own_lock = {'token': str(uuid.uuid4()), 'supervisor_pid': os.getpid(), 'supervision_run': args.supervision_run,
                'supervisor_birth': supervisor_birth, 'supervisor_command_argv': supervisor_argv,
                'supervisor_source_sha256': digest(__file__), 'wrapper_command_argv': static['command_argv'],
                'stage': args.stage, 'chunk': args.chunk, 'created_utc': now()}
    with ACTIVE.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(own_lock, ensure_ascii=False, indent=2) + '\n')
    path = root / 'supervisor-outcome.json'
    receipt = {'status': 'supervisor_started_child_not_started', 'started_utc': now(), 'supervisor_pid': os.getpid(),
               'supervisor_parent_pid': os.getppid(), 'stage': args.stage, 'chunk': args.chunk,
               'command_argv': static['command_argv'], 'cwd': str(ROOT), 'source_identity': static,
               'CUDA_initialized_by_supervisor': False, 'retry_or_next_stage_policy': 'Never restart,retry,change budget or dispatch another stage',
               'GPU_child_started': False,
               'durability_limit': 'No survival promise after app/job closure,logout or reboot.', 'active_stage_lock': own_lock}
    child = None
    terminal = False
    seen_descendants = set()
    start = time.perf_counter()
    try:
        gate = dispatch_gate(args.stage, args.chunk, own_lock=own_lock)
        write(root / 'dispatch-audit.json', gate)
        job_helper = load('_supervisor006_job_membership', JOB_HELPER)
        receipt['windows_job_membership'] = job_helper.current_job_membership()
        write(path, receipt)
        observer = load('_supervisor006_observed_argv', OBSERVER)
        identity_seen = False
        identity_errors = []
        with (root / 'stage-wrapper.log').open('xb') as log:
            child = subprocess.Popen(static['command_argv'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            birth = child_birth(child)
            own_lock['actual_child_birth'] = birth
            write(ACTIVE, own_lock)
            receipt.update(status='child_started', child_pid=child.pid, actual_child_birth=birth, GPU_child_started=True)
            write(path, receipt)
            while True:
                code = child.poll()
                observation = {'observed_utc': now(), 'child_handle_poll_exit_code': code, 'child_pid': child.pid}
                try:
                    snapshot = process_snapshot()
                    descendants = observed_descendants(snapshot, child.pid)
                    seen_descendants.update((p['ProcessId'], creation_ms(p['CreationDate'])) for p in descendants)
                    current = next((p for p in snapshot if p['ProcessId'] == child.pid), None)
                    if current:
                        actual_argv = observer.windows_argv(current['CommandLine'])
                        matches = creation_ms(current['CreationDate']) == birth['creation_Unix_ms'] and exact_child_argv(actual_argv, static['command_argv'])
                        observation.update(actual_child_argv=actual_argv, actual_child_birth_and_argv_match=matches)
                        if matches:
                            identity_seen = True
                        else:
                            identity_errors.append(observation['observed_utc'])
                    observation.update(actual_processes=snapshot, actual_child_process=current,
                                       actual_observed_child_descendants=descendants)
                except Exception as error:
                    # A CIM timeout or missing PID is never evidence of exit and never causes restart.
                    observation['process_observation_error'] = f'{type(error).__name__}: {error}'
                with (root / 'process-observations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(observation, ensure_ascii=False) + '\n')
                write(root / 'latest-process-observation.json', observation)
                receipt.update(last_observed_utc=observation['observed_utc'], exact_child_birth_argv_observed=identity_seen,
                               identity_mismatch_observations=identity_errors)
                write(path, receipt)
                if code is not None:
                    terminal = True
                    break
                time.sleep(15)
        receipt.update(status='child_terminal', child_exit_code=code)
        write(path, receipt)
        snapshot = process_snapshot()
        busy = busy_processes(snapshot)
        descendants = live_seen_descendants(snapshot, seen_descendants)
        receipt.update(terminal_actual_processes=snapshot, terminal_actual_gpu_stage_processes=busy,
                       terminal_live_observed_descendants=descendants,
                       GPU_stage_processes_released=not busy and not descendants, child_handle_terminal=True)
        require(code == 0, 'Actual stage child exited nonzero; preserve without retry')
        require(identity_seen and not identity_errors, 'Actual launched child birth/exact argv identity unproven or mismatched')
        require(not busy, 'GPU-stage process still live after wrapper exit; completion/orphan-free release unproven')
        require(not descendants, 'Observed child descendant remains live; no orphan-free completion')
        completed = final_receipt(args.stage, args.chunk)
        receipt.update(status='completed_one_actual_stage_only', actual_final_stage=completed, next_stage_dispatched=False)
    except BaseException as error:
        poll = child.poll() if child is not None else None
        terminal = child is None or poll is not None
        receipt.update(status='failed_or_interrupted_preserved_no_retry', error_type=type(error).__name__, error=str(error),
                       actual_child_handle_poll_exit_code=poll, child_handle_terminal=terminal,
                       next_stage_dispatched=False)
        try:
            snapshot = process_snapshot()
            busy = busy_processes(snapshot)
            descendants = live_seen_descendants(snapshot, seen_descendants)
            receipt.update(terminal_actual_processes=snapshot, terminal_actual_gpu_stage_processes=busy,
                           terminal_live_observed_descendants=descendants,
                           GPU_stage_processes_released=terminal and not busy and not descendants)
        except Exception as observation_error:
            receipt['terminal_process_observation_error'] = repr(observation_error)
    finally:
        released = bool(terminal and receipt.get('GPU_stage_processes_released') and ACTIVE.exists() and read(ACTIVE) == own_lock)
        if released:
            ACTIVE.unlink()
        receipt.update(active_stage_lock_released=released, ended_utc=now(), elapsed_seconds=time.perf_counter()-start)
        write(path, receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt['status'] != 'completed_one_actual_stage_only':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
