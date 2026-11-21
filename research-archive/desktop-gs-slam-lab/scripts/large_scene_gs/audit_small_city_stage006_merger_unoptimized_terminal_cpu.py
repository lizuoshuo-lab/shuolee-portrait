"""Selected first unoptimized Merger source/Popen terminal peer, CPU only.

--prepare reads receipts without arrays or ownership changes. The actual audit
requires real exit0 and supervisorial release, then verifies both Popen births,
argv, immutable sources, original Creator input hashes, and full fresh CIM.
It never dispatches, releases a lock, retries, or calls global no-conflict.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
RUN = 'city006-merge-unoptimized-r2-stage-001'
SUP = BASE / 'official-hierarchy-stage-supervision-006-revision-002' / RUN
EXTRA = SUP.parent / 'runtime-source-supplements' / RUN
LAUNCH = BASE / 'official-hierarchy-stage-supervision-launch-006-revision-002' / RUN / 'launch-receipt.json'
ORDER = ['0_0', '0_1', '1_0', '1_1']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--audit-run')
    args = p.parse_args()
    native_path = LOCK / 'merge-unoptimized-outcome.json'
    if args.prepare:
        print(json.dumps({'status': 'CPU_terminal_peer_prepared_not_audited', 'run': RUN,
                          'source_sha256': sha(__file__), 'current_native_status': read(native_path)['status'] if native_path.exists() else 'native_receipt_not_yet_created',
                          'GPU_started': False, 'arrays_read': False, 'lock_control_performed': False}))
        return
    require(args.audit_run and args.audit_run.replace('-', '').isalnum() and len(args.audit_run) <= 110, 'Unique safe CPU audit namespace required')
    out = BASE / 'recovery-audit' / args.audit_run
    require(not out.exists(), 'Never overwrite a previous terminal audit')
    sup, native, launch, runtime = read(SUP / 'supervisor-outcome.json'), read(native_path), read(LAUNCH), read(EXTRA / 'runtime-control-revision.json')
    require(sup['status'] == 'completed_one_actual_stage_only' and sup['stage'] == 'merge-unoptimized' and sup['chunk'] is None, 'Actual Merger supervision not completed')
    require(sup['child_exit_code'] == 0 and sup['child_handle_terminal'] and sup['exact_child_birth_argv_observed'] and not sup['identity_mismatch_observations'], 'Actual wrapper handle/birth/argv terminal missing')
    require(sup['GPU_stage_processes_released'] and sup['active_stage_lock_released'] and not sup['next_stage_dispatched'] and not sup['terminal_actual_gpu_stage_processes'] and not sup['terminal_live_observed_descendants'], 'Actual release/no orphan/no next stage missing')
    require(native['status'] == 'completed' and native['source_kind'] == 'unoptimized' and native['exit_code'] == native['actual_child_poll_exit_code'] == 0, 'Actual native Merger completion not exit0')
    require(native['chunk_order'] == ORDER and native['skybox_in_tree'] == 0 and native['hierarchy']['unowned_tail'] == 0, 'Original order/source kind/no Sky in tree contract differs')
    require(sup['actual_final_stage']['actual_receipt'] == native and sup['actual_final_stage']['receipt_sha256'] == sha(native_path), 'Actual native receipt detached')
    require(runtime == sup['runtime_control_revision'] == sup['active_stage_lock']['runtime_control_revision'] == sup['active_stage_lock']['supervisor_birth']['runtime_control_revision'], 'Actual composite source provenance detached')
    entry = load('_merger_terminal_revision002_entry', runtime['actual_runtime_entry_path'])
    baseline, current_runtime = entry.configure()
    require(current_runtime == runtime, 'Actual frozen composite source differs')
    pinned = [(baseline.DRIVER, baseline.DRIVER_SHA), (LOCK / 'driver-source.py', baseline.DRIVER_SHA), (LOCK / 'configuration.json', baseline.CONFIG_SHA), (LOCK / 'plan.json', baseline.PLAN_SHA), (baseline.BINDING, baseline.BINDING_SHA), (baseline.LOCAL, baseline.LOCAL_SHA), (baseline.OBSERVER, baseline.OBSERVER_SHA), (baseline.JOB_HELPER, baseline.JOB_HELPER_SHA)]
    for path, digest in pinned:
        require(sha(path) == digest, 'Current frozen source/config/binding changed: ' + str(path))
    driver = load('_merger_terminal_native006_source', baseline.DRIVER)
    driver.execution_source_gate()  # Source/receipt bytes only, no lineage arrays or global GPU-role gate.
    config = read(LOCK / 'configuration.json')
    binding = read(baseline.BINDING)
    manifest = read(BASE / 'official-inputs/small-city-training-002/manifest.json')
    require(binding['ordered_chunks'][-1]['model']['sha256'] == '367bef34e584616daeda938e2e0b5826a8e4bbf30d13b4d987db32c9f72abcb4' and binding['ordered_chunks'][-1]['formal_training_run'] is False and not binding['formal_observation_reclassification'], 'Observed11 reclassified or invalid source substituted')
    require(sup['source_identity'] == read(SUP / 'configuration.json') and sup['command_argv'] == sup['active_stage_lock']['wrapper_command_argv'] == baseline.command_for('merge-unoptimized', None, config), 'Actual wrapper argv/source configuration differs')
    variant_path = BASE / 'hierarchy-build/merger-unoptimized/manifest.json'
    variant = read(variant_path)
    require(sha(variant_path) == config['unoptimized_filename_variant_manifest_sha256'] and native['input_filename_only_variant'] == variant, 'Filename-only variant source differs')
    exe = Path(variant['executable']['path'])
    require(sha(exe) == variant['executable']['sha256'] == native['merger_executable_sha256'] == native['executable_sha256'] == native['executable_after_sha256'], 'Actual native executable changed')
    command = [str(exe), str(LOCK / 'trained_chunks'), '0', str(BASE / 'official-inputs/small-city-training-002/chunks'), str(LOCK / 'merged-unoptimized.hier'), *ORDER]
    require(native['command_argv'] == command and native['ownership'] == config['merge_ownership'] and native['chunk_centers'] == {c['name']: c['center'] for c in manifest['chunks']}, 'Original Merger argv/centers/ownership differs')
    protected = [native_path, SUP / 'supervisor-outcome.json', SUP / 'process-observations.jsonl', SUP / 'configuration.json', SUP / 'dispatch-audit.json', LAUNCH, EXTRA / 'runtime-control-revision.json', EXTRA / 'actual-runtime-entry-source.py', EXTRA / 'actual-time-parser-source.py', LOCK / 'merged-unoptimized.hier', variant_path, exe, Path(runtime['source_seal_path']), Path(runtime['actual_runtime_entry_path']), Path(runtime['actual_parser_path']), Path(baseline.__file__), *[v[0] for v in pinned]]
    require([r['chunk'] for r in native['input_hierarchies']] == ORDER, 'Actual ordered four original inputs missing')
    for row, origin in zip(native['input_hierarchies'], binding['ordered_chunks']):
        path = LOCK / 'trained_chunks' / row['chunk'] / 'hierarchy.hier'
        receipt_path = path.parent / 'creator-outcome.json'
        receipt = read(receipt_path)
        require(Path(row['path']) == path and row['source_kind'] == 'unoptimized' and row['sha256'] == sha(path) == receipt['hierarchy_sha256'] and row['source_stage_receipt_sha256'] == sha(receipt_path), 'Actual original Creator input bytes/receipt differ')
        require(receipt['status'] == 'completed' and receipt['exit_code'] == receipt['actual_child_poll_exit_code'] == 0 and row['training_source_kind'] == receipt['training_source_kind'] == origin['training_source_kind'] and receipt['training_origin'] == origin and row['training_lineage_binding_sha256'] == baseline.BINDING_SHA, 'Actual source input lineage or handle completion differs')
        protected += [path, receipt_path]
    require(sha(LOCK / 'merged-unoptimized.hier') == native['hierarchy_sha256'] == sup['actual_final_stage']['model_sha256'], 'Actual native output bytes detached')
    launcher = ROOT / 'scripts/large_scene_gs/launch_small_city_hierarchy_stage_006_revision_002.ps1'
    require(sha(launcher) == launch['launcher_source_sha256'] == 'b8c45ec8c3b0f658a7bf41926d6581964b167f8c5547678cd07ef175944f9238' and launch['supervisor_source_sha256'] == runtime['actual_runtime_entry_sha256'] and launch['start_process_pid'] == sup['supervisor_parent_pid'], 'Actual Hidden launch provenance differs')
    require(sha(EXTRA / 'actual-runtime-entry-source.py') == runtime['actual_runtime_entry_sha256'] and sha(EXTRA / 'actual-time-parser-source.py') == runtime['actual_parser_sha256'], 'Actual runtime source snapshots differ')
    protected.append(launcher)
    before = {str(path): sha(path) for path in protected}
    parser = load('_merger_terminal_creation_parser', runtime['actual_parser_path'])
    argv_parser = driver.load_module('_merger_terminal_exact_windows_argv', driver.PROCESS_HELPER).windows_argv
    identities = {}
    for birth, role in [(sup['active_stage_lock']['supervisor_birth'], 'supervisor'), (sup['actual_child_birth'], 'wrapper_Popen'), (native['actual_child_birth'], 'native_Popen')]:
        require((birth['creation_FILETIME_100ns'] - 116444736000000000) // 10000 == birth['creation_Unix_ms'], 'Real FILETIME epoch floor differs')
        identities[(birth['pid'], birth['creation_Unix_ms'])] = role
    launch_key = (launch['start_process_pid'], parser.creation_ms(launch['start_process_creation_utc']))
    identities[launch_key] = 'Hidden_StartProcess_shim'
    observations = [json.loads(line) for line in (SUP / 'process-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    exact_seen, parent_pids = set(), set()
    exact_expected = {(sup['child_pid'], sup['actual_child_birth']['creation_Unix_ms']): sup['command_argv'], (native['actual_child_pid'], native['actual_child_birth']['creation_Unix_ms']): command}
    for obs in observations:
        for row in obs.get('actual_observed_child_descendants', []):
            identities[(int(row['ProcessId']), parser.creation_ms(row['CreationDate']))] = 'observed_descendant'
        for row in obs.get('actual_processes', []):
            key = (int(row['ProcessId']), parser.creation_ms(row['CreationDate']))
            if key in exact_expected:
                require(argv_parser(row['CommandLine']) == exact_expected[key], 'Observed actual Popen argv differs')
                exact_seen.add(key)
            if key == launch_key:
                parent_pids.add(int(row['ParentProcessId']))
    require(exact_seen == set(exact_expected) and observations[-1]['child_handle_poll_exit_code'] == 0, 'Actual native/wrapper birth+argv or real last poll missing')
    query = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 4 -Compress"
    snapshots = []
    for _ in range(2):
        result = subprocess.run([str(baseline.PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', query], env=baseline.PW_ENV, capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        require(result.returncode == 0, 'Fresh full CIM failed: ' + result.stderr)
        rows = json.loads(result.stdout or '[]')
        rows = rows if isinstance(rows, list) else [rows]
        for row in rows:
            require((int(row['ProcessId']), parser.creation_ms(row['CreationDate'])) not in identities and int(row['ProcessId']) not in parent_pids, 'Selected historical chain or recorded launch parent still alive')
        snapshots.append({'UTC': baseline.now(), 'full_CIM_rows': rows, 'other_current_GPU_roles': driver.gpu_process_roles([r for r in rows if r['Name'].lower().startswith(('python', 'gaussianhierarchy'))], argv_parser)})
    active = read(baseline.ACTIVE) if baseline.ACTIVE.exists() else None
    require(active is None or active['token'] != sup['active_stage_lock']['token'], 'Selected Merger lock remains ACTIVE')
    require(before == {str(path): sha(path) for path in protected} and 'torch' not in sys.modules, 'Protected source/input/output bytes changed or Torch imported')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'audit-source.py').write_bytes(Path(__file__).read_bytes())
    report = {'status': 'actual_first_unoptimized_merger_source_and_owned_terminal_peer_passed', 'run': RUN, 'UTC': baseline.now(), 'audit_source_sha256': sha(__file__), 'actual_native_Popen_birth': native['actual_child_birth'], 'actual_wrapper_Popen_birth': sup['actual_child_birth'], 'actual_exact_native_and_wrapper_argv_seen': True, 'selected_seen_process_births_absent': [{'pid': k[0], 'creation_Unix_ms': k[1], 'role': v} for k, v in identities.items()], 'recorded_Hidden_parent_PIDs_absent_birth_not_retained': sorted(parent_pids), 'runtime_source_supplement': runtime, 'actual_source_input_output_receipts_SHA_before_after_unchanged': before, 'actual_output_receipt_counts_pending_independent_physical_audit': native['hierarchy'], 'two_fresh_full_CIM_snapshots': snapshots, 'current_ACTIVE_lock': active, 'selected_owned_lock_released_by_actual_supervisor': True, 'all4_original_Creator_inputs_only': True, 'observation11_reclassified_formal': False, 'post_or_pilot_used_as_input': False, 'global_GPU_empty_claimed': False, 'arrays_read': False, 'process_GPU_or_lock_control_performed': False, 'next_stage_dispatched': False, 'paired_post_merge_and1404_completion_claimed': False}
    path = out / 'terminal-peer.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(path), 'SHA': sha(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
