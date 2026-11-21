"""Read-only selected post00 process/source terminal peer; never dispatch or release.

This verifies the real recorded Popen identities, exact argv observations, actual
Hidden launch provenance, composite sources, and two fresh full CIM snapshots.
It does not re-audit model arrays or assert other parent-owned GPU stages absent.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
RUN = 'city006-post00-i15000-r2-stage-001'
STAGE = BASE / 'official-hierarchy/small-city-hierarchy-006/trained_chunks/0_0'
SUP = BASE / 'official-hierarchy-stage-supervision-006-revision-002' / RUN
EXTRA = SUP.parent / 'runtime-source-supplements' / RUN
LAUNCH = BASE / 'official-hierarchy-stage-supervision-launch-006-revision-002' / RUN / 'launch-receipt.json'
OUT = BASE / 'recovery-audit/stage006-formalpost00-supervision-terminal-CPU-001'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


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
    sup, native, launch, runtime = read(SUP / 'supervisor-outcome.json'), read(STAGE / 'post-outcome.json'), read(LAUNCH), read(EXTRA / 'runtime-control-revision.json')
    require(sup['status'] == 'completed_one_actual_stage_only' and sup['stage'] == 'post' and sup['chunk'] == '0_0', 'Selected actual supervision terminal missing')
    require(sup['child_exit_code'] == 0 and sup['child_handle_terminal'] and sup['exact_child_birth_argv_observed'] and not sup['identity_mismatch_observations'], 'Real child handle/birth/argv terminal missing')
    require(sup['GPU_stage_processes_released'] and sup['active_stage_lock_released'] and not sup['next_stage_dispatched'], 'Actual release or one-stage policy missing')
    require(not sup['terminal_actual_gpu_stage_processes'] and not sup['terminal_live_observed_descendants'], 'Supervisor saw an orphan stage at terminal')
    require(native['status'] == 'completed' and native['source_kind'] == 'post_optimized' and native['exit_code'] == native['actual_child_poll_exit_code'] == 0, 'Native completion is not a real exit0')
    require(native['requested_iterations'] == 15000 and native['optimizer_update_calls'] == 14999 and native['optimizer_call_count_source_derived'], 'Original full post budget changed')
    require(sup['actual_final_stage']['actual_receipt'] == native and sup['actual_final_stage']['receipt_sha256'] == sha(STAGE / 'post-outcome.json'), 'Selected native receipt detached')
    entry = load('_selected_post00_terminal_revision002', runtime['actual_runtime_entry_path'])
    baseline, actual_runtime = entry.configure()
    require(actual_runtime == runtime == sup['runtime_control_revision'] == sup['active_stage_lock']['runtime_control_revision'] == sup['active_stage_lock']['supervisor_birth']['runtime_control_revision'], 'Composite source provenance differs')
    driver, config, binding, manifest, _ = baseline.fixed_sources('post')
    driver.execution_source_gate()  # Source only; do not call the global GPU-role gate.
    require(sup['source_identity'] == read(SUP / 'configuration.json'), 'Actual supervision source configuration differs')
    expected_source = {'supervisor_source_sha256': sha(baseline.__file__), 'fixed_driver_sha256': baseline.DRIVER_SHA, 'formal_configuration_sha256': baseline.CONFIG_SHA, 'formal_CPU_plan_sha256': baseline.PLAN_SHA, 'lineage_binding_sha256': baseline.BINDING_SHA, 'run_local_sha256': baseline.LOCAL_SHA, 'one_stage_only': True, 'automatic_retry_or_next_stage': False, 'parent_exclusive_dispatch_required': True}
    require(all(sup['source_identity'].get(k) == v for k, v in expected_source.items()), 'Frozen source identity differs')
    require(sup['command_argv'] == sup['active_stage_lock']['wrapper_command_argv'] == baseline.command_for('post', '0_0', config), 'Actual wrapper argv differs')
    require(native['command_argv'] == driver.post_command(STAGE, BASE / 'official-inputs/small-city-training-002/chunks/0_0', Path(binding['coarse']['model']['path']).parent, manifest, 15000), 'Actual native post argv differs')
    require(launch['stage'] == 'post' and launch['chunk'] == '0_0' and launch['supervision_run'] == RUN and launch['single_stage_only'] and not launch['automatic_retry_or_next_stage'], 'Hidden launch stage contract differs')
    require(launch['start_process_pid'] == sup['supervisor_parent_pid'] and launch['supervisor_source_sha256'] == runtime['actual_runtime_entry_sha256'] and launch['actual_runtime_parser_sha256'] == runtime['actual_parser_sha256'] and launch['actual_runtime_source_seal_sha256'] == runtime['source_seal_sha256'], 'Actual Hidden launch source differs')
    launcher_path = ROOT / 'scripts/large_scene_gs/launch_small_city_hierarchy_stage_006_revision_002.ps1'
    require(sha(launcher_path) == launch['launcher_source_sha256'] == 'b8c45ec8c3b0f658a7bf41926d6581964b167f8c5547678cd07ef175944f9238', 'Frozen Hidden launcher differs')
    require(sha(EXTRA / 'actual-runtime-entry-source.py') == runtime['actual_runtime_entry_sha256'] and sha(EXTRA / 'actual-time-parser-source.py') == runtime['actual_parser_sha256'], 'Actual stage source snapshots differ')
    require(sha(STAGE / 'hierarchy.hier_opt') == native['hierarchy_sha256'] == sup['actual_final_stage']['model_sha256'] == 'a387578911cc4d630e1105659010de4af7bd7b7add3d21be483e73243cd712a6', 'Physical fullpost file hash differs')
    require(sha(STAGE / 'hierarchy.hier') == '9b3e8ad0901601410e7e73d902cdb448051258d582e41574f6b0fd98f7a30cd7', 'Original Creator input differs')
    parser = load('_selected_post00_terminal_creation_parser', runtime['actual_parser_path'])
    argv_parser = driver.load_module('_selected_post00_terminal_exact_argv', driver.PROCESS_HELPER).windows_argv
    identities = {}

    def remember(pid, birth_ms, role, argv=None):
        key = (int(pid), int(birth_ms))
        row = identities.setdefault(key, {'pid': key[0], 'creation_Unix_ms': key[1], 'roles': [], 'observed_exact_argv': []})
        if role not in row['roles']:
            row['roles'].append(role)
        if argv is not None and argv not in row['observed_exact_argv']:
            row['observed_exact_argv'].append(argv)

    for birth, role in [(sup['active_stage_lock']['supervisor_birth'], 'supervisor'), (sup['actual_child_birth'], 'wrapper_owned_Popen_handle'), (native['actual_child_birth'], 'train_owned_Popen_handle')]:
        require((birth['creation_FILETIME_100ns'] - 116444736000000000) // 10000 == birth['creation_Unix_ms'], 'Real FILETIME floor mismatch')
        remember(birth['pid'], birth['creation_Unix_ms'], role)
    remember(launch['start_process_pid'], parser.creation_ms(launch['start_process_creation_utc']), 'Hidden_StartProcess_venv_shim')
    observations = [json.loads(line) for line in (SUP / 'process-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    wrapper_match, train_match, shim_parent_pids = False, False, set()
    for obs in observations:
        for child in obs.get('actual_observed_child_descendants', []):
            remember(child['ProcessId'], parser.creation_ms(child['CreationDate']), 'actual_seen_descendant', argv_parser(child['CommandLine']))
        for row in obs.get('actual_processes', []):
            key = (int(row['ProcessId']), parser.creation_ms(row['CreationDate']))
            parsed = argv_parser(row['CommandLine'])
            if key in identities:
                remember(*key, 'actual_CIM_identity_observed', parsed)
            if key == (sup['child_pid'], sup['actual_child_birth']['creation_Unix_ms']):
                require(parsed == sup['command_argv'], 'Actual observed wrapper argv mismatch')
                wrapper_match = True
            if key == (native['actual_child_pid'], native['actual_child_birth']['creation_Unix_ms']):
                require(parsed == native['command_argv'], 'Actual observed train argv mismatch')
                train_match = True
            if key == (launch['start_process_pid'], parser.creation_ms(launch['start_process_creation_utc'])):
                shim_parent_pids.add(int(row['ParentProcessId']))
    require(wrapper_match and train_match, 'No actual exact argv observation for both real Popen births')
    require(observations[-1]['child_handle_poll_exit_code'] == 0, 'Last real owned poll not terminal0')
    tracked = [SUP / 'supervisor-outcome.json', SUP / 'process-observations.jsonl', SUP / 'configuration.json', SUP / 'dispatch-audit.json', LAUNCH, STAGE / 'post-outcome.json', STAGE / 'hierarchy.hier', STAGE / 'hierarchy.hier_opt', EXTRA / 'runtime-control-revision.json', EXTRA / 'actual-runtime-entry-source.py', EXTRA / 'actual-time-parser-source.py', launcher_path, Path(runtime['source_seal_path']), Path(runtime['actual_runtime_entry_path']), Path(runtime['actual_parser_path']), Path(baseline.__file__), baseline.DRIVER, baseline.LOCK / 'configuration.json', baseline.LOCK / 'plan.json', baseline.BINDING]
    before = {str(p): sha(p) for p in tracked}
    query = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 4 -Compress"
    snapshots = []
    for _ in range(2):
        result = subprocess.run([str(baseline.PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', query], env=baseline.PW_ENV, capture_output=True, text=True, encoding='utf-8', errors='strict', timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        require(result.returncode == 0, 'Fresh full CIM failed: ' + result.stderr)
        rows = json.loads(result.stdout or '[]')
        rows = rows if isinstance(rows, list) else [rows]
        for row in rows:
            require((int(row['ProcessId']), parser.creation_ms(row['CreationDate'])) not in identities, 'Selected historical process/descendant still alive')
            require(int(row['ProcessId']) not in shim_parent_pids, 'Recorded Hidden-launch parent PID still alive; no birth was retained for this parent')
        snapshots.append({'observed_utc': baseline.now(), 'full_CIM_rows': rows, 'other_current_GPU_roles': driver.gpu_process_roles([r for r in rows if r['Name'].lower().startswith(('python', 'gaussianhierarchy'))], argv_parser)})
    active = read(baseline.ACTIVE) if baseline.ACTIVE.exists() else None
    require(active is None or active['token'] != sup['active_stage_lock']['token'], 'Selected owned ACTIVE lock still exists')
    require({str(p): sha(p) for p in tracked} == before and 'torch' not in sys.modules, 'Protected bytes changed or Torch imported')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'audit-source.py').write_bytes(Path(__file__).read_bytes())
    report = {'status': 'actual_formalpost00_owned_handle_source_and_no_orphan_terminal_peer_passed', 'run': RUN, 'UTC': baseline.now(), 'audit_source_sha256': sha(__file__), 'native_receipt_sha256': before[str(STAGE / 'post-outcome.json')], 'supervisor_receipt_sha256': before[str(SUP / 'supervisor-outcome.json')], 'launch_receipt_sha256': before[str(LAUNCH)], 'runtime_control_revision': runtime, 'actual_exact_wrapper_and_train_argv_observed': True, 'actual_same_Popen_exit_codes': [sup['child_exit_code'], native['actual_child_poll_exit_code']], 'all_selected_historical_births_absent': list(identities.values()), 'recorded_Hidden_launch_parent_PIDs_absent_no_birth_retained': sorted(shim_parent_pids), 'two_fresh_full_CIM_snapshots': snapshots, 'current_ACTIVE_lock': active, 'own_active_lock_released_by_actual_supervisor': True, 'source_receipt_and_model_bytes_unchanged': before, 'requested_iterations': 15000, 'optimizer_calls_14999_source_derived_not_instrumented': True, 'model_arrays_not_read': True, 'global_GPU_empty_claimed': False, 'process_or_GPU_or_lock_control_performed': False, 'next_stage_dispatched': False, 'all4_formalpost_merge_or1404_completion_claimed': False}
    path = OUT / 'terminal-peer.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(path), 'SHA': sha(path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
