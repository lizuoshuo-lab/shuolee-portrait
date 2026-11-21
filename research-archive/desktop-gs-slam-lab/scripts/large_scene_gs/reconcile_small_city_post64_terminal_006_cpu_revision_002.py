"""CPU-only reconciliation of actual post64 completion and exact owned stale006 lock.

Default audits without releasing. Explicit --release-exact-owned-lock requires
two fresh complete CIM snapshots and exact lock bytes after all physical audits.
No process launch, stop, retry, model rewrite, source rewrite, or GPU stage.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
LOCK = BASE / 'official-hierarchy-stage-supervision-006/active-stage.lock.json'
SUP = BASE / 'official-hierarchy-stage-supervision-006/city006-post00-i64-stage-001'
PILOT = BASE / 'official-post-preflight/small-city-post00-i64-r4-006'
OLD = ROOT / 'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
PARSER = ROOT / 'scripts/large_scene_gs/006-supervision-revision-002/parse_windows_creation.py'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require(ok, text):
    if not ok:
        raise RuntimeError(text)


def snapshot():
    # Observe ALL processes, so no surviving recorded Python descendant can be hidden by a script filter.
    query = "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    result = subprocess.run([str(s.PW), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', query],
        env=s.PW_ENV, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW)
    require(result.returncode == 0, 'Complete actual process query failed')
    rows = json.loads(result.stdout or '[]')
    return rows if isinstance(rows, list) else [rows]


def absent(rows, ownership, seen, driver, argv_parser):
    relevant = [r for r in rows if r['Name'].lower().startswith('python') or r['Name'].lower().startswith('gaussianhierarchy')]
    require(not driver.gpu_process_roles(relevant, argv_parser), 'Actual GPU owner still live; refuse reconciliation release')
    identities = [(launch['start_process_pid'], creation.creation_ms(launch['start_process_creation_utc'])), (ownership['supervisor_pid'], ownership['supervisor_birth']['creation_Unix_ms']),
                  (ownership['actual_child_birth']['pid'], ownership['actual_child_birth']['creation_Unix_ms']),
                  (pilot['child_pid'], pilot['actual_child_birth']['creation_Unix_ms'])]
    for row in rows:
        require(int(row['ProcessId']) not in recorded_stage_parent_PIDs, 'A recorded actual stage launcher/supervisor shim parent PID is still present')
        key = (int(row['ProcessId']), creation.creation_ms(row['CreationDate']))
        require(key not in identities and key not in seen, 'Exact owned process/observed descendant remains live')
    return {'queried_utc': datetime.now(timezone.utc).isoformat(), 'all_process_count': len(rows),
            'actual_relevant_processes': relevant, 'actual_matching_locked_processes': [], 'actual_observed_descendants': [],
            'actual_GPU_process_roles': [], 'scope': 'Actual full Get-Cim snapshot, exact PID+birth and parsed GPU script/native roles'}


def main():
    global s, creation, pilot, launch, recorded_stage_parent_PIDs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    parser.add_argument('--release-exact-owned-lock', action='store_true')
    args = parser.parse_args()
    require(sha(OLD) == '3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7', 'Frozen006 supervisor changed')
    s = load('_reconcile006_fixed_supervisor', OLD)
    creation = load('_reconcile006_zoned_creation_parser', PARSER)
    root = BASE / 'recovery-audit' / s.identifier(args.audit_run)
    root.mkdir(parents=True, exist_ok=False)
    before_bytes = LOCK.read_bytes()
    before_sha = hashlib.sha256(before_bytes).hexdigest()
    own = json.loads(before_bytes)
    supervision = read(SUP / 'supervisor-outcome.json')
    pilot = read(PILOT / 'outcome.json')
    launch_path = BASE / 'official-hierarchy-stage-supervision-launch-006' / SUP.name / 'launch-receipt.json'
    launch = read(launch_path)
    require(launch['stage'] == 'post64' and launch['chunk'] == '0_0' and launch['supervision_run'] == SUP.name
            and launch['supervisor_source_sha256'] == sha(OLD)
            and launch['launcher_source_sha256'] == '974aabeb16fc0d52a22ac9e77dd44365235bcfc78b68f25ddb55f663cf915851'
            and launch['start_process_pid'] == supervision['supervisor_parent_pid'], 'Actual launcher birth/source/supervisor-parent identity differs')
    recorded_stage_parent_PIDs = {int(launch['start_process_pid']), int(supervision['supervisor_parent_pid'])}
    require(own['stage'] == 'post64' and own['chunk'] == '0_0' and own['supervision_run'] == SUP.name,
            'Not the explicitly authorized old post64 ownership lock')
    require(own['supervisor_source_sha256'] == sha(OLD) and own['supervisor_pid'] == supervision['supervisor_pid']
            and own['actual_child_birth'] == supervision['actual_child_birth']
            and own['wrapper_command_argv'] == supervision['command_argv'] and own == supervision['active_stage_lock'],
            'Actual stale lock and failed supervision receipt ownership differ')
    require(supervision['status'] == 'failed_or_interrupted_preserved_no_retry'
            and supervision['child_exit_code'] == 0 and supervision['actual_child_handle_poll_exit_code'] == 0
            and supervision['exact_child_birth_argv_observed'] and not supervision['identity_mismatch_observations']
            and supervision['active_stage_lock_released'] is False and supervision['error_type'] == 'ValueError'
            and supervision['error'].startswith('Invalid isoformat string:'), 'Actual failure is not the known CPU timestamp terminal failure')
    require(pilot['status'] == 'completed_post64_preflight_only' and pilot['exit_code'] == 0
            and pilot['actual_child_poll_exit_code'] == 0 and pilot['actual_child_birth']['pid'] == pilot['child_pid']
            and pilot['requested_iterations'] == 64 and pilot['optimizer_update_calls'] == 63
            and pilot['source_kind'] == 'post64_preflight' and pilot['eligible_for_formal_merge'] is False,
            'Actual pilot64 completion/real handle exit/source-kind failed')
    driver, config, binding, manifest, _ = s.fixed_sources('post64')
    helper, codec, training, _ = driver.frozen_helpers(s.LOCK)
    binder, _ = driver.fixed_sources()
    for ref in [binding['coarse'], *binding['ordered_chunks']]:
        actual = binder.bind_one(Path(ref['training_directory']), ref['stage'], ref['chunk'], manifest,
            driver, training, codec, observed=ref['training_source_kind'] == 'observed_local_official_training')
        require(actual == ref, 'Actual all5 saved mixed lineage differs')
    driver.input_split_audit(binder.INPUT, manifest)
    creator = driver.creator_gate('0_0', config, manifest, training, codec, helper, read(s.LOCK / 'plan.json')['camera_archives'])
    source_binding = read(PILOT / 'source-binding.json')
    require(source_binding['source_creator_receipt_sha256'] == sha(s.LOCK / 'trained_chunks/0_0/creator-outcome.json'), 'Pilot real Creator receipt changed')
    for name, expected in source_binding['source_file_sha256'].items():
        require(sha(s.LOCK / 'trained_chunks/0_0' / name) == expected and sha(PILOT / 'creator-input' / name) == expected,
                'Creator-to-pilot copied bytes differ')
    directory = PILOT / 'creator-input'
    before = codec.read_hierarchy(directory / 'hierarchy.hier')
    after = codec.read_hierarchy(directory / 'hierarchy.hier_opt', expected_unowned_tail=100000)
    anchors = codec.read_anchors(directory / 'anchors.bin', len(before.positions))
    freeze = helper.post_freeze_audit(before, after, anchors, Path(source_binding['actual_coarse_directory']), 100000)
    numeric = helper.hierarchy_numeric_audit(after)
    require(freeze == pilot['frozen_fields'] and numeric == pilot['numeric_audit'] and after.summary() == pilot['hierarchy']
            and sha(after.path) == pilot['output_hierarchy_sha256'], 'Actual physical post64 tree/freeze/half-Sky/numeric differs')
    require(sha(directory / 'exposure.json') == creator['copied_exposure_sha256']
            and pilot['frozen_trained_exposure']['sha256'] == creator['copied_exposure_sha256'], 'Actual frozen trained exposure differs')
    command = driver.post_command(directory, binder.INPUT / 'chunks/0_0', Path(source_binding['actual_coarse_directory']), manifest, 64)
    require(pilot['command_argv'] == command, 'Actual original post64 argv differs')
    log = (directory / 'post-stage/stage.log').read_text(encoding='utf-8', errors='replace')
    require('[ITER 64] Saving Gaussians' in log and 'Training complete.' in log, 'Actual final64-save/normal return missing')
    observations = [json.loads(line) for line in (SUP / 'process-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    seen = set()
    observation_stage_PIDs = {int(own['supervisor_pid']), int(launch['start_process_pid']), int(own['actual_child_birth']['pid'])}
    for row in observations:
        for p in row.get('actual_observed_child_descendants', []):
            seen.add((int(p['ProcessId']), creation.creation_ms(p['CreationDate'])))
        for p in row.get('actual_processes', []):
            if int(p['ProcessId']) in observation_stage_PIDs:
                parent = int(p['ParentProcessId'])
                if parent not in observation_stage_PIDs:
                    recorded_stage_parent_PIDs.add(parent)
    argv_parser = s.load('_reconcile006_exact_windows_argv', s.OBSERVER).windows_argv
    first = absent(snapshot(), own, seen, driver, argv_parser)
    protected = [OLD, s.DRIVER, s.LOCK / 'configuration.json', s.LOCK / 'plan.json', s.BINDING,
                 SUP / 'supervisor-outcome.json', launch_path, PILOT / 'outcome.json', Path(after.path),
                 directory / 'anchors.bin', directory / 'exposure.json']
    protected_sha = {str(p): sha(p) for p in protected}
    (root / 'exact-stale-lock-preserved.json').write_bytes(before_bytes)
    second = absent(snapshot(), own, seen, driver, argv_parser)
    require(LOCK.read_bytes() == before_bytes, 'Exact owned lock bytes changed during audit; never release a replacement lock')
    require(all(sha(p) == expected for p, expected in protected_sha.items()), 'Protected source/result changed during CPU audit')
    released = False
    if args.release_exact_owned_lock:
        LOCK.unlink()
        released = True
    result = {'status': 'actual_completed_post64_CPU_terminal_reconciled_exact_stale_lock_released' if released else 'actual_completed_post64_CPU_terminal_reconciled_release_pending',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': sha(__file__), 'parser_sha256': sha(PARSER),
        'original_supervisor_failure_preserved': True, 'old_failure_receipt_sha256': protected_sha[str(SUP / 'supervisor-outcome.json')],
        'physical_pilot_receipt_sha256': protected_sha[str(PILOT / 'outcome.json')], 'exact_stale_lock_sha256': before_sha,
        'exact_stale_lock_backup': str(root / 'exact-stale-lock-preserved.json'), 'exact_stale_lock_released': released,
        'protected_before_after_sha256': protected_sha, 'all5_mixed_training_models_recomputed': True,
        'actual_before_after_tree': {'before': before.summary(), 'after': after.summary()},
        'actual_frozen_anchor_count': len(anchors), 'actual_freeze': freeze, 'actual_numeric': numeric,
        'actual_launch_receipt_sha256': sha(launch_path), 'actual_launch_StartProcess_pid_and_creation_utc': {'pid': launch['start_process_pid'], 'creation_utc': launch['start_process_creation_utc']},
        'all_recorded_stage_launcher_supervisor_parent_or_shim_PIDs_absent': sorted(recorded_stage_parent_PIDs),
        'actual_Popen_births': {'supervisor': own['supervisor_birth'], 'wrapper': own['actual_child_birth'], 'train_post': pilot['actual_child_birth']},
        'actual_wrapper_and_train_post_handle_exit_codes': [0, 0], 'birth_and_exact_argv_previously_observed': True,
        'two_fresh_full_CIM_terminal_audits': [first, second], 'all_observed_descendant_identity_count': len(seen),
        'timestamp_defect': 'Python3.10 limited fraction parsing; original parser did not pad legal five-digit PS7 ISO fractional seconds',
        'source_kind': 'post64_preflight', 'requested_iterations': 64, 'optimizer_update_calls_source_derived': 63,
        'eligible_for_formal_merge': False, 'GPU_started': False, 'process_stopped_or_restarted': False,
        'original_outcomes_models_configs_or_sources_rewritten': False, 'next_stage_dispatched': False}
    path = root / 'terminal-reconciliation.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'path': str(path), 'sha256': sha(path), 'exact_lock_sha256': before_sha}, ensure_ascii=False))


if __name__ == '__main__':
    main()
