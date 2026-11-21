"""Independent CPU controls for stage006 supervision; never a GPU stage."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
SOURCE = ROOT / 'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
BASE = ROOT / 'outputs/large-scene-gs-001'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-run', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('_stage006_independent_cpu_controls', SOURCE)
    supervisor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = supervisor
    spec.loader.exec_module(supervisor)
    root = BASE / 'recovery-audit' / supervisor.identifier(args.audit_run)
    root.mkdir(parents=True, exist_ok=False)
    controls = {}
    ast.parse(SOURCE.read_text(encoding='utf-8'))
    controls['supervisor_AST_parse'] = True
    observer = supervisor.load('_stage006_control_argv', supervisor.OBSERVER)
    synthetic = []
    for name in ('GaussianHierarchyCreator.exe', 'GaussianHierarchyMerger.exe', 'GaussianHierarchyMergerUnoptimized.exe'):
        synthetic.append({'Name': name, 'CommandLine': name})
    for script in ('train_coarse.py', 'train_single.py', 'train_post.py', 'evaluate_city.py',
                   'run_small_city_hierarchy_006.py', 'run_small_city_post_preflight_006.py'):
        argv = [str(supervisor.PYTHON), '-u', str(ROOT / 'scripts/large_scene_gs' / script), '--execute']
        synthetic.append({'Name': 'python.exe', 'CommandLine': subprocess.list2cmdline(argv)})
    rejected = supervisor.busy_processes(synthetic)
    supervisor.require(len(rejected) == len(synthetic), 'GPU role negative controls failed')
    harmless = [{'Name': 'python.exe', 'CommandLine': subprocess.list2cmdline([str(supervisor.PYTHON), '-c', "print('train_post.py run_small_city_hierarchy_006.py --execute')"])}]
    supervisor.require(not supervisor.busy_processes(harmless), 'Literal source text mistaken for actual script argv')
    controls['synthetic_busy_GPU_role_controls'] = {'all_rejected': True, 'cases': synthetic,
        'scope': 'Synthetic command-line classification only; no such process was started'}
    controls['literal_python_c_text_not_GPU_script'] = True
    actual_before = supervisor.process_snapshot()
    supervisor.require(not supervisor.busy_processes(actual_before), 'Actual GPU process present; CPU control not allowed')
    command = [str(supervisor.PYTHON), '-u', '-c', 'import time; time.sleep(5)']
    start = time.perf_counter()
    child = subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    birth = supervisor.child_birth(child)
    timeout = False
    try:
        child.wait(timeout=.01)
    except subprocess.TimeoutExpired:
        timeout = True
    supervisor.require(timeout and child.poll() is None, 'CPU handle timeout must leave same live process')
    snapshot = supervisor.process_snapshot()
    current = next((p for p in snapshot if p['ProcessId'] == child.pid), None)
    supervisor.require(current is not None, 'CPU actual Popen handle PID not observed')
    actual_argv = observer.windows_argv(current['CommandLine'])
    supervisor.require(supervisor.creation_ms(current['CreationDate']) == birth['creation_Unix_ms']
                       and supervisor.exact_child_argv(actual_argv, command), 'CPU actual Popen birth/exact argv mismatch')
    supervisor.require(not supervisor.exact_child_argv(actual_argv, command + ['different']), 'Changed argv was accepted')
    code = child.wait(timeout=20)
    supervisor.require(code == 0 and child.poll() == 0, 'CPU Popen actual terminal handle failed')
    actual_after = supervisor.process_snapshot()
    supervisor.require(not supervisor.busy_processes(actual_after), 'Actual GPU processes appeared during CPU control')
    supervisor.require(not any(p['ProcessId'] == child.pid and supervisor.creation_ms(p['CreationDate']) == birth['creation_Unix_ms'] for p in actual_after), 'CPU child process not released')
    controls['actual_harmless_CPU_Popen_handle_control'] = {
        'command_argv': command, 'actual_child_birth': birth, 'actual_observed_process': current,
        'actual_observed_argv': actual_argv, 'exact_birth_argv_verified': True,
        'changed_argv_rejected': True, 'wait_timeout_observed_same_child_live': timeout,
        'restart_count': 0, 'child_poll_exit_code': child.poll(), 'child_released': True,
        'elapsed_seconds': time.perf_counter()-start,
        'scope': 'Only harmless Python sleep child; no trainer,native stage,CUDA or hierarchy model'}
    refusals = {}
    for stage, chunk in [('post', '0_0'), ('merge', None), ('merge-unoptimized', None)]:
        try:
            supervisor.preconditions(stage, chunk, None, None, None, None)
        except (RuntimeError, FileNotFoundError) as error:
            refusals[stage] = {'refused': True, 'error_type': type(error).__name__, 'error': str(error), 'GPU_started': False}
        else:
            raise RuntimeError('Expected actual missing prior-stage refusal: ' + stage)
    controls['actual_missing_prior_stage_refusals'] = refusals
    source_records = {str(p): supervisor.digest(p) for p in [SOURCE, Path(__file__),
        ROOT / 'scripts/large_scene_gs/launch_small_city_hierarchy_stage_006.ps1', supervisor.DRIVER,
        supervisor.LOCK / 'configuration.json', supervisor.LOCK / 'plan.json', supervisor.BINDING,
        supervisor.LOCAL, supervisor.OBSERVER] if p.is_file()}
    outcome = {'status': 'actual_CPU_supervision_controls_passed_no_GPU', 'created_utc': supervisor.now(),
        'controls': controls, 'actual_processes_before': actual_before, 'actual_processes_after': actual_after,
        'source_sha256': source_records, 'CUDA_initialized': False, 'GPU_stage_started': False,
        'real_GPU_stage_receipt_created': False, 'scope': 'CPU process/control evidence only; not Creator/post/Merger completion'}
    supervisor.write(root / 'control-audit.json', outcome)
    print(json.dumps({'status': outcome['status'], 'root': str(root),
        'receipt_sha256': supervisor.digest(root / 'control-audit.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
