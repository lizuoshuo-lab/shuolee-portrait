"""Seal independent CPU stage006 review and Creator00 dispatch evidence."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = ROOT / 'outputs/large-scene-gs-001'
OUT = BASE / 'recovery-audit/stage006-supervision-controls-CPU-001'
SCRIPTS = ROOT / 'scripts/large_scene_gs'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    output = OUT / 'creator00-ready.json'
    if output.exists():
        raise RuntimeError('Existing ready review; no overwrite')
    records = {}
    for p in [SCRIPTS / 'supervise_small_city_hierarchy_stage_006.py', SCRIPTS / 'launch_small_city_hierarchy_stage_006.ps1',
              SCRIPTS / 'run_small_city_hierarchy_006.py', LOCK / 'configuration.json', LOCK / 'plan.json',
              LOCK / 'binding-audits/lineage006-complete-cpu-001/lineage-binding.json', OUT / 'control-audit.json']:
        records[str(p)] = sha(p)
    text = (OUT / 'default-launcher-check-stdout.log').read_text(encoding='utf-8-sig').strip()
    decoder = json.JSONDecoder()
    values = []
    while text:
        value, end = decoder.raw_decode(text)
        values.append(value)
        text = text[end:].strip()
    gate, default = values
    assert gate['status'] == 'actual006_source_lineage_input_prior_stage_disk_noGPU_gates_passed'
    assert gate['actual_all5_model_binding_recheck']['all_five_saved_model_bindings_rechecked']
    assert gate['GPU_started'] is False and default['GPU_started'] is False
    assert default['status'] == 'CPU_dispatch_check_only' and default['requires_Execute_to_start']
    assert not (OUT / 'default-launcher-check-stderr.log').read_text(encoding='utf-8-sig').strip()
    assert not (BASE / 'official-hierarchy-stage-supervision-launch-006/city006-creator00-stage-001').exists()
    assert not (LOCK / 'trained_chunks/0_0').exists()
    cpu_plan = BASE / 'official-hierarchy-stage-supervision-006/city006-creator00-CPU-plan-001'
    before = {str(p):sha(p) for p in cpu_plan.iterdir() if p.is_file()}
    repeat = subprocess.run([str(PYTHON), '-X', 'utf8', str(SCRIPTS / 'supervise_small_city_hierarchy_stage_006.py'),
        '--stage', 'creator', '--chunk', '0_0', '--supervision-run', cpu_plan.name], capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
    assert repeat.returncode != 0 and 'FileExistsError' in repeat.stderr
    after = {str(p):sha(p) for p in cpu_plan.iterdir() if p.is_file()}
    assert before == after
    for path, expected in records.items():
        assert sha(path) == expected
    native = SCRIPTS / 'run_small_city_hierarchy_006.py'
    ast.parse(native.read_text(encoding='utf-8-sig'))
    findings = [
        'Driver006 independently rechecks all5 actual trained models/finite activations/shared sky/exposures through immutable1514 mixed lineage; legacy calls unknown and observed11 remains non-formal.',
        'New exact process guard covers006/old GPU wrappers/trainers/evaluators/all native stages including MergerUnoptimized; only actual locked supervisor birth+source+exact argv and real Popen ancestor chain can be allowed.',
        'Creator gates require physical tree/anchors/exposure, actual Popen handle birth/poll, fixed five-argument argv, native executable SHA and full origin/config; no CPU plan is a completion.',
        'Formal post gates require original15000 final-save/normal-return,14999 source-derived calls, frozen anchors/raw fields/exposure/single100k sky; formal Merger rejects short64 pilot.',
        'Supervisor is one explicitly selected stage,80GiB/fresh gates,atomic live lock,no overwrite/retry/auto-nextstage; observation timeout never infers child exit or restarts.',
        'Default verified PS7 launcher is CPU-only; only explicit-Execute uses Start-Process -WindowStyle Hidden, with process-local Windows PSModulePath and fixed supervisor SHA.'
    ]
    result = {'status':'Creator00_actual_CPU_gates_ready_for_root_independent_single_stage_dispatch',
        'created_utc':datetime.now(timezone.utc).isoformat(), 'stage':'creator', 'chunk':'0_0',
        'source_sha256':records, 'actual_default_launcher_gate':gate, 'actual_default_launcher_result':default,
        'actual_cpu_control_receipt':str(OUT / 'control-audit.json'),
        'actual_cpu_control_receipt_sha256':sha(OUT / 'control-audit.json'),
        'CPU_plan_only':str(cpu_plan), 'CPU_plan_sha256':sha(cpu_plan / 'plan.json'),
        'existing_CPU_plan_reprepare_actual_refused':{'exit_code':repeat.returncode,'error':'FileExistsError','before_after_sha256_exact':before == after,'GPU_started':False},
        'independent_native_peer_review':{'source_sha256':sha(native),'blocking_findings':[],'reviewed_contracts':findings},
        'GPU_started_by_this_task':False,'Creator_post_Merger_completion_claimed':False,
        'parent_independent_review_and_explicit_single_stage_execute_required':True,
        'root_PS7_command_argv':[r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe',
            '-NoLogo','-NoProfile','-NonInteractive','-File',str(SCRIPTS / 'launch_small_city_hierarchy_stage_006.ps1'),
            '-Stage','creator','-Chunk','0_0','-SupervisionRun','city006-creator00-stage-001','-Execute'],
        'historical_CPUplan004_supervisor002_or005_results_are_not006_completion':True}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'path':str(output),'sha256':sha(output),'source_sha256':records},ensure_ascii=False))


if __name__ == '__main__':
    main()
