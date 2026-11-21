"""CPU-only parser, real owned-controller gates, and no-GPU Popen-boundary controls."""
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
CODE = ROOT / 'scripts/large_scene_gs'
ENTRY = CODE / '006-supervision-revision-002/supervise_small_city_hierarchy_stage_006.py'
OUT = BASE / 'recovery-audit/stage006-revision002-controller-CPU-001'
PYTHON = ROOT / '.venv-large-gs/Scripts/python.exe'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    entry = load('_actual_revision002_CPU_controls', ENTRY)
    s, identity = entry.configure()
    assert identity['actual_runtime_entry_sha256'] == '6e8e5f111aefdd6e3762d05ce6222de54589fe41b7745aae9edf23f023ee8002'
    before = {str(p):sha(p) for p in [ENTRY, entry.PARSER, entry.SEAL, entry.BASELINE,
        CODE / 'launch_small_city_hierarchy_stage_006_revision_002.ps1', s.DRIVER, s.LOCK / 'configuration.json', s.LOCK / 'plan.json']}
    # Independent epoch base; fractional tests exercise all legal decimal lengths, not just the old six-digit fixture.
    epoch = int(datetime(2026, 10, 12, 9, 7, 45, tzinfo=timezone.utc).timestamp()) * 1000
    cases = []
    for length in range(10):
        digits = '786460123'[:length]
        text = '2026-10-12T17:07:45' + ('.' + digits if length else '') + '+08:00'
        expected = epoch + (int((digits + '000')[:3]) if length else 0)
        actual = s.creation_ms(text)
        assert actual == expected
        cases.append({'input':text, 'expected_ms':expected, 'actual_ms':actual})
    assert s.creation_ms('/Date(-1+0800)/') == -1
    assert s.creation_ms('2026-10-12T09:07:45.78646Z') == epoch + 786
    rejected = []
    for text in ['2026-10-12T17:07:45.78646', '2026-10-12T17:07:45.1234567890+08:00',
                 '2026-10-12T17:07:45.+08:00', '2026-10-12T17:07:45.1+25:00', 'arbitrary']:
        try:
            s.creation_ms(text)
        except ValueError:
            rejected.append(text)
        else:
            raise RuntimeError('Malformed/unzoned birth timestamp accepted')
    assert not s.ACTIVE.exists()
    # The control entry has the exact recognized006 basename. Its actual source is explicit,
    # and a local subprocess proxy makes the GPU wrapper Popen boundary unconditionally fail.
    directory = OUT / 'cpu-control-entry'
    directory.mkdir()
    harness = directory / 'supervise_small_city_hierarchy_stage_006.py'
    source = '''import hashlib,importlib.util,json,pathlib,subprocess,sys
sys.dont_write_bytecode=True
entry_path=pathlib.Path(r"C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\scripts\\large_scene_gs\\006-supervision-revision-002\\supervise_small_city_hierarchy_stage_006.py")
spec=importlib.util.spec_from_file_location("_CPU_probe_actual_revision002",entry_path);e=importlib.util.module_from_spec(spec);sys.modules[spec.name]=e;spec.loader.exec_module(e)
m,identity=e.configure()
old_birth=m.current_process_identity;old_write=m.write
probe={"CPU_only_actual_ownership_probe":True,"actual_CPU_probe_entry":str(pathlib.Path(__file__).resolve()),"actual_CPU_probe_entry_sha256":hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),"GPU_Popen_unconditionally_disabled":True}
def birth():
 b,a=old_birth();b["CPU_control_source_supplement"]=probe;return b,a
def write(p,d):
 if pathlib.Path(p).name in {"supervisor-outcome.json","dispatch-audit.json","active-stage.lock.json"}:d["CPU_control_source_supplement"]=probe
 old_write(p,d)
class CPUOnlyStageSubprocess:
 def __getattr__(self,name):return getattr(subprocess,name)
 def Popen(self,*args,**kwargs):raise RuntimeError("CPU_ONLY_PROBE_STOPPED_BEFORE_GPU_WRAPPER_POPEN")
m.current_process_identity=birth;m.write=write;m.subprocess=CPUOnlyStageSubprocess()
m.main()
'''
    harness.write_text(source, encoding='utf-8')
    run = 'city006-r2-creator01-ownership-control-CPU-001'
    argv = [str(PYTHON), '-X', 'utf8', str(harness), '--stage', 'creator', '--chunk', '0_1', '--supervision-run', run, '--execute']
    result = subprocess.run(argv, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=120,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT / 'ownership-probe-stdout.log').write_text(result.stdout, encoding='utf-8')
    (OUT / 'ownership-probe-stderr.log').write_text(result.stderr, encoding='utf-8')
    receipt_path = s.SUPERVISION / run / 'supervisor-outcome.json'
    receipt = s.read(receipt_path)
    assert result.returncode != 0 and receipt['status'] == 'failed_or_interrupted_preserved_no_retry'
    assert receipt['error'] == 'CPU_ONLY_PROBE_STOPPED_BEFORE_GPU_WRAPPER_POPEN'
    assert receipt['GPU_child_started'] is False and receipt['child_handle_terminal'] is True
    assert receipt['GPU_stage_processes_released'] is True and receipt['active_stage_lock_released'] is True
    assert receipt['CPU_control_source_supplement']['actual_CPU_probe_entry_sha256'] == sha(harness)
    assert receipt['runtime_control_revision'] == identity and not s.ACTIVE.exists()
    assert not (s.LOCK / 'trained_chunks/0_1').exists()
    actual = s.process_snapshot()
    assert not s.busy_processes(actual)
    assert all(sha(p) == h for p,h in before.items())
    final = {'status':'actual_revision002_CPU_parser_and_owned_single_execute_Popen_boundary_controls_passed_no_GPU',
        'created_utc':s.now(), 'actual_runtime_identity':identity, 'source_before_after_sha256':before,
        'parser_legal_fraction_cases':cases, 'malformed_or_unzoned_cases_rejected':rejected,
        'actual_original_defective_five_fraction_parse_passed':True,
        'actual_CPU_only_ownership_probe':{'argv':argv,'actual_return_code':result.returncode,'receipt_path':str(receipt_path),
            'receipt_sha256':sha(receipt_path),'CPU_probe_entry_sha256':sha(harness),'real_supervisor_birth':receipt['active_stage_lock']['supervisor_birth'],
            'all5_real_model_and_native_exact_birth_argv_active_ownership_gates_passed':True,
            'Popen_GPU_child_started':False,'original_GPU_stage_or_native_receipt_created':False,'actual_lock_released':True},
        'actual_processes_after':actual,'CUDA_initialized':False,'GPU_started':False,'next_stage_dispatched':False}
    path = OUT / 'control-audit.json'
    path.write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':final['status'],'path':str(path),'sha256':sha(path)},ensure_ascii=False))


if __name__ == '__main__':
    main()
