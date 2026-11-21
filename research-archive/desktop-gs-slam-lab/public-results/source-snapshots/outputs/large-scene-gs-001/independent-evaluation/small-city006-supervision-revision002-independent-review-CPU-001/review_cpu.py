"""Read-only peer review + actual own CIM/FILETIME parser agreement; no dispatch."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
HERE=Path(__file__).resolve().parent
ENTRY=ROOT/'scripts/large_scene_gs/006-supervision-revision-002/supervise_small_city_hierarchy_stage_006.py'
PARSER=ENTRY.parent/'parse_windows_creation.py'
BASELINE=ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
LAUNCHER=ROOT/'scripts/large_scene_gs/launch_small_city_hierarchy_stage_006_revision_002.ps1'
SEAL=BASE/'official-hierarchy-stage-supervision-006-revision-002/source-lock-r2-final.json'
NATIVE=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
QUERY=BASE/'independent-evaluation/city-local-hierarchy-evaluation-driver-CPU-006/query_windows_gpu_roles_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
pins={str(ENTRY):'6e8e5f111aefdd6e3762d05ce6222de54589fe41b7745aae9edf23f023ee8002',str(PARSER):'68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a',str(BASELINE):'3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7',str(LAUNCHER):'b8c45ec8c3b0f658a7bf41926d6581964b167f8c5547678cd07ef175944f9238',str(SEAL):'c3d54d7f447b1ee316ee9d56d3f92b85fdda6303bfcbaa6837f5bd832f43f9cc',str(NATIVE):'a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6',str(QUERY):'eb9bab149ebc03f289c8e17bdfcd223b9b5d3168cb1ee3adf9804b496e3b689e'}
assert all(sha(p)==h for p,h in pins.items()) and not (HERE/'report.json').exists()
entry=load('readonly_revision002_peer_entry',ENTRY);parser=load('readonly_revision002_peer_parser',PARSER)
baseline=load('readonly_revision002_original_baseline',BASELINE);native=load('readonly_native006_birth_only',NATIVE)
module,runtime=entry.configure()
assert module.ACTIVE==baseline.ACTIVE and module.SUPERVISION!=baseline.SUPERVISION
assert module.creation_ms is not baseline.creation_ms and module.main.__code__.co_code==baseline.main.__code__.co_code
assert runtime['actual_runtime_entry_sha256']==pins[str(ENTRY)] and runtime['actual_parser_sha256']==pins[str(PARSER)] and runtime['decision_logic_overrides']==['creation_ms']
cases=[('2026-10-12T09:00:00.12345Z',1791795600123),('2026-10-12T17:00:00.1234567+08:00',1791795600123),('2026-10-12T09:00:00Z',1791795600000),('/Date(1791795600123+0800)/',1791795600123),('/Date(-1+0800)/',-1)]
controls=[]
for value,expected in cases:
    actual=parser.creation_ms(value);assert actual==expected;controls.append({'value':value,'expected_integer_epoch_ms':expected,'actual':actual})
rejected=[]
for value in ['2026-10-12T09:00:00.12345','2026-10-12T09:00:00.1234567890Z','2026-10-12T09:00:00+25:00','not-a-creation-date']:
    try:parser.creation_ms(value)
    except ValueError as ex:rejected.append({'value':value,'rejected':True,'error':str(ex)})
    else:raise AssertionError('Ambiguous/malformed birth time accepted')
try:
    baseline.creation_ms('2026-10-12T09:00:00.12345Z');old_fraction5={'old_parser_rejected':False}
except ValueError as ex:old_fraction5={'old_parser_rejected':True,'error':str(ex)}
query=load('readonly_actual_private_ps7_query',QUERY).query('terminal')
own=next(p for p in query['actual_process_snapshot'] if p['ProcessId']==os.getpid())
birth=native.process_birth(os.getpid());actual_ms=parser.creation_ms(own['CreationDate']);assert actual_ms==birth['creation_Unix_ms']
actual5=[]
for row in query['actual_process_snapshot']:
    value=row.get('CreationDate');match=re.search(r'\.(\d{5})(?:Z|[+-]\d\d:\d\d)$',str(value))
    if match:
        parsed=parser.creation_ms(value)
        try:baseline.creation_ms(value);old_rejected=False
        except ValueError:old_rejected=True
        actual5.append({'PID':row['ProcessId'],'actual_CIM_creation':value,'revised_epoch_ms':parsed,'old_parser_rejected':old_rejected})
source_current_birth,source_argv=module.current_process_identity()
assert source_current_birth['runtime_control_revision']==runtime and {k:source_current_birth[k] for k in ('pid','creation_FILETIME_100ns','creation_Unix_ms')}==birth
mockdir=HERE/'recording-hook-only';mockdir.mkdir();fixture={'status':'CPU_recording_hook_fixture_no_stage_or_GPU_execution'}
module.write(mockdir/'supervisor-outcome.json',fixture)
record=json.loads((mockdir/'supervisor-outcome.json').read_text(encoding='utf-8'));assert record['runtime_control_revision']==runtime
assert all(sha(p)==h for p,h in pins.items()) and 'torch' not in sys.modules
report={'status':'independent_CPU_revision002_source_parser_actual_CIM_and_recording_review_passed','source_pins':pins,'actual_runtime_identity':runtime,'baseline_main_decision_bytecode_unchanged':True,'only_decision_override':'creation_ms','shared_ACTIVE_path_unchanged':str(module.ACTIVE),'new_receipt_namespace':str(module.SUPERVISION),'integer_epoch_ms_controls':controls,'malformed_or_ambiguous_time_negative_controls':rejected,'old_fraction5_control':old_fraction5,'actual_CIM_own_process':own,'actual_GetProcessTimes_birth':birth,'actual_parser_CIM_ms_equals_FILETIME_floor_ms':actual_ms==birth['creation_Unix_ms'],'actual_fraction5_CIM_samples':actual5[:8],'actual_current_identity_has_composite_runtime_supplement':True,'actual_recording_hook_fixture_path':str(mockdir/'supervisor-outcome.json'),'recording_fixture_sha256':sha(mockdir/'supervisor-outcome.json'),'main_launcher_or_GPU_dispatch_called':False,'source_bytes_unchanged':True,'Torch_imported':False,'CUDA_called':False,'scope':'Independent source review and real own process creation-time identity. CPU input/recording controls are marked fixtures and do not prove child dispatch, post/Merger or GPU completion. Sealed eval260/5c9 remains unchanged; role prefix already has separate actual CPU rejection evidence.'}
(HERE/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(HERE/'report.json'),'sha256':sha(HERE/'report.json'),'actual_CIM_ms':actual_ms,'actual5samples':len(actual5),'Torch_imported':False,'CUDA_called':False}))
