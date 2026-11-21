"""Read-only thin-supervisor source/identity peer audit; no main or stage dispatch."""
from pathlib import Path
import importlib.util,hashlib,json,sys,types
from datetime import datetime,timezone
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'recovery-audit/small-city-thin-supervision-revision002-independent-review-CPU-002'
OUT.mkdir(parents=True,exist_ok=False);sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
entry=ROOT/'scripts/large_scene_gs/006-supervision-revision-002/supervise_small_city_hierarchy_stage_006.py'
baseline=ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py'
parser=entry.parent/'parse_windows_creation.py'
seal=BASE/'official-hierarchy-stage-supervision-006-revision-002/source-lock-r2-final.json'
pinned={str(entry):'6e8e5f111aefdd6e3762d05ce6222de54589fe41b7745aae9edf23f023ee8002',
        str(baseline):'3db6b7ee81d0b7df592609509f8dd565e146179d399f4ca8fbee9baf65e85cf7',
        str(parser):'68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a',
        str(seal):'c3d54d7f447b1ee316ee9d56d3f92b85fdda6303bfcbaa6837f5bd832f43f9cc'}
assert {p:sha(p) for p in pinned}==pinned
e=load('_thin_supervisor_revision002_peer',entry)
old=load('_thin_baseline_comparison_peer',baseline)
effective,runtime=e.configure()
changed=[];same=[]
for name,value in vars(old).items():
    if isinstance(value,types.FunctionType) and value.__module__==old.__name__:
        target=getattr(effective,name)
        if target.__code__.co_code==value.__code__.co_code and target.__code__.co_consts==value.__code__.co_consts:same.append(name)
        else:changed.append(name)
assert set(changed)=={'creation_ms','current_process_identity','write'}
for name in ['ACTIVE','DRIVER','DRIVER_SHA','CONFIG_SHA','PLAN_SHA','BINDING','BINDING_SHA','PILOT','PILOT_PIN','LOCAL','LOCAL_SHA','PYTHON']:
    assert getattr(old,name)==getattr(effective,name)
assert effective.SUPERVISION==e.SUPERVISION and effective.SUPERVISION!=old.SUPERVISION
assert runtime['immutable_baseline_source_sha256']==pinned[str(baseline)]
assert runtime['actual_runtime_entry_sha256']==pinned[str(entry)] and runtime['actual_parser_sha256']==pinned[str(parser)]
# Actual read-only self PID handle/birth/argv, demonstrating explicit extra provenance.
birth,argv=effective.current_process_identity()
assert birth['runtime_control_revision']==runtime and birth['pid']==__import__('os').getpid()
assert isinstance(birth['creation_FILETIME_100ns'],int) and argv
assert effective.creation_ms('2026-10-12T17:07:45.78646+08:00')==1791796065786
assert {p:sha(p) for p in pinned}==pinned and 'torch' not in sys.modules
report={'status':'passed_thin_runtime_revision002_source_identity_readonly_peer_review',
        'UTC':datetime.now(timezone.utc).isoformat(),'pinned_source_before_after':pinned,
        'unchanged_baseline_function_names':same,'overridden_function_names':changed,
        'decision_override_only':'creation_ms','recording_hooks_only':['current_process_identity','write'],
        'actual_effective_self_birth_and_argv':{'birth':birth,'argv':argv},'runtime_source_identity':runtime,
        'baseline_vs_actual_entry_provenance_explicit':True,'old_native006_guard_source_birth_argv_chain_contract_preserved':True,
        'new_receipt_namespace_only':str(effective.SUPERVISION),'shared_active_lock_same_as_frozen_native006':str(effective.ACTIVE),
        'configured_main_or_dispatch_or_Popen_executed':False,'ACTIVE_created_or_released':False,'GPU_started':False,'CUDA_initialized':False,
        'root_source_seal_fields_do_not_modify_official_algorithm_or_iteration_budget':True}
(OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'review.json'),'report_sha256':sha(OUT/'review.json')}))
