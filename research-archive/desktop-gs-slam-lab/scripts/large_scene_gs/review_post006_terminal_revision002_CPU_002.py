"""Read-only peer controls for revised timestamp parsing and exact stale-lock reconciliation."""
from pathlib import Path
import hashlib,importlib.util,sys,json
from datetime import datetime,timezone
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'recovery-audit/post006-terminal-revision002-independent-review-CPU-002'
OUT.mkdir(parents=True,exist_ok=False);sys.dont_write_bytecode=True
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
parser_file=ROOT/'scripts/large_scene_gs/006-supervision-revision-002/parse_windows_creation.py'
reconcile_file=ROOT/'scripts/large_scene_gs/reconcile_small_city_post64_terminal_006_cpu_revision_002.py'
assert sha(parser_file)=='68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a'
assert sha(reconcile_file)=='fad85300492cf1e431046e8ba7bbd43bc72564ef769cc18e4abbfbd0a764ef55'
before={str(p):sha(p) for p in [parser_file,reconcile_file]}
c=load('_terminal_revision002_parser_CPUpeer',parser_file)
r=load('_terminal_revision002_reconciliation_CPUpeer',reconcile_file)
d=load('_terminal_revision002_native006_CPUpeer',ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py')
argv=d.load_module('_terminal_revision002_argparser_CPUpeer',d.PROCESS_HELPER).windows_argv
tests=[]
for n in range(1,10):
    f=('786460123')[:n]
    expected=1791796065000+int((f+'000')[:3])
    value='2026-10-12T17:07:45.'+f+'+08:00'
    got=c.creation_ms(value);assert got==expected
    tests.append({'input':value,'actual_Unix_ms':got,'expected_Unix_ms':expected})
for value,expected in [('2026-10-12T09:07:45.78646Z',1791796065786),('2026-10-12T05:07:45.78646-04:00',1791796065786),
                       ('2026-10-12T09:07:45Z',1791796065000),('/Date(1791796065786+0800)/',1791796065786),
                       ('/Date(-1)/',-1),('1969-12-31T23:59:59.999999Z',-1)]:
    got=c.creation_ms(value);assert got==expected;tests.append({'input':value,'actual_Unix_ms':got,'expected_Unix_ms':expected})
rejected=[]
for value in ['2026-10-12T09:07:45.78646','not-a-date',None,'2026-13-12T09:07:45Z','/Date(abc)/','2026-10-12T09:07:45.1234567890Z']:
    try:c.creation_ms(value)
    except ValueError:rejected.append({'input':value,'rejected':True})
    else:raise AssertionError('Unsafe/invalid parser accepted')
ownership=read(BASE/'recovery-audit/stage006-post64-terminal-reconciliation-CPU-001/exact-stale-lock-preserved.json')
r.creation=c;r.pilot=read(BASE/'official-post-preflight/small-city-post00-i64-r4-006/outcome.json')
r.launch=read(BASE/'official-hierarchy-stage-supervision-launch-006/city006-post00-i64-stage-001/launch-receipt.json')
r.recorded_stage_parent_PIDs={7940}
actual_seen={(int(ownership['actual_child_birth']['pid']),ownership['actual_child_birth']['creation_Unix_ms'])}
assert r.absent([],ownership,actual_seen,d,argv)['actual_GPU_process_roles']==[]
def row(pid,ms,cmd,name='python.exe'):
    return {'ProcessId':pid,'ParentProcessId':1,'Name':name,'CreationDate':'/Date('+str(ms)+')/','CommandLine':cmd}
terminal_controls=[]
def no(name,rows):
    try:r.absent(rows,ownership,actual_seen,d,argv)
    except RuntimeError as e:terminal_controls.append({'case':name,'rejected':True,'error':str(e),'scope':'isolatedCPUfixture'})
    else:raise AssertionError('Invalid terminal accepted: '+name)
no('actual_locked_supervisor_birth',[row(ownership['supervisor_pid'],ownership['supervisor_birth']['creation_Unix_ms'],'python.exe harmless.py')])
no('actual_Popen_wrapper_birth',[row(ownership['actual_child_birth']['pid'],ownership['actual_child_birth']['creation_Unix_ms'],'python.exe harmless.py')])
no('actual_train_post_birth',[row(r.pilot['child_pid'],r.pilot['actual_child_birth']['creation_Unix_ms'],'python.exe harmless.py')])
no('recorded_launcher_shim_PID',[row(7940,1,'python.exe harmless.py')])
no('any_live_native_Creator',[row(1001,1,'GaussianHierarchyCreator.exe','GaussianHierarchyCreator.exe')])
no('any_live_MergerUnoptimized',[row(1002,1,'GaussianHierarchyMergerUnoptimized.exe','GaussianHierarchyMergerUnoptimized.exe')])
no('any_official_Gaussian_trainer',[row(1003,1,'python.exe train_single.py')])
no('any_GPU_eval',[row(1004,1,'python.exe evaluate_local_city_hierarchy.py --gpu')])
no('any_new006_execute_wrapper',[row(1005,1,'python.exe run_small_city_hierarchy_006.py --stage creator --execute')])
assert {p:sha(p) for p in before}==before and 'torch' not in sys.modules
report={'status':'passed_readonly_parser_and_terminal_release_source_review','created_utc':datetime.now(timezone.utc).isoformat(),
        'source_before_after':before,'parser_valid_actual_fraction_controls':tests,'parser_rejected_controls':rejected,
        'terminal_fixture_negative_controls':terminal_controls,'release_source_review':{
           'exact_preserved_active_lock_compared_to_actualfailed_supervisor':True,'actualPopen_handle_exit0_not_inferred':True,
           'all5_original_mixed_sources_and_Creator_and_pilot_arrays_recomputed':True,
           'actuallaunch_StartProcess7940_and_shimparents_absent_required':True,
           'two_full_fresh_CIM_snapshots_including_recorded_descendants':True,
           'no_release_unless_explicit_flag_and_final_exact_lock_bytes_equal':True,
           'originalfailed_and_success_and_sources_no_rewrite':True},
        'root_actual_default_reconciliation_receipt_path':str(BASE/'recovery-audit/stage006-post64-terminal-reconciliation-CPU-001/terminal-reconciliation.json'),
        'reviewer_executed_reconciliation_main':False,'reviewer_released_lock':False,'GPU_started':False,'CUDA_initialized':False,
        'scope':'Actual immutable source review + isolated CPU fixtures. Actual final release remains owner/root decision.'}
(OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'review.json'),'report_sha256':sha(OUT/'review.json')}))
