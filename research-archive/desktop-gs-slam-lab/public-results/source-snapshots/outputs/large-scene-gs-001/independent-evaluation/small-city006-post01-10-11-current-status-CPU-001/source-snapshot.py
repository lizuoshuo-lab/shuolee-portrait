"""One fresh read-only native-file/PS7-CIM snapshot, not a dispatch/GPU-free gate."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
OUT=BASE/'independent-evaluation/small-city006-post01-10-11-current-status-CPU-001'
QUERY=BASE/'independent-evaluation/city-local-hierarchy-evaluation-driver-CPU-006/query_windows_gpu_roles_006.py'
PARSER=ROOT/'scripts/large_scene_gs/006-supervision-revision-002/parse_windows_creation.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(QUERY)=='eb9bab149ebc03f289c8e17bdfcd223b9b5d3168cb1ee3adf9804b496e3b689e'
assert sha(PARSER)=='68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a'
q=load('current_status_private_PS7',QUERY);parser=load('current_status_owned_birth_parser',PARSER)
snapshot=q.query('terminal');processes={int(p['ProcessId']):p for p in snapshot['actual_process_snapshot']}
records=[]
for chunk,label in [('0_1','01'),('1_0','10'),('1_1','11')]:
    folder=LOCK/'trained_chunks'/chunk;receiptpath=folder/'post-outcome.json';model=folder/'hierarchy.hier_opt';log=folder/'post-stage/stage.log'
    record={'chunk':chunk,'native_post_receipt_present':receiptpath.is_file(),'actual_hier_opt_present':model.is_file(),'actual_hier_opt_bytes':model.stat().st_size if model.is_file() else None}
    if receiptpath.is_file():
        r=read(receiptpath);pid=r.get('actual_child_pid');row=processes.get(pid);matching_birth=False
        if row and r.get('actual_child_birth'):matching_birth=parser.creation_ms(row['CreationDate'])==r['actual_child_birth']['creation_Unix_ms']
        record.update(native_receipt_path=str(receiptpath),native_receipt_sha256=sha(receiptpath),native_status=r.get('status'),actual_exit_code=r.get('exit_code'),actual_poll_exit_code=r.get('actual_child_poll_exit_code'),requested_iterations=r.get('requested_iterations'),source_kind=r.get('source_kind'),model_sha256_in_receipt=r.get('hierarchy_sha256'),actual_hierarchy_summary_in_receipt=r.get('hierarchy'),actual_child_pid=pid,actual_child_birth_in_receipt=r.get('actual_child_birth'),same_PID_and_birth_in_fresh_CIM=matching_birth,PID_in_fresh_CIM=row is not None)
        if matching_birth:record['actual_owned_child_CIM']=row
        elif row:record['PID_reused_current_CIM']=dict(ProcessId=row['ProcessId'],Name=row['Name'],CreationDate=row['CreationDate'])
    run='city006-post'+label+'-i15000-r2-stage-001';sup=BASE/'official-hierarchy-stage-supervision-006-revision-002'/run/'supervisor-outcome.json'
    record['supervision_outcome_present']=sup.is_file()
    if sup.is_file():
        s=read(sup);record.update(supervision_outcome_path=str(sup),supervision_outcome_sha256=sha(sup),supervision_status=s.get('status'),supervision_child_exit_code=s.get('child_exit_code'),supervision_owned_child_terminal=s.get('child_handle_terminal'),supervision_stage_processes_released=s.get('GPU_stage_processes_released'))
    if log.is_file():
        # Bounded tail only, not full logs/model arrays.
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-4096));tail=f.read().decode('utf-8','replace')
        record['actual_stage_log_tail']=tail
    relevant=[]
    for row in snapshot['actual_classified_processes']:
        argv=row['parsed_argv'];normalized=[v.replace('\\','/').lower() for v in argv]
        path_match=any('/small-city-hierarchy-006/trained_chunks/'+chunk+'/' in v for v in normalized)
        chunk_match=any(argv[i]=='--chunk' and i+1<len(argv) and argv[i+1]==chunk for i in range(len(argv))) and any('run_small_city_hierarchy_006.py' in v or 'supervise_small_city_hierarchy_stage_006.py' in v for v in normalized)
        if path_match or chunk_match:relevant.append(row)
    record['actual_selected_stage_role_processes_at_query']=relevant
    records.append(record)
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'source-snapshot.py')
report={'status':'completed_readonly_post01_10_11_actual_native_files_and_fresh_PS7_CIM_snapshot_only','observed_utc':snapshot['observed_utc'],'file_read_finished_utc':datetime.now(timezone.utc).isoformat(),'records':records,
 'process_query_source_sha256':sha(QUERY),'time_parser_source_sha256':sha(PARSER),'fresh_Win32_Process_query':snapshot['fresh_Win32_Process_query'],'verified_PS7_sha256':snapshot['verified_PS7_sha256'],'parent_PSModulePath_unchanged':snapshot['parent_PSModulePath_unchanged'],
 'scope':'Receipt status plus physical presence and one timestamped PID/birth/argv query only; completed data counts are recorded receipt summaries, not a new numerical tree audit. Missing receipt/model describes file state, not proof no future dispatch is pending. No global GPU-free claim.',
 'global_GPU_free_claimed':False,'GPU_or_process_or_lock_control_executed':False,'CUDA_called':False,'Torch_imported':'torch' in sys.modules,'model_arrays_read':False,'original_sources_or_reports_edited':False}
assert not report['Torch_imported']
(OUT/'status-snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'observed_utc':report['observed_utc'],'report_path':str(OUT/'status-snapshot.json'),'report_sha256':sha(OUT/'status-snapshot.json'),'records':[{'chunk':r['chunk'],'receipt_present':r['native_post_receipt_present'],'model_present':r['actual_hier_opt_present'],'native_status':r.get('native_status'),'owned_child_live':r.get('same_PID_and_birth_in_fresh_CIM'),'selected_roles':len(r['actual_selected_stage_role_processes_at_query'])} for r in records]}))
