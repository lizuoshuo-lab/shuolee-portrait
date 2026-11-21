"""Selected Creator terminal/model/source audit; other parent-authorized GPU stages may be live."""
from pathlib import Path
import argparse,importlib.util,hashlib,json,sys,subprocess
from datetime import datetime,timezone
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def require(v,m):
    if not v:raise RuntimeError(m)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chunk',choices=['0_1','1_0'],required=True);a=p.parse_args()
    label=a.chunk.replace('_','');run='city006-creator'+label+'-r2-stage-001'
    SUP=BASE/'official-hierarchy-stage-supervision-006-revision-002'/run
    LAUNCH=BASE/'official-hierarchy-stage-supervision-launch-006-revision-002'/run/'launch-receipt.json'
    EXTRA=SUP.parent/'runtime-source-supplements'/run
    OUT=BASE/'recovery-audit'/('city006-creator'+label+'-independent-terminal-CPU-002');OUT.mkdir(parents=True,exist_ok=False)
    source=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
    require(sha(source)=='a44e97310e02b742ed3222d5e9d5e6777526d8485412662c0b46ced0ce1f50a6','Native006 changed')
    d=load('_selected_creator_terminal_CPU_native',source)
    config,plan=read(LOCK/'configuration.json'),read(LOCK/'plan.json')
    manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
    sup=read(SUP/'supervisor-outcome.json');launch=read(LAUNCH)
    runtime=read(EXTRA/'runtime-control-revision.json')
    require(sup['status']=='completed_one_actual_stage_only' and sup['child_exit_code']==0 and sup['child_handle_terminal'] and sup['GPU_stage_processes_released'] and sup['active_stage_lock_released'],'Selected Creator supervision not terminal/verified')
    require(sup['exact_child_birth_argv_observed'] and not sup['identity_mismatch_observations'],'Selected Creator actual child birth/argv unproven')
    require(sup['stage']=='creator' and sup['chunk']==a.chunk and launch['supervision_run']==run and launch['start_process_pid']==sup['supervisor_parent_pid'],'Selected actual launcher/supervisor/stage mismatch')
    require(runtime==sup['runtime_control_revision']==sup['active_stage_lock']['runtime_control_revision']==sup['active_stage_lock']['supervisor_birth']['runtime_control_revision'],'Composite source supplements differ')
    require(runtime['actual_runtime_entry_sha256']=='6e8e5f111aefdd6e3762d05ce6222de54589fe41b7745aae9edf23f023ee8002' and runtime['actual_parser_sha256']=='68cdbccd72881203236481f839defa01ad0d9387c45c8fc7cae869d363d6c63a' and runtime['source_seal_sha256']=='c3d54d7f447b1ee316ee9d56d3f92b85fdda6303bfcbaa6837f5bd832f43f9cc','Reviewed runtime sources not selected')
    for actual,key,snapshot in [(runtime['actual_runtime_entry_path'],'actual_runtime_entry_sha256',EXTRA/'actual-runtime-entry-source.py'),(runtime['actual_parser_path'],'actual_parser_sha256',EXTRA/'actual-time-parser-source.py')]:
        require(sha(actual)==sha(snapshot)==runtime[key],'Actual/supplement source bytes differ')
    require(sha(runtime['source_seal_path'])==runtime['source_seal_sha256'],'Runtime seal changed')
    require(launch['supervisor_source_sha256']==runtime['actual_runtime_entry_sha256'] and launch['actual_runtime_parser_sha256']==runtime['actual_parser_sha256'] and launch['actual_runtime_source_seal_sha256']==runtime['source_seal_sha256'],'Actual Hidden launcher source mismatch')
    directory=LOCK/'trained_chunks'/a.chunk;receipt_file=directory/'creator-outcome.json'
    tracked=[source,LOCK/'driver-source.py',LOCK/'configuration.json',LOCK/'plan.json',d.BINDING_PATH,receipt_file,
             directory/'hierarchy.hier',directory/'anchors.bin',directory/'exposure.json',SUP/'supervisor-outcome.json',LAUNCH,
             EXTRA/'runtime-control-revision.json',EXTRA/'actual-runtime-entry-source.py',EXTRA/'actual-time-parser-source.py']
    before={str(f):sha(f) for f in tracked}
    helper,codec,training,_=d.frozen_helpers(LOCK)
    # No broad all5 tensor scan or global no-conflict call: only this completed Creator.
    receipt=d.creator_gate(a.chunk,config,manifest,training,codec,helper,plan['camera_archives'])
    require(sup['actual_final_stage']['receipt_path']==str(receipt_file) and sup['actual_final_stage']['receipt_sha256']==sha(receipt_file) and sup['actual_final_stage']['model_sha256']==receipt['hierarchy_sha256'],'Native model receipt differs from actual terminal supervision')
    origin=receipt['training_origin'];require(sha(origin['model']['path'])==origin['model']['sha256'],'Selected original model changed')
    query="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine | ConvertTo-Json -Depth 4 -Compress"
    baseline=load('_selected_creator_readonly_PS7_runtime',ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py')
    raw=subprocess.run([str(baseline.PW),'-NoProfile','-NonInteractive','-Command',query],env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True).stdout.strip()
    rows=json.loads(raw) if raw else [];rows=rows if isinstance(rows,list) else [rows]
    exact_args=d.load_module('_selected_creator_terminal_exactargv',d.PROCESS_HELPER).windows_argv
    parser=load('_selected_creator_zoned_parser',runtime['actual_parser_path'])
    identities={(int(launch['start_process_pid']),parser.creation_ms(launch['start_process_creation_utc'])),
                (sup['supervisor_pid'],sup['active_stage_lock']['supervisor_birth']['creation_Unix_ms']),
                (sup['child_pid'],sup['actual_child_birth']['creation_Unix_ms']),
                (receipt['actual_child_pid'],receipt['actual_child_birth']['creation_Unix_ms'])}
    observations=[json.loads(x) for x in (SUP/'process-observations.jsonl').read_text(encoding='utf-8').splitlines()]
    for x in observations:
        for child in x.get('actual_observed_child_descendants',[]):identities.add((int(child['ProcessId']),parser.creation_ms(child['CreationDate'])))
    conflicts=[]
    for row in rows:
        if int(row['ProcessId']) in {q[0] for q in identities}:
            if (int(row['ProcessId']),parser.creation_ms(row['CreationDate'])) in identities:conflicts.append(row)
    require(not conflicts,'Selected original process or descendant PID+birth still live')
    relevant=[r for r in rows if r['Name'].lower().startswith(('python','gaussianhierarchy'))]
    other_owners=d.gpu_process_roles(relevant,exact_args)
    active_state={'exists':d.ACTIVE_STAGE.exists(),'read_only':True,'released_by_this_audit':False}
    if d.ACTIVE_STAGE.exists():
        lock=read(d.ACTIVE_STAGE);active_state.update(current_supervision_run=lock['supervision_run'],stage=lock['stage'],chunk=lock['chunk'],sha256=sha(d.ACTIVE_STAGE))
        require(lock['supervision_run']!=run,'Selected completed stage still owns shared active lock')
    require({f:sha(f) for f in before}==before and 'torch' not in sys.modules,'Selected artifacts/source changed during CPU audit')
    report={'status':'actual_selected_Creator_terminal_source_model_and_runtime_supplements_CPU_verified',
          'UTC':datetime.now(timezone.utc).isoformat(),'chunk':a.chunk,'supervision_run':run,'source_before_after':before,
          'actual_hierarchy':receipt['hierarchy'],'actual_anchor_count':receipt['anchor_count'],'actual_tree_SHA':receipt['hierarchy_sha256'],
          'actual_training_origin_kind':origin['training_source_kind'],'actual_training_model_SHA':origin['model']['sha256'],
          'actual_train_camera_archive_heldout_count':receipt['creator_train_camera_archive']['heldout_pose_count'],
          'numeric_audit':receipt['numeric_audit'],'shared_sky_source':receipt['shared_skybox'],'actual_source_exposure':receipt['source_exposure'],
          'runtime_control_revision':runtime,'actual_final_supervisor_receipt_SHA':sha(SUP/'supervisor-outcome.json'),
          'selected_original_launch_supervisor_wrapper_native_and_seen_descendant_identities_absent':sorted(identities),
          'actual_current_other_GPU_stage_owners':other_owners,'current_shared_lock_observation':active_state,
          'global_GPU_free_claimed':False,'no_process_or_lock_control_performed':True,'GPU_started':False,'CUDA_initialized':False,
          'broad_all5_array_audit_repeated':False,'formalpost_or_Merger_or1404_complete_claimed':False}
    (OUT/'terminal-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'path':str(OUT/'terminal-audit.json'),'SHA':sha(OUT/'terminal-audit.json'),'chunk':a.chunk,'G':receipt['hierarchy']['gaussian_count'],'other_GPU_owner_count':len(other_owners)}))
if __name__=='__main__':main()
