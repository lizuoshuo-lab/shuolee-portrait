"""Independent read-only source/start/process observation for coarse run 003.

No Torch import, GPU query, process control or training mutation. Observations
are timestamped evidence only; they are not a durability or completion claim.
"""
from datetime import datetime, timezone
import argparse, hashlib, json, os, re, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
RUN='small-city-coarse-i30000-r4-003'
TRAIN=BASE/'official-training'/RUN
SUP=BASE/'official-training-supervision'/RUN
LAUNCH=BASE/'official-training-supervision-launch'/RUN
OUT=Path(__file__).resolve().parent
EXPECTED={
 'scripts/large_scene_gs/supervise_small_city_stage.py':'b7c0015374521b201a4e28845cb9ff79ca9b4c597d5c05f80bf69aa03f58378d',
 'scripts/large_scene_gs/launch_small_city_coarse_003.ps1':'b8b5e8a4768b8bb74458a3838ce595a0110fcd480bb814ab3877df96f57ebe66',
 'scripts/large_scene_gs/run_small_city_stage.py':'dead1eab49ff455d3da04bfbd97d99570ecbb5235e9523caab457c5ab87bf704',
 'scripts/large_scene_gs/run_local.py':'1fa39a1af880809d9a8cabc62f219458358acad6b661780c6f7321df33dcc2a2',
 'repos/hierarchical-3d-gaussians/train_coarse.py':'af2ed6483b1a9f8548086f34216c04f9eeea4bdef23331023da6eb5fb41bdf52',
 'outputs/large-scene-gs-001/official-inputs/small-city-training-002/manifest.json':'2dd5c04122e290d27934a224921b2506b7955a34d801a0482e0408af536de3d5',
}

def digest(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
    return d.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def require(value,message):
    if not value:raise RuntimeError(message)
def save(path,value):
    require(not path.exists(),'Independent artifact already exists: '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sources():
    launch=read(LAUNCH/'launch-receipt.json');supervisor=read(SUP/'supervisor-outcome.json')
    config=read(TRAIN/'configuration.json');manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
    snapshots={
      'scripts/large_scene_gs/supervise_small_city_stage.py':SUP/'supervisor-source.py',
      'scripts/large_scene_gs/launch_small_city_coarse_003.ps1':LAUNCH/'launch-source.ps1',
      'scripts/large_scene_gs/run_small_city_stage.py':SUP/'stage-launcher-source.py',
      'scripts/large_scene_gs/run_local.py':SUP/'environment-launcher-source.py',
    }
    rows=[]
    for relative,expected in EXPECTED.items():
        original=ROOT/relative;actual=digest(original);require(actual==expected,'Source/input SHA changed: '+relative)
        snap=snapshots.get(relative)
        if snap:require(digest(snap)==expected,'Launch snapshot mismatch: '+str(snap))
        rows.append(dict(path=str(original),sha256=actual,snapshot_path=str(snap) if snap else None))
    require(digest(TRAIN/'launcher-source.py')==EXPECTED['scripts/large_scene_gs/run_small_city_stage.py'],'Stage-local source snapshot mismatch')
    require(digest(TRAIN/'environment-launcher-source.py')==EXPECTED['scripts/large_scene_gs/run_local.py'],'Stage-local env snapshot mismatch')
    require(launch['supervisor_source_sha256']==EXPECTED['scripts/large_scene_gs/supervise_small_city_stage.py'] and launch['launch_source_sha256']==EXPECTED['scripts/large_scene_gs/launch_small_city_coarse_003.ps1'],'Launch identity mismatch')
    require(supervisor['stage_launcher_sha256']==config['launcher_source_sha256']==EXPECTED['scripts/large_scene_gs/run_small_city_stage.py'],'Wrapper identity mismatch')
    require(supervisor['environment_launcher_sha256']==config['environment_launcher_sha256']==EXPECTED['scripts/large_scene_gs/run_local.py'],'Environment launcher identity mismatch')
    require(supervisor['input_manifest_sha256']==config['input_manifest_sha256']==launch['input_manifest_sha256']==EXPECTED['outputs/large-scene-gs-001/official-inputs/small-city-training-002/manifest.json'],'Input identity mismatch')
    require(Path(launch['working_directory']).resolve()==Path(supervisor['cwd']).resolve()==ROOT.resolve(),'Wrong cwd')
    require(supervisor['run']==RUN and supervisor['stage']==config['stage']=='coarse' and config['requested_iterations']==supervisor['requested_iterations']==30000 and config['expected_optimizer_update_calls']==29999,'Wrong stage/iteration contract')
    args=config['effective_official_arguments']
    require(args['eval'] is True and args['train_test_exp'] is False and args['resolution']==4 and config['resolution_pixels']==[256,172] and config['exposure_parameter_updates_enabled'] is False,'Wrong frozen eval/resolution/exposure configuration')
    require(args['skybox_num']==100000 and not args['hierarchy'] and not args['pretrained'] and not args['scaffold_file'] and not args['bounds_file'],'Wrong coarse inputs')
    require(config['training_names']==manifest['train_names'] and config['stage_test_names']==manifest['test_names'] and config['frozen_global_test_names']==manifest['test_names'] and len(manifest['train_names'])==5705 and len(manifest['test_names'])==117,'Frozen split mismatch')
    require(Path(args['model_path']).resolve()==(TRAIN/'model').resolve(),'Wrong model destination')
    command=supervisor['command_argv']
    require('--execute' in command and command[command.index('--run')+1]==RUN and command[command.index('--input')+1]=='small-city-training-002' and command[command.index('--iterations')+1]=='30000','Wrong actual child argv')
    require(supervisor['recovery_kind']=='from_scratch' and not supervisor['checkpoint_resume'] and not launch['checkpoint_resume'],'This run must be new from scratch, not resume')
    require(config['official_commit']=='596dc7081ee74e57a66502f4746328911f763b34' and config['trainer_source_sha256']==EXPECTED['repos/hierarchical-3d-gaussians/train_coarse.py'] and not config['source_changed'],'Wrong unchanged official trainer')
    interruption_path=BASE/'recovery-audit/small-city-coarse-i30000-r4-002-interruption-001/interruption-audit.json'
    interruption=read(interruption_path)
    require(interruption['status']=='interrupted_no_live_python_no_saved_training_state' and not interruption['optimizer_checkpoint_files'] and not interruption['final_model_files'],'Recovery evidence differs')
    return dict(status='independent_CPU_source_launch_split_contract_passed',source_rows=rows,launch_receipt_sha256=digest(LAUNCH/'launch-receipt.json'),configuration_sha256=digest(TRAIN/'configuration.json'),
        from_scratch=True,checkpoint_resume=False,requested_iterations=30000,source_optimizer_calls_if_completed=29999,train_count=5705,test_count=117,
        sky_prefix=100000,source_GaussianModel_SH_degree=1,config_sh_degree_3_overridden_by_fixed_source_constructor=True,
        supervision_source_review='CPU-only wrapper starts unchanged stage through fixed environment; polls child returncode, preserves errors, no checkpoint/restart/code alteration. Completion receipt requires childexit0 plus stagecompleted.',
        interrupted_002_audit_path=str(interruption_path),interrupted_002_audit_sha256=digest(interruption_path),interrupted_002_exit_code=interruption['exit_code'],interrupted_002_cause=interruption['cause'],
        windows_job_membership_scope='Membership only. No promise for Codex app exit/jobclosure, reboot or logout.',cuda_called=False)

def observation(label):
    # No process mutation and no interpolated shell command.
    command="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    ps=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    queried=datetime.now(timezone.utc).isoformat()
    result=subprocess.run([str(ps),'-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
    require(result.returncode==0,'Fresh GetCim failed: '+result.stderr)
    all_processes=json.loads(result.stdout or '[]');all_processes=all_processes if isinstance(all_processes,list) else [all_processes]
    matches=[p for p in all_processes if p['Name'].lower()=='python.exe' and RUN in (p.get('CommandLine') or '')]
    trainer=[p for p in matches if str(ROOT/'repos/hierarchical-3d-gaussians/train_coarse.py').replace('/','\\') in (p.get('CommandLine') or '')]
    sup=[p for p in matches if 'supervise_small_city_stage.py' in (p.get('CommandLine') or '')]
    launch=read(LAUNCH/'launch-receipt.json');launch_pid=int(launch['launching_powershell_pid'])
    launch_alive=any(p['ProcessId']==launch_pid for p in all_processes)
    path=TRAIN/'training.log';stat=path.stat()
    with path.open('rb') as f:
        f.seek(max(0,stat.st_size-65536));tail=f.read().decode('utf-8',errors='replace')
    progress=re.findall(r'(\d+)/(\d+)\s*\[',tail)
    require(progress,'No actual TQDM progress observed')
    actual=int(progress[-1][0]);budget=int(progress[-1][1]);require(budget==30000,'Actual log budget mismatch')
    status='independent_live_pipeline_and_log_observed' if trainer and sup and not launch_alive else 'independent_process_observation_requires_review'
    return dict(status=status,label=label,observed_utc=queried,fresh_Win32_Process_query=True,actual_processes=matches,
        actual_trainer_pids=[p['ProcessId'] for p in trainer],actual_supervisor_pids=[p['ProcessId'] for p in sup],launching_powershell_pid=launch_pid,launching_powershell_still_present=launch_alive,
        training_log_path=str(path),training_log_bytes=stat.st_size,training_log_modified_utc=datetime.fromtimestamp(stat.st_mtime,timezone.utc).isoformat(),last_logged_iterations=actual,last_logged_budget=budget,
        recorded_stage_outcome=read(TRAIN/'outcome.json'),liveness_from_fresh_process_query_not_running_status=True,
        limitations=['Only process existence/creationtime/command and log observation at this timestamp.','TQDM iteration is not independently counted optimizer calls.','PowerShell launch exited while downstream processes exist; no promise across app/job closure, reboot or logout.','No completion/model binding inferred from live process or log progress.'],cuda_called=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',required=True);a=p.parse_args();require(a.sample.replace('-','').isalnum(),'Invalid sample')
    source=sources();path=OUT/'independent-003-source-launch-audit.json'
    if not path.exists():save(path,source)
    else:require(read(path)==source,'Source audit changed between samples')
    result=observation(a.sample);save(OUT/f'independent-003-process-{a.sample}.json',result)
    print(json.dumps({k:result[k] for k in ('status','observed_utc','actual_trainer_pids','actual_supervisor_pids','launching_powershell_still_present','last_logged_iterations','cuda_called')}))

if __name__=='__main__':main()
