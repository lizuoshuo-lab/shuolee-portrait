"""Read-only PS7/CIM GPU-role gates; no Torch, CUDA or process dispatch."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[4]
PW=Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')
PW_SHA='362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139'
ARGV=ROOT/'scripts/large_scene_gs/observe_small_city_process_roles.py'
ARGV_SHA='cae238d2e692230d40c91c72d7a5a7de05ba0d759506e6c002a40fa078deed8a'
NATIVE={'gaussianhierarchycreator.exe','gaussianhierarchymerger.exe','mergerunoptimized.exe','gaussianhierarchymergerunoptimized.exe'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise RuntimeError(m)

def query(mode,evaluator_source=None,launcher_source=None):
    require(mode in ('pre_gpu','terminal'),'Unknown process-query mode')
    require(sha(PW)==PW_SHA and sha(ARGV)==ARGV_SHA,'Verified PS7/exact Windows argv reader changed')
    inherited=os.environ.get('PSModulePath')
    environment=os.environ.copy()
    environment['PSModulePath']=os.pathsep.join([str(PW.parent/'Modules'),str(Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/Modules')])
    command="$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CreationDate,CommandLine) | ConvertTo-Json -Depth 3 -Compress"
    r=subprocess.run([str(PW),'-NoLogo','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW,env=environment)
    require(r.returncode==0,'Actual private-module PS7/CIM query failed: '+r.stderr)
    processes=json.loads(r.stdout or '[]');processes=processes if isinstance(processes,list) else [processes]
    spec=importlib.util.spec_from_file_location('_exact_windows_argv006',ARGV);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    byid={p['ProcessId']:p for p in processes};own={os.getpid()};ancestor_proofs=[]
    target=Path(evaluator_source).resolve() if evaluator_source is not None else None
    launcher=Path(launcher_source).resolve() if launcher_source is not None else ROOT/'scripts/large_scene_gs/run_local.py'
    if mode=='pre_gpu':
        require(target is not None,'Exact evaluator source required for its ancestor exclusion')
        parent=os.getppid()
        while parent in byid:
            row=byid[parent]
            if (row.get('Name') or '').lower() not in {'python.exe','pythonw.exe'}:break
            argv=parser.windows_argv(row.get('CommandLine') or '')
            scripts=[Path(v).resolve() for v in argv[1:] if v.lower().endswith('.py')]
            entry=scripts[0] if scripts else None
            same_eval=entry==target
            same_launcher=entry==launcher.resolve() and target in scripts
            if not(same_eval or same_launcher):break
            own.add(parent);ancestor_proofs.append(dict(row,parsed_argv=argv,reason='exact own evaluator script/launcher and real ancestor chain'));parent=row['ParentProcessId']
    conflicting=[];classified=[]
    for row in processes:
        name=(row.get('Name') or '').lower()
        python=name in {'python.exe','pythonw.exe'}
        if name not in NATIVE and not python:continue
        argv=parser.windows_argv(row.get('CommandLine') or '')
        names={Path(s).name.lower() for s in argv[1:] if s.lower().endswith('.py')}
        roles=[]
        if name in NATIVE:roles.append('native_hierarchy_stage')
        if python:
            if names&{'train_coarse.py','train_single.py','train_post.py'}:roles.append('official_trainer')
            if '--gpu' in argv and any(n.startswith('evaluate_') for n in names):roles.append('GPU_evaluator')
            if '--execute' in argv and any(n.startswith(('run_small_city_hierarchy','run_small_city_post_preflight','run_small_city_observation','observe_small_city_train_single','supervise_small_city_hierarchy_stage')) or n=='run_small_city_stage.py' for n in names):roles.append('GPU_stage_wrapper_or_supervisor')
        if roles:
            entry=dict(row,parsed_argv=argv,roles=roles,excluded_own_pid=row['ProcessId'] in own)
            classified.append(entry)
            if row['ProcessId'] not in own:conflicting.append(entry)
    require(os.environ.get('PSModulePath')==inherited,'Parent PSModulePath unexpectedly changed')
    return dict(observed_utc=datetime.now(timezone.utc).isoformat(),fresh_Win32_Process_query=True,
        verified_PS7_path=str(PW),verified_PS7_sha256=PW_SHA,child_local_PSModulePath=environment['PSModulePath'],parent_PSModulePath_unchanged=True,
        exact_argv_reader_sha256=ARGV_SHA,actual_conflicting_GPU_stage_processes=conflicting,actual_classified_processes=classified,
        actual_process_snapshot=processes,excluded_own_pids=sorted(own),exact_ancestor_proofs=ancestor_proofs,
        scope='Actual process PID/parent/birth/argv at this timestamp. Temporary CPU role controls are identified separately. No dispatch/termination, Torch, CUDA or GPU completion query.')
