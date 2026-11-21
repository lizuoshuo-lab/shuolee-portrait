"""Preserve001 CPU CIM encoding failure; use explicit UTF8/PS7 private environment in002."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
src=ROOT/'scripts/large_scene_gs/audit_city006_selected_creator_terminal_CPU_001.py'
fail=ROOT/'outputs/large-scene-gs-001/recovery-audit/city006-creator01-independent-terminal-CPU-001/failure.json'
assert not fail.exists();fail.write_text(json.dumps({'status':'CPU_terminal_audit_failed_preserved','actual_exit_code':1,
    'source_SHA':hashlib.sha256(src.read_bytes()).hexdigest(),'error':'GBK stdout decoding of full Win32 CIM Unicode commandlines failed in CPU observer, then stdout None',
    'completed_creator_model_source_and_runtime_checks_reached':True,'model_or_stage_fault_claimed':False,
    'GPU_or_lock_or_process_control_performed':False},indent=2)+'\n',encoding='utf-8')
s=src.read_text(encoding='utf-8').replace("'-independent-terminal-CPU-001'","'-independent-terminal-CPU-002'")
s=s.replace("query=\"Get-CimInstance", "query=\"$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-CimInstance")
s=s.replace("raw=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],capture_output=True,text=True,check=True).stdout.strip()", "baseline=load('_selected_creator_readonly_PS7_runtime',ROOT/'scripts/large_scene_gs/supervise_small_city_hierarchy_stage_006.py')\n    raw=subprocess.run([str(baseline.PW),'-NoProfile','-NonInteractive','-Command',query],env=baseline.PW_ENV,capture_output=True,text=True,encoding='utf-8',errors='strict',check=True).stdout.strip()")
target=ROOT/'scripts/large_scene_gs/audit_city006_selected_creator_terminal_CPU_002.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
