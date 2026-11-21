"""One local dependency chain for the already-running City006 evaluation.

Never launches, restarts, kills, modifies or replaces the GPU evaluation.
Waits on its birth-bound Windows process handle, requires actual exit zero
and the completed outcome, then runs the three sealed CPU verifiers once.
The boundary output is an image index; manual visual review remains required.
"""
from pathlib import Path
from datetime import datetime, timezone
from ctypes import wintypes
import argparse, ctypes, hashlib, json, os, subprocess, sys, time, traceback

ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
EVALS=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
RESEARCH=ROOT/'outputs/large-scene-gs-001/research/city-heldout-postprocessing-018'
PYTHON=ROOT/'.venv-large-gs/Scripts/python.exe'
RUN='small-city006-heldout-1404-001'
POST='small-city006-heldout-posterior-CPU-001'
METRIC='small-city006-heldout-perceptual-CPU-001'
BOUNDARY='small-city006-heldout-boundary-CPU-001'
PID=22296
# GetProcessTimes FILETIME, preserving the final 100 ns digit omitted by CIM.
BIRTH=134362982304856438
EXPECTED_STATUS='complete117_each_requested_actual_local_pre_post_setting'
SCRIPTS=[
 (EVALS/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py','5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c'),
 (EVALS/'city-perceptual-CPU-reference-driver-006/recompute_city_perceptual_cpu_006.py','9e42b40079f6d7ea73c069997fdb23121b5e24f00f5409cfd4f09830d40e8a0c'),
 (EVALS/'city-boundary-image-binder-CPU-006/audit_city_boundary_images_006.py','f16ccb274707515a29cf3466be9c294b206b08a2cd1404f05321a99c88669379'),
]
ADAPTER_SHA='260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3'
CORE_SHA='672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec'
def now(): return datetime.now(timezone.utc).isoformat()
def require(v,m):
 if not v: raise RuntimeError(m)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
 temp=p.with_suffix('.tmp')
 temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 os.replace(temp,p)
def source_check():
 for p,h in SCRIPTS: require(p.is_file() and sha(p)==h,'Sealed CPU source missing or changed: '+str(p))
 require(PYTHON.is_file(),'Pinned Python missing')
def api():
 k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
 k.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4;k.GetProcessTimes.restype=wintypes.BOOL
 k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
 k.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)];k.GetExitCodeProcess.restype=wintypes.BOOL
 k.CloseHandle.argtypes=[wintypes.HANDLE];k.CloseHandle.restype=wintypes.BOOL
 return k
def birth_of(k,h):
 parts=[wintypes.FILETIME() for _ in range(4)]
 if not k.GetProcessTimes(h,*[ctypes.byref(p) for p in parts]): raise ctypes.WinError(ctypes.get_last_error())
 return (parts[0].dwHighDateTime<<32)|parts[0].dwLowDateTime
def steps():
 return [
  dict(name='posterior',run=POST,script=SCRIPTS[0][0],args=['--audit-run',POST,'--evaluation-run',RUN],status='actual1404_complete_independent_CPU_posterior_audit_passed'),
  dict(name='perceptual',run=METRIC,script=SCRIPTS[1][0],args=['--audit-run',METRIC,'--scope','own1404','--threads','4','--evaluation-run',RUN,'--posterior-audit',str(EVALS/POST/'completion-audit.json')],status='completed_future_actual1404_CPU_perceptual_wrapper'),
  dict(name='boundary',run=BOUNDARY,script=SCRIPTS[2][0],args=['--mode','dense','--run',BOUNDARY,'--evaluation-run',RUN,'--posterior-audit-run',POST],status='completed_actual_city_geometry_selected_original_boundary_image_index'),
 ]
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');a=parser.parse_args()
 source_check()
 for name in (POST,METRIC,METRIC+'-actual1404',BOUNDARY):require(not (EVALS/name).exists(),'Output namespace already exists; do not rerun: '+name)
 require(not RESEARCH.exists(),'Dependency-chain directory already exists; inspect existing process instead')
 k=api();h=k.OpenProcess(0x1000|0x100000,False,PID)
 if not h: raise ctypes.WinError(ctypes.get_last_error())
 try:
  actual_birth=birth_of(k,h);require(actual_birth==BIRTH,'Evaluation PID birth mismatch')
  require(k.WaitForSingleObject(h,0)==258,'Evaluation no longer running; inspect its real terminal result before adapting')
  plan=dict(status='plan_checked_no_child_started',evaluation_run=RUN,evaluation_pid=PID,evaluation_birth_filetime=BIRTH,CPU_steps=[dict(name=s['name'],argv=[str(PYTHON),'-B','-X','utf8',str(s['script'])]+s['args'],run=s['run'],expected_status=s['status']) for s in steps()],sealed_sources=[dict(path=str(p),sha256=q) for p,q in SCRIPTS],no_GPU_restart=True,manual_image_review_still_required=True)
  if not a.run:
   print(json.dumps(plan,ensure_ascii=False));return
  RESEARCH.mkdir(parents=True,exist_ok=False)
  save(RESEARCH/'plan.json',plan)
  (RESEARCH/'source-snapshot.py').write_bytes(Path(__file__).read_bytes())
  state=dict(status='waiting_for_bound_GPU_evaluation',started_utc=now(),supervisor_pid=os.getpid(),evaluation_run=RUN,evaluation_pid=PID,evaluation_birth_filetime=BIRTH,source_SHA=sha(__file__),steps=[],manual_boundary_review_complete=False,final_site_update_complete=False,final_PPT_complete=False)
  save(RESEARCH/'outcome.json',state)
  try:
   while True:
    wait=k.WaitForSingleObject(h,0)
    if wait==0:break
    require(wait==258,'Unexpected Windows process-wait result')
    time.sleep(30)
   code=wintypes.DWORD()
   if not k.GetExitCodeProcess(h,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
   require(code.value==0,'GPU evaluation actual process exit was not zero: '+str(code.value))
   outcome=read(EVALS/RUN/'outcome.json')
   require(outcome.get('status')==EXPECTED_STATUS and outcome.get('cuda_used') is True,'GPU result has not completed all actual settings')
   require(outcome.get('driver_sha256')==ADAPTER_SHA and outcome.get('core_evaluator_sha256')==CORE_SHA,'GPU evaluator identity differs')
   state.update(status='running_CPU_verifiers',evaluation_process_exit=int(code.value),evaluation_outcome_sha256=sha(EVALS/RUN/'outcome.json'),GPU_completed_utc=now())
   save(RESEARCH/'outcome.json',state)
   for s in steps():
    source_check();require(not (EVALS/s['run']).exists(),'Target output appeared before dispatch: '+s['run'])
    argv=[str(PYTHON),'-B','-X','utf8',str(s['script'])]+s['args']
    record=dict(name=s['name'],run=s['run'],argv=argv,status='starting',started_utc=now());state['steps'].append(record)
    with (RESEARCH/(s['name']+'.log')).open('xb') as log:
     child=subprocess.Popen(argv,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
     record.update(status='running',child_pid=child.pid);save(RESEARCH/'outcome.json',state)
     returncode=child.wait()
    record.update(exit_code=returncode,ended_utc=now());require(returncode==0,'CPU stage failed; original outputs preserved: '+s['name'])
    result=read(EVALS/s['run']/'outcome.json')
    require(result['status']==s['status'],'CPU stage has unexpected result status: '+s['name']+' '+str(result['status']))
    record.update(status='completed',outcome_SHA=sha(EVALS/s['run']/'outcome.json'));save(RESEARCH/'outcome.json',state)
   state.update(status='CPU_verifiers_complete_manual_review_and_delivery_pending',ended_utc=now());save(RESEARCH/'outcome.json',state)
   print(json.dumps(dict(status=state['status'],directory=str(RESEARCH)),ensure_ascii=False))
  except BaseException as e:
   state.update(status='failed_preserved_no_automatic_restart',ended_utc=now(),error=str(e),traceback=traceback.format_exc());save(RESEARCH/'outcome.json',state);raise
 finally:k.CloseHandle(h)
if __name__=='__main__':main()
