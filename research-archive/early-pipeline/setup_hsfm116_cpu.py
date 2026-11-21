"""Install only small, pinned CPU interface packages into an isolated target."""
from pathlib import Path
import datetime as dt, json, os, subprocess, sys, time
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116')
H=R/'hsfm'; TARGET=Path('/opt/gs-slam-lab/shuolee-hsfm116/python')
def save(p,data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
now=dt.datetime.now(dt.timezone.utc)
q=json.loads((R/'queue-state.json').read_text())
assert q['state']=='P3_source_contract_reviewed_input_adapter_pending'
assert not (H/'cpu-setup-result.json').exists()
deadline=now+dt.timedelta(hours=2)
q.update(updated=now.isoformat(),state='P3_input_adapter_passed_isolated_dependency_setup',
         active_pid=os.getpid(),active_process_kind='P3_CPU_dependencies',
         setup_deadline_utc=deadline.isoformat(),p3_dependency_started_at=now.isoformat(),
         active_log=str(H/'cpu-setup.log'))
save(R/'queue-state.json',q)
packages=['rtmlib==0.0.16','onnxruntime==1.23.2','flatbuffers==25.2.10',
          'coloredlogs==15.0.1','humanfriendly==10.0','protobuf==4.25.8']
argv=[sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:',
      '--disable-pip-version-check','--target',str(TARGET),'--report',str(H/'cpu-pip-report.json'),*packages]
save(H/'cpu-setup-protocol.json',dict(created=now.isoformat(),deadline=deadline.isoformat(),
    pid=os.getpid(),command=argv,isolated_target=str(TARGET),model_weights_downloaded=False,
    base_environments_modified=False,timeout_seconds=600))
t=time.monotonic()
with (H/'cpu-setup.log').open('w') as log:
    try:
        p=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=600);code=p.returncode
    except subprocess.TimeoutExpired:code=124
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='completed' if code==0 else 'failed',
            exit_code=code,seconds=time.monotonic()-t,pid=os.getpid(),gpu=False)
save(H/'cpu-setup-result.json',result)
q=json.loads((R/'queue-state.json').read_text());q.update(updated=result['updated'],
    active_pid=None,active_process_kind=None,state='P3_input_adapter_passed_asset_runtime_check_pending' if code==0 else 'P3_dependency_failure',
    setup_state=result,next_action='Verify RTMW ONNX body23 schema in isolated target. Read hsfm/adapter-check.json and asset-compatibility.json. No source or subject fitting has run; DUSt3R checkpoint acquisition remains pending.')
save(R/'queue-state.json',q);print(json.dumps(result),flush=True)
raise SystemExit(code)
