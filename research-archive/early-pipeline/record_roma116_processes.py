"""Read actual relevant process identities and GPU state, never start a job."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess

R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/roma')
matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0');name=Path(args[0].decode()).name;cmd=b' '.join(args).decode()
    except (OSError,UnicodeDecodeError,IndexError):continue
    if ('python' in name or name in ['bash','sh','timeout']) and any(x in cmd for x in ['run_pixsfm116_ba.py','run_roma116_diagnostic.py','train_smplx.py','train_avatar_pair_114','supervise_avatar_pair_114']):
        matches.append(dict(pid=int(p.name),command=cmd))
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),matching_processes=matches,gpu_query_exit_code=gpu.returncode,gpu_compute_processes=gpu.stdout.strip().splitlines(),gpu_query_stderr=gpu.stderr.strip(),new_job_started=False)
(R/'process-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
