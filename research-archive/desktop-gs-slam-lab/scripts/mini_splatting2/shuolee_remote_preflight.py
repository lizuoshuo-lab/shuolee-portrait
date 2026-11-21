"""Read-only Linux preflight; --prepare creates only the explicit shuolee namespace."""
import argparse
import getpass
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
from datetime import datetime, timezone


def command(argv):
    try:
        p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=20)
        return {'exit_code':p.returncode,'output':p.stdout.strip()}
    except (FileNotFoundError,subprocess.TimeoutExpired) as error:
        return {'error':type(error).__name__}


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');args=p.parse_args()
    assert platform.system()=='Linux', 'This file is for the authenticated Linux server.'
    home=Path('/home/zhouhaoyi').resolve(strict=True)
    assert getpass.getuser()=='zhouhaoyi' and os.getuid()!=0, 'Use the existing regular training account.'
    root=home/'shuolee'
    assert not root.is_symlink()
    if root.exists(): assert root.resolve().parent==home
    result={'owner':'shuolee','checked_at':datetime.now(timezone.utc).isoformat(),
        'hostname':socket.gethostname(),'account':getpass.getuser(),'platform':platform.platform(),
        'workspace':str(root),'disk':shutil.disk_usage(str(home))._asdict(),
        'gpus':command(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
        'gpu_processes':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']),
        'existing_envs':sorted(x.name for x in (home/'miniconda3/envs').glob('*') if x.is_dir()),
        'training_started':False,'prepared':False}
    if args.prepare:
        marker=root/'workspace.json'
        if root.exists() and any(root.iterdir()):
            assert marker.is_file() and json.loads(marker.read_text()).get('owner')=='shuolee', 'Unrecognized existing directory; inspect before reuse.'
        root.mkdir(exist_ok=True)
        for name in ['code','data','runs','best']:
            target=root/name
            assert not target.is_symlink()
            target.mkdir(exist_ok=True)
        result['prepared']=True
        marker.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
