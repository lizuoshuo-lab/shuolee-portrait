"""Bounded P2 source/runtime provenance; no downloads of model weights or subject inference."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116')
OUT=ROOT/'roma'
BASE=Path('/opt/gs-slam-lab/shuolee-roma116')
REPO=BASE/'RoMa'
COMMIT='77f8d68803526dcddfd9b7a46bc76125bdc25f15'
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def write(path,data):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8');p.replace(path)
def command(args,timeout=90):
    p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    if p.returncode:raise RuntimeError(f'{args[:3]} failed {p.returncode}: {p.stdout[-5000:]}')
    return p.stdout.strip()
def main():
    assert not OUT.exists(),'Do not overwrite or rerun P2 preflight blindly'
    OUT.mkdir();BASE.mkdir(exist_ok=True)
    pids=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:args=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except (OSError,UnicodeDecodeError):continue
        if any(x in args for x in ('run_pixsfm116_ba.py','evaluate_pixsfm116_candidate.py','run_roma116_diagnostic.py','train_smplx.py','train_avatar')):pids.append(dict(pid=int(p.name),args=args))
    assert not pids,pids
    gpu=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    prereg=dict(created=now(),candidate='P2',stage='bounded_source_runtime_preflight',pid=os.getpid(),setup_budget_hours=2,
        repository='https://github.com/Parskatt/RoMa',commit=COMMIT,one_gpu_task=True,weight_downloads=False,
        competing_hypotheses=['Existing matches shift between different material points as view changes','Estimated cameras still misproject a correctly identified material point','Sequential subject changes or view-dependent appearance break a static observation model'],
        fixed_conditions=['Original raw image identities and source074 immutable','P1 frozen validation bank unchanged','No 2D portrait warps or generated images','No Gaussian training'],
        next='Inspect official source and known cache paths; preregister fixed image triples and geometric stopping criteria before inference',
        stop_on=['Missing required weights: record exact official URLs/size and do not silently substitute another model','Unchanged setup failure: do not repeatedly retry','Changed source or incompatible APIs'],
        process_check=pids,gpu_processes=gpu)
    write(OUT/'preregistration.json',prereg)
    if not REPO.exists():
        command(['git','clone','--filter=blob:none','--no-checkout','https://github.com/Parskatt/RoMa',str(REPO)],180)
        command(['git','-C',str(REPO),'checkout','--detach',COMMIT],120)
    assert command(['git','-C',str(REPO),'rev-parse','HEAD'])==COMMIT
    assert not command(['git','-C',str(REPO),'status','--porcelain'])
    import torch
    hub=Path(torch.hub.get_dir())/'checkpoints'
    weights=[]
    for filename in ['roma_outdoor.pth','dinov2_vitl14_pretrain.pth']:
        path=hub/filename
        item=dict(filename=filename,path=str(path),exists=path.is_file(),bytes=path.stat().st_size if path.is_file() else None)
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
            item['sha256']=h.hexdigest()
        weights.append(item)
    result=dict(updated=now(),candidate='P2',state='source_checked_runtime_import_pending',python=sys.version,executable=sys.executable,
        repo=str(REPO),commit=COMMIT,torch=torch.__version__,cuda=torch.version.cuda,weights=weights,
        dependencies={x:bool(importlib.util.find_spec(x)) for x in ['torchvision','timm','kornia','cv2','einops','loguru','PIL','matplotlib']},
        model_download_started=False,subject_inference_started=False,gaussian_training_started=False)
    write(OUT/'preflight.json',result)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
