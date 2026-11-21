"""Build Mini-Splatting2's own CUDA kernels in an isolated local copy."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/mini-splatting2-mip-depth-001/native-runtime'
RUNTIME=Path('/opt/gs-slam-lab/mini-splatting2-native-001')

def main():
    if os.name=='nt':raise RuntimeError('Run inside WSL with CUDA environment sourced')
    OUT.mkdir(exist_ok=True);RUNTIME.mkdir(exist_ok=True)
    repo=RUNTIME/'repo';source=LAB/'repos/mini-splatting2'
    if not repo.exists():
        shutil.copytree(source,repo,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__'))
    # CUDA 12.8 no longer supplies FLT_MAX transitively. Explicit standard
    # header only; the original Windows checkout stays untouched.
    knn=repo/'submodules/simple-knn/simple_knn.cu'
    original=knn.read_text()
    if '#include <cfloat>' not in original:knn.write_text('#include <cfloat>\n'+original)
    env=os.environ.copy();env.update(TORCH_CUDA_ARCH_LIST='12.0',MAX_JOBS='2')
    target=RUNTIME/'deps';target.mkdir(exist_ok=True)
    env['PYTHONPATH']=str(target)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    mp=OUT/'manifest.json';attempt=1
    if mp.exists():
        previous=json.loads(mp.read_text());attempt=previous.get('attempt',1)+1
        (OUT/f'manifest-attempt-{attempt-1}.json').write_text(mp.read_text())
    manifest=dict(status='running',attempt=attempt,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  runtime=str(RUNTIME),source_commit='5ec9202c2db7d900728bfc09733e85467cff885e',
                  cuda_arch='12.0',shared_environment_modified=False,builds=[],
                  compatibility_patches=['simple_knn.cu: explicit cfloat header for CUDA12.8'])
    def save():mp.write_text(json.dumps(manifest,indent=2))
    save()
    for name in ['simple-knn','diff-gaussian-rasterization','diff-gaussian-rasterization_ms','fused-ssim']:
        command=[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(target),str(repo/'submodules'/name)]
        manifest['current']=name;save();start=time.time()
        with (OUT/f'{name}-attempt-{attempt}.log').open('w') as log:
            result=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        manifest['builds'].append(dict(name=name,returncode=result.returncode,elapsed_seconds=time.time()-start))
        save();print(manifest['builds'][-1],flush=True)
        if result.returncode:
            manifest['status']='failed';save();return result.returncode
    code='import torch; from simple_knn._C import distCUDA2; import diff_gaussian_rasterization, diff_gaussian_rasterization_ms, fused_ssim; x=torch.rand(100,3,device="cuda"); y=distCUDA2(x); assert torch.isfinite(y).all(); print(torch.__version__,torch.cuda.get_device_name(),y.shape)'
    result=subprocess.run([sys.executable,'-c',code],env=env,cwd=repo,capture_output=True,text=True)
    manifest['import_probe']=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
    manifest['status']='complete' if result.returncode==0 else 'probe_failed';save();print(manifest['status'],flush=True)
    return result.returncode

if __name__=='__main__':raise SystemExit(main())
