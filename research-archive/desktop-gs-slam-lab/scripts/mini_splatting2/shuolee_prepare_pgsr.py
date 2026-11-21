"""Prepare an isolated PGSR runtime without replacing existing training libraries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
HOME=LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
SOURCE=HOME/'code/pgsr'
RUNTIME=Path('/opt/shuolee/pgsr-runtime')
OUT=HOME/'current/pgsr-surface'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=False)
    assert not RUNTIME.exists(),'Inspect an existing runtime rather than overwrite it'
    RUNTIME.mkdir(parents=True)
    shutil.copytree(SOURCE,RUNTIME/'repo',ignore=shutil.ignore_patterns('.git','assets','__pycache__','.DS_Store'))
    assert (RUNTIME/'repo/submodules/diff-plane-rasterization/third_party/glm/glm/glm.hpp').is_file()
    manifest=dict(status='building',source_commit='de24f1a38b350387e8d8fe381b2cd70c1ae946e7',
        source_url='https://github.com/zju3dv/PGSR',owner='shuolee',runtime=str(RUNTIME),
        baseline='Unmodified official renderer; experiment-specific masks will be tracked separately.',
        script_sha256=sha(Path(__file__)),started=time.time())
    dump(OUT/'runtime.json',manifest)
    command=[sys.executable,'-m','pip','install','--no-build-isolation','--no-deps','--target',str(RUNTIME/'deps'),
        str(RUNTIME/'repo/submodules/diff-plane-rasterization')]
    with (OUT/'build.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    manifest.update(status='compiled' if result.returncode==0 else 'build_failed',returncode=result.returncode,
        elapsed_seconds=time.time()-manifest['started'])
    dump(OUT/'runtime.json',manifest)
    assert result.returncode==0,'Inspect build.log'
    sys.path[:0]=[str(RUNTIME/'deps'),str(RUNTIME/'repo'),'/opt/gs-slam-lab/mini-splatting2-native-001/deps']
    import torch
    import diff_plane_rasterization as dpr
    from pytorch3d.transforms import quaternion_to_matrix
    from simple_knn._C import distCUDA2
    q=torch.tensor([[1.,0,0,0]],device='cuda')
    assert torch.allclose(quaternion_to_matrix(q)[0],torch.eye(3,device='cuda'))
    distances=distCUDA2(torch.rand(64,3,device='cuda'))
    assert torch.isfinite(distances).all() and (distances>0).all()
    manifest.update(status='imports_verified',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        rasterizer=dict(path=dpr._C.__file__,sha256=sha(Path(dpr._C.__file__))),cuda_ready=True,
        simple_knn='Reuse existing compatible distCUDA2 library; no shared package replaced')
    dump(OUT/'runtime.json',manifest);print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
