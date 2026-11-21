"""Pinned sources through process-local paths, no upstream setup.py or weights."""
from pathlib import Path
import datetime as dt, hashlib, importlib.util, json, sys, zipfile
H=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/hsfm')
R=Path('/opt/gs-slam-lab/shuolee-hsfm116')
spec=json.loads((H/'runtime-sources.json').read_text()); receipts=[]
for item in spec['packages']:
    src=H/item['archive'];destroot=R/item['name'];prefix=item['name']+'-'+item['commit']+'/'
    assert not destroot.exists(), 'Inspect existing source; do not replace'
    with zipfile.ZipFile(src) as z:
        for info in z.infolist():
            assert info.filename.startswith(prefix)
            rel=info.filename[len(prefix):]
            if not rel:continue
            dest=destroot/rel
            assert dest.resolve().is_relative_to(destroot.resolve())
            assert ((info.external_attr>>16)&0o170000)!=0o120000
            if info.is_dir():dest.mkdir(parents=True,exist_ok=True)
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(info))
    receipts.append(dict(**item,path=str(destroot),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
sys.path[:0]=[str(R/'dust3r'),str(R/'croco'),str(R/'python')]
import torch
def forbid(*args,**kwargs):raise RuntimeError('Model downloads forbidden during source import checks')
torch.hub.load=forbid;torch.hub.load_state_dict_from_url=forbid
deps={x:bool(importlib.util.find_spec(x)) for x in ['dust3r','croco','torch','torchvision','roma','trimesh','scipy','einops','huggingface_hub','pyglet','tensorboard','matplotlib']}
try:
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    from dust3r.inference import inference
    imported=True;error=None
except Exception as e: imported=False;error=repr(e)
out=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),sources=receipts,dependencies=deps,
    imports_verified=imported,error=error,torch_version=torch.__version__,cuda=torch.version.cuda,
    state='imports_ready_weight_validation_pending' if imported else 'import_failure',
    gpu_used=False,subject_inference=False,official_old_torch_pin_not_installed=True,
    disclosure='Pinned HSfM forks with Torch2.7.1 and official PyTorch RoPE fallback; runtime compatibility needs actual model forward.')
(H/'runtime-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
