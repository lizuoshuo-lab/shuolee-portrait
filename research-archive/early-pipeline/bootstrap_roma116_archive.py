"""Recover only failed source transport; inspect the pinned official model without fetching weights."""
from pathlib import Path
import datetime as dt
import hashlib
import importlib.metadata as md
import importlib.util
import inspect
import json
import sys
import zipfile

OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/roma')
BASE=Path('/opt/gs-slam-lab/shuolee-roma116')
REPO=BASE/'RoMa'
COMMIT='77f8d68803526dcddfd9b7a46bc76125bdc25f15'
ZIP=OUT/'RoMa-source-77f8d688.zip'
def write(path,data):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');p.replace(path)
assert (OUT/'source-transport-failure.json').is_file()
assert not (OUT/'preflight.json').exists()
assert not REPO.exists()
with zipfile.ZipFile(ZIP) as z:
    prefix='RoMa-'+COMMIT+'/'
    assert all(x.filename.startswith(prefix) for x in z.infolist())
    for item in z.infolist():
        rel=item.filename[len(prefix):]
        if not rel:continue
        dest=REPO/rel
        assert dest.resolve().is_relative_to(REPO.resolve())
        assert ((item.external_attr>>16)&0o170000)!=0o120000,'No archive symlink allowed'
        if item.is_dir():dest.mkdir(parents=True,exist_ok=True)
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
import torch
def no_download(*a,**kw):raise RuntimeError('Network model downloads disabled in P2 source/coordinate preflight')
torch.hub.load_state_dict_from_url=no_download
torch.hub.load=no_download
sys.path.insert(0,str(REPO))
dependencies={x:bool(importlib.util.find_spec(x)) for x in ['torch','torchvision','romatch','einops','kornia','timm','cv2','loguru','h5py','albumentations','wandb','poselib']}
imports={'ok':False}
try:
    import romatch
    from romatch.models.matcher import RegressionMatcher
    imports=dict(ok=True,roma_outdoor_signature=str(inspect.signature(romatch.roma_outdoor)),module_path=str(romatch.__file__))
    (OUT/'coordinate-api-source.txt').write_text(inspect.getsource(RegressionMatcher.to_pixel_coordinates)+'\n'+inspect.getsource(RegressionMatcher.match_keypoints),encoding='utf8')
except Exception as e:imports=dict(ok=False,error=repr(e))
weights=[]
for name in ['roma_outdoor.pth','dinov2_vitl14_pretrain.pth']:
    p=Path(torch.hub.get_dir())/'checkpoints'/name
    weights.append(dict(filename=name,path=str(p),exists=p.is_file(),bytes=p.stat().st_size if p.is_file() else 0))
result=dict(updated=dt.datetime.now(dt.timezone.utc).isoformat(),state='imports_ready_weights_missing' if imports['ok'] and not all(x['exists'] for x in weights) else 'inspect',
    candidate='P2',repo=str(REPO),commit=COMMIT,archive_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),archive_bytes=ZIP.stat().st_size,
    source_transport='Exact pinned official GitHub archive via Windows HTTPS; WSL git TLS failure recorded',
    python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,dependencies=dependencies,imports=imports,weights=weights,
    environment_policy='Existing base PyTorch environment remains unchanged; exact pinned source supplied only through process-local sys.path',
    model_download_started=False,subject_inference_started=False,gaussian_training_started=False)
write(OUT/'preflight.json',result)
print(json.dumps(result,ensure_ascii=False))
