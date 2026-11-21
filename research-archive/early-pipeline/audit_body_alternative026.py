"""Read public upstream metadata only; do not upload subject images or fetch weights."""
import concurrent.futures
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

OUT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/smplx-observation-uncertainty-fresh-026/research')
OUT.mkdir(exist_ok=True)
repos={'sam3dbody':'facebookresearch/sam-3d-body','mhr':'facebookresearch/MHR','better-rigs':'dcaustin33/better_rigs_not_bigger_networks'}

def metadata(item):
    name,repo=item
    r=requests.get(f'https://api.github.com/repos/{repo}/commits/main',timeout=(10,25))
    return name,{'repository':f'https://github.com/{repo}','status':r.status_code,'commit':r.json().get('sha') if r.status_code==200 else None}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    commits=dict(pool.map(metadata,repos.items()))
sources=[('sam3dbody','README.md'),('sam3dbody','INSTALL.md'),
    ('sam3dbody','sam_3d_body/sam_3d_body_estimator.py'),('sam3dbody','sam_3d_body/build_models.py'),
    ('mhr','tools/mhr_smpl_conversion/README.md'),('mhr','tools/mhr_smpl_conversion/smpl_mhr.py'),
    ('better-rigs','README.md'),('better-rigs','pyproject.toml')]

def fetch_source(item):
    name,path=item
    commit=commits[name]['commit']
    if not commit:return {'repo':name,'path':path,'state':'commit_unavailable'}
    url=f'https://raw.githubusercontent.com/{repos[name]}/{commit}/{path}'
    r=requests.get(url,timeout=(10,25))
    target=OUT/(name+'-'+path.replace('/','__'))
    if r.status_code==200:
        assert len(r.content)<2000000
        target.write_bytes(r.content)
    return {'repo':name,'path':path,'url':url,'status':r.status_code,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest() if r.status_code==200 else None}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    sources=list(pool.map(fetch_source,sources))
model='facebook/sam-3d-body-dinov3'
info=requests.get(f'https://huggingface.co/api/models/{model}?blobs=true',timeout=(10,25))
md=info.json() if info.status_code==200 else {}
weights=[]
for f in md.get('siblings',[]):
    if f['rfilename'] in ['model.ckpt','assets/mhr_model.pt','model_config.yaml']:
        weights.append({'name':f['rfilename'],'size':f.get('size',f.get('lfs',{}).get('size')),'lfs':f.get('lfs')})
head=requests.head(f'https://huggingface.co/{model}/resolve/main/model.ckpt',allow_redirects=True,timeout=(10,25))
report=dict(checked=datetime.now(timezone.utc).isoformat(),repositories=commits,sources=sources,
    model=model,model_revision=md.get('sha'),gated=md.get('gated'),model_files=weights,
    anonymous_checkpoint_http_status=head.status_code,weights_downloaded=False,executed_on_subject=False,
    purpose='Independent image-conditioned body geometry and official conversion to the existing SMPL-X topology.',
    verified_public_capabilities=['Source estimator accepts existing bbox, masks and calibrated camera intrinsics.',
        'Official MHR toolkit supports SAM3D output to SMPLX and returning conversion fit errors.'],
    validation_required=['Native predicted geometry under known camera before fitting','Unit and camera-axis conversion','Mesh transfer residual before Gaussian initialization','Source overlays and novel fixed-camera sides'],
    private_subject_data_uploaded=False)
(OUT/'prior-alternative-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
