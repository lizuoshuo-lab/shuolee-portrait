"""Read bounded public primary sources for design review. No downloaded code executes."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,urllib.request,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/generic-2d3d-research'
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Shuolee-Primary-Source-Review','Accept':'application/vnd.github+json' if 'api.github.com' in url else 'text/plain'})
    with urllib.request.urlopen(req,timeout=25) as r:
        data=r.read(2_000_001)
        if len(data)>2_000_000:raise ValueError('Research source size budget exceeded')
        return data
repos={
 'vchoutas/smplify-x':['README.md','LICENSE','smplifyx/utils.py','smplifyx/data_parser.py','smplifyx/camera.py','smplifyx/fitting.py','smplifyx/fit_single_frame.py','cfg_files/fit_smplx.yaml'],
 'vchoutas/smplx':['README.md','LICENSE','smplx/body_models.py','smplx/lbs.py','smplx/joint_names.py'],
 'open-mmlab/mmhuman3d':['LICENSE','docs/keypoints_convention.md','configs/smplify/smplifyx.py','mmhuman3d/core/conventions/keypoints_mapping/__init__.py'],
 'zju3dv/EasyMocap':['LICENSE','Readme.md','doc/quickstart.md','easymocap/pyfitting/smpl.py','easymocap/pyfitting/lossfactory.py'],
 'naver/multi-hmr':['LICENSE','README.md','model.py'],
 'facebookresearch/DensePose':['LICENSE','README.md'],
 'aipixel/GaussianAvatar':['README.md','LICENSE.md'],
 'jmaraval/GaussianAvatars_SMPLX':['README.md','LICENSE.md']}
manifest=O/'research-sources.json';assert not manifest.exists()
heads={};errors=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    fut={pool.submit(fetch,f'https://api.github.com/repos/{repo}/commits?per_page=1'):repo for repo in repos}
    for f in as_completed(fut):
        repo=fut[f]
        try:
            data=json.loads(f.result())[0];heads[repo]=dict(commit=data['sha'],commit_time=data['commit']['committer']['date'])
        except Exception as e:errors.append(dict(repo=repo,phase='commit',error=str(e)))
records=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    fut={}
    for repo,paths in repos.items():
        if repo not in heads:continue
        for p in paths:
            url=f'https://raw.githubusercontent.com/{repo}/{heads[repo]["commit"]}/{p}'
            fut[pool.submit(fetch,url)]=(repo,p,url)
    for f in as_completed(fut):
        repo,p,url=fut[f]
        try:
            data=f.result();local=O/'sources'/repo.replace('/','__')/(p+'.txt');local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(data)
            records.append(dict(repo=repo,path=p,url=url,browse_url=f'https://github.com/{repo}/blob/{heads[repo]["commit"]}/{p}',local=str(local),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        except Exception as e:errors.append(dict(repo=repo,path=p,url=url,phase='file',error=str(e)))
doc=dict(updated=datetime.now(timezone.utc).isoformat(),heads=heads,records=sorted(records,key=lambda r:(r['repo'],r['path'])),errors=errors,
    code_executed=False,weights_downloaded=False,person_data_transmitted=False,read_only_primary_source_review=True)
manifest.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(repos=len(heads),sources=len(records),total_bytes=sum(r['bytes'] for r in records),errors=errors),ensure_ascii=False))
