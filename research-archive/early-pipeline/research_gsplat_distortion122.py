"""Bounded source snapshot only; never installs or imports downloaded code."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, urllib.request, subprocess
from concurrent.futures import ThreadPoolExecutor

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
O=R/'gsplat-distortion-research'
G=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/gsplat')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'Shuolee-public-source-audit'})
    with urllib.request.urlopen(req,timeout=30) as response:
        data=response.read(4*1024*1024+1)
        if len(data)>4*1024*1024:raise ValueError('Bounded text source limit')
        return data
if __name__=='__main__':
    if O.exists():raise FileExistsError('Snapshot exists; inspect, do not repeat')
    O.mkdir()
    latest=json.loads(get('https://api.github.com/repos/nerfstudio-project/gsplat/commits/main'))
    commit=latest['sha'];save(O/'official-main-head.json',dict(sha=commit,date=latest['commit']['committer']['date'],url=latest['html_url']))
    local_commit=subprocess.check_output(['git','-C',str(G),'rev-parse','HEAD'],text=True).strip()
    names=['gsplat/rendering.py','gsplat/cuda/_wrapper.py','gsplat/cuda/include/Cameras.cuh',
           'docs/3dgut.md','gsplat/sensors/design-models.md','examples/simple_trainer.py']
    def fetch(name):
        url=f'https://raw.githubusercontent.com/nerfstudio-project/gsplat/{commit}/{name}'
        dest=O/'official-main'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            data=get(url);dest.write_bytes(data)
            return dict(path=name,url=url,sha256=sha(dest),bytes=len(data),state='retrieved')
        except Exception as e:return dict(path=name,url=url,state='unavailable',error=str(e))
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(fetch,names))
    local=[G/p for p in names if (G/p).exists()]
    save(O/'sources.json',dict(updated=datetime.now(timezone.utc).isoformat(),official_main=commit,local_release='1.5.3',
        local_commit=local_commit,files=files,local_hashes={str(p):sha(p) for p in local},
        adapter_hashes={str(H/p):sha(H/p) for p in ['native_portrait_renderer122_v2.py','native_training122_v2.py','native_scene_package122_v3.py']},
        downloaded_code_executed=False,environment_changed=False,weights_downloaded=False,
        docs=['https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html','https://docs.gsplat.studio/main/apis/rasterization.html',
              'https://github.com/nerfstudio-project/gsplat/releases','https://github.com/nerfstudio-project/gsplat/blob/main/docs/3dgut.md']))
    print(json.dumps(dict(output=str(O),main=commit,local=local_commit,files=files)))
