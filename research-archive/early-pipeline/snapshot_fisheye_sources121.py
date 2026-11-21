"""Bounded public-source snapshot for research; no install, weights or training."""
from pathlib import Path
from urllib.request import Request,urlopen
from datetime import datetime,timezone
import json,hashlib,math,base64

R=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee\current\raw-domain-121')
out=R/'fisheye-research-source';out.mkdir(exist_ok=True)
repo='Metaverse-AI-Lab-THU/DirectFisheye-GS'
def fetch(url):
    req=Request(url,headers={'User-Agent':'Shuolee-Research-ReadOnly','Accept':'application/vnd.github+json'})
    with urlopen(req,timeout=30) as r:
        content=r.read(2_000_001)
    if len(content)>2_000_000:raise RuntimeError('Source size limit exceeded')
    return content

head=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/main'))['sha']
files=['scene/dataset_readers.py','scene/cameras.py','utils/graphics_utils.py','submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu','submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h']
rows=[]
for name in files:
    url='https://raw.githubusercontent.com/'+repo+'/'+head+'/'+name
    try:
        api_url='https://api.github.com/repos/'+repo+'/contents/'+name+'?ref='+head
        payload=json.loads(fetch(api_url))
        if payload.get('encoding')!='base64':raise RuntimeError('Unexpected public source encoding')
        data=base64.b64decode(payload['content']);path=out/name.replace('/','__');path.write_bytes(data)
        rows.append({'path':name,'url':url,'permalink':'https://github.com/'+repo+'/blob/'+head+'/'+name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'snapshot':str(path)})
    except Exception as exc:
        rows.append({'path':name,'url':url,'error':str(exc)})

checks=[]
for subject,wh,cx,cy in [('portrait',[1080,1920],540.7900597125944,963.5492474072441),('landscape',[2704,1520],1352.8626255714278,763.7770175739016)]:
    centers=[[wh[0]/2,wh[1]/2],[(wh[0]-1)/2,(wh[1]-1)/2]]
    checks.append({'subject':subject,'conditional_K_principal_point':[cx,cy],
      'center_replacement_examples':[{'replacement_center':c,'same_ray_offset_px':math.hypot(cx-c[0],cy-c[1])} for c in centers],
      'kind':'Analytic same-ray principal-point replacement, not actual repo CUDA runtime or measured video error'})
result={'updated':datetime.now(timezone.utc).isoformat(),'repo':repo,'commit':head,'files':rows,'principal_point_examples':checks,'runtime_executed':False,'training_started':False}
(R/'fisheye-source-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
