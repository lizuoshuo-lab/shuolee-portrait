from pathlib import Path
import json, hashlib, re, concurrent.futures
import requests
import pymupdf as fitz

LAB = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
HOME = LAB/'outputs/mini-splatting2-mip-depth-001/shuolee'
OUT = HOME/'code/presentation-sources'
OUT.mkdir(parents=True, exist_ok=True)

def fetch(item):
    name, url = item
    target = OUT/name
    if not target.exists():
        res=requests.get(url,timeout=60); res.raise_for_status()
        if name.endswith('.pdf') and not res.content.startswith(b'%PDF'): raise RuntimeError(url)
        target.write_bytes(res.content)
    return {'file':str(target),'url':url,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}

sources=[('pgsr.pdf','https://arxiv.org/pdf/2406.06521'),
         ('gaussianobject-pipeline.png','https://gaussianobject.github.io/static/images/pipe.png'),
         ('gaussianobject-bonsai.png','https://gaussianobject.github.io/static/images/comparison/mip360_bonsai_4.png'),
         ('viewextrapolator-overview.jpg','https://kunhao-liu.github.io/ViewExtrapolator/assets/images/overview.jpg'),
         ('viewextrapolator-teaser.jpg','https://kunhao-liu.github.io/ViewExtrapolator/assets/images/teaser.jpg'),
         ('difix-pipeline.jpg','https://research.nvidia.com/labs/toronto-ai/difix3d/assets/pipeline.jpg'),
         ('difix-model.jpg','https://research.nvidia.com/labs/toronto-ai/difix3d/assets/model.jpg'),
         ('gaussianobject.html','https://gaussianobject.github.io/'),
         ('viewextrapolator.html','https://kunhao-liu.github.io/ViewExtrapolator/'),
         ('difix.html','https://research.nvidia.com/labs/toronto-ai/difix3d/')]
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(fetch,sources): records.append(result)
for name in ['gaussianobject','viewextrapolator','difix']:
    html=(OUT/f'{name}.html').read_text(encoding='utf-8',errors='ignore')
    print(name, re.findall(r'(?:src|href)=["\']([^"\']+\.(?:png|jpg|jpeg|svg|pdf))(?:["\'])', html)[:30])
doc=fitz.open(OUT/'pgsr.pdf')
(OUT/'pgsr.txt').write_text('\n'.join(f'--- PAGE {i+1} ---\n'+p.get_text(sort=True) for i,p in enumerate(doc)),encoding='utf-8')
for i in range(min(7,len(doc))):
    doc[i].get_pixmap(matrix=fitz.Matrix(1,1)).save(OUT/f'pgsr-page-{i+1}.png')
(OUT/'sources.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf-8')
print('sources_ready')
