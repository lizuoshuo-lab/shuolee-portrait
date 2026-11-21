"""Read the two imported helpers required to interpret the public CVO loop."""
from pathlib import Path
from urllib.request import Request,urlopen
import json,hashlib,base64
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/directfisheye-research')
s=json.loads((O/'sources.json').read_text()); tree={x['path']:x for x in json.loads((O/'tree.json').read_text())['tree'] if x['type']=='blob'}
for name in ['utils/find_camera_groups.py','utils/loss_utils.py']:
    if any(x['path']==name for x in s['files']):raise RuntimeError('Helper already inspected')
    row=tree[name]
    with urlopen(Request(row['url'],headers={'User-Agent':'Shuolee-ReadOnly-Research'}),timeout=30) as r: raw=r.read(500001)
    if len(raw)>500000:raise RuntimeError('Source bound exceeded')
    data=base64.b64decode(json.loads(raw)['content'])
    digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert digest==row['sha']
    p=O/name.replace('/','__');p.write_bytes(data)
    s['files'].append(dict(path=name,snapshot=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob_sha=digest,url=row['url'],permalink='https://github.com/'+s['repo']+'/blob/'+s['commit']+'/'+name))
(O/'sources.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
print('Verified public source files:',len(s['files']))
