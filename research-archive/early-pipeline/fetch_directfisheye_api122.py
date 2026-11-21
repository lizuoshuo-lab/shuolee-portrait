"""Bounded public API fallback after raw host connection reset; no install."""
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json, hashlib, base64, shutil

O = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/directfisheye-research')
old = json.loads((O/'sources.json').read_text())
if (O/'sources-raw-failed.json').exists():
    raise RuntimeError('Fallback already attempted; inspect saved results')
shutil.copy2(O/'sources.json', O/'sources-raw-failed.json')
blobs = {x['path']: x for x in json.loads((O/'tree.json').read_text())['tree'] if x['type']=='blob'}

def one(row):
    name = row['path']
    if name not in blobs:
        return row
    blob = blobs[name]
    url = blob['url']
    try:
        with urlopen(Request(url, headers={'User-Agent':'Shuolee-ReadOnly-Research','Accept':'application/vnd.github+json'}), timeout=30) as r:
            raw = r.read(2800001)
        if len(raw)>2800000:
            raise RuntimeError('Source size limit exceeded')
        response = json.loads(raw)
        data = base64.b64decode(response['content'])
        if len(data)>2000000 or response['sha']!=blob['sha']:
            raise RuntimeError('Size or blob identity mismatch')
        git_sha = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if git_sha!=blob['sha']:
            raise RuntimeError('Git blob hash mismatch')
        dest = O/name.replace('/','__')
        dest.write_bytes(data)
        return dict(path=name, permalink='https://github.com/'+old['repo']+'/blob/'+old['commit']+'/'+name, url=url, bytes=len(data), git_blob_sha=git_sha, sha256=hashlib.sha256(data).hexdigest(), snapshot=str(dest))
    except Exception as exc:
        return dict(path=name, url=url, error=repr(exc))

with ThreadPoolExecutor(max_workers=4) as pool:
    rows = list(pool.map(one, old['files']))
old.update(updated=datetime.now(timezone.utc).isoformat(), files=rows, raw_host_failure_receipt='sources-raw-failed.json', fallback='Official GitHub blob API with verified Git blob identity')
(O/'sources.json').write_text(json.dumps(old,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(commit=old['commit'], downloaded=sum('sha256' in x for x in rows), errors=[x for x in rows if 'error' in x]),indent=2))
