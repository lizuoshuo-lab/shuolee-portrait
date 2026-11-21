"""Fallback public GitHub contents API after raw-host connection resets."""
from research_gsplat_distortion122 import O, get, save, sha
from concurrent.futures import ThreadPoolExecutor
import json,base64

s=json.loads((O/'sources.json').read_text(encoding='utf-8'));commit=s['official_main']
def fetch(row):
    name=row['path'];dest=O/'official-main'/name
    url=f'https://api.github.com/repos/nerfstudio-project/gsplat/contents/{name}?ref={commit}'
    try:
        d=json.loads(get(url));data=base64.b64decode(d['content'],validate=False)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        return dict(path=name,url=url,blob_sha=d['sha'],sha256=sha(dest),bytes=len(data),state='retrieved')
    except Exception as e:return dict(path=name,url=url,state='unavailable',error=str(e))
if (O/'api-sources.json').exists():raise FileExistsError('Already attempted')
with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,s['files']))
save(O/'api-sources.json',dict(commit=commit,files=rows,downloaded_code_executed=False))
print(json.dumps(rows))
