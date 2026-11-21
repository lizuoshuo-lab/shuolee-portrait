"""Fetch pinned official Difix reference checkpoint; no user data uploaded."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
REV='d4830559772a5795c9d136302c5b197d6418d3fb'
OUT=ROOT/'sources/difix_ref'


def fetch(entry):
    name=entry['rfilename'];p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    expected=entry.get('lfs',{}).get('sha256')
    if p.exists() and (not expected or hashlib.sha256(p.read_bytes()).hexdigest()==expected):
        return dict(name=name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cached=True)
    url=f'https://huggingface.co/nvidia/difix_ref/resolve/{REV}/{name}'
    h=hashlib.sha256()
    with requests.get(url,stream=True,timeout=(30,90)) as r:
        r.raise_for_status()
        with p.with_suffix(p.suffix+'.part').open('wb') as f:
            for b in r.iter_content(4*1024*1024):f.write(b);h.update(b)
    if expected and h.hexdigest()!=expected:raise ValueError(f'Hash mismatch: {name}')
    p.with_suffix(p.suffix+'.part').replace(p)
    print(name,p.stat().st_size,flush=True)
    return dict(name=name,bytes=p.stat().st_size,sha256=h.hexdigest(),expected_sha256=expected)


if __name__=='__main__':
    r=requests.get(f'https://huggingface.co/api/models/nvidia/difix_ref/revision/{REV}?blobs=true',timeout=30);r.raise_for_status()
    entries=r.json()['siblings']
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(fetch,entries))
    (ROOT/'sources/difix-download-receipt.json').write_text(json.dumps(dict(repository='nvidia/difix_ref',revision=REV,files=records),indent=2))
