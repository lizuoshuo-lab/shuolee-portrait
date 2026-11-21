import argparse, urllib.request, time, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('dest');p.add_argument('--sha');p.add_argument('--size',type=int);a=p.parse_args()
dest=Path(a.dest);dest.parent.mkdir(exist_ok=True,parents=True);part=Path(str(dest)+'.part')
if dest.exists():
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    if a.sha and digest==a.sha and (not a.size or dest.stat().st_size==a.size):
        print(json.dumps({'state':'already_verified','file':str(dest),'sha256':digest}),flush=True);raise SystemExit
    raise RuntimeError('Destination already exists; refusing overwrite')
for attempt in range(1,9):
    offset=part.stat().st_size if part.exists() else 0
    try:
        req=urllib.request.Request(a.url,headers={'User-Agent':'shuolee-research/1.0',**({'Range':f'bytes={offset}-'} if offset else {})})
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status==206:
                assert r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                mode='ab'
            elif r.status==200: mode='wb';offset=0
            else: raise RuntimeError(f'HTTP {r.status}')
            print(json.dumps({'attempt':attempt,'resume_bytes':offset,'http':r.status}),flush=True)
            with part.open(mode) as f:
                while block:=r.read(1024*1024): f.write(block)
        if a.size and part.stat().st_size!=a.size: raise RuntimeError('Incomplete length')
        digest=hashlib.sha256(part.read_bytes()).hexdigest()
        if a.sha and digest!=a.sha: raise RuntimeError('SHA mismatch')
        part.rename(dest)
        print(json.dumps({'state':'download_verified','bytes':dest.stat().st_size,'sha256':digest,'file':str(dest)}),flush=True)
        break
    except Exception as e:
        print(json.dumps({'attempt':attempt,'error_type':type(e).__name__,'partial_bytes':part.stat().st_size if part.exists() else 0}),flush=True)
        if attempt==8: raise
        time.sleep(3)
