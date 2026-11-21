from pathlib import Path
import json,hashlib,subprocess,requests,time
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038'
REPO=LAB/'repos/sam2';DEST=LAB/'models/sam2';DEST.mkdir(exist_ok=True)
url='https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'
assert url in (REPO/'README.md').read_text(encoding='utf8')
target=DEST/'sam2.1_hiera_tiny.pt';temp=target.with_suffix('.pt.partial')
if not target.exists():
    start=time.time()
    with requests.get(url,proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'},stream=True,timeout=(20,60)) as r:
        r.raise_for_status();expected=int(r.headers.get('Content-Length','0'));size=0
        with temp.open('wb') as f:
            for chunk in r.iter_content(2**20):
                if chunk:f.write(chunk);size+=len(chunk)
        assert not expected or size==expected
        assert size>100_000_000
    temp.replace(target)
    print('Downloaded',size,'bytes in',round(time.time()-start,1),'seconds',flush=True)
h=hashlib.sha256()
with target.open('rb') as f:
    for b in iter(lambda:f.read(2**20),b''):h.update(b)
result=dict(status='official_source_and_checkpoint_local',source_url='https://github.com/facebookresearch/sam2',
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
    acquisition='Official sparse Git clone through existing system HTTPproxy; two earlier direct Git attempts failed and were not substituted with third-partyPyPIpackage.',
    checkpoint_url=url,checkpoint=str(target.relative_to(LAB)),bytes=target.stat().st_size,sha256=h.hexdigest(),
    license='Apache2.0 per official README and LICENSE for code and checkpoints',purpose='CPU-only segmentation candidate; no RGB enhancement or human-model acceptance')
(OUT/'source-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
