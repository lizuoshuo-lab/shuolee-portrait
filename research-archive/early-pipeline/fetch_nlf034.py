"""Fetch one public official NLF release asset and verify its published digest."""
from pathlib import Path
import datetime, hashlib, json, os, time
import requests

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/nlf-camera-prior-034')
DEST=Path('D:/GS-SLAM-Data/models/nlf')
NAME='nlf_l_multi_0.3.2.patch6.torchscript'
EXPECTED='2f9e377817acdd47e35224aeb70299b3f53ab574e0829372a094920515352a65'
SIZE=549527206
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(p,d):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

ROOT.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
assert not (ROOT/'model-asset.json').exists()
r=requests.get('https://api.github.com/repos/isarandi/nlf/releases/tags/v0.3.2',timeout=30);r.raise_for_status()
meta=r.json();asset=next(a for a in meta['assets'] if a['name']==NAME)
assert asset['size']==SIZE and asset['digest']=='sha256:'+EXPECTED
url=asset['browser_download_url'];assert url.startswith('https://github.com/isarandi/nlf/releases/download/v0.3.2/')
final=DEST/NAME;partial=DEST/(NAME+'.partial')
assert not partial.exists(), 'Inspect a preexisting partial before retrying'
if not final.exists():
    save(ROOT/'download-state.json',dict(updated=now(),state='downloading_official_asset',running=True,bytes=0,total=SIZE))
    with requests.get(url,stream=True,timeout=(25,60)) as response:
        response.raise_for_status();h=hashlib.sha256();count=0;last=time.monotonic()
        with partial.open('xb') as f:
            for block in response.iter_content(4*1024*1024):
                if not block:continue
                f.write(block);h.update(block);count+=len(block)
                if time.monotonic()-last>10:
                    msg=dict(updated=now(),state='downloading_official_asset',running=True,bytes=count,total=SIZE)
                    save(ROOT/'download-state.json',msg);print(json.dumps(msg),flush=True);last=time.monotonic()
    assert count==SIZE and h.hexdigest()==EXPECTED,(count,h.hexdigest())
    os.replace(partial,final)
assert final.stat().st_size==SIZE and sha(final)==EXPECTED
summary=dict(updated=now(),model='NLF-L',release='v0.3.2',asset=NAME,path=str(final),bytes=SIZE,
    sha256=EXPECTED,official_digest_verified=True,download_url=url,release_url=meta['html_url'],
    official_source='https://github.com/isarandi/nlf',source_commit='29046738a8101c15dc366ab45945b45e40debf17',
    code_license='MIT',weight_scope='Noncommercial research use per official README. Not a commercial deployment approval.',
    actual_subject_inference_done=False,quality_accepted=False,local_research_only=True)
save(ROOT/'model-asset.json',summary);save(DEST/'provenance.json',summary)
save(ROOT/'download-state.json',dict(updated=now(),state='official_asset_verified',running=False,bytes=SIZE,total=SIZE))
print(json.dumps(summary,ensure_ascii=False),flush=True)
