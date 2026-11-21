"""Resumable acquisition of one inspected 1.26 GB official AV2 street log."""
from pathlib import Path
import json,requests,concurrent.futures,hashlib,os,shutil,time,threading

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003/street-preflight'
TARGET=LAB/'data/dynamic-human/argoverse2/sensor/train/00a6ffc1-6ce9-3bc3-a060-6006e9893a1a'

def main():
    index=json.loads((OUT/'argoverse-complete-object-index.json').read_text());items=index['objects'];size=sum(i['size'] for i in items)
    assert size<2_000_000_000 and shutil.disk_usage(LAB).free>size*2
    TARGET.mkdir(parents=True,exist_ok=True);receipt=OUT/'street-log-data-receipt.json';records=[];local=threading.local()
    def download(item):
        assert item['key'].startswith(index['prefix']);relative=item['key'][len(index['prefix']):];path=(TARGET/relative).resolve();assert path.is_relative_to(TARGET.resolve());path.parent.mkdir(parents=True,exist_ok=True);etag=None
        if not path.exists() or path.stat().st_size!=item['size']:
            if not hasattr(local,'session'):local.session=requests.Session()
            for attempt in range(3):
                try:
                    with local.session.get(index['source']+item['key'],stream=True,timeout=(15,90)) as r:
                        r.raise_for_status();etag=r.headers.get('ETag');partial=path.with_suffix(path.suffix+'.partial');count=0
                        with partial.open('wb') as f:
                            for chunk in r.iter_content(262144):f.write(chunk);count+=len(chunk)
                        assert count==item['size'];os.replace(partial,path)
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(attempt+1)
        return dict(path=relative,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),remote_etag=etag)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for future in concurrent.futures.as_completed([pool.submit(download,i) for i in items]):
            records.append(future.result())
            if len(records)%100==0:print('DOWNLOADED',len(records),'/',len(items),flush=True)
    assert len(records)==len(items)
    receipt.write_text(json.dumps(dict(status='one_public_street_log_acquired',source=index['source'],prefix=index['prefix'],files=len(records),bytes=size,license='CC BY-NC-SA 4.0',terms='https://www.argoverse.org/about.html#terms-of-use',attribution='© 2021 Argo AI, LLC',raw_anonymization_preserved=True,training_status='not trained; SMPL template and preprocessing still required for the full OmniRe human branch',members=sorted(records,key=lambda r:r['path'])),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print('STREET LOG COMPLETE',len(records),size,flush=True)

if __name__=='__main__':main()
