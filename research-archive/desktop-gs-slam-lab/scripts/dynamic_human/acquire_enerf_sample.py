"""Acquire the author's public ENeRF-Outdoor sample; keep data local."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse
import hashlib, json, time, zipfile, requests, os

LAB = Path(__file__).resolve().parents[2]
OUT = LAB / 'outputs/dynamic-human-enerf-data-021'
DEST = LAB / 'data/dynamic-human/enerf_outdoor'
ARCHIVE = LAB / 'data/downloads/actor1_4_subseq.zip'
FILE_ID = '1XxeO7TnAPvDugnxguEF5Jp89ERS9CAia'
MAX_BYTES = 8 * 1024**3

class Form(HTMLParser):
    def __init__(self):
        super().__init__(); self.action=None; self.fields={}
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if tag=='form': self.action=a.get('action')
        if tag=='input' and a.get('type')=='hidden' and a.get('name'):
            self.fields[a['name']]=a.get('value','')

def write(name, data):
    p=OUT/name; tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8'); os.replace(tmp,p)

def main():
    OUT.mkdir(parents=True,exist_ok=True); DEST.mkdir(parents=True,exist_ok=True); ARCHIVE.parent.mkdir(exist_ok=True)
    if (OUT/'acquisition.json').exists(): raise RuntimeError('Completed acquisition exists; inspect instead of overwrite')
    s=requests.Session()
    source='https://raw.githubusercontent.com/zju3dv/EasyVolcap/main/readme.md'
    r=s.get(source,timeout=(15,60));r.raise_for_status()
    assert FILE_ID in r.text
    (OUT/'official-readme.txt').write_text(r.text,encoding='utf8')
    write('source.json',dict(readme=source,public_sample=f'https://drive.google.com/file/d/{FILE_ID}/view',local_only=True,scope='Author public sample; not a full outdoor sequence or a city-scale benchmark',verified_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if not ARCHIVE.exists():
        part=ARCHIVE.with_suffix('.zip.partial')
        if part.exists(): raise RuntimeError('Partial archive exists; inspect before restarting')
        r=s.get('https://drive.usercontent.google.com/download',params={'id':FILE_ID,'export':'download'},stream=True,timeout=(20,90));r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''):
            doc=r.text;f=Form();f.feed(doc);r.close()
            if not f.action or urlparse(f.action).hostname!='drive.usercontent.google.com' or f.fields.get('id')!=FILE_ID:
                raise RuntimeError('Expected public large-file confirmation form missing')
            r=s.get(f.action,params=f.fields,stream=True,timeout=(20,90));r.raise_for_status()
        if 'text/html' in r.headers.get('Content-Type',''):
            reason='google_drive_public_download_quota_exceeded' if 'Quota exceeded' in r.text else 'unexpected_html_response'
            write('state.json',dict(status='download_unavailable',reason=reason,archive_downloaded=False,
                observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid()))
            raise RuntimeError(reason)
        expected=int(r.headers.get('Content-Length',0));assert expected<=MAX_BYTES
        total=0;tick=0;h=hashlib.sha256()
        with part.open('xb') as out:
            for chunk in r.iter_content(1024*1024):
                total+=len(chunk)
                if total>MAX_BYTES: raise RuntimeError('Archive exceeds declared 8GiB download bound')
                out.write(chunk);h.update(chunk)
                if time.monotonic()-tick>10:
                    write('state.json',dict(status='downloading',bytes=total,expected_bytes=expected,pid=os.getpid()));print('DOWNLOADED',total,expected,flush=True);tick=time.monotonic()
        r.close()
        if expected: assert total==expected
        assert zipfile.is_zipfile(part)
        part.replace(ARCHIVE)
    files=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        assert sum(i.file_size for i in z.infolist())<20*1024**3
        for info in z.infolist():
            if info.is_dir(): continue
            target=(DEST/info.filename).resolve();assert target.is_relative_to(DEST.resolve())
            assert not ((info.external_attr>>16)&0o170000)==0o120000
            data=z.read(info);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): assert target.read_bytes()==data
            else: target.write_bytes(data)
            files.append(dict(path=info.filename,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=info.CRC))
    digest=hashlib.file_digest(ARCHIVE.open('rb'),'sha256').hexdigest()
    write('acquisition.json',dict(status='archive_and_all_entries_verified',archive_bytes=ARCHIVE.stat().st_size,archive_sha256=digest,source=f'https://drive.google.com/file/d/{FILE_ID}/view',files=files,local_only=True))
    write('state.json',dict(status='download_extraction_complete_audit_pending',files=len(files),pid=os.getpid()))
    print('COMPLETE',len(files),ARCHIVE.stat().st_size,flush=True)

if __name__=='__main__':main()
