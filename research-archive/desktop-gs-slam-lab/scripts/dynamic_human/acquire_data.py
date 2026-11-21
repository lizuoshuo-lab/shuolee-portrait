"""Fetch the authors' public cleaned Panoptic release; retain source and checksums."""
from pathlib import Path
import requests,json,hashlib,zipfile,time

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003'
URL='https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip'
archive=LAB/'data/downloads/dynamic3dgaussians-data.zip'
dest=LAB/'data/dynamic-human/dynamic3dgaussians'
def main():
    archive.parent.mkdir(exist_ok=True);dest.mkdir(parents=True,exist_ok=True)
    head=requests.head(URL,timeout=30);head.raise_for_status();size=int(head.headers['Content-Length']);partial=archive.with_suffix('.zip.partial')
    if not archive.exists():
        start=partial.stat().st_size if partial.exists() else 0
        if start>size:raise RuntimeError('Partial file larger than remote archive')
        with requests.get(URL,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(20,60)) as response:
            response.raise_for_status()
            if start and response.status_code!=206:raise RuntimeError('Server did not honor resume range')
            with partial.open('ab' if start else 'wb') as f:
                last=time.monotonic()
                for chunk in response.iter_content(4*1024*1024):
                    f.write(chunk)
                    if time.monotonic()-last>15:print('DOWNLOAD',f.tell(),size,flush=True);last=time.monotonic()
        if partial.stat().st_size!=size:raise RuntimeError('Download size mismatch')
        partial.replace(archive)
    if archive.stat().st_size!=size:raise RuntimeError('Archive size mismatch')
    files=[]
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target=(dest/item.filename).resolve()
            if not target.is_relative_to(dest.resolve()):raise RuntimeError('Archive escapes destination')
            if item.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            data=z.read(item) # zipfile validates member CRC.
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(data)
            elif target.read_bytes()!=data:raise RuntimeError('Existing extracted data differs')
            files.append(dict(path=item.filename,bytes=item.file_size,crc32=item.CRC))
    sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt=dict(status='downloaded_and_crc_verified',url=URL,archive_bytes=size,sha256=sha,etag=head.headers.get('ETag'),files=files,source='authors cleaned CMU Panoptic release linked by Dynamic3DGaussians README',scope='multiview human-motion benchmark; does not establish street-scale reconstruction')
    (OUT/'data-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print('DATA COMPLETE',len(files),'files',size,'archive bytes',flush=True)

if __name__=='__main__':main()
