"""Read only one sequence from the official ZIP using cached HTTP byte ranges."""
from pathlib import Path
import requests,io,zipfile,json,hashlib,time
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-reconstruction-003'
URL='https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip'
DEST=LAB/'data/dynamic-human/dynamic3dgaussians';CACHE=LAB/'data/downloads/dynamic3dgs-ranges';PARTIAL=LAB/'data/downloads/dynamic3dgaussians-data.zip.partial'
class Remote(io.RawIOBase):
    def __init__(self):
        self.pos=0;self.session=requests.Session();r=self.session.head(URL,timeout=30);r.raise_for_status();self.size=int(r.headers['Content-Length']);self.etag=r.headers.get('ETag');self.block=4*1024*1024;CACHE.mkdir(exist_ok=True);self.partial_size=PARTIAL.stat().st_size if PARTIAL.exists() else 0
    def seekable(self):return True
    def seek(self,n,whence=0):self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n;return self.pos
    def tell(self):return self.pos
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if not n:return b''
        pieces=[]
        while n>0:
            start=self.pos//self.block*self.block;end=min(self.size,start+self.block);offset=self.pos-start;count=min(n,end-self.pos)
            if self.pos+count<=self.partial_size:
                with PARTIAL.open('rb') as f:f.seek(self.pos);pieces.append(f.read(count))
            else:
                file=CACHE/f'{start:012d}-{end:012d}.bin'
                if not file.exists():
                    r=self.session.get(URL,headers={'Range':f'bytes={start}-{end-1}','If-Range':self.etag},timeout=(20,90));r.raise_for_status()
                    if r.status_code!=206 or len(r.content)!=end-start:raise RuntimeError('Invalid ranged response')
                    file.write_bytes(r.content);print('RANGE',start,end,flush=True)
                with file.open('rb') as f:f.seek(offset);pieces.append(f.read(count))
            self.pos+=count;n-=count
        return b''.join(pieces)
def main():
    reader=Remote();files=[]
    with zipfile.ZipFile(reader) as z:
        for item in z.infolist():
            if not item.filename.startswith('data/basketball/') or item.is_dir():continue
            target=(DEST/item.filename).resolve();assert target.is_relative_to(DEST.resolve())
            data=z.read(item);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
            files.append(dict(path=item.filename,bytes=item.file_size,sha256=hashlib.sha256(data).hexdigest(),crc32=item.CRC))
    r=dict(status='single_sequence_extracted_and_crc_verified',sequence='basketball',url=URL,remote_archive_bytes=reader.size,etag=reader.etag,files=files,full_archive_downloaded=False,scope='authors cleaned multiview Panoptic benchmark; not an outdoor large-scene result')
    (OUT/'data-receipt.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print('BASKETBALL COMPLETE',len(files),flush=True)
if __name__=='__main__':main()
