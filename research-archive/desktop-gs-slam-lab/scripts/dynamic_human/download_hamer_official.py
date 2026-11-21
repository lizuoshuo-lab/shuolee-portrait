"""Four resumable ranges from the author URL, exact range/ETag checks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,time,json,hashlib,shutil,http.client
ROOT=Path('D:/GS-SLAM-Data/models/hamer').resolve();ROOT.mkdir(parents=True,exist_ok=True);PARTS=ROOT/'ranges';PARTS.mkdir(exist_ok=True)
URL='https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz';TOTAL=6037554929;ETAG='"167ddc6f1-60c69e4dd5f93"';N=4;SIZE=(TOTAL+N-1)//N
save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8');old=ROOT/'hamer_demo_data.tar.gz.part';first=PARTS/'00.part'
if old.exists() and not first.exists():assert old.stat().st_size<SIZE;shutil.copy2(old,first)
def download(i):
 start=i*SIZE;end=min((i+1)*SIZE,TOTAL)-1;path=PARTS/f'{i:02d}.part';expected=end-start+1
 failures=0
 while failures<40:
  done=path.stat().st_size if path.exists() else 0;assert done<=expected
  if done==expected:return
  try:
   segment_end=min(start+done+8*1024*1024-1,end)
   req=urllib.request.Request(URL,headers={'Range':f'bytes={start+done}-{segment_end}','If-Range':ETAG})
   with urllib.request.urlopen(req,timeout=45) as r:
    assert r.status==206 and r.headers.get('Content-Range')==f'bytes {start+done}-{segment_end}/{TOTAL}' and r.headers.get('ETag')==ETAG
    with path.open('ab') as f:
     while True:
      chunk=r.read(1024*1024)
      if not chunk:break
      assert f.tell()+len(chunk)<=segment_end-start+1;f.write(chunk);f.flush()
   if path.stat().st_size != segment_end-start+1:raise OSError('Response ended before the advertised range; resume from verified bytes')
   failures=0
  except (OSError,TimeoutError,http.client.HTTPException) as e:
   failures+=1;save(PARTS/f'{i:02d}-retry.json',dict(consecutive_failures=failures,error=str(e),time=time.time()));time.sleep(min(2+failures,10))
 raise RuntimeError(f'range{i} exhausted retries;partial preserved')
assert not (ROOT/'hamer_demo_data.tar.gz').exists(),'Completed archive already present'
began=time.time()
with ThreadPoolExecutor(max_workers=N) as pool:
 fs=[pool.submit(download,i) for i in range(N)]
 while not all(f.done() for f in fs):
  sizes=[p.stat().st_size if p.exists() else 0 for p in [PARTS/f'{i:02d}.part' for i in range(N)]];save(ROOT/'download-progress.json',dict(status='downloading_author_archive',url=URL,expected_bytes=TOTAL,bytes=sum(sizes),range_bytes=sizes,elapsed_seconds=time.time()-began,updated_unix=time.time()));time.sleep(10)
 for f in fs:f.result()
out=ROOT/'hamer_demo_data.tar.gz.assembled.part';assert out.resolve().parent==ROOT
with out.open('wb') as target:
 for i in range(N):
  with (PARTS/f'{i:02d}.part').open('rb') as source:shutil.copyfileobj(source,target,8*1024*1024)
assert out.stat().st_size==TOTAL;digest=hashlib.sha256()
with out.open('rb') as f:
 while chunk:=f.read(8*1024*1024):digest.update(chunk)
dest=ROOT/'hamer_demo_data.tar.gz';assert dest.resolve().parent==ROOT;out.replace(dest)
save(ROOT/'download-progress.json',dict(status='download_complete_archive_not_yet_extracted',url=URL,expected_bytes=TOTAL,bytes=TOTAL,sha256=digest.hexdigest(),etag=ETAG,elapsed_seconds=time.time()-began,updated_unix=time.time()));print('DOWNLOAD_COMPLETE',digest.hexdigest(),flush=True)
