import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'
relative='distill-view-assets/elevation-ab.mp4'
content=(ROOT/'viewer'/relative).read_bytes()
url='http://127.0.0.1:8820/'+relative
cases=[('bytes=0-99',206,content[:100]),('bytes=100-199',206,content[100:200]),
       ('bytes=-100',206,content[-100:]),(f'bytes={len(content)-100}-',206,content[-100:]),
       (f'bytes={len(content)}-',416,b''),('bytes=200-100',416,b'')]
rows=[]
for interval,status,expected in cases:
    try: response=urlopen(Request(url,headers={'Range':interval}),timeout=10)
    except HTTPError as exc: response=exc
    with response:
        body=response.read(); assert response.status==status,(interval,response.status)
        assert body==expected,interval
        assert response.headers.get('Accept-Ranges')=='bytes'
        rows.append(dict(range=interval,status=response.status,bytes=len(body),content_range=response.headers.get('Content-Range')))
with urlopen(Request(url,method='HEAD'),timeout=10) as response:
    assert response.status==200 and int(response.headers['Content-Length'])==len(content)
    assert response.read()==b''
result=dict(status='passed',created=time.time(),file=relative,file_bytes=len(content),cases=rows,head_passed=True)
(ROOT/'viewer-range-support-001/http-check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
