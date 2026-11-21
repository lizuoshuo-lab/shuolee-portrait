"""Bounded public file metadata check; never downloads a model in full."""
import datetime
import json
import os
from pathlib import Path
import urllib.request

out = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116/pixsfm')
if os.name != 'nt':
    out = Path('/mnt/c')/str(out)[3:]
url = 'https://www.dropbox.com/s/hnv51iwu4hn82rj/s2dnet_weights.pth?raw=1'
record = {'updated': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'official_readme': 'https://github.com/germain-hug/S2DNet-Minimal/blob/main/README.md',
          'readme_sha': '14b9e0a5afcc176787ec783502136edcdb89efce',
          'request_url': url, 'platform':os.name, 'full_weight_downloaded': False}
try:
    request = urllib.request.Request(url, headers={'Range':'bytes=0-31', 'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=25) as r:
        prefix = r.read(32)
        record.update(status=r.status, content_type=r.headers.get('Content-Type'),
                      content_length=r.headers.get('Content-Length'),
                      content_range=r.headers.get('Content-Range'), final_url=r.geturl(),
                      prefix_hex=prefix.hex(), bytes_read=len(prefix))
except Exception as e:
    record['error'] = type(e).__name__+': '+str(e)
(out/f's2dnet-download-preflight-{os.name}.json').write_text(json.dumps(record, indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
