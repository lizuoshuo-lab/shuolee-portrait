"""Verify exact source PNGs and local gallery links without modifying the images."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlparse
import hashlib,json
from PIL import Image
root=Path('/mnt/d/00/工作日志/2026-10-29/灰衣_SMPL-X对齐原图').resolve()
manifest=json.loads((root/'清单.json').read_text())
checks=[]
for frame in manifest['frames']:
    for record in frame['files']:
        path=(root/record['path']).resolve();assert root in path.parents
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
        with Image.open(path) as im:
            assert list(im.size)==record['wh'];im.verify()
        checks.append(record['path'])
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ['src','href'] and value:self.links.append(value)
link_count=0
for html in root.glob('*.html'):
    parser=Links();parser.feed(html.read_text())
    for link in parser.links:
        url=urlparse(link)
        if not url.scheme and url.path:
            target=(html.parent/unquote(url.path)).resolve()
            assert root in target.parents or target==root
            assert target.exists(),target;link_count+=1
report=dict(source_pngs_verified=len(checks),source_frames=len(manifest['frames']),local_links_verified=link_count,
            native_full_frame_wh=[1520,2704],source_images_modified=False,passed=True)
(root/'文件检查.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report))
