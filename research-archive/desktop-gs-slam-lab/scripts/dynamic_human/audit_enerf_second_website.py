"""Verify both changed pages and the explicit new public asset list."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,urllib.request
from audit_street_site import Links
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda b:hashlib.sha256(b).hexdigest()
def get(url):
 with urllib.request.urlopen('http://127.0.0.1:8799'+url,timeout=30) as r:assert r.status==200;return r.read()
p=Links();pages=[]
for slug,terms in {'dynamic-human-scene-reconstruction':['时间基函数与局部运动','另一运动片段的真实对照','监督中的背景污染'], 'dynamic-human-reconstruction-benchmark':['另一运动片段中的时间表示','24.637','22.699','共52张','训练掩码中的墙面阴影与张手细节']}.items():
 raw=get('/gs-lab/read/'+slug);html=raw.decode('utf8');assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
 p.feed(html);pages.append(dict(slug=slug,sha256=sha(raw)))
expected={f'outputs/{root}/{f}' for root,files in read(R/'published-resources.json')['paths'].items() for f in files}
c=read(S/'lib/gs-lab-content.json');assert not c['warnings'];resources=[r for r in c['resources'] if r['path'] in expected]
assert len(resources)==len(expected)==10
assert all(r['url'] in p.urls for r in resources)
assert not any(r['path'].endswith(('.npz','.pkl','.pt','.ply','.onnx','.zip')) for r in resources)
def check(r):
 assert sha((L/r['path']).read_bytes())==r['source_sha256'];raw=get(r['url']);assert sha(raw)==r['sha256'] and len(raw)==r['bytes']
 return dict(path=r['path'],url=r['url'],sha256=sha(raw))
with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(check,resources))
out=dict(status='two_pages_and_ten_new_assets_HTTP_verified',pages=pages,resources=checks,math_parse_errors=0,licensed_models_exported=False,goal_complete=False)
(R/'website-qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(out['status'])
