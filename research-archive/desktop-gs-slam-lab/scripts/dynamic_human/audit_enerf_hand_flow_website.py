from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,urllib.request
from audit_street_site import Links
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-open-hand-flow-083';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda b:hashlib.sha256(b).hexdigest()
def get(url):
    with urllib.request.urlopen('http://127.0.0.1:8799'+url,timeout=30) as r:assert r.status==200;return r.read()
links=Links();pages=[]
for slug,terms in {'dynamic-human-scene-reconstruction':['图像运动约束与重建清晰度','图像等亮度线的法向','训练点的运动残差下降约35%'],'dynamic-human-reconstruction-benchmark':['手部法向运动与真实重建对照','1527','本轮候选不替换原基线']}.items():
    data=get('/gs-lab/read/'+slug);html=data.decode('utf8');assert 'class="katex-error"' not in html
    for term in terms:assert term in html,(slug,term)
    links.feed(html);pages.append(dict(slug=slug,sha256=sha(data)))
expected={f'outputs/{root}/{f}' for root,files in read(O/'published-resources.json')['paths'].items() for f in files}
content=read(S/'lib/gs-lab-content.json');assert not content['warnings'];resources=[r for r in content['resources'] if r['path'] in expected];assert len(resources)==10
assert all(r['url'] in links.urls for r in resources)
assert not any(r['path'].endswith(('.pkl','.onnx','.pt','.npz','.ply','.zip')) for r in resources)
def verify(r):
    assert sha((L/r['path']).read_bytes())==r['source_sha256'];data=get(r['url']);assert sha(data)==r['sha256'] and len(data)==r['bytes'];return dict(path=r['path'],url=r['url'],sha256=sha(data))
with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(verify,resources))
out=dict(status='two_pages_ten_new_assets_HTTP_verified',pages=pages,resources=checks,math_errors=0,licensed_model_exports=False,goal_complete=False)
(O/'website-qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(out['status'])
