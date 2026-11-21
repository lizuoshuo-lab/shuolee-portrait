"""Check changed pages and all11new resources after successful build."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,urllib.request
from audit_street_site import Links
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-temporal-basis-078';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda b:hashlib.sha256(b).hexdigest()
def get(url):
    with urllib.request.urlopen('http://127.0.0.1:8799'+url,timeout=30) as response:
        assert response.status==200;return response.read()
parser=Links();pages=[]
for slug,terms in {
    'dynamic-human-scene-reconstruction':['时间基函数与局部运动','局部时间基函数与人物细节'],
    'dynamic-human-reconstruction-benchmark':['局部时间基函数与人物细节','223105','26.504','23.818','共51张']
}.items():
    raw=get('/gs-lab/read/'+slug);html=raw.decode('utf8');assert 'class="katex-error"' not in html
    for term in terms:assert term in html,(slug,term)
    parser.feed(html);pages.append(dict(slug=slug,sha256=sha(raw)))
content=read(S/'lib/gs-lab-content.json');assert not content['warnings']
resources=[r for r in content['resources'] if 'outputs/'+R.name+'/' in r['path']]
paths=read(R/'published-resources.json')['paths'];assert len(resources)==len(paths)==11
assert set(r['path'].split(R.name+'/')[1] for r in resources)==set(paths)
assert all(r['url'] in parser.urls for r in resources)
assert not any(r['path'].endswith(('.pt','.npz','.pkl','.onnx','.zip','.ply')) for r in resources)
def check(r):
    assert sha((L/r['path']).read_bytes())==r['source_sha256']
    data=get(r['url']);assert sha(data)==r['sha256'] and len(data)==r['bytes']
    return dict(path=r['path'],url=r['url'],sha256=sha(data))
with ThreadPoolExecutor(max_workers=4) as executor:checks=list(executor.map(check,resources))
report=dict(status='temporal_basis_knowledge_evidence_and11new_assets_verified',pages=pages,resources=checks,
    all_exported_resources=len(content['resources']),math_parse_errors=0,licensed_models_exported=False,goal_complete=False)
(R/'website-qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k not in ['pages','resources']}))
