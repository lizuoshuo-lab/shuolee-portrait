from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,hashlib,urllib.request
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-enerf-hand-trajectories-095';S=Path('D:/Projects/EngineLab/engine-home');read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda b:hashlib.sha256(b).hexdigest()
class Page(HTMLParser):
 def __init__(self):super().__init__();self.urls=set();self.ids=set()
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v and v.startswith('/gs-lab/'):self.urls.add(v)
   if k=='id':self.ids.add(v)
def get(url):
 with urllib.request.urlopen('http://127.0.0.1:8799'+url,timeout=30) as r:assert r.status==200;return r.read()
pages={};proof=[]
for slug,terms in {'dynamic-human-scene-reconstruction':['为什么单视角拟合不等于三维重建','时间函数连续','逐高斯模型还有局部黑斑'],'dynamic-human-reconstruction-benchmark':['局部连续运动自由度的同预算对照','24.637','22.722','132930']}.items():
 raw=get('/gs-lab/read/'+slug);html=raw.decode('utf8');assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
 p=Page();p.feed(html);pages['/gs-lab/read/'+slug]=p;proof.append(dict(slug=slug,sha256=sha(raw)))
assert '为什么单视角拟合不等于三维重建' in pages['/gs-lab/read/dynamic-human-scene-reconstruction'].ids
target='/gs-lab/read/dynamic-human-reconstruction-benchmark#局部连续运动自由度的同预算对照';urls={unquote(u) for p in pages.values() for u in p.urls};assert target in urls
assert urlsplit(target).fragment in pages[urlsplit(target).path].ids
expected={f'outputs/{root}/{f}' for root,files in read(O/'published-resources.json')['paths'].items() for f in files};content=read(S/'lib/gs-lab-content.json');assert not content['warnings'];rs=[r for r in content['resources'] if r['path'] in expected];assert len(rs)==7
assert all(unquote(r['url']) in urls for r in rs);assert not any(r['path'].endswith(('.pt','.pkl','.npz','.zip','.ply','.onnx')) for r in rs)
def check(r):
 assert sha((L/r['path']).read_bytes())==r['source_sha256'];raw=get(r['url']);assert sha(raw)==r['sha256'] and len(raw)==r['bytes'];return dict(path=r['path'],url=r['url'],sha256=sha(raw))
with ThreadPoolExecutor(max_workers=4) as pool:assets=list(pool.map(check,rs))
out=dict(status='two_pages_seven_assets_HTTP_hash_formula_andnewanchor_verified',pages=proof,resources=assets,math_errors=0,new_anchor_verified=True,licensed_model_exports=False,goal_complete=False);(O/'website-qa.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(out['status'])
