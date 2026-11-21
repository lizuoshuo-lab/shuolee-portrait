from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import json,hashlib,urllib.request
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-hand-photo-pose-105';S=Path('D:/Projects/EngineLab/engine-home')
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda b:hashlib.sha256(b).hexdigest()
class Page(HTMLParser):
 def __init__(self):super().__init__();self.urls=set();self.ids=set()
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ['href','src'] and v and v.startswith('/gs-lab/'):self.urls.add(unquote(v))
   if k=='id':self.ids.add(v)
def get(url):
 with urllib.request.urlopen('http://127.0.0.1:8799'+url,timeout=30) as r:assert r.status==200;return r.read()
pages={};proof=[]
for slug,terms in {'dynamic-human-scene-reconstruction':['手部表面高斯与关节运动','有限的姿态修正不足','姿态修正的原生图像检验'],'dynamic-human-reconstruction-benchmark':['固定外观后的手部姿态修正','0.18%','20.585']}.items():
 raw=get('/gs-lab/read/'+slug);html=raw.decode('utf8');assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
 p=Page();p.feed(html);pages['/gs-lab/read/'+slug]=p;proof.append(dict(slug=slug,sha256=sha(raw)))
urls=set.union(*(p.urls for p in pages.values()));target='/gs-lab/read/dynamic-human-reconstruction-benchmark#固定外观后的手部姿态修正';assert target in urls and urlsplit(target).fragment in pages[urlsplit(target).path].ids
expected={f'outputs/{root}/{f}' for root,files in read(O/'published-resources.json')['paths'].items() for f in files};content=read(S/'lib/gs-lab-content.json');assert not content['warnings'];rs=[r for r in content['resources'] if r['path'] in expected];assert len(rs)==len(expected) and all(unquote(r['url']) in urls for r in rs)
assets=[]
for r in rs:
 assert not r['path'].endswith(('.pt','.pkl','.npz','.zip','.ply'));assert sha((L/r['path']).read_bytes())==r['source_sha256'];raw=get(r['url']);assert sha(raw)==r['sha256'] and len(raw)==r['bytes'];assets.append(dict(path=r['path'],url=r['url'],sha256=sha(raw)))
(O/'website-qa.json').write_text(json.dumps(dict(status='two_pages_resources_math_andanchor_verified',pages=proof,resources=assets,math_errors=0,licensed_model_exports=False,goal_complete=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(f'2pages{len(assets)}assets and formula/anchor checks passed')
