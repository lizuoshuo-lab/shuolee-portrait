"""Read back served documents and exact allowed resources from the local website."""
from __future__ import annotations
import concurrent.futures
import datetime as dt
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.error
import urllib.request

LAB = Path(__file__).resolve().parent.parent
SITE = Path(r'D:\Projects\EngineLab\engine-home')
OUT = LAB/'outputs/slam-drift-mechanism-001'
BASE = 'http://127.0.0.1:8799'
SLUG = 'slam-drift-mechanism-support-vio'
def sha(data): return hashlib.sha256(data).hexdigest()
class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources=[]
    def handle_starttag(self,tag,attrs):
        if tag=='img': self.sources.append(dict(attrs).get('src'))
def get(url):
    try:
        with urllib.request.urlopen(BASE+url, timeout=40) as r: return r.status, r.read()
    except urllib.error.HTTPError as e: return e.code, e.read()
def main():
    content_bytes=(SITE/'lib/gs-lab-content.json').read_bytes()
    content=json.loads(content_bytes.decode('utf8'))
    assert content['warnings']==[]
    assert len(content['documents'])==56 and len(content['results'])==47
    docs={d['slug']:d for d in content['documents']}
    assert SLUG in docs
    assert '全部 36 个非基准条件' in docs[SLUG]['body_markdown']
    resources=content['resources']
    checks=[]
    def resource_check(r):
        original=LAB/r['path']
        assert sha(original.read_bytes())==r['source_sha256'], r['path']
        status,data=get(r['url'])
        assert status==200 and sha(data)==r['sha256'] and len(data)==r['bytes'],r['url']
        return {'path':r['path'],'url':r['url'],'status':status,'served_sha256':sha(data),'source_sha256':r['source_sha256'],'passed':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for r in executor.map(resource_check,resources):checks.append(r)
    page_checks=[]
    def doc_check(d):
        status,data=get('/gs-lab/read/'+d['slug']);html=data.decode('utf8')
        assert status==200 and '<h1' in html,d['slug']
        return {'slug':d['slug'],'status':status,'html_sha256':sha(data),'passed':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        page_checks=list(executor.map(doc_check,content['documents']))
    status,data=get('/gs-lab/read/'+SLUG);html=data.decode('utf8')
    terms=['38 个条件','152 次 renderer forward','全部 36 个非基准条件','737','736','0.164355','0.179215','没有复现','固定基准像素']
    numeric_checks={term:term in html for term in terms}
    assert all(numeric_checks.values()),numeric_checks
    image_parser=Images()
    image_parser.feed(docs[SLUG]['html'])
    inline=[r for r in resources if r['media_type']=='image/png' and r['url'] in image_parser.sources]
    assert len(inline)==5,len(inline)
    assert all(r['url'] in html for r in inline)
    assert 'C:/Users/Administrator' not in html and 'C:\\Users\\Administrator' not in html
    hs,hd=get('/gs-lab');home=hd.decode('utf8')
    assert hs==200 and '/gs-lab/read/'+SLUG in home and '第 32 篇' in home
    assert home.index('/gs-lab/read/'+SLUG) < home.index('/gs-lab/read/splatam-tracking-ablation-repeat')
    unknown_status,_=get('/gs-lab/read/diagnostic-unknown-route-probe')
    assert unknown_status==404
    OUT.mkdir(parents=True,exist_ok=True)
    output=OUT/'website-http-and-source-audit.json'
    assert not output.exists(),output
    receipt={'status':'passed_actual_local_HTTP_and_source_hash_readback','created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'local_only':True,'audited_loopback_origin':BASE,'website_content_sha256':sha(content_bytes),'documents':len(content['documents']),
        'resources':len(resources),'results':len(content['results']),'resource_checks':checks,'page_checks':page_checks,
        'new_article_numeric_checks':numeric_checks,'new_article_inline_PNGs':[r['url'] for r in inline],
        'home_new_article_link_pinned_first':True,'unknown_route_status':unknown_status,
        'no_absolute_private_path_in_new_article_HTML':True,'source_sha256':sha(Path(__file__).read_bytes())}
    output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':receipt['status'],'documents':receipt['documents'],'resources':receipt['resources'],
        'results':receipt['results'],'new_inline_PNGs':len(inline),'output':str(output)},ensure_ascii=False))
if __name__=='__main__':main()
