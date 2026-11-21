"""Check the served updated theory/evidence pages and their linked resources."""
from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request

LAB=Path(__file__).resolve().parents[2]
SITE=Path('D:/Projects/EngineLab/engine-home')
OUT=LAB/'outputs/dynamic-human-appearance-refinement-008'
BASE='http://127.0.0.1:8799'

def sha(data):return hashlib.sha256(data).hexdigest()
def fetch(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as r:
        assert r.status==200
        return r.read()

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls=set()
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        for key in ['href','src']:
            if key in d and d[key].startswith('/gs-lab/'):
                self.urls.add(d[key])

def main():
    content=json.loads((SITE/'lib/gs-lab-content.json').read_text(encoding='utf8'))
    assert not content['warnings']
    pages=[]
    parser=Links()
    for slug,terms in {
        'dynamic-human-scene-reconstruction':['空间颜色一致性不能代替表面一致性','29.33','2,366','原流程会用测试照片'],
        'dynamic-human-reconstruction-benchmark':['固定几何的颜色再训练对照','987','105','0.00560','训练／测试之前']
    }.items():
        raw=fetch('/gs-lab/read/'+slug)
        html=raw.decode('utf8')
        # Math parser errors would be actual rendered elements; data strings may contain CSS class names.
        assert 'class="katex-error"' not in html
        for term in terms:assert term in html,(slug,term)
        parser.feed(html)
        pages.append(dict(slug=slug,status=200,sha256=sha(raw),required_terms=terms))
    resources=[r for r in content['resources'] if r['url'] in parser.urls]
    assert any('appearance-refinement-008/assets/camera15-person.png' in r['path'] for r in resources)
    def check(r):
        assert sha((LAB/r['path']).read_bytes())==r['source_sha256'],r['path']
        raw=fetch(r['url'])
        assert sha(raw)==r['sha256'] and len(raw)==r['bytes'],r['url']
        return dict(path=r['path'],url=r['url'],sha256=sha(raw),bytes=len(raw))
    with ThreadPoolExecutor(max_workers=6) as executor:
        checks=list(executor.map(check,resources))
    report=dict(status='updated_pages_and_all_linked_exported_resources_http_verified',
        local_only=True,checked_pages=pages,checked_resources=len(checks),resources=checks,
        all_exported_resources=len(content['resources']),
        knowledge_body='Theory, equations and measured comparison; runtime details in collapsed appendix/evidence page',
        visual_browser_review='pending')
    (OUT/'website-http-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['resources','checked_pages']},ensure_ascii=False))

if __name__=='__main__':main()
