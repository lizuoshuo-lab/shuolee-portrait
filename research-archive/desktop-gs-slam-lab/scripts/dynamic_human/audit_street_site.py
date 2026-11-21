"""Verify local knowledge/evidence pages and every linked exported resource."""
from pathlib import Path
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request

LAB=Path(__file__).resolve().parents[2]
SITE=Path('D:/Projects/EngineLab/engine-home')
OUT=LAB/'outputs/dynamic-human-street-joint-009'
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
        for key,value in attrs:
            if key in ('href','src') and value and value.startswith('/gs-lab/'):
                self.urls.add(value)

def main():
    content=json.loads((SITE/'lib/gs-lab-content.json').read_text(encoding='utf8'))
    assert not content['warnings']
    pages=[]
    parser=Links()
    for slug,terms in {
        'dynamic-human-scene-reconstruction':['空间颜色一致性不能代替表面一致性','14 张测试照片','最近的有效激光点','街景对照训练协议','官方 SMPL 中性模板已接入','对齐尚未验收','全部 105 张最终图与 435 个人物框'],
        'dynamic-human-reconstruction-benchmark':['测试照片与初始化隔离','435','13 个区域','三个被丢弃的优化步骤','86,914','官方 SMPL 模板接入与初始对齐','实际人体节点检查','清晰人物重建未通过']
    }.items():
        raw=fetch('/gs-lab/read/'+slug)
        html=raw.decode('utf8')
        assert 'class="katex-error"' not in html
        for term in terms:assert term in html,(slug,term)
        parser.feed(html)
        pages.append(dict(slug=slug,status=200,sha256=sha(raw),required_terms=terms))
    resources=[r for r in content['resources'] if r['url'] in parser.urls]
    assert any('training-only-initialization-audit.json' in r['path'] for r in resources)
    assert any('street-joint-009/baseline-bounded/protocol.json' in r['path'] for r in resources)
    assert any('resume-10000-expandable/recovery-audit.json' in r['path'] for r in resources)
    assert any('dynamic-human-smpl-010/visual-review.json' in r['path'] for r in resources)
    assert any('dynamic-human-smpl-010/smpl-node-smoke.json' in r['path'] for r in resources)
    assert any('final-image-audit/independent-audit.json' in r['path'] for r in resources)
    assert any('paired-final-audit/frame130-camera4-person79-comparison.png' in r['path'] for r in resources)
    assert any('paired-final-audit/visual-review.json' in r['path'] for r in resources)
    assert any('geometry-decomposition-15000/receipt.json' in r['path'] for r in resources)
    assert any('street-temporal-012/independent-audit-v2/independent-audit.json' in r['path'] for r in resources)
    assert len([r for r in resources if 'street-temporal-012/' in r['path'] and r['path'].endswith('.mp4')]) == 8
    assert any('pose-refinement-014/independent-audit/visual-review.json' in r['path'] for r in resources)
    assert any('pose-batch-015/independent-state-audit.json' in r['path'] for r in resources)
    assert any('pose-batch-015/independent-audit/frame130-camera4-person79-comparison.png' in r['path'] for r in resources)
    assert any('pose-convergence-016/independent-state-audit.json' in r['path'] for r in resources)
    assert any('pose-convergence-016/independent-audit/visual-review.json' in r['path'] for r in resources)
    assert any('footprint-diagnostic-017/linear-compositing-v2/independent-image-audit.json' in r['path'] for r in resources)
    assert any('footprint-diagnostic-017/linear-compositing-v2/renders/frame130-camera4-scales.png' in r['path'] for r in resources)
    assert any('surface-refit-018/appearance_only/independent-state-audit.json' in r['path'] for r in resources)
    assert any('surface-refit-018/surface-retry-v2/surface_and_appearance/independent-state-audit.json' in r['path'] for r in resources)
    assert any('surface-refit-018/paired-audit/visual-review.json' in r['path'] for r in resources)
    assert any('surface-refit-018/paired-audit/frame130-camera4-person79-comparison.png' in r['path'] for r in resources)
    assert any('surface-temporal-019/independent-audit/visual-review.json' in r['path'] for r in resources)
    assert len([r for r in resources if 'surface-temporal-019/' in r['path'] and r['path'].endswith('.mp4')]) == 8
    assert any('local-capacity-020/paired-audit/frame129-camera4-person79-comparison.png' in r['path'] for r in resources)
    assert any('local-capacity-020/paired-audit/frame130-camera4-person79-comparison.png' in r['path'] for r in resources)
    assert any('local-capacity-020/adjacent_pair/independent-state-audit.json' in r['path'] for r in resources)
    assert any('capacity-components-022/independent-image-audit.json' in r['path'] for r in resources)
    assert any('capacity-components-022/paired-audit/frame129-camera4-geometry-color.png' in r['path'] for r in resources)
    assert any('capacity-components-022/paired-audit/frame140-camera4-color-opacity.png' in r['path'] for r in resources)
    assert any('box-bottom-diagnostic/visual-review.json' in r['path'] for r in resources)
    assert any('street-smpl-joint-011/preflight/completion.json' in r['path'] for r in resources)
    assert any('paired-updates-10000-audit/independent-audit.json' in r['path'] for r in resources)
    assert any('geometry-03000-cpu/receipt.json' in r['path'] for r in resources)
    assert any('training-alignment-013/keypoint-visual-review.json' in r['path'] for r in resources)
    assert not any(r['path'].lower().endswith(('.pkl','.zip','.npz','.pt'))
        for r in content['resources'] if 'dynamic-human-smpl-010' in r['path'])
    def check(r):
        assert sha((LAB/r['path']).read_bytes())==r['source_sha256'],r['path']
        raw=fetch(r['url'])
        assert sha(raw)==r['sha256'] and len(raw)==r['bytes'],r['url']
        return dict(path=r['path'],url=r['url'],sha256=sha(raw),bytes=len(raw))
    with ThreadPoolExecutor(max_workers=6) as executor:
        checks=list(executor.map(check,resources))
    report=dict(status='street_joint_comparisons_and_smpl_readiness_pages_http_verified',
        local_only=True,checked_pages=pages,checked_resources=len(checks),resources=checks,
        all_exported_resources=len(content['resources']),
        knowledge_body='Real street observation/reconstruction example in limitations; runtime and SMPL initial-offset records in collapsed appendix/evidence page; licensed templates excluded')
    (OUT/'website-http-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(status=report['status'],checked_resources=len(checks))))

if __name__=='__main__':main()
