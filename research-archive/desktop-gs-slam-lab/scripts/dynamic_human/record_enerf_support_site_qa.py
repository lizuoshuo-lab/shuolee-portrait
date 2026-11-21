"""Verify newly published content without claiming unavailable browser screenshots."""
from pathlib import Path
from datetime import datetime, timezone
import json, urllib.request

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-enerf-support-030'
SITE=Path('D:/Projects/EngineLab/engine-home')
content=json.loads((SITE/'lib/gs-lab-content.json').read_text(encoding='utf8'))
assert not content['warnings']
new_resources=[r for r in content['resources'] if any(s in r['path'] for s in ['enerf-support-030','enerf-performer-selection-031','enerf-background-032'])]
assert len(new_resources)==20
assert all(Path(r['path']).suffix in ['.json','.png','.mp4'] for r in new_resources)
terms={
    'dynamic-human-scene-reconstruction':['尺度与不透明度能改善覆盖','25.38','63 张改善','训练机位也存在相同类型的缺陷'],
    'dynamic-human-reconstruction-benchmark':['运动阶段的尺度与不透明度对照','行走遮挡片段与背景几何准备','57,600','6,480','清晰度和完整场景目标仍未完成']
}
pages=[]
for slug, required in terms.items():
    with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:
        assert response.status==200
        html=response.read().decode('utf8')
    assert all(term in html for term in required)
    assert 'class="katex-error"' not in html
    pages.append(dict(slug=slug,terms_verified=required,status=200))
audit=json.loads((OUT/'website-http-audit.json').read_text(encoding='utf8'))
qa=dict(status='build_content_resource_checks_passed_browser_visual_check_incomplete',updated_utc=datetime.now(timezone.utc).isoformat(),
    pages=pages,new_public_resources=len(new_resources),total_site_resources=len(content['resources']),
    linked_resource_http_audit='website-http-audit.json',linked_resources_checked=274,build_exit_code=0,
    browser_dom=dict(title='大场景中的动态人物重建：清晰外观、运动与遮挡 | GS 知识库',heading_present=True,katex_errors=0,horizontal_page_overflow=False,viewport=[633,450]),
    browser_limit='Input.dispatchMouseEvent timed out; the two new lazy-loaded images remained incomplete in background DOM. No browser screenshot or successful on-screen image layout verification is claimed. HTTP bytes/source hashes passed; actual saved comparison images were separately inspected.',
    actual_image_review='visual-review.json',preview_parent_pid=20064,
    public_data='Only explicit JSON reports and actual PNG/MP4 comparisons; no models, raw archives or SMPL templates exported.')
(OUT/'website-qa.json').write_text(json.dumps(qa,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
manifest_path=LAB/'manifests/dynamic-human-work.json'; manifest=json.loads(manifest_path.read_text(encoding='utf8'))
manifest.update(updated=qa['updated_utc'],outdoor_support_website_qa='outputs/dynamic-human-enerf-support-030/website-qa.json')
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=qa['status'],resources=20,linked=274,build=0)))
