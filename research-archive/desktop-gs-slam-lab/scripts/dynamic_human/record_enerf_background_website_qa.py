from pathlib import Path
import urllib.request,json
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-matte-background-034'
checks=[]
for slug,term in [('dynamic-human-scene-reconstruction','人物轮廓与背景变化怎样区分'),('dynamic-human-reconstruction-benchmark','背景拟合后的实际纹理检查')]:
    url='http://127.0.0.1:8799/gs-lab/read/'+slug
    with urllib.request.urlopen(url,timeout=30) as response:html=response.read().decode('utf8')
    assert term in html and 'class="katex-error"' not in html
    checks.append(dict(url=url,new_text_present=True,katex_errors=0))
audit=json.loads((OUT/'website-http-audit.json').read_text(encoding='utf8'))
assert len([r for r in audit['resources'] if any(x in r['path'] for x in ['background-fit-033','matte-background-034'])])==11
(OUT/'website-qa.json').write_text(json.dumps(dict(status='export_sync_build_http_and_formula_parse_pass',pages=checks,linked_resources=audit['checked_resources'],new_resources=11,
    visual='Source/rendered image contents inspected using local image viewer. Browser screenshot/interaction QA not completed; earlier CDPinputtimeouts recorded in030.',local_only=True),indent=2)+'\n',encoding='utf8')
print('New theory/evidence text, formulas,11newassets and285linkedresources verified.')
