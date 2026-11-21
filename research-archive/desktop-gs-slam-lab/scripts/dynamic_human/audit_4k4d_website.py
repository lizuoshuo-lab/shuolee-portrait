from pathlib import Path
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-4k4d-official-035'
base.main()
url='http://127.0.0.1:8799/gs-lab/read/dynamic-human-reconstruction-benchmark'
with urllib.request.urlopen(url,timeout=30) as response:html=response.read().decode('utf8')
assert '官方 4K4D 的原生单帧人物验证' in html and 'class="katex-error"' not in html
audit=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[r for r in audit['resources'] if 'dynamic-human-4k4d-official-035' in r['path']]
assert len(new)==7
report=dict(status='export_sync_build_http_and_formula_parse_pass',new_text_present=True,katex_errors=0,
    checked_resources=audit['checked_resources'],new_resources=len(new),preview_pid=43332,
    visual='Actual local source/render PNGs reviewed. Browser interaction/screenshot QA not completed.',local_only=True)
(base.OUT/'website-qa.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report))
