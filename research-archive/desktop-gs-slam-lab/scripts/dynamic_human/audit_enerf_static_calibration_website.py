import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-static-calibration-037'
base.main()
with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/dynamic-human-reconstruction-benchmark',timeout=30) as r:html=r.read().decode('utf8')
for term in ['全部训练相机的几何检查','另一段时间的静态特征','它们没有进入人体外观训练','当前模型仍使用原相机']:assert term in html
assert 'class="katex-error"' not in html
audit=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[r for r in audit['resources'] if any(t in r['path'] for t in ['static-camera-036','static-calibration-037'])]
assert len(new)==4
report=dict(status='export_sync_build_http_content_and_resource_hashes_pass',new_resources=4,linked_resources=audit['checked_resources'],
    new_content='Static geometry findings, disjoint-time calibration-data access, and unapplied candidate scope explicitly stated.',
    visual='No new figures. Browser screenshot/interaction verification not performed.',goal_complete=False,local_only=True)
(base.OUT/'website-qa.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
