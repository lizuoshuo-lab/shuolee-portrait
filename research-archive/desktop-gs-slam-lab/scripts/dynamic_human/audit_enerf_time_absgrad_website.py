from pathlib import Path
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-absgrad-052'
base.main()
for slug,terms in {
 'dynamic-human-scene-reconstruction':['增密梯度与细节恢复','AbsGS','相反方向的细节需求'],
 'dynamic-human-reconstruction-benchmark':['单机位时间偏移与联合相机图','79,659','三条变差']
}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r:html=r.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for t in terms:assert t in html,(slug,t)
report=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[r for r in report['resources'] if any(s in r['path'] for s in ['time-050','graph-051','absgrad-052'])]
assert len(new)==13
report['status']='time_control_and_absgrad_theory_evidence_HTTP_verified'
report['new_resources_verified']=len(new)
report['visual_qualification']='Actual native six-view comparison panels and17detail regions reviewed in each control; browser layout screenshot not performed.'
(base.OUT/'website-http-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],checked_resources=report['checked_resources'],new_resources=len(new))))
