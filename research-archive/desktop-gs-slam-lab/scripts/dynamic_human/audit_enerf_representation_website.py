from pathlib import Path
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-representation-048'
base.main()
slugs={'dynamic-human-scene-reconstruction':['纹理清楚与运动表面正确需要分别检验','手臂出现拉长'],
       'dynamic-human-reconstruction-benchmark':['行走人物：训练、覆盖范围与表示对照','142,735','9 组改善、5 组变差']}
for slug,terms in slugs.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r:html=r.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
report=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[r for r in report['resources'] if any(s in r['path'] for s in ['training-044','geometry-045','footprints-046','visibility-047','representation-048','time-049'])]
assert len(new)>=11
report['status']='current_singleframe_representation_theory_and_evidence_HTTP_verified'
report['new_resources_verified']=len(new)
report['visual_qualification']='Underlying actual comparison images reviewed; no browser layout screenshot QA in this update.'
(base.OUT/'website-http-audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],resources=report['checked_resources'],new_resources=len(new))))
