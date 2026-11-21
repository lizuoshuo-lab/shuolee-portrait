from pathlib import Path
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-forearm-control-061';base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['人体关键点约束的是结构，不是真实衣物表面','人体关节或脚部标志点'], 'dynamic-human-reconstruction-benchmark':['全身关键点与人体结构先验的修正','裸露前臂的柔性表面约束','208,468']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:html=response.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'));prefixes=['wholebody-059','wholebody-prior-060','forearm-control-061'];new=[x for x in r['resources'] if any(s in x['path'] for s in prefixes)]
for s in prefixes:assert any(s in x['path'] for x in new)
assert not any(x['path'].endswith(('.pkl','.npz','.pt','.ply','.zip')) for x in new)
r.update(status='wholebody_prior_and_forearm_control_pages_HTTP_verified',new_resources_verified=len(new),visual_qualification='Actual saved panels separately reviewed; browserlayout screenshot notclaimed.')
(base.OUT/'website-http-audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(dict(status=r['status'],checked_resources=r['checked_resources'],new_resources=len(new))))
