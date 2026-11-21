from pathlib import Path
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-surface-long-056'
base.main()
for slug,terms in {
 'dynamic-human-scene-reconstruction':['表面表示与表面一致性约束','掩码中的未知区域与几何观测覆盖','2D Gaussian Splatting'],
 'dynamic-human-reconstruction-benchmark':['二维高斯表面与几何正则项','更长训练与持续增密','训练照片中的人体关节与粗略人体先验','800,528']
}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:html=response.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
prefixes=['mask-uncertainty-053','surface-tracks-054','surface-gs-055','surface-long-056','body-observations-057','body-prior-058']
new=[x for x in r['resources'] if any(s in x['path'] for s in prefixes)]
for s in prefixes:assert any(s in x['path'] for x in new),s
assert not any(x['path'].endswith(('.pkl','.npz','.pt','.ply','.zip')) for x in new)
r.update(status='surface_regularization_long_training_and_body_prior_pages_HTTP_verified',new_resources_verified=len(new),
 visual_qualification='Actual saved native result panels and bounded body-prior projection panels separately reviewed. No browser layout screenshot claimed.')
(base.OUT/'website-http-audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=r['status'],checked_resources=r['checked_resources'],new_resources=len(new))))
