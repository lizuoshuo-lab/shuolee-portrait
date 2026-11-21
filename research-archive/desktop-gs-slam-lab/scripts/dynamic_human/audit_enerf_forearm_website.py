"""Verify the built pages and all linked source bytes, including071/072."""
import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-forearm-matte-repair-072'
base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['运动控制节点的密度与观测支持','掩码怎样改变颜色监督','当前可见的重建效果'], 'dynamic-human-reconstruction-benchmark':['运动控制节点密度的同预算对照','三张训练图的局部掩码修正']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r:html=r.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[x for x in r['resources'] if 'fine-graph-071' in x['path'] or 'forearm-matte-repair-072' in x['path']]
assert len(new)>=20 and not any(x['path'].endswith(('.pt','.npz','.pkl','.zip','.ply')) for x in new)
qa=dict(status='knowledge_evidence_and_all_linked_resources_exact_bytes_HTTP_verified',checked_resources=r['checked_resources'],new_resources=len(new),math_parse_errors=0,build='website-build.log',visual_qualification='Actual scientific image review recorded separately; browser layout not yet asserted.',goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa))
