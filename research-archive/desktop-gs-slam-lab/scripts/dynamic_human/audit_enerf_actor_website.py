import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-actor-supervision-070'
base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['人物归属监督怎样作用于联合渲染','未恢复明显的精细结构'], 'dynamic-human-reconstruction-benchmark':['人物归属监督的同预算训练对照','不替换067开发基线']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r: html=r.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[x for x in r['resources'] if 'actor-supervision-070' in x['path']]
assert len(new)>=17 and not any(x['path'].endswith(('.pt','.npz','.pkl','.zip','.ply')) for x in new)
qa=dict(status='knowledge_evidence_exact_resource_bytes_HTTP_verified',checked_resources=r['checked_resources'],new_resources=len(new),math_parse_errors=0,build='website-build.log',visual_qualification='Actual image review recorded; no browser layout screenshot claimed.',goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa))
