import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-residual-graph-067'
base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['连续人体运动如何带动高斯形状','骨架之外的局部变形如何进入渲染'], 'dynamic-human-reconstruction-benchmark':['连续人体运动与拍摄时间的实际重建对照','骨架之后的局部变形对照','清晰人物重建仍未通过']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:html=response.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
new=[x for x in r['resources'] if any(s in x['path'] for s in ['continuous-body-063','articulated-control-064','residual-graph-067'])]
for prefix in ['continuous-body-063','articulated-control-064','residual-graph-067']:assert any(prefix in x['path'] for x in new)
assert not any(x['path'].endswith(('.pkl','.npz','.pt','.ply','.zip')) for x in new)
r.update(status='residual_graph_knowledge_evidence_HTTP_verified',new_resources_verified=len(new),visual_qualification='Actual reconstruction panels and fixed details visually reviewed in recorded scopes; no browserlayout screenshot claimed.')
(base.OUT/'website-http-audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
qa=dict(status=r['status'],checked_resources=r['checked_resources'],new_resources=len(new),build='website-build.log',math_parse_errors=0,goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n')
print(json.dumps(qa))
