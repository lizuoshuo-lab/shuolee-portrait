import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-pose-sequence-062';base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['人体关键点约束的是结构，不是真实衣物表面','从一个人物估计的偏移应当有助于解释另一个人物的运动'], 'dynamic-human-reconstruction-benchmark':['全身关键点与人体结构先验的修正','裸露前臂的柔性表面约束','连续人体关键点中的跨相机时间偏移','2,262']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:html=response.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'));prefixes=['wholebody-059','wholebody-prior-060','forearm-control-061','pose-sequence-062'];new=[x for x in r['resources'] if any(s in x['path'] for s in prefixes)]
for s in prefixes:assert any(s in x['path'] for x in new)
assert not any(x['path'].endswith(('.pkl','.npz','.pt','.ply','.zip')) for x in new)
r.update(status='wholebody_forearm_and_continuous_pose_pages_HTTP_verified',new_resources_verified=len(new),visual_qualification='Saved actual projection/result panels reviewed in bounded scopes; no browserlayout screenshot claimed.')
(base.OUT/'website-http-audit.json').write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
qa=dict(status=r['status'],checked_resources=r['checked_resources'],new_resources=len(new),build='website-build.log',export_documents=71,synced_documents=68,synced_resources=1178,math_parse_errors=0,goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n');p=base.LAB/'manifests/dynamic-human-work.json';m=json.loads(p.read_text(encoding='utf8'));m['outdoor_pose_sequence_website_qa']='outputs/dynamic-human-enerf-pose-sequence-062/website-qa.json';p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps(qa))
