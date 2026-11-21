import json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-heldout-cameras-068'
base.main()
for slug,terms in {'dynamic-human-scene-reconstruction':['留出视角中的时间对应与外观验证','不透明度、人物归属与遮挡要分别检查'],
 'dynamic-human-reconstruction-benchmark':['留出相机与人物层诊断','18张前景PSNR均改善','手指、鞋部和严重遮挡仍未通过清晰度验收']}.items():
 with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r:html=r.read().decode('utf8')
 assert 'class="katex-error"' not in html
 for term in terms:assert term in html,(slug,term)
r=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
for suffix in ['heldout-cameras-068','actor-layers-069']:
 new=[x for x in r['resources'] if suffix in x['path']]
 assert len(new)>5 and not any(x['path'].endswith(('.pt','.npz','.pkl','.zip','.ply')) for x in new)
 out=base.LAB/('outputs/dynamic-human-enerf-'+suffix)
 qa=dict(status='knowledge_evidence_and_exact_resource_bytes_HTTP_verified',checked_resources=r['checked_resources'],new_resources=len(new),math_parse_errors=0,
  build='dynamic-human-enerf-heldout-cameras-068/website-build.log',visual_qualification='Actual output images visually reviewed in named scopes. No browser-layout screenshot claimed.',goal_complete=False)
 (out/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
 print(suffix,json.dumps(qa))
