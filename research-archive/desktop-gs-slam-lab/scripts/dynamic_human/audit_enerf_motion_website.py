"""Verify the bounded image-correspondence diagnosis served by the site."""
import contextlib,json,urllib.request
import audit_street_site as base
base.OUT=base.LAB/'outputs/dynamic-human-enerf-native-motion-tracks-075'
with (base.OUT/'website-audit.log').open('w',encoding='utf8') as output:
    with contextlib.redirect_stdout(output):
        base.main()
for slug,terms in {
    'dynamic-human-scene-reconstruction':['局部图像跟踪也不能直接替代手指观测'],
    'dynamic-human-reconstruction-benchmark':['局部图像对应的歧义','5117','5072','0.793','0.801']
}.items():
    with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as response:
        html=response.read().decode('utf8')
    assert 'class="katex-error"' not in html
    for term in terms: assert term in html,(slug,term)
report=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
paths=json.loads((base.OUT/'published-resources.json').read_text(encoding='utf8'))['paths']
checked={r['path'] for r in report['resources']}
for path in paths:
    assert 'outputs/'+base.OUT.name+'/'+path in checked,path
assert not any(p.endswith(('.pt','.npz','.pkl','.onnx','.zip','.ply')) for p in paths)
qa=dict(status='motion_diagnosis_knowledge_evidence_and_linked_assets_verified',checked_resources=report['checked_resources'],new_resources=len(paths),math_parse_errors=0,build='website-build.log',licensed_models_exported=False,goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa))
