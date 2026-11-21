"""Verify completed RGB control is served with exact linked artifact bytes."""
import contextlib
import json
import urllib.request
import audit_street_site as base

base.OUT=base.LAB/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
with (base.OUT/'website-audit.log').open('w',encoding='utf8') as output:
    with contextlib.redirect_stdout(output):
        base.main()
for slug,terms in {
    'dynamic-human-scene-reconstruction':['手部观测必须区分可见性与预测分数','本地修复姿态网络的RGB输入后'],
    'dynamic-human-reconstruction-benchmark':['正确颜色输入后的完整重建对照','228414','实际查看38张对照','23.745']
}.items():
    with urllib.request.urlopen('http://127.0.0.1:8799/gs-lab/read/'+slug,timeout=30) as r:
        html=r.read().decode('utf8')
    assert 'class="katex-error"' not in html
    for term in terms:
        assert term in html,(slug,term)
report=json.loads((base.OUT/'website-http-audit.json').read_text(encoding='utf8'))
published=json.loads((base.OUT/'published-resources.json').read_text(encoding='utf8'))['paths']
checked={r['path'] for r in report['resources']}
for path in published:
    assert 'outputs/'+base.OUT.name+'/'+path in checked,path
all_new=[p for p in checked if 'hand-observability-073/' in p or 'rgb-pose-correction-074/' in p]
assert len(all_new)==21
assert not any(p.endswith(('.pt','.npz','.pkl','.onnx','.zip','.ply')) for p in all_new)
qa=dict(status='completed_RGB_control_knowledge_evidence_and_linked_assets_HTTP_verified',
    checked_resources=report['checked_resources'],new_completed_control_resources=len(published),
    hand_and_RGB_resources=len(all_new),math_parse_errors=0,
    build='website-build.log',licensed_models_exported=False,
    visual_qualification='38 rendered comparison PNGs actually inspected; web layout inspection recorded separately.',
    goal_complete=False)
(base.OUT/'website-qa.json').write_text(json.dumps(qa,indent=2)+'\n',encoding='utf8')
print(json.dumps(qa))
