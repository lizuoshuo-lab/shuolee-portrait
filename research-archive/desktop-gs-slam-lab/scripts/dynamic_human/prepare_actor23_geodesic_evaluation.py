"""Freeze stage107 evaluation from the inspected stage106 implementation."""
from pathlib import Path
import json, hashlib, difflib, py_compile
from datetime import datetime, timezone

L = Path(__file__).resolve().parents[2]
S = L / 'scripts/dynamic_human'
I = L / 'outputs/dynamic-human-actor23-independent-106'
O = L / 'outputs/dynamic-human-actor23-geodesic-107'
read = lambda p: json.loads(p.read_text(encoding='utf8'))
save = lambda p, x: p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n', encoding='utf8')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'evaluation-implementation.json').exists()
old_contract = read(I/'evaluation-implementation.json')
for name, digest in old_contract['scripts'].items():
    assert sha(S/name) == digest, name
for name, digest in old_contract['plans'].items():
    assert sha(I/name) == digest, name

def replace(text, old, new, count=1):
    assert text.count(old)==count, (old, text.count(old), count)
    return text.replace(old, new)

old = (S/'evaluate_actor23_control.py').read_text(encoding='utf8')
t = replace(old, "O=L/'outputs/dynamic-human-actor23-independent-106';S=", "I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';S=")
t = replace(t, "choices=['static','dynamic_nominal','dynamic_clock']", "choices=['path_rigidity','path_rigidity_binding']")
t = replace(t, "contract=read(O/'comparison-protocol.json')", "contract=read(O/'training-source-contract.json')")
t = replace(t, "g=ActorGraph(O/'model/graph.npz',cp['scene_scale'])", "assert sha(B/'graph.npz')==read(O/'protocol.json')['variants'][a.branch]['graph_sha256'];g=ActorGraph(B/'graph.npz',cp['scene_scale'])")
t = replace(t, "if a.branch=='static':assert all(torch.count_nonzero(v)==0 for v in g.parameters())\n", '')
t = replace(t, "read(O/('evaluation-protocol.json'", "read(I/('evaluation-protocol.json'")
t = replace(t, "O/'evaluation-clock-decision.json'", "I/'evaluation-clock-decision.json'", 2)
t = replace(t, "O/'review-plan/protocol.json'", "I/'review-plan/protocol.json'")
t = replace(t, "path=O/r['input']", "path=I/r['input']")
t = replace(t, "time_index=612 if a.branch=='static' else f+(offsets[c] if a.branch=='dynamic_clock' else 0)", "time_index=f+offsets[c]")
new = S/'evaluate_actor23_geodesic.py'
if new.exists():assert new.read_text(encoding='utf8')==t
else:new.write_text(t, encoding='utf8')
py_compile.compile(str(new), doraise=True)
(O/'evaluator-diff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),t.splitlines(True),fromfile='frozen106',tofile='107')),encoding='utf8')

# Preserve the old comparison's independent float/PNG checks and exact-pixel panels.
old = (S/'compare_actor23_control.py').read_text(encoding='utf8')
t = replace(old, "O=L/'outputs/dynamic-human-actor23-independent-106';D=O/'comparison'", "I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';D=O/'comparison'")
# Token substitution is simultaneous: baseline, edge-only, edges-plus-binding.
for a,b in [('static','BASE_PLACEHOLDER'),('dynamic_nominal','EDGE_PLACEHOLDER'),('dynamic_clock','BIND_PLACEHOLDER')]:
    t=t.replace("'"+a+"'", "'"+b+"'")
for a,b in [('BASE_PLACEHOLDER','dynamic_clock'),('EDGE_PLACEHOLDER','path_rigidity'),('BIND_PLACEHOLDER','path_rigidity_binding')]:
    t=t.replace("'"+a+"'", "'"+b+"'")
t=replace(t, "labels=['Original','Static GS','Dynamic GS','Dynamic GS + time offset']", "roots={b:(I/b if b=='dynamic_clock' else O/b) for b in B};labels=['Original','Euclidean graph','Surface-path edges','Surface-path edges + binding']")
t=t.replace("O/b/", "roots[b]/")
t=replace(t,"O/'static/sampling-schedule.npz'", "I/'dynamic_clock/sampling-schedule.npz'")
for f in ['evaluation-protocol.json','review-plan/protocol.json']:
    t=t.replace("O/'"+f+"'", "I/'"+f+"'")
t=t.replace("O/r['input']", "I/r['input']").replace("O/inputs[c,f]['input']", "I/inputs[c,f]['input']")
t=replace(t,"assert initials[b]['initial']==initials['dynamic_clock']['initial'] and initials[b]['initial_graph']==initials['dynamic_clock']['initial_graph']", "assert read(O/'matched-training-audit.json')['initial_and_sampling_match'];assert all(initials[b]['initial'][k]==v for k,v in initials['dynamic_clock']['initial'].items() if k not in ['node_indices','node_weights'])")
t=t.replace('dynamic_vs_static_db','change_vs_baseline_db').replace('clock_vs_nominal','binding_vs_edges')
t=t.replace("identical_initial_parameters_and_sampling=True", "identical_initial_Gaussian_attributes_and_sampling=True,graph_changes_audited_separately=True")
t=t.replace("scope='Independent actors, same geographic recording environment.", "scope='Inspected106validation is development for these graph controls, not a new blind test. Independent actors, same geographic recording environment.")
new=S/'compare_actor23_geodesic.py';assert not new.exists();new.write_text(t,encoding='utf8');py_compile.compile(str(new),doraise=True)
(O/'comparison-diff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),t.splitlines(True),fromfile='frozen106',tofile='107')),encoding='utf8')

old=(S/'audit_actor23_continuity.py').read_text(encoding='utf8')
t=replace(old,"O=L/'outputs/dynamic-human-actor23-independent-106';T=O/'continuity'", "I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';T=O/'continuity'")
t=replace(t,"plan=read(O/'continuity-protocol.json')", "plan=read(I/'continuity-protocol.json')")
t=replace(t,"B=['static','dynamic_nominal','dynamic_clock'];labels=['Original','Static GS','Dynamic GS','Dynamic GS + time offset']", "B=['dynamic_clock','path_rigidity','path_rigidity_binding'];roots={b:(I/b if b=='dynamic_clock' else O/b) for b in B};labels=['Original','Euclidean graph','Surface-path edges','Surface-path edges + binding']")
t=t.replace("O/b/", "roots[b]/").replace("O/r['input']", "I/r['input']")
new=S/'audit_actor23_geodesic_continuity.py';assert not new.exists();new.write_text(t,encoding='utf8');py_compile.compile(str(new),doraise=True)
(O/'continuity-diff.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),t.splitlines(True),fromfile='frozen106',tofile='107')),encoding='utf8')
scripts=['evaluate_actor23_geodesic.py','compare_actor23_geodesic.py','audit_actor23_geodesic_continuity.py']
save(O/'evaluation-implementation.json',dict(status='frozen_before107rendering',recorded_utc=datetime.now(timezone.utc).isoformat(),scripts={n:sha(S/n) for n in scripts},shared106plans=old_contract['plans'],baseline_checkpoint_sha256=read(O/'protocol.json')['base_checkpoint_sha256'],evaluation_images_per_branch=274,scope='Same106source-defined full-person/detail/temporal windows; now development, not blind confirmation.',goal_complete=False))
print('GEODESIC_EVALUATION_PREPARED')
