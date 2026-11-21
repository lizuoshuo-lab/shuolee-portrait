"""Derive a tightly scoped trainer without modifying the frozen106source."""
from pathlib import Path
import json,hashlib,difflib
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(S/'train_actor23_control.py')==read(I/'comparison-protocol.json')['source_sha256']['train_actor23_control.py'];source=(S/'train_actor23_control.py').read_text(encoding='utf8');s=source
def replace(old,new,count=1):
    global s
    assert s.count(old)==count,(old,s.count(old),count)
    s=s.replace(old,new)
replace('Matched static, nominal-time dynamic and clock-corrected dynamic baselines.','Matched surface-path rigidity and material-binding controls, from106baseline.')
replace("O=L/'outputs/dynamic-human-actor23-independent-106';M=O/'model'","I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';M=I/'model'")
replace("choices=['static','dynamic_nominal','dynamic_clock'],default='dynamic_clock'","choices=['path_rigidity','path_rigidity_binding'],default='path_rigidity'")
replace("a=ap.parse_args();torch.set_num_threads(6)","a=ap.parse_args();R=O/a.branch;G=R/'graph.npz';control=read(O/'protocol.json');assert sha(G)==control['variants'][a.branch]['graph_sha256'];assert sha(M/'initial.npz')==control['initial_sha256'];torch.set_num_threads(6)")
replace("sha(M/'graph.npz')==init['graph_sha256']","sha(I/'model/graph.npz')==init['graph_sha256']")
replace("M/'graph.npz'","G",3)
replace("O/'geometry-time/candidate-clock-offsets.json'","I/'geometry-time/candidate-clock-offsets.json'",2)
replace("O/'training-input-audit.json'","I/'training-input-audit.json'",2)
replace("O/'validation-reservation.json'","I/'validation-reservation.json'")
replace("path=O/r['input']","path=I/r['input']")
replace("M/'runtime-smoke.json'","R/'runtime-smoke.json'",3)
replace("Static freezes all motion coefficients at zero. Dynamic nominal uses image index; dynamic clock uses image index + training-only fitted camera offset. All three share geometry/appearance initialization and195native photos, same15000draws and backgrounds. No heldout model selection.","Reuse106dynamic_clock initialization,195nativephotos,clocks,drawsandbackgrounds,15000updatesandmotionbounds. path_rigiditychangesgraphneighbors/weights only;path_rigidity_bindingalsochangesfixedGaussianmaterialbinding. Source106validationisnowdevelopment,notanewblindtest.")
replace("g.requires_grad_(a.branch!='static')","g.requires_grad_(True)")
replace("index=612 if a.branch=='static' else o['frame']+(clocks[o['camera']] if a.branch=='dynamic_clock' else 0)","index=o['frame']+clocks[o['camera']]")
replace("    if a.branch!='static':go.step();go.zero_grad(set_to_none=True);go.param_groups[0]['lr']=.003*(.1**((i+1)/steps))","    go.step();go.zero_grad(set_to_none=True);go.param_groups[0]['lr']=.003*(.1**((i+1)/steps))")
replace("if a.branch=='static':assert all(torch.count_nonzero(v)==0 for v in g.parameters())\n","")
replace(",static_motion_exactly_zero=a.branch=='static'",",graph_variant=a.branch")
replace("for name in ['train_actor23_control.py','actor23_graph_model.py']","for name in ['train_actor23_geodesic.py','actor23_graph_model.py']")
assert "a.branch=='static'" not in s and "M/'graph.npz'" not in s
out=S/'train_actor23_geodesic.py';assert not out.exists();out.write_text(s,encoding='utf8');(O/'trainer-diff.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='frozen106trainer',tofile='107geodesictrainer')),encoding='utf8')
files=['train_actor23_geodesic.py','actor23_graph_model.py','enerf_residual_graph.py','enerf_temporal_graph.py','enerf_graph_renderer.py']
(O/'training-source-contract.json').write_text(json.dumps(dict(status='trainer_derived_from_hash_verified106source_before_smoke_ortraining',source_sha256={n:sha(S/n) for n in files},baseline_trainer_sha256=sha(S/'train_actor23_control.py'),shared_native_data_manifest_sha256=sha(I/'training-input-audit.json'),shared_clock_sha256=sha(I/'geometry-time/candidate-clock-offsets.json'),goal_complete=False),indent=2)+'\n',encoding='utf8');print('107trainerwritten;frozen106sourceunchanged;reviewdiffbeforelaunch.')
