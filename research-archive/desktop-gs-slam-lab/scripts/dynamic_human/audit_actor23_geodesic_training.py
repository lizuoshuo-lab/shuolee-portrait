"""Verify changed graph buffers and unchanged appearance/sampling, then record state."""
from pathlib import Path
import json, hashlib, argparse
import numpy as np
from datetime import datetime, timezone
L=Path(__file__).resolve().parents[2];I=L/'outputs/dynamic-human-actor23-independent-106';O=L/'outputs/dynamic-human-actor23-geodesic-107';S=L/'scripts/dynamic_human'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--complete',action='store_true');a=ap.parse_args()
contract=read(O/'training-source-contract.json');protocol=read(O/'protocol.json')
for n,h in contract['source_sha256'].items():assert sha(S/n)==h,n
assert sha(I/'dynamic_clock/resume.pt')==protocol['base_checkpoint_sha256']
base=read(I/'dynamic_clock/initial-and-sampling-audit.json');sampling=np.load(I/'dynamic_clock/sampling-schedule.npz');bg=dict(np.load(I/'model/graph.npz'));reports={}
for b in ['path_rigidity','path_rigidity_binding']:
    R=O/b
    if not (R/'initial-and-sampling-audit.json').exists():
        assert not a.complete;reports[b]={'status':'queued'};continue
    audit=read(R/'initial-and-sampling-audit.json');s=np.load(R/'sampling-schedule.npz');g=dict(np.load(R/'graph.npz'));assert sha(R/'graph.npz')==protocol['variants'][b]['graph_sha256']
    assert audit['draw_sha256']==base['draw_sha256'] and all(np.array_equal(s[k],sampling[k]) for k in sampling.files)
    for k,v in base['initial'].items():
        if b=='path_rigidity' or k not in ['node_indices','node_weights']:assert audit['initial'][k]==v,k
    for k,v in base['initial_graph'].items():
        if k not in ['edges','edge_weights']:assert audit['initial_graph'][k]==v,k
    for k in ['node_points','node_anchor']:
        assert np.array_equal(g[k],bg[k]),k
    assert len(g['edges'])==len(bg['edges'])==3072
    expected=['edges','edge_weights']+([] if b=='path_rigidity' else ['point_node_indices','point_node_weights'])
    changed=[k for k in bg if not np.array_equal(g[k],bg[k])]
    assert sorted(changed)==sorted(expected),(changed,expected)
    completion=read(R/'completion.json') if (R/'completion.json').exists() else None
    if completion:assert completion['steps']==15000 and sha(R/'resume.pt')==completion['checkpoint_sha256']
    assert not a.complete or completion
    reports[b]=dict(status='completed' if completion else 'running',changed_graph_arrays=changed,completion=completion)
result=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),initial_and_sampling_match=True,all_training_complete=all(r['status']=='completed' for r in reports.values()),branches=reports,unchanged='Gaussian attributes, node locations, initial zero motion, training images, sampling order and background draws; only declared graph buffers differ.',goal_complete=False)
save(O/'matched-training-audit.json',result)
m=read(L/'manifests/dynamic-human-work.json');m['outdoor_actor23_graph_control'].update(status='training_complete_evaluation_pending' if result['all_training_complete'] else 'matched_training_running',training_started=True,training_audit=str((O/'matched-training-audit.json').relative_to(L)).replace('\\','/'));m['status']='107_graph_controls_training_complete' if result['all_training_complete'] else '107_graph_controls_training_running';m['current_runs']=[] if result['all_training_complete'] else [{'stage':'107','session_id':52418,'runner':'scripts/dynamic_human/run_actor23_geodesic_training.sh'}];save(L/'manifests/dynamic-human-work.json',m)
print(json.dumps(result,indent=2))
