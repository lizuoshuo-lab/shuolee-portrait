"""Prepare a fixed105-image temporal-basis control and preserve audit sources."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human'
R=L/'outputs/dynamic-human-enerf-temporal-basis-078';A=L/'outputs/dynamic-human-author-reference-077'
B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074'
R.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save(A/'reference-audit.json',dict(status='author_public_checkpoint_downloads_quota_limited',
    repo_commit='157b62aa0538181ec2f2f9ab67f72938d2423e92',
    source='https://github.com/zju3dv/4K4D',
    files=[dict(name='99.npz',id='1yKnWkMQtnZPHraUE5zkPgrx64zMsZNQT',display_size='171.6MB'),dict(name='latest.pt',id='172Z9F8bFz93rm5OuTbvrS-UNcKQmgPjZ',display_size='1.08GB')],
    observed='Both unauthenticated official downloads reported too many users; no checkpoint received. Folder metadata accessible. No repeated bypass attempts.',
    local_baselines=dict(original035_history_updates=10000,corrected_source_pipeline044_history_updates=3000),
    author_config=dict(foreground_epochs=1600,ep_iter=500,joint_epochs=100),
    interpretation='Single-frame local diagnostics are not a complete official fullsequence reproduction. Original10000step run predates mask repair. Author checkpoint would provide a reference on its own frame span, not a fair untouched heldout score.',
    goal_complete=False))
protocol=dict(status='fixed_before_training',
    hypothesis='Global cubic residual motion may underfit local changes of moving extremities. Add compact smooth temporal bases while retaining the exact old motion family as a zero-extension case.',
    baseline='074correctRGBmotion with original photometricloss; not076weightedloss',
    temporal_basis='tau=(capture_frame-750)/6; old[tau,tau^2,tau^3] plus B3((tau-k)/0.5)-B3(-k/0.5), k=-1.5,-1,-0.5,0,0.5,1,1.5. Centered cardinal cubic B-spline, C2continuous.',
    local_rotation_coefficient_bound_rad=0.03,local_translation_coefficient_bound_scene_scale=0.01,
    basis_l1_bound=2,reference_identity_exact=True,
    changes='Adds7rotation and7translation coefficients/node, zero initialized. Same spatial topology,4material bindings,bodyaffines,rasterizer,loss,AdamLR,sampling,Gaussians and15000updatebudget.',
    unchanged='105originaleven-timeimages and original masks only. Same52109initialpoints,384nodes,seed73,randombackgrounds,densityschedule. No075flow or076ROIweight.90oddknown-camera and18other-camera images stay out of appearance training.',
    regularizers='Unchanged:10*graph_rigidity +0.001*rotation_size +0.1*translation_size, evaluated on total motion. No new parameter tuned to validation results.',
    acceptance='All120native renders; samefixedfullperson/detailpanels+expandedhands andworstchanges. Keep all failed cases. Not accepted from trainingloss alone.',
    limitation='Local research extension, single seed, reused development times/cameras; no claim of independent final test or fullscene completion.',
    body_motion_sha256=sha(B/'head/capture-affines.npz'),graph_sha256=sha(B/'graph.npz'),
    prepared_utc=datetime.now(timezone.utc).isoformat(),goal_complete=False)
assert not (R/'protocol.json').exists()
save(R/'protocol.json',protocol)
shutil.copy2(B/'graph.npz',R/'graph.npz')
review=read(L/'outputs/dynamic-human-enerf-detail-weight-076/visual-review-plan.json');review['status']='fixed_before_temporal_basis_training';review['decision']='Compare078to074; keep all prior failure views.';save(R/'visual-review-plan.json',review)
shutil.copytree(L/'outputs/dynamic-human-enerf-detail-weight-076/expanded-detail',R/'expanded-detail-plan',ignore=shutil.ignore_patterns('*.png'))

def once(text,a,b):
    assert text.count(a)==1,(a,text.count(a));return text.replace(a,b,1)

text=(S/'train_enerf_graph_control.py').read_text(encoding='utf8')
text=once(text,'from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer,bind_graph','from enerf_temporal_graph import GraphMotion,ResidualGraphRenderer,bind_graph')
text=once(text,'nodes=384,coefficients=6912','nodes=384,coefficients=23040,temporal_basis="globalcubic_plus7compact_cubic_Bsplines"')
text=once(text,"for n in ['train_enerf_graph_control.py','enerf_articulated_model.py','enerf_graph_renderer.py','enerf_residual_graph.py']:","for n in ['train_enerf_temporal_control.py','enerf_temporal_graph.py','enerf_articulated_model.py','enerf_graph_renderer.py','enerf_residual_graph.py']:")
(S/'train_enerf_temporal_control.py').write_text(text,encoding='utf8')
text=(S/'evaluate_enerf_graph_control.py').read_text(encoding='utf8')
text=once(text,'from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer','from enerf_temporal_graph import GraphMotion,ResidualGraphRenderer')
text=once(text,"['rotation_raw','translation_raw']","['rotation_raw','translation_raw','local_rotation_raw','local_translation_raw']")
text=once(text,'assert gsteps==[15000,15000]','assert gsteps==[15000,15000,15000,15000]')
(S/'evaluate_enerf_temporal_control.py').write_text(text,encoding='utf8')
runner='''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
python "$L/scripts/dynamic_human/train_enerf_temporal_control.py" --branch residual_graph --result-root "$L/outputs/dynamic-human-enerf-temporal-basis-078" --motion-file "$L/outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz"
python "$L/scripts/dynamic_human/evaluate_enerf_temporal_control.py" --branch residual_graph --result-root "$L/outputs/dynamic-human-enerf-temporal-basis-078" --motion-file "$L/outputs/dynamic-human-enerf-rgb-pose-correction-074/head/capture-affines.npz"
'''
(S/'run_enerf_temporal_control.sh').write_text(runner,encoding='utf8',newline='\n')
save(R/'source-change-audit.json',dict(baseline_trainer_sha256=sha(S/'train_enerf_graph_control.py'),candidate_trainer_sha256=sha(S/'train_enerf_temporal_control.py'),temporal_model_sha256=sha(S/'enerf_temporal_graph.py'),only_training_model_import_and_protocol_snapshot_changed=True,goal_complete=False))
shutil.copy2(__file__,R/Path(__file__).name)
print('077reference evidence recorded;078local temporal basis prepared. Runtime probe required before training.')
