"""Prepare a single-factor graph-resolution ablation; never replace baseline sources."""
from pathlib import Path
import json,subprocess,sys
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-fine-graph-071';R.mkdir(exist_ok=True)
assert not (R/'graph.npz').exists()
def write(name,text): (S/name).write_text(text,encoding='utf8',newline='\n')
p=(S/'prepare_enerf_residual_graph.py').read_text(encoding='utf8').replace('dynamic-human-enerf-residual-graph-067','dynamic-human-enerf-fine-graph-071').replace('192','768').replace('384','1536')
write('prepare_enerf_fine_graph.py',p)
subprocess.run([sys.executable,str(S/'prepare_enerf_fine_graph.py')],check=True)
p=(S/'smoke_enerf_graph_renderer.py').read_text(encoding='utf8').replace('dynamic-human-enerf-residual-graph-067','dynamic-human-enerf-fine-graph-071').replace("'smoke_enerf_graph_renderer.py'","'smoke_enerf_fine_graph.py'")
write('smoke_enerf_fine_graph.py',p)
p=(S/'train_enerf_graph_control.py').read_text(encoding='utf8').replace('nodes=384,coefficients=6912','nodes=1536,coefficients=27648').replace("'train_enerf_graph_control.py'","'train_enerf_fine_graph.py'")
p=p.replace("graph_file=a.result_root/'graph.npz';", "assert read(root/'initial-and-sampling-audit.json')==read(LAB/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph/initial-and-sampling-audit.json')\n assert read(a.result_root/'runtime-smoke.json')['actual_duplicate_split_material_binding_check']\n graph_file=a.result_root/'graph.npz';")
write('train_enerf_fine_graph.py',p)
p=(S/'evaluate_enerf_graph_control.py').read_text(encoding='utf8').replace("p['node_indices'].max()<384","p['node_indices'].max()<len(graph.node_points)")
write('evaluate_enerf_fine_graph.py',p)
p=(S/'render_enerf_actor_heldout.py').read_text(encoding='utf8').replace('dynamic-human-enerf-actor-supervision-070','dynamic-human-enerf-fine-graph-071').replace('Evaluate070','Evaluate071').replace('for070','for071')
write('render_enerf_fine_graph_heldout.py',p)
p=(S/'compare_enerf_actor_heldout.py').read_text(encoding='utf8').replace('dynamic-human-enerf-actor-supervision-070','dynamic-human-enerf-fine-graph-071').replace('Instance-supervised graph','Finer residual graph')
write('compare_enerf_fine_graph_heldout.py',p)
protocol=dict(status='fixed_before_training',baseline='067',hypothesis='Finer spatial motion support may reduce averaging of differently moving fine structures; this is unproven.',nodes_before=384,nodes_after=1536,per_person_before=192,per_person_after=768,trainable_motion_coefficients_before=6912,trainable_motion_coefficients_after=27648,updates=15000,images=105,initial_gaussians=52109,unchanged=['Original native RGB, matte, calibration','Frozen066bodyandclocks','InitialGaussianparameters','Drawscheduleandbackground','Lossweights, learningrates anddensityschedule','Four material-node weights per Gaussian, eight graph edges per node','Same105evenimages,90oddvalidation and18newcamera validation'],intervention='Graph node density and resulting neighbor distances/weights. Edge-averaged rigidity uses the same coefficient; its effective spatial smoothing is not identical as distances change.',qualification='No ownership labels from070; local research extension; same updates not same compute; all current validation is development data.',visual_review=['Fixed12knownfull24details','Fixed9newcamera full18details','Worstchangedview ineachsplit'],goal_complete=False)
(R/'ablation-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
write('run_enerf_fine_graph.sh','''#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
S="$LAB/scripts/dynamic_human"
RESULT="$LAB/outputs/dynamic-human-enerf-fine-graph-071"
MOTION="$LAB/outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz"
python "$S/smoke_enerf_fine_graph.py"
python "$S/train_enerf_fine_graph.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/evaluate_enerf_fine_graph.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/render_enerf_fine_graph_heldout.py"
''')
print('Prepared fixed1536node ablation, actual training not yet started.')
