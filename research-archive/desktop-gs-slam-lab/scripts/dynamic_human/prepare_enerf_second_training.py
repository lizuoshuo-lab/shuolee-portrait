"""Prepare shared new graph and timestamp-rebased matched trainers."""
from pathlib import Path
import json,hashlib,subprocess,sys
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-second-interval-079'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert (R/'initial.npz').exists() and not (R/'graph.npz').exists()
text=(S/'prepare_enerf_residual_graph.py').read_text().replace('dynamic-human-enerf-articulated-control-064','dynamic-human-enerf-second-interval-079').replace('dynamic-human-enerf-residual-graph-067','dynamic-human-enerf-second-interval-079').replace('Same064train-only visual-hull initialization','New079frame726train-only visual-hull initialization')
(S/'prepare_enerf_second_graph.py').write_text(text);subprocess.run([sys.executable,str(S/'prepare_enerf_second_graph.py')],check=True)
base=(S/'enerf_graph_renderer.py').read_text();assert base.count('750')==2
(S/'enerf_second_renderer.py').write_text(base.replace('750','726'))
local=(S/'enerf_temporal_graph.py').read_text();assert local.count('from enerf_graph_renderer import')==2
(S/'enerf_second_temporal_graph.py').write_text(local.replace('from enerf_graph_renderer import','from enerf_second_renderer import'))
text=(S/'train_enerf_graph_control.py').read_text()
def once(a,b):
 global text
 assert text.count(a)==1,(a,text.count(a));text=text.replace(a,b,1)
once('from enerf_graph_renderer import GraphMotion,ResidualGraphRenderer,bind_graph','from enerf_second_renderer import bind_graph')
once("OUT=LAB/'outputs/dynamic-human-enerf-articulated-control-064'","OUT=LAB/'outputs/dynamic-human-enerf-second-interval-079'")
once("choices=['residual_graph']","choices=['global_cubic','local_bspline']")
once("assert read(OUT/'runtime-smoke.json')['status']=='identity_affine_covariance_gradients_and_density_bindings_verified'","""assert read(OUT/'runtime-smoke.json')['status']=='new_interval_forward_gradients_zero_extension_verified'
 if a.branch=='global_cubic':
  from enerf_second_renderer import GraphMotion,ResidualGraphRenderer
 else:
  from enerf_second_temporal_graph import GraphMotion,ResidualGraphRenderer""")
once('nodes=384,coefficients=6912','nodes=384,coefficients=6912 if a.branch==\'global_cubic\' else 23040')
once('reference_physical_frame=750','reference_physical_frame=726')
once("for n in ['train_enerf_graph_control.py','enerf_articulated_model.py','enerf_graph_renderer.py','enerf_residual_graph.py']:","for n in ['train_enerf_second_interval.py','enerf_articulated_model.py','enerf_second_renderer.py','enerf_second_temporal_graph.py','enerf_residual_graph.py']:")
(S/'train_enerf_second_interval.py').write_text(text)
save(R/'training-source-audit.json',dict(status='prepared_before_runtime_probe_and_training',
 new_cubic_renderer_sha256=sha(S/'enerf_second_renderer.py'),new_temporal_module_sha256=sha(S/'enerf_second_temporal_graph.py'),trainer_sha256=sha(S/'train_enerf_second_interval.py'),
 renderer_change='Exactlytwo750->726time-origin replacements; timeunit6andallmathunchanged. Temporal module imports rebaselocalrenderer. Original modules unchanged.',
 training_change='Newinputroot,twobranchselector,source/protocol names andactual runtimegate; same105image sampler,seed73,15000updates,loss/rates/densitysettings.',
 initial_sha256=sha(R/'initial.npz'),graph_sha256=sha(R/'graph.npz'),body_motion_sha256=sha(R/'head/capture-affines.npz'),goal_complete=False))
(S/'run_enerf_second_training.sh').write_text('''#!/bin/bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
L=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
for branch in global_cubic local_bspline; do
 python "$L/scripts/dynamic_human/train_enerf_second_interval.py" --branch "$branch" --result-root "$L/outputs/dynamic-human-enerf-second-interval-079" --motion-file "$L/outputs/dynamic-human-enerf-second-interval-079/head/capture-affines.npz"
done
''',newline='\n')
print('Matched new-interval trainers prepared; runtimeprobe still required.')
