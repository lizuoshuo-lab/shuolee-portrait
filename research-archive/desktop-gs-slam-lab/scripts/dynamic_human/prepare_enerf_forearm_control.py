"""Fixed 072 intervention: three local training mattes, original067model/sampling."""
from pathlib import Path
import json,shutil,hashlib
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';B=L/'outputs/dynamic-human-enerf-residual-graph-067'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda n,p:(S/n).write_text(p,encoding='utf8',newline='\n')
audit=read(R/'mask-independent-audit.json');assert audit['accepted_frames']==[750,752,754]
assert not (R/'graph.npz').exists()
shutil.copy2(B/'graph.npz',R/'graph.npz');assert sha(R/'graph.npz')==sha(B/'graph.npz')
p=(S/'train_enerf_graph_control.py').read_text(encoding='utf8')
old="records=[r for r in read(OUT/'input-audit.json')['records'] if r['split']=='train'];"
new="repair_audit=read(a.result_root/'mask-independent-audit.json'); repairs={(x['camera'],x['frame']):x for x in repair_audit['records'] if x['accepted']}; assert set(repairs)=={('01',750),('01',752),('01',754)}\n "+old
assert old in p;p=p.replace(old,new,1)
old="K[0,2]-=x0;K[1,2]-=y0\n  obs.append"
new="""K[0,2]-=x0;K[1,2]-=y0
  train_matte=z['matte']
  if (r['camera'],r['frame']) in repairs:
   row=repairs[r['camera'],r['frame']];candidate=a.result_root/'candidates'/f"camera{r['camera']}-frame{r['frame']:06d}.npz"
   assert sha(f)==row['source_sha256'] and sha(candidate)==row['candidate_sha256']
   train_matte=np.load(candidate)['matte'];assert train_matte.shape==z['matte'].shape
  obs.append"""
assert old in p;p=p.replace(old,new,1)
p=p.replace("matte=tensor(z['matte'][y0:y1,x0:x1])","matte=tensor(train_matte[y0:y1,x0:x1])")
p=p.replace("graph_file=a.result_root/'graph.npz';","assert read(root/'initial-and-sampling-audit.json')==read(LAB/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph/initial-and-sampling-audit.json')\n graph_file=a.result_root/'graph.npz';")
p=p.replace("'train_enerf_graph_control.py'","'train_enerf_forearm_control.py'")
write('train_enerf_forearm_control.py',p)
for src,dst in [('render_enerf_fine_graph_heldout.py','render_enerf_forearm_heldout.py'),('compare_enerf_fine_graph_heldout.py','compare_enerf_forearm_heldout.py')]:
 p=(S/src).read_text(encoding='utf8').replace('dynamic-human-enerf-fine-graph-071',R.name).replace('Finer residual graph','Local matte repair').replace('for071','for072').replace('Evaluate071','Evaluate072')
 write(dst,p)
p=(S/'audit_trained_enerf_fine_graph.py').read_text(encoding='utf8').replace('dynamic-human-enerf-fine-graph-071',R.name).replace('1536','384')
write('audit_trained_enerf_forearm.py',p)
protocol=dict(status='fixed_before_training',baseline='067',updates=15000,training_images=105,accepted_camera='01',accepted_even_frames=[750,752,754],intervention='Only local training matte changed on three images. Both composited RGB target and alpha target follow that matte. Original RGB untouched.',graph_nodes=384,unchanged=['Initial52109Gaussians','Graphfile,materialweights,bodymotionandclock','Randomdraws and background','Lossweights andlearningrates','Densityschedule','Allother102trainingmattes','Alloriginalevaluationinputsandmetrics'],additional_assessment='Raw native photograph at visible forearm interiors and actual rendered crops; released matte is not GT and may penalize correctly restored pixels.',qualification='Predicted training labels reviewed visually. One seed, development views/times. Prior039/053mask experiments had limited or mixed benefits; no success assumed.',goal_complete=False)
(R/'ablation-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf8')
write('run_enerf_forearm_control.sh','''#!/usr/bin/env bash
set -euo pipefail
source /opt/gs-slam-lab/surface-gs/env.sh
LAB=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab
S="$LAB/scripts/dynamic_human"
RESULT="$LAB/outputs/dynamic-human-enerf-forearm-matte-repair-072"
MOTION="$LAB/outputs/dynamic-human-enerf-head-motion-066/capture-affines.npz"
python "$S/train_enerf_forearm_control.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/evaluate_enerf_graph_control.py" --branch residual_graph --result-root "$RESULT" --motion-file "$MOTION"
python "$S/render_enerf_forearm_heldout.py"
''')
print('072prepared: three training mattes only, original384node graph.')
