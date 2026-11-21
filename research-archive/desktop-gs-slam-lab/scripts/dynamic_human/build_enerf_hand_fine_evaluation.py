from pathlib import Path
import json,shutil
L=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');D=L/'scripts/dynamic_human';O=L/'outputs/dynamic-human-enerf-hand-fine-085';J=L/'outputs/dynamic-human-enerf-hand-joint-084'
def common(s):
    return s.replace('dynamic-human-enerf-hand-joint-084','dynamic-human-enerf-hand-fine-085').replace("'joint_photo'","'fine_photo'").replace("'joint_flow'","'fine_flow'").replace("'hand_joint_084'","'hand_fine_085'").replace('HAND_JOINT','HAND_FINE')
s=common((D/'evaluate_enerf_hand_joint.py').read_text())
s=s.replace('from enerf_hand_joint_model import HandJointRenderer','from enerf_hand_fine_model import HandFineGraph,HandFineRenderer')
s=s.replace("O/'support/","L/'outputs/dynamic-human-enerf-hand-joint-084/support/")
insert="""fine=HandFineGraph(O/'fine-graph.npz',cp['scene_scale'])
assert cp['fine_graph_source_sha256']==sha(O/'fine-graph.npz')
base=fine.state_dict()
for k,v in base.items():
 if k in dynamic:assert torch.isfinite(cp['fine_graph'][k]).all() and cp['fine_graph'][k].abs().max()>0
 else:assert torch.equal(v.cpu(),cp['fine_graph'][k])
fine.load_state_dict(cp['fine_graph'])
"""
s=s.replace("p={k:v.cuda() for k,v in cp['params'].items()};",insert+"\np={k:v.cuda() for k,v in cp['params'].items()};")
s=s.replace("HandJointRenderer(S/'head/capture-affines.npz',graph,ref,active.cuda())","HandFineRenderer(S/'head/capture-affines.npz',graph,ref,active.cuda(),fine)")
s=s.replace("HandJointRenderer(motion,graph,ref,active.cuda())","HandFineRenderer(motion,graph,ref,active.cuda(),fine)")
s=s.replace("active_nodes=int(nodes.sum()),","active_nodes=int(nodes.sum()),additional_local_nodes=32,")
(D/'evaluate_enerf_hand_fine.py').write_text(s,encoding='utf8')
s=common((D/'compare_enerf_hand_joint.py').read_text()).replace("['Original','Parent model','Joint photo','Joint + motion']","['Original','Parent model','Fine photo','Fine + motion']")
s=s.replace("Original079eval RGB/mattes; targethandadditionalwindows source-defined before084rendering.","Original079eval RGB/mattes; targethandadditionalwindows reused unchanged from084.")
(D/'compare_enerf_hand_fine.py').write_text(s,encoding='utf8')
shutil.copy2(J/'target-hand-review-plan.json',O/'target-hand-review-plan.json')
(O/'temporal').mkdir(exist_ok=True);shutil.copy2(J/'temporal/protocol.json',O/'temporal/protocol.json')
s=common((D/'render_enerf_hand_joint_temporal.py').read_text())
s=s.replace('from enerf_hand_joint_model import HandJointRenderer','from enerf_hand_joint_model import HandJointRenderer\nfrom enerf_hand_fine_model import HandFineGraph,HandFineRenderer')
s=s.replace("O/'support/","L/'outputs/dynamic-human-enerf-hand-joint-084/support/")
s=s.replace("model=HandJointRenderer(S/'heldout/capture-affines-estimated.npz',g,ref,active);dest=T/branch;dest.mkdir(exist_ok=True)","""fine=HandFineGraph(O/'fine-graph.npz',cp['scene_scale'])
  if branch!='parent':fine.load_state_dict(cp['fine_graph'])
  model=HandFineRenderer(S/'heldout/capture-affines-estimated.npz',g,ref,active,fine);dest=T/branch;dest.mkdir(exist_ok=True)""")
(D/'render_enerf_hand_fine_temporal.py').write_text(s,encoding='utf8')
s=common((D/'audit_enerf_hand_joint_temporal.py').read_text()).replace("['Original','Parent model','Joint photo','Joint + motion']","['Original','Parent model','Fine photo','Fine + motion']")
(D/'audit_enerf_hand_fine_temporal.py').write_text(s,encoding='utf8')
for name,commands in [('evaluation','python "$L/scripts/dynamic_human/evaluate_enerf_hand_fine.py" --branch fine_photo\npython "$L/scripts/dynamic_human/evaluate_enerf_hand_fine.py" --branch fine_flow'),('temporal','python "$L/scripts/dynamic_human/render_enerf_hand_fine_temporal.py"')]:
    (D/('run_enerf_hand_fine_'+name+'.sh')).write_text('#!/usr/bin/env bash\nset -euo pipefail\nsource /opt/gs-slam-lab/surface-gs/env.sh\nL=/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab\n'+commands+'\n',encoding='utf8',newline='\n')
for name in ['evaluate_enerf_hand_fine.py','compare_enerf_hand_fine.py','render_enerf_hand_fine_temporal.py','audit_enerf_hand_fine_temporal.py']:
    compile((D/name).read_text(),name,'exec')
p=L/'manifests/dynamic-human-work.json';m=json.loads(p.read_text(encoding='utf8'));m.update(status='085hand_fine_32nodes_matched_training_running',current_runs=[dict(stage='085training',session_id=87528,command='run_enerf_hand_fine_training.sh')],outdoor_hand_fine='outputs/'+O.name+'/protocol.json',goal_turn_classification='Progress:32localcontrols projected in39trainingcontexts,13sheetsreviewed;exactinitialrender/nonhandisolation/gradient smoke passed; matchedactualtraining started.');p.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('085evaluation and fixed protocols ready')

