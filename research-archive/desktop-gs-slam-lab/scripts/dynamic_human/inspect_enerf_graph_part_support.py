"""Read-only training-geometry diagnosis of graph-node support by body region."""
from pathlib import Path
import json,pickle,shutil
import numpy as np
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-residual-graph-067'
with open('/opt/gs-slam-lab/dynamic-human/repos/drivestudio/smpl_models/SMPL_NEUTRAL.pkl','rb') as f:body=pickle.load(f,encoding='latin1')
weights=np.asarray(body['weights']);assert weights.shape==(6890,24)
groups={'torso':[0,3,6,9,13,14],'head_neck':[12,15],'upper_arm':[16,17],'forearm':[18,19],'hand_wrist':[20,21,22,23],'thigh':[1,2],'lower_leg':[4,5],'foot_ankle':[7,8,10,11]}
names=list(groups);region=np.stack([weights[:,groups[n]].sum(1) for n in names],1).argmax(1)
g=dict(np.load(R/'graph.npz'));z=dict(np.load(L/'outputs/dynamic-human-enerf-articulated-control-064/initial.npz'))
node_region=region[g['node_anchor']%6890];point_region=region[z['anchor']%6890];node_person=g['node_anchor']//6890;point_person=z['anchor']//6890
rows=[]
for person in [0,1]:
 for ri,name in enumerate(names):
  sel=(point_person==person)&(point_region==ri);nodes=(node_person==person)&(node_region==ri)
  linked_regions=node_region[g['point_node_indices'][sel]];same=(linked_regions==ri);w=g['point_node_weights'][sel];distance=np.linalg.norm(z['points'][sel,None]-g['node_points'][g['point_node_indices'][sel]],axis=-1)
  rows.append(dict(person=person,region=name,initial_points=int(sel.sum()),graph_nodes=int(nodes.sum()),mean_weight_on_same_region=float((w*same).sum(1).mean()),same_region_node_available_fraction=float(same.any(1).mean()),weighted_distance_world_units_median=float(np.median((distance*w).sum(1))),weighted_distance_world_units_p90=float(np.percentile((distance*w).sum(1),90))))
out=dict(status='read_only_training_initialization_region_support_diagnosis',rows=rows,qualification='Regions assigned by summed SMPL skinning weights at nearest surface anchor, not semanticGT. Clothes/toys can inherit anatomically misleading labels. Counts alone do not establish cause of image blur.',next='Use actual fixed-region images together with these aggregate counts before choosing adaptive graph refinement; no new training performed.',goal_complete=False)
(R/'part-support-diagnostic.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(out))
