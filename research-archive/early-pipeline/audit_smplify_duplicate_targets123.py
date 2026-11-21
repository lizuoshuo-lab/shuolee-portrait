"""Read-only contradictory-target audit; no observations dropped or fitted."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
R=Path('/opt/gs-slam-lab/shuolee-local-surface-123/grass-exploratory')
REF=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control')
O=REF/'conditioned-reference';M=REF.parent/'grass-exploratory/provenance-audit/mapping-audit/official/smplifyx/utils.py'
from datetime import datetime,timezone
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def stats(a):return {'n':a.size,'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}
assert not (O/'duplicate-target-audit.json').exists()
save(O/'duplicate-target-plan.json',{'registered':datetime.now(timezone.utc).isoformat(),'scope':'All94 frozen native detector arrays plus3 completed fits; identify identical official SMPL-X joint IDs receiving different 2D targets. No deletion, threshold change, fitting or camera update.',
 'input_sha256':hashlib.sha256((R/'data/observations.npz').read_bytes()).hexdigest(),'reason':'Direct body and hand maps can repeat the same wrist joint; a correct map does not guarantee mutually consistent detector observations.',
 'budget':{'frames':94,'fit_steps':0,'cpu_seconds':30}})
tree=ast.parse(M.read_text());func=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='smpl_to_openpose'];ns={'np':np}
exec(compile(ast.Module(body=func,type_ignores=[]),str(M),'exec'),ns)
mapping=ns['smpl_to_openpose'](model_type='smplx',use_hands=True,use_face=True,use_face_contour=False,openpose_format='coco25')
coco=np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11]);names=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle']
obs=dict(np.load(R/'data/observations.npz'));gt=np.zeros((len(obs['keypoints']),118,2));ids=np.r_[coco,np.arange(19,25),np.arange(25,67),np.arange(67,118)];src=np.r_[np.arange(23),np.arange(91,133),np.arange(40,91)]
gt[:,ids]=obs['keypoints'][:,src]
duplicates=[]
for joint in np.unique(mapping):
    locations=np.where(mapping==joint)[0]
    if len(locations)<2:continue
    assert len(locations)==2
    a,b=locations;distance=np.linalg.norm(gt[:,a]-gt[:,b],axis=-1)
    duplicates.append({'smplx_joint_index':int(joint),'target_indices':[int(a),int(b)],'gap_px':stats(distance),'minimum_possible_max_residual_px':stats(distance/2),'per_frame_gap_px':distance.tolist(),'three_reviewed_view_gaps':{str(s):float(distance[s]) for s in [0,31,62]}})
views=[]
for slot in [0,31,62]:
    f=dict(np.load(O/f'slot{slot:03d}-fit.npz'));err=np.linalg.norm(f['projected'][0]-f['targets'][0],axis=-1)
    order=np.argsort(err[coco])[::-1]
    views.append({'slot':slot,'body_sorted':[{'name':names[i],'residual_px':float(err[coco[i]]),'confidence':float(f['confidence'][0,coco[i]])} for i in order],
      'duplicate_projected_coordinate_max_difference_px':max(float(np.linalg.norm(f['projected'][0,d['target_indices'][0]]-f['projected'][0,d['target_indices'][1]])) for d in duplicates)})
out={'created':datetime.now(timezone.utc).isoformat(),'duplicates':duplicates,'views':views,'claim':'Triangle inequality lower bound is algebraic, not a physical ground-truth error. The detector inputs can disagree even with an exact semantic map. Visibility/material identity remain unverified.','gaussian_steps':0,'observations_changed':False}
save(O/'duplicate-target-audit.json',out)
print(json.dumps({'duplicates':[{k:v for k,v in d.items() if k!='per_frame_gap_px'} for d in duplicates],'views':views}))
