"""Exercise the actual candidate's reviewed 2D/3D input loader, not real fitting."""
from pathlib import Path
import json,hashlib,copy,importlib.util,sys
import numpy as np
import torch,smplx
from smplx_direct_correspondence123 import load_reviewed_face,sha
OUT=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/direct-smplx-bindings')
FIX=OUT/'synthetic-input-fixture'
MODEL='/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz'
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not FIX.exists();(FIX/'data').mkdir(parents=True)
z=dict(np.load(OUT/'synthetic-known-correspondences.npz'))
model=smplx.SMPLX(MODEL,num_betas=10,num_expression_coeffs=10,use_pca=False,
                  flat_hand_mean=False,use_face_contour=False,batch_size=1)
rows=[dict(frame=str(i),wh=[1520,2704],w2c=z['E'][i].tolist(),K_integer_centers=z['K_integer'][i].tolist()) for i in range(3)]
obs=dict(frame_ids=np.array([str(i) for i in range(3)]),keypoints=np.zeros((3,133,2)),
         joint_names=np.array([f'face-{i-23}' if 23<=i<91 else f'other-{i}' for i in range(133)]))
obs['keypoints'][:,40:91]=z['target_uv']
np.savez_compressed(FIX/'data/observations.npz',**obs);save(FIX/'data/inputs.json',dict(records=rows))
records=[]
for i in range(3):
    for j in range(51):
        records.append(dict(frame=str(i),ibug68_index=j+17,anchor_id=f'ibug68:{j+17}',
            visibility='unknown' if i==1 else 'observed',semantic_identity_verified=True,
            split='HELD' if j%7==0 else 'FIT',physical_family_id=f'known-template:{j}',
            review_evidence='synthetic-known-correspondences.npz',visibility_source='synthetic_known',
            uv_native=z['target_uv'][i,j].tolist(),localization_sigma_px=2.,reliability=1.))
doc=dict(synthetic_control=True,coordinate_domain='native_integer_pixel_centres',
         observations_sha256=sha(FIX/'data/observations.npz'),inputs_sha256=sha(FIX/'data/inputs.json'),records=records)
path=FIX/'data/smplx-face-observations.json';save(path,doc)
objective,receipt=load_reviewed_face(FIX,rows,obs,model,device='cpu',synthetic=True)
checks=[dict(name='all153_states_consumed',passed=len(receipt['records'])==153),
        dict(name='unknown_camera_entirely_inactive',passed=bool((objective.fit_weight[1]==0).all())),
        dict(name='eyes_present_in_loss',passed=bool((objective.fit_weight[:,19:31]>0).any()))]
def reject(name,mutator):
    trial=copy.deepcopy(doc);mutator(trial);save(path,trial)
    try:load_reviewed_face(FIX,rows,obs,model,device='cpu',synthetic=True)
    except (ValueError,KeyError) as exc:checks.append(dict(name=name,passed=True,message=str(exc)))
    else:checks.append(dict(name=name,passed=False))
reject('mismatched_anatomical_identity',lambda d:d['records'][1].update(anchor_id='ibug68:19'))
reject('changed_native_pixel',lambda d:d['records'][1].update(uv_native=[5.,5.]))
reject('same_physical_family_split_leak',lambda d:d['records'][51].update(split='FIT'))
reject('confidence_is_not_observed_visibility',lambda d:d['records'][1].update(visibility_source='detector_score'))
reject('incomplete_unknown_states',lambda d:d['records'].pop())
reject('changed_input_hash',lambda d:d.update(inputs_sha256='0'*64))
save(path,doc)
try:load_reviewed_face(FIX,rows,obs,model,device='cpu',synthetic=False)
except ValueError:checks.append(dict(name='synthetic_cannot_enter_real_fitter',passed=True))
else:checks.append(dict(name='synthetic_cannot_enter_real_fitter',passed=False))
# Import the actual separately generated fitter with its unchanged dependencies.
project=Path('/opt/gs-slam-lab/shuolee-local-surface-123/runtime')
sys.path.insert(0,str(project/'src'))
from shuolee_portrait import runtime
runtime.configure('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets')
spec=importlib.util.spec_from_file_location('shuolee_portrait.direct_fitting123',OUT/'candidate_fitting.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
checks.append(dict(name='real_candidate_imports_direct_objective',passed=module.load_reviewed_face is load_reviewed_face))
save(OUT/'input-contract-result.json',dict(checks=checks,passed=all(x['passed'] for x in checks),
    actual_fitter_imported=True,real_fitter_executed=False,synthetic_loader_executed=True,
    candidate_sha256=sha(OUT/'candidate_fitting.py'),module_sha256=sha(Path(__file__).with_name('smplx_direct_correspondence123.py')),
    training_started=False,quality_accepted=False))
assert all(x['passed'] for x in checks)
print(json.dumps(dict(checks=len(checks),passed=True,real_fit=False)))
