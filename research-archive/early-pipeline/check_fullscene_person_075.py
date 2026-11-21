"""Check the new calibration path without claiming measured-camera accuracy."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab') if sys.platform!='win32' else Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
B=L/'outputs/mini-splatting2-mip-depth-001/shuolee';R=B/'current/person-static-fullscene-camera-fresh-075'
sys.path.insert(0,str(L/'repos/PGSR-shuolee'))
from camera_contract import require_registered_cameras
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
d=read(R/'data/training-inputs.json');counts=require_registered_cameras(d)
assert counts['kinds']=={'fresh_full_scene_sfm':175} and counts['experimental_training_authorized']
E=np.load(B/'current/full-scene-localization-074/camera-comparison.npz')['new_aligned']
assert np.array_equal(E,np.array([r['w2c'] for r in d['records']]))
obs=np.load(R/'data/observations.npz');assert np.array_equal(obs['w2c'],E)
assert np.array_equal(obs['K'],np.array([r['K'] for r in d['records']]))
assert [r['source_frame_index'] for r in d['records']]==list(range(1,176))
assert len(np.load(R/'clothing-init/points.npz')['xyz'])==200000
for name in ['source-inputs.json','clothing-init/audit.json']:
 a=read(R/name);assert not a['old_gaussian_loaded'] and not a['old_subject_fit_loaded']
assert read(R/'clothing-support/audit.json')['old_sparse_or_subject_fit_loaded'] is False
assert read(R/'camera-source.json')['remap_roundtrip_max_px']<1e-5
print(json.dumps(dict(passed=True,cameras=counts,frames=175,new_points=200000,old_learned_state=False,camera_accuracy_accepted=False)))
