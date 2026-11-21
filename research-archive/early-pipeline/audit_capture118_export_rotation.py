from pathlib import Path
import sys,json,numpy as np,hashlib
PROJECT=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')
ROOT=Path('/opt/gs-slam-lab/shuolee-new-capture-118-native-fixedk')
REC=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
sys.path.insert(0,str(PROJECT/'src'))
from shuolee_portrait.export import rigid_rotation64
raw=np.asarray(np.load(ROOT/'fit/fitted.npz')['root_rotation'][0],float).T
proper,details=rigid_rotation64(raw)
assert np.linalg.det(proper)>0.999999999 and abs(proper@proper.T-np.eye(3)).max()<1e-12
rejected=[]
for name,mat in [('reflection',np.diag([-1.,1.,1.])),('scale',np.diag([1.,1.001,1.])),('shear',np.array([[1.,.001,0],[0,1,0],[0,0,1.]]))]:
    try:rigid_rotation64(mat)
    except RuntimeError:rejected.append(name)
assert len(rejected)==3
evidence=dict(error='Export requires a rigid proper rotation.',root=str(ROOT),input_dtype='float32',
              numerical_cause=details,rejected_nonrigid_controls=rejected,
              source_sha256=hashlib.sha256((PROJECT/'src/shuolee_portrait/export.py').read_bytes()).hexdigest(),
              training_modified_or_resumed=False,render_parity_still_required=True)
(REC/'fixedk-export-roundoff-repair.json').write_text(json.dumps(evidence,indent=2))
failed=ROOT/'export';archive=ROOT/'export-attempt-001-roundoff-rejected'
assert failed.resolve().parent==ROOT.resolve() and archive.resolve().parent==ROOT.resolve()
assert failed.is_dir() and not list(failed.iterdir()) and not archive.exists()
failed.rename(archive)
print(json.dumps(evidence,indent=2))
