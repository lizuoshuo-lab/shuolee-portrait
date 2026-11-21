from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess

P=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
O=R/'generic-2d3d-research/adapter-a0'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'plan.json').exists(), 'A0 already registered; inspect instead of restarting.'
O.mkdir(exist_ok=True)
files=[P/'src/shuolee_portrait/cli.py',P/'src/shuolee_portrait/fitting.py',P/'src/shuolee_portrait/admission.py',
       H/'local_surface123_alignment_gate.py',H/'smplx_direct_correspondence123.py',
       R/'direct-smplx-bindings/candidate_fitting.py',R/'generic-2d3d-research/adaptation-plan.json']
dirty=subprocess.check_output(['git','status','--porcelain'],cwd=P,text=True,encoding='utf-8')
save(O/'plan.json',dict(created=datetime.now(timezone.utc).isoformat(),stage='A0',
    hypothesis='An explicit backend and a single hash-bound native observation adapter remove the observed call-path/face-stream disconnect without changing cameras or subject-specific settings.',
    variable='Only interface, call dispatch and the recorded fixed-51 semantic/pixel mapping; no optimization or detector rerun.',
    repository=str(P),baseline_git_status=dirty,before={str(p):sha(p) for p in files},
    preserves=['All pre-existing dirty files','Frozen WSL runtime','Existing 113 deployment','Camera/fit/training admission thresholds','Original observation arrays and review decisions'],
    work=['Portable camera/observation adapter accepts explicit COCO133 or IBUG68 face sources; exact ordered frame identity and native float coordinates.',
          'Reuse the existing fixed SMPL-X51 binding/loss candidate; port the unchanged reviewed admission conditions, remove host-specific path resolution.',
          'Explicit opt-in fit backend in actual CLI; read-only fit-inputs preflight shares the same consumer.',
          'CPU tests of the new interface only, including negative controls and real177-frame intake without cameras/assigned families.'],
    budget=dict(CPU_test_seconds=120,real_fits=0,new_detectors=0,GPU_jobs=0,new_weights=0,training=0),
    stops=['Frame/order/hash/semantic mismatch','FIT/HELD family overlap','Unexplained camera convention change','Any input mutation','Any unintended runtime or prior dirty-file change'],
    claims_excluded=['Real geometry aligned','Physical calibration passed','Full official SMPLify-X reproduction','Ghosting resolved','New trained model']))
print(O)
