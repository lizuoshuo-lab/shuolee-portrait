"""Finite postprocessing of the fixed071 experiment; no model selection or retraining."""
from pathlib import Path
import subprocess,sys,time,json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-fine-graph-071';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph';A=R/'residual_graph'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def wait(p):
 deadline=time.monotonic()+3600
 while not p.exists():
  if time.monotonic()>deadline:raise TimeoutError(str(p))
  time.sleep(10)
def run(cmd):
 print('START',cmd[0],flush=True);subprocess.run([sys.executable,str(S/cmd[0]),*cmd[1:]],check=True);print('DONE',cmd[0],flush=True)
wait(A/'native-evaluation/metrics.json')
assert read(A/'initial-and-sampling-audit.json')==read(B/'initial-and-sampling-audit.json')
old=np.load(B/'sampling-schedule.npz');new=np.load(A/'sampling-schedule.npz');assert old.files==new.files
for k in old.files:assert np.array_equal(old[k],new[k]),k
assert read(A/'protocol.json')==read(B/'protocol.json')
g=read(A/'graph-protocol.json');b=read(B/'graph-protocol.json')
assert {k:v for k,v in g.items() if k not in ['graph_sha256','nodes','coefficients']}=={k:v for k,v in b.items() if k not in ['graph_sha256','nodes','coefficients']}
assert g['nodes']==1536 and g['coefficients']==27648
audit=dict(status='same105nativeimages_initialGaussianparameters_draws_background_motion_and_training_schedule_verified',updates=15000,nodes=1536,coefficients=27648,baseline_nodes=384,intervention='Finer node sampling and material bindings; physical regularization neighborhood also changes.',goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
for cmd in [
 ['audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',str(R)],
 ['compare_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Finer residual graph'],
 ['temporal_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Finer residual graph']]:run(cmd)
wait(R/'heldout/native-evaluation/metrics.json')
run(['compare_enerf_fine_graph_heldout.py'])
run(['temporal_enerf_heldout_cameras.py','--output-root',str(R/'heldout'),'--before-root',str(L/'outputs/dynamic-human-enerf-heldout-cameras-068/graph_estimated'),'--after-root',str(R/'heldout'),'--before-label','Residual graph baseline','--after-label','Finer residual graph'])
shutil.copy2(__file__,R/Path(__file__).name)
print('071 numeric postprocessing complete; visual review required.',flush=True)
