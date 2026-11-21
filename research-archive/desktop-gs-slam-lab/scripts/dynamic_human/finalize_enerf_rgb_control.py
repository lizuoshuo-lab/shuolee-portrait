"""Original image targets and frozen review regions, with changed pose only."""
from pathlib import Path
import subprocess,sys,time,json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph';A=R/'residual_graph'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def wait(p):
 deadline=time.monotonic()+3600
 while not p.exists():
  if time.monotonic()>deadline:raise TimeoutError(str(p))
  time.sleep(10)
def run(cmd):
 print('START',cmd[0],flush=True);subprocess.run([sys.executable,str(S/cmd[0]),*cmd[1:]],check=True);print('DONE',cmd[0],flush=True)
wait(A/'native-evaluation/metrics.json')
assert read(A/'initial-and-sampling-audit.json')==read(B/'initial-and-sampling-audit.json')
a=np.load(A/'sampling-schedule.npz');b=np.load(B/'sampling-schedule.npz');assert a.files==b.files
for k in a.files:assert np.array_equal(a[k],b[k])
old=read(B/'protocol.json');new=read(A/'protocol.json');assert old['motion_sha256']!=new['motion_sha256']
assert {k:v for k,v in old.items() if k!='motion_sha256'}=={k:v for k,v in new.items() if k!='motion_sha256'}
assert new['motion_sha256']==sha(R/'head/capture-affines.npz')
assert read(A/'graph-protocol.json')==read(B/'graph-protocol.json')
assert sha(R/'graph.npz')==sha(B.parent/'graph.npz')
audit=dict(status='sameinitial_graph_sampling_background_learning_rates_density_and105originaltargets_only_frozen_body_motion_changed',updates=15000,images=105,motion_sha256=new['motion_sha256'],graph_sha256=sha(R/'graph.npz'),hand_supervision_added=False,matte_repair_added=False,goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
for cmd in [
 ['audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',str(R)],
 ['compare_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Corrected RGB pose'],
 ['temporal_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Corrected RGB pose']]:run(cmd)
wait(R/'heldout/native-evaluation/metrics.json')
run(['compare_enerf_rgb_heldout.py'])
run(['temporal_enerf_heldout_cameras.py','--output-root',str(R/'heldout'),'--before-root',str(L/'outputs/dynamic-human-enerf-heldout-cameras-068/graph_estimated'),'--after-root',str(R/'heldout'),'--before-label','Residual graph baseline','--after-label','Corrected RGB pose'])
shutil.copy2(__file__,R/Path(__file__).name)
print('074 numeric postprocessing complete; actual native rendering review still required.',flush=True)
