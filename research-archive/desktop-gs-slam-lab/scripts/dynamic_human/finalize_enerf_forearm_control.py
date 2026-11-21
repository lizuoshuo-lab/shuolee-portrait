"""Bounded CPU comparison; keep all original evaluation mattes."""
from pathlib import Path
import subprocess,sys,time,json,hashlib,shutil
import numpy as np
L=Path(__file__).resolve().parents[2];S=L/'scripts/dynamic_human';R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph';A=R/'residual_graph'
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
old=np.load(B/'sampling-schedule.npz');new=np.load(A/'sampling-schedule.npz');assert old.files==new.files
for k in old.files:assert np.array_equal(old[k],new[k]),k
assert read(A/'protocol.json')==read(B/'protocol.json')
assert read(A/'graph-protocol.json')==read(B/'graph-protocol.json')
assert sha(R/'graph.npz')==sha(B.parent/'graph.npz')
m=read(R/'mask-independent-audit.json');accepted=[r for r in m['records'] if r['accepted']]
for r in m['records']:
 assert sha(L/'outputs/dynamic-human-enerf-articulated-control-064/native-input'/f"camera01-frame{r['frame']:06d}.npz")==r['source_sha256']
 assert sha(R/'candidates'/f"camera01-frame{r['frame']:06d}.npz")==r['candidate_sha256']
count=sum(c=='01' and int(f) in m['accepted_frames'] for c,f in zip(new['camera'],new['frame']))
audit=dict(status='sameinitial_graph_motion_sampling_background_and_schedule_three_local_training_matte_overrides_verified',updates=15000,images=105,modified_training_images=3,matte_supervised_updates=int(count),changed_native_pixels=m['accepted_changed_pixels'],mask_independent_audit_sha256=sha(R/'mask-independent-audit.json'),original_sources_unchanged=True,evaluation_mattes='Original unchanged for comparability, not GT; raw foreground interior check reported separately.',goal_complete=False)
(R/'training-protocol-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
for cmd in [
 ['audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',str(R)],
 ['compare_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Local matte repair'],
 ['temporal_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Local matte repair']]:run(cmd)
wait(R/'heldout/native-evaluation/metrics.json')
run(['compare_enerf_forearm_heldout.py'])
run(['temporal_enerf_heldout_cameras.py','--output-root',str(R/'heldout'),'--before-root',str(L/'outputs/dynamic-human-enerf-heldout-cameras-068/graph_estimated'),'--after-root',str(R/'heldout'),'--before-label','Residual graph baseline','--after-label','Local matte repair'])
shutil.copy2(__file__,R/Path(__file__).name)
print('072numeric postprocessing complete; actual visual review still required.',flush=True)
