from pathlib import Path
import time,subprocess,sys
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-heldout-cameras-068';S=L/'scripts/dynamic_human';deadline=time.monotonic()+1200
while not (R/'graph_zero/native-evaluation/metrics.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Frozen model renders not finished; inspect existing process, do not restart blindly')
 time.sleep(5)
for script,args in [('audit_compare_enerf_heldout_images.py',[]),('temporal_enerf_heldout_cameras.py',['--before-label','Head motion baseline','--after-label','Local residual motion'])]:
 print('START',script,flush=True);subprocess.run([sys.executable,str(S/script),*args],check=True)
print('54nativeimage audits and15temporal differences/3decoded videos complete; visual review required',flush=True)
