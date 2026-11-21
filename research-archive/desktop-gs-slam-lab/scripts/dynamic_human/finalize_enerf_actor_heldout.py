from pathlib import Path
import json,time,subprocess,sys
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070';S=L/'scripts/dynamic_human'
while not (R/'heldout/native-evaluation/metrics.json').exists():time.sleep(10)
commands=[['compare_enerf_actor_heldout.py'],['temporal_enerf_heldout_cameras.py','--output-root',str(R/'heldout'),'--before-root',str(L/'outputs/dynamic-human-enerf-heldout-cameras-068/graph_estimated'),'--after-root',str(R/'heldout'),'--before-label','Residual graph baseline','--after-label','Instance-supervised graph']]
for cmd in commands:
 print('START',cmd[0],flush=True);subprocess.run([sys.executable,str(S/cmd[0]),*cmd[1:]],check=True);print('DONE',cmd[0],flush=True)
while not (R/'actor-layers/render-summary.json').exists():time.sleep(10)
subprocess.run([sys.executable,str(S/'audit_enerf_supervised_layers.py')],check=True)
