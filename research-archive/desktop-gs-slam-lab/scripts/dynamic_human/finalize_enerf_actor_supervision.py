"""Wait for specific070evaluation output, then audit and compare known cameras."""
from pathlib import Path
import json,subprocess,time,sys
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070';S=L/'scripts/dynamic_human';B=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph';A=R/'residual_graph'
while not (A/'native-evaluation/metrics.json').exists():time.sleep(10)
commands=[['audit_enerf_articulated_images.py','--branch','residual_graph','--result-root',str(R)],
 ['compare_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Instance-supervised graph'],
 ['temporal_enerf_articulated_control.py','--before-root',str(B),'--after-root',str(A),'--output-root',str(R),'--before-label','Residual graph baseline','--after-label','Instance-supervised graph']]
for cmd in commands:
 print('START',cmd[0],flush=True);subprocess.run([sys.executable,str(S/cmd[0]),*cmd[1:]],check=True);print('DONE',cmd[0],flush=True)
