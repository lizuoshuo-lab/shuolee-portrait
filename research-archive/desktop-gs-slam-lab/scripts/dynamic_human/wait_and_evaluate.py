"""Run the frozen held-out evaluation only after both successful full budgets."""
from pathlib import Path
import json,time,subprocess,sys,hashlib

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-reconstruction-003'
deadline=time.monotonic()+6*3600
while True:
    states=[]
    for mode in ['static','dynamic']:
        p=OUT/mode/'completion.json'
        if p.exists():
            state=json.loads(p.read_text())
            assert state['status']=='trained_evaluation_pending' and state['gradient_updates']==70000 and state['frames']==31
            states.append(mode)
    if len(states)==2:break
    if time.monotonic()>deadline:raise TimeoutError('Both training completion records did not arrive within six hours')
    time.sleep(10)
receipt=json.loads((OUT/'implementation-receipt.json').read_text())
for item in receipt['files']:
    assert hashlib.sha256((LAB/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
print('Both training budgets complete; starting frozen evaluation.',flush=True)
subprocess.run([str(LAB/'.venv-large-gs/Scripts/python.exe'),str(LAB/'scripts/large_scene_gs/run_local.py'),str(LAB/'scripts/dynamic_human/evaluate_benchmark.py')],check=True,cwd=LAB)
print('Evaluation saved; all-view visual review still required.',flush=True)
