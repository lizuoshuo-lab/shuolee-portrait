"""Sequential, restartable equal-budget repeats; does not overwrite runs."""
import json
from pathlib import Path
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/mini-splatting2-mip-depth-001'
SCRIPT=Path(__file__).with_name('local_refinement.py')

queue=[(f'{kind}-refine-s{seed}',mode,weight,seed) for seed in [1,2]
       for kind,mode,weight in [('classic','classic',0),('mip','mip',0),('depth','classic',.01),('mip-depth','mip',.01)]]
receipt=dict(status='running',started=time.time(),jobs=[])
for name,mode,weight,seed in queue:
    status=OUT/name/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status')=='complete':
        receipt['jobs'].append(dict(name=name,status='previously_complete'));continue
    if (OUT/name).exists():raise RuntimeError(f'Incomplete run requires inspection: {name}')
    command=[sys.executable,str(SCRIPT),'train','--name',name,'--mode',mode,'--depth-weight',str(weight),'--seed',str(seed),'--steps','2000']
    receipt['current']=name
    (OUT/'repeat-queue.json').write_text(json.dumps(receipt,indent=2))
    with (OUT/f'{name}.log').open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    receipt['jobs'].append(dict(name=name,returncode=result.returncode,status='complete' if result.returncode==0 else 'failed'))
    if result.returncode:
        receipt['status']='failed';break
else:receipt['status']='complete'
receipt['finished']=time.time()
(OUT/'repeat-queue.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
