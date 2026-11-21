"""Real exact-ancestor CPU controls; role arguments do not run any GPU code."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
HERE=Path(__file__).resolve().parent
PY=ROOT/'.venv-large-gs/Scripts/python.exe'
ADAPTER=OUT/'city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ADAPTER)=='260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3'
assert not (HERE/'report.json').exists()
own=HERE/'own';own.mkdir(exist_ok=False)
script=own/'evaluate_local_city_hierarchy_006.py'
script.write_text('''import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index("--control-root")+1]);adapter=Path(sys.argv[sys.argv.index("--adapter")+1])
if "--chain-parent" in sys.argv:
    args=[sys.executable,str(Path(__file__).resolve()),"--gpu","--chain-child","--control-root",str(root),"--adapter",str(adapter)]
    p=subprocess.Popen(args,creationflags=subprocess.CREATE_NO_WINDOW)
    (root/"parent-ready.json").write_text(json.dumps({"pid":os.getpid(),"child_popen_pid":p.pid,"argv":sys.argv,"CPU_only":True}),encoding="utf-8")
    p.wait(timeout=25)
else:
    s=importlib.util.spec_from_file_location("CPU_exact_ancestor_adapter",adapter);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    # Only source-path metadata is replaced for this CPU control. The actual
    # production no_live_trainers/helper bodies and exact Windows query run.
    m.__file__=str(Path(__file__).resolve())
    (root/"child-ready.json").write_text(json.dumps({"pid":os.getpid(),"parent_pid":os.getppid(),"argv":sys.argv,"Torch_imported":"torch" in sys.modules,"CUDA_called":False}),encoding="utf-8")
    deadline=time.monotonic()+15
    while not (root/"go.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
    rows=[]
    for case in ("self-exact-ancestor","other-independent-evaluator"):
        d=root/case;d.mkdir()
        try:m.no_live_trainers(d,case);row={"case":case,"status":"allowed"}
        except Exception as ex:row={"case":case,"status":"rejected","error":str(ex)}
        rows.append(row)
        (root/(case+"-returned.flag")).write_text("actual CPU call returned",encoding="utf-8")
        if case=="self-exact-ancestor":
            deadline=time.monotonic()+15
            while not (root/"other-ready.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
    (root/"child-results.json").write_text(json.dumps({"results":rows,"Torch_imported":"torch" in sys.modules,"CUDA_called":False}),encoding="utf-8")
''',encoding='utf-8')
parent=subprocess.Popen([str(PY),str(script),'--gpu','--chain-parent','--control-root',str(own),'--adapter',str(ADAPTER)],creationflags=subprocess.CREATE_NO_WINDOW)
other=None
def await_file(p,seconds=10):
    deadline=time.monotonic()+seconds
    while not p.exists() and time.monotonic()<deadline:time.sleep(.05)
    assert p.exists(),'CPU control timeout: '+str(p)
try:
    await_file(own/'child-ready.json');await_file(own/'parent-ready.json')
    (own/'go.flag').write_text('go',encoding='utf-8');await_file(own/'self-exact-ancestor-returned.flag')
    otherdir=HERE/'other';otherdir.mkdir();otherscript=otherdir/'evaluate_local_city_hierarchy_006.py'
    otherscript.write_text('''import json,os,sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index("--root")+1]);(root/"ready.json").write_text(json.dumps({"pid":os.getpid(),"argv":sys.argv,"CPU_only":True}),encoding="utf-8")
deadline=time.monotonic()+15
while not (root/"stop.flag").exists() and time.monotonic()<deadline:time.sleep(.05)
''',encoding='utf-8')
    other=subprocess.Popen([str(PY),str(otherscript),'--gpu','--root',str(otherdir)],creationflags=subprocess.CREATE_NO_WINDOW)
    await_file(otherdir/'ready.json');(own/'other-ready.flag').write_text('ready',encoding='utf-8')
    await_file(own/'child-results.json');parent.wait(timeout=10)
finally:
    if other is not None:
        (HERE/'other/stop.flag').write_text('natural CPU fixture exit',encoding='utf-8');other.wait(timeout=10)
rows=json.loads((own/'child-results.json').read_text())
assert rows['results'][0]['status']=='allowed' and rows['results'][1]['status']=='rejected'
proofs=[]
for case in ('self-exact-ancestor','other-independent-evaluator'):
    path=own/case/(case+'-preGPU-actual-training-process-audit.json');data=json.loads(path.read_text());proofs.append({'case':case,'path':str(path),'sha256':sha(path),'excluded_own_pids':data['excluded_own_pids'],'actual_conflicting_pids':[p['ProcessId'] for p in data['actual_conflicting_GPU_stage_processes']],'exact_ancestor_proof_count':len(data['exact_ancestor_proofs'])})
assert proofs[0]['exact_ancestor_proof_count']>=1 and not proofs[0]['actual_conflicting_pids'] and proofs[1]['actual_conflicting_pids']
result={'status':'actual_CPU_exact_own_ancestor_allowed_other_independent_evaluator_rejected','adapter_sha256':sha(ADAPTER),'parent_ready':json.loads((own/'parent-ready.json').read_text()),'child_ready':json.loads((own/'child-ready.json').read_text()),'other_ready':json.loads((HERE/'other/ready.json').read_text()),'actual_query_proofs':proofs,'parent_terminal_exit_code':parent.returncode,'other_terminal_exit_code':other.returncode,'Torch_imported':rows['Torch_imported'],'CUDA_called':False,'actual1404_completed':False,'scope':'Real short CPU mock processes. Production no_live_trainers invoked with CPU fixture source-path metadata; exact real ancestor/argv query body unchanged. Not any GPU/model/render completion.'}
(HERE/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert sha(ADAPTER)=='260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3'
print(json.dumps({'status':result['status'],'report':str(HERE/'report.json'),'sha256':sha(HERE/'report.json'),'query_proofs':proofs,'Torch_imported':rows['Torch_imported'],'CUDA_called':False},ensure_ascii=False))
