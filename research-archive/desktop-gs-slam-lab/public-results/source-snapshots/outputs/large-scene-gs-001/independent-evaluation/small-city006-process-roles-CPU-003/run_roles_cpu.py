"""Real short CPU process-role controls; never execute a GPU-role implementation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
HERE=Path(__file__).resolve().parent
PY=ROOT/'.venv-large-gs/Scripts/python.exe'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
adapter=OUT/'city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py'
posterior=OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py'
assert not (HERE/'report.json').exists()
modules=[('adapter',load('role_control_eval006',adapter)),('posterior',load('role_control_post006',posterior))]
template='''import json,os,sys,time
from pathlib import Path
stop=Path(sys.argv[sys.argv.index("--stop-file")+1]);ready=Path(sys.argv[sys.argv.index("--ready-file")+1])
ready.write_text(json.dumps({"pid":os.getpid(),"argv":sys.argv,"CPU_role_fixture":True,"Torch_imported":False,"CUDA_called":False}),encoding="utf-8")
end=time.monotonic()+20
while not stop.exists() and time.monotonic()<end:time.sleep(.05)
'''
roles=[('native006', 'run_small_city_hierarchy_006.py',['--stage','merge','--execute']),
       ('otherEval006','evaluate_local_city_hierarchy_006.py',['--gpu']),
       ('nativeMergerUnoptimized','cpu_native_wait.py',[])]
rows=[]
for role,filename,flags in roles:
    folder=HERE/role;folder.mkdir(exist_ok=False);script=folder/filename;script.write_text(template,encoding='utf-8')
    ready=folder/'ready.json';stop=folder/'stop.flag'
    cmd=[str(PY),str(script),*flags,'--ready-file',str(ready),'--stop-file',str(stop)]
    if role=='nativeMergerUnoptimized':
        # Harmless command processor bytes, deliberately renamed only inside
        # this CPU fixture. This is NOT the author Merger and never loads CUDA.
        native=folder/'GaussianHierarchyMergerUnoptimized.exe';shutil.copyfile(Path(os.environ['SystemRoot'])/'System32/cmd.exe',native)
        quoted=' '.join('"'+v+'"' for v in cmd)
        batch=folder/'cpu_wait.cmd';batch.write_text('@'+quoted+'\r\n',encoding='utf-8')
        cmd=[str(native),'/d','/c','cpu_wait.cmd']
    process=subprocess.Popen(cmd,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
    deadline=time.monotonic()+8
    while not ready.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.05)
    row={'role':role,'CPU_mock_process_argv':cmd,'popen_pid':process.pid,'script_sha256':sha(script),'ready':json.loads(ready.read_text()) if ready.exists() else None,'results':[]}
    try:
        assert ready.exists(),'CPU fixture failed to become ready'
        for name,m in modules:
            dest=folder/name;dest.mkdir()
            try:
                if name=='adapter':m.no_live_trainers(dest,'CPU-role-'+role)
                else:m.terminal_query(dest,'CPU-role-'+role)
                result={'module':name,'status':'UNSAFE_allow_of_live_mock_GPU_role'}
            except Exception as exc:
                result={'module':name,'status':'rejected_live_mock_role','error':str(exc)}
            artifacts=list(dest.glob('*.json'))
            result['actual_query_artifacts']=[{'path':str(p),'sha256':sha(p)} for p in artifacts]
            row['results'].append(result)
    finally:
        stop.write_text('CPU fixture natural exit requested\n',encoding='utf-8')
        stdout,stderr=process.communicate(timeout=30)
        row.update(process_exit_code=process.returncode,stderr=stderr.decode(errors='replace'),stdout=stdout.decode(errors='replace'))
    rows.append(row)
result={'status':'actual_CPU_role_controls_finished','roles':rows,'adapter_sha256':sha(adapter),'posterior_sha256':sha(posterior),'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'actual1404_completed':False,'scope':'Real temporary CPU processes with GPU-role names/argv only; no author executable, training, renderer or metric run.'}
result['unsafe_allow_count']=sum(r['status'].startswith('UNSAFE') for row in rows for r in row['results'])
result['status']='CPU_role_classification_failures_found' if result['unsafe_allow_count'] else 'CPU_live_role_rejections_passed'
(HERE/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'report':str(HERE/'report.json'),'sha256':sha(HERE/'report.json'),'status':result['status'],'unsafe_allow_count':result['unsafe_allow_count'],'roles':[{'role':r['role'],'results':r['results'],'exit_code':r['process_exit_code']} for r in rows],'Torch_imported':result['Torch_imported'],'CUDA_called':False},ensure_ascii=False))
