"""Actually invoke both frozen query functions, recording CPU-only evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
adapter=OUT/'city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py'
posterior=OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py'
assert sha(adapter)=='260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3'
assert sha(posterior)=='5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c'
assert not (HERE/'report.json').exists()
result={'status':'actual_CPU_query_functions_running','actual1404_completed':False,'CUDA_called':False,'sources':{'adapter':sha(adapter),'posterior':sha(posterior)},'calls':[]}
for name,path,function,args in [('adapter',adapter,'no_live_trainers',(HERE,'actualCPU')),
                                ('posterior',posterior,'terminal_query',(HERE,'actualCPU-no-evaluation'))]:
    m=load('actual_'+name+'_query_cpu',path)
    original=m.subprocess.run;events=[]
    def capture(*a,**k):
        r=original(*a,**k)
        events.append({'argv':a[0],'returncode':r.returncode,'stderr':r.stderr,'stdout_bytes':len(r.stdout.encode('utf-8')),'private_env_supplied':'env' in k})
        return r
    m.subprocess.run=capture
    row={'function':function,'source_sha256':sha(path)}
    try:
        value=getattr(m,function)(*args)
        row.update(status='actual_query_function_returned_pass',query_proof_returned=value)
    except Exception as ex:
        row.update(status='actual_query_function_failed',error=str(ex),traceback=traceback.format_exc())
    finally:
        m.subprocess.run=original
    row['actual_subprocess_queries']=events;result['calls'].append(row)
result.update(status='actual_CPU_query_functions_finished',Torch_imported='torch' in sys.modules)
(HERE/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert sha(adapter)==result['sources']['adapter'] and sha(posterior)==result['sources']['posterior']
print(json.dumps({'report':str(HERE/'report.json'),'sha256':sha(HERE/'report.json'),'calls':[{'function':r['function'],'status':r['status'],'stderr':[q['stderr'][:500] for q in r['actual_subprocess_queries']]} for r in result['calls']],'Torch_imported':result['Torch_imported'],'CUDA_called':False},ensure_ascii=False))
