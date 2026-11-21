"""Actual saved-model gates plus isolated CPU fixtures for supervised006 role/chain guard."""
from pathlib import Path
import importlib.util
import hashlib
import json
import subprocess
import sys
import copy
from types import SimpleNamespace
from datetime import datetime, timezone
sys.dont_write_bytecode = True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab'); BASE=ROOT/'outputs/large-scene-gs-001'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'; OUT=LOCK/'cpu-validation-002'
OUT.mkdir(exist_ok=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
source=ROOT/'scripts/large_scene_gs/run_small_city_hierarchy_006.py'
spec=importlib.util.spec_from_file_location('_actual006_CPU002',source)
d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
config,plan=read(LOCK/'configuration.json'),read(LOCK/'plan.json')
manifest=read(BASE/'official-inputs/small-city-training-002/manifest.json')
helper,codec,training,_=d.frozen_helpers(LOCK)
tracked=[source,LOCK/'driver-source.py',LOCK/'configuration.json',LOCK/'plan.json',d.BINDING_PATH,d.BINDER_PATH,d.BINDER_SNAPSHOT]
before={str(p):sha(p) for p in tracked}
assert sha(source)==sha(LOCK/'driver-source.py')==config['driver_sha256']
d.execution_source_gate()
positive=d.lineage_gate(config,manifest,training,codec)
controls=[]
def rejected(name,fn,scope='actual_stage_or_model_gate'):
    try:fn()
    except (RuntimeError,FileNotFoundError,ValueError,KeyError) as e:
        controls.append({'name':name,'rejected':True,'error':str(e),'scope':scope});return
    raise AssertionError('Gate accepted invalid condition: '+name)
for c in d.ORDER:
    rejected('actual_missing_Creator_'+c,lambda c=c:d.creator_gate(c,config,manifest,training,codec,helper,plan['camera_archives']))
    rejected('actual_missing_formalpost_'+c,lambda c=c:d.post_gate(c,config,manifest,training,codec,helper,plan['camera_archives']))
for key,value in [('training_lineage_binding_sha256','0'*64),('training_lineage_kind','four_legacy_formal_training'),('input_manifest_sha256','0'*64)]:
    b=dict(config);b[key]=value
    rejected('invalid_'+key,lambda b=b:d.lineage_gate(b,manifest,training,codec))
actual_guard=d.no_conflicting_gpu_stage(); actual_birth=d.process_birth(__import__('os').getpid())
assert actual_guard['actual_conflicting_gpu_processes']==[]
# Toy data are never installed in the real active lock, passed to an executable,
# or represented as actual process evidence. They exercise only the guard's CPU logic.
parser=d.load_module('_CPU002_process_parser',d.PROCESS_HELPER).windows_argv
wrapper=[str(d.TRAINING_PYTHON),'-u',str(d.ENV_LAUNCHER),str(source),'--stage','creator','--run',d.FIXED_RUN,
         '--input','small-city-training-002','--coarse-run','small-city-coarse-i30000-r4-003']
for c in d.ORDER:wrapper+=['--chunk-run',c+'='+config['chunk_runs'][c]]
wrapper+=['--chunk','0_0','--execute']
sup=[str(d.TRAINING_PYTHON),str(d.SUPERVISOR_SOURCE),'--stage','creator','--chunk','0_0','--supervision-run','CPU-toy-only','--execute']
def row(pid,parent,argv,name='python.exe'):
    return {'ProcessId':pid,'ParentProcessId':parent,'Name':name,'CreationDate':'CPU-fixture-not-actual',
            'CommandLine':subprocess.list2cmdline(argv)}
def birth(pid):return {'pid':pid,'creation_FILETIME_100ns':116444736000000000+pid*10000,'creation_Unix_ms':pid}
rows=[row(90,1,sup),row(100,90,sup),row(200,100,wrapper),row(201,200,wrapper),
      row(300,201,[wrapper[0],*wrapper[3:]]),row(301,300,[wrapper[0],*wrapper[3:]])]
lock={'supervisor_pid':100,'supervisor_birth':{**birth(100),'identity_source':'CPU-toy-only'},
      'supervisor_command_argv':sup,'supervisor_source_sha256':d.digest(d.SUPERVISOR_SOURCE),
      'wrapper_command_argv':wrapper,'stage':'creator','chunk':'0_0',
      'actual_child_birth':{**birth(200),'identity_source':'CPU-toy-only'}}
original_run=d.subprocess.run; original_pid=d.os.getpid; original_birth=d.process_birth;original_active=d.ACTIVE_STAGE
fixture=OUT/'ownership-fixture.json'
toy_controls=[]
def toy(variant_rows,variant_lock,current=301):
    if variant_lock is None:
        if fixture.exists():fixture.unlink()
    else:fixture.write_text(json.dumps(variant_lock),encoding='utf-8')
    d.ACTIVE_STAGE=fixture; d.os.getpid=lambda:current;d.process_birth=lambda pid:birth(pid)
    d.subprocess.run=lambda *a,**kw:SimpleNamespace(stdout=json.dumps(variant_rows))
    try:return d.no_conflicting_gpu_stage()
    finally:
        d.ACTIVE_STAGE=original_active;d.os.getpid=original_pid;d.process_birth=original_birth;d.subprocess.run=original_run
positive_fixture=toy(rows,lock)
assert positive_fixture['allowed_current_supervised_ancestor_PIDs']==[90,100,200,201,300,301]
supervisor_before_child=dict(lock);supervisor_before_child.pop('actual_child_birth')
positive_supervisor_fixture=toy(rows[:2],supervisor_before_child,current=100)
assert positive_supervisor_fixture['allowed_current_supervised_ancestor_PIDs']==[90,100]
def toyreject(name,r,l,current=301):rejected(name,lambda:toy(r,l,current),'isolated_CPU_process_fixture_not_actual_live_evidence')
toyreject('own_execute_without_active_lock',rows,None)
bad=copy.deepcopy(lock);bad['supervisor_birth']['creation_FILETIME_100ns']+=1;toyreject('supervisor_birth_reused',rows,bad)
bad=copy.deepcopy(lock);bad['actual_child_birth']['creation_FILETIME_100ns']+=1;toyreject('Popen_child_birth_reused',rows,bad)
bad=copy.deepcopy(rows);bad[-2]['ParentProcessId']=999;toyreject('driver_ancestor_chain_broken',bad,lock)
bad=copy.deepcopy(rows);bad[-1]['CommandLine']=subprocess.list2cmdline([wrapper[0],*wrapper[3:],'--chunk','1_1']);toyreject('driver_exact_argv_changed',bad,lock)
toyreject('foreign_same_stage_wrapper',rows+[row(401,1,[wrapper[0],*wrapper[3:]])],lock)
toyreject('foreign006_supervisor',rows+[row(402,1,sup)],lock)
toyreject('live_native_Creator',rows+[row(403,1,['GaussianHierarchyCreator.exe'],'GaussianHierarchyCreator.exe')],lock)
toyreject('live_native_MergerUnoptimized',rows+[row(404,1,['GaussianHierarchyMergerUnoptimized.exe'],'GaussianHierarchyMergerUnoptimized.exe')],lock)
toyreject('live_original_trainpost',rows+[row(405,1,[wrapper[0],'train_post.py'])],lock)
toyreject('live_original_trainsingle',rows+[row(406,1,[wrapper[0],'train_single.py'])],lock)
toyreject('live_gpu_eval',rows+[row(407,1,[wrapper[0],'evaluate_local_city_hierarchy.py','--gpu'])],lock)
toyreject('foreign_new_post64_wrapper',rows+[row(408,1,[wrapper[0],str(d.POST64_SOURCE),'--stage','post','--execute'])],lock)
bad=copy.deepcopy(lock);bad['stage']='post64';toyreject('formal_wrapper_uses_post64_stage',rows,bad)
assert 'torch' not in sys.modules
after={p:sha(p) for p in before};assert after==before
report={'status':'passed_actual006_CPU_models_missing_stages_and_supervised_role_controls','UTC':datetime.now(timezone.utc).isoformat(),
        'source_before':before,'source_after':after,'all_five_actual_model_bindings':positive,'controls':controls,
        'actual_no_GPU_process_guard':actual_guard,'actual_CPU_process_birth':actual_birth,
        'ownership_fixture_scope':'isolated mocked CIM rows/PID births; no launcher, native or training execution',
        'own_wrapper_fixture_passed':positive_fixture,'own_supervisor_before_child_fixture_passed':positive_supervisor_fixture,
        'negative_control_count':len(controls),'Torch_imported':False,'CUDA_initialized':False,'GPU_started':False,
        'Creator_executed':False,'post15000_executed':False,'Merge_executed':False,'original_files_changed':False}
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(LOCK/'cpu-binding-verification.json').write_text(json.dumps({'status':report['status'],'verification_path':str(OUT/'verification.json'),
        'verification_sha256':sha(OUT/'verification.json'),'driver_sha256':sha(source),
        'configuration_sha256':sha(LOCK/'configuration.json'),'negative_control_count':len(controls),'GPU_started':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'verification.json'),'report_sha256':sha(OUT/'verification.json'),
                 'negative_controls':len(controls)}))
