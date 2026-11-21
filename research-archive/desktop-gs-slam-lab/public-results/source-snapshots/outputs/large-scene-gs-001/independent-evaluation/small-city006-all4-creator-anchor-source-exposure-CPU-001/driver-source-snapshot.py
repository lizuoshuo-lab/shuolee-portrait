"""Read-only serial all-four actual Creator ledger, not formal post or metrics."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
ROOT=D.parents[3]
BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'independent-evaluation'
LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
SOURCE=D/'audit_city_hierarchy_lineage_006.py'
SOURCE_SHA='b275c765ce7aec4e96e49dabbc575761939105f2da237a2d9e51007085c27cb9'
RUN='small-city006-all4-creator-anchor-source-exposure-CPU-001'
OUTPUT=OUT/RUN
CHUNKS=['0_0','0_1','1_0','1_1']
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(ok,msg):
    if not ok:raise RuntimeError(msg)
OUTPUT.mkdir(exist_ok=False)
shutil.copyfile(__file__,OUTPUT/'driver-source-snapshot.py')
shutil.copyfile(SOURCE,OUTPUT/'ledger-source-snapshot.py')
start=time.perf_counter();records=[];evidence=[]
outcome={'status':'running_actual_all4_Creator_CPU_ledger','CUDA_called':False,'Torch_imported':False,'formal_post_completed':False,'actual_1404_complete':False}
save(OUTPUT/'outcome.json',outcome)
try:
    require(sha(SOURCE)==SOURCE_SHA,'Creator source changed before launch')
    for c in CHUNKS:
        r=read(LOCK/'trained_chunks'/c/'creator-outcome.json')
        require(r['status']=='completed' and r['exit_code']==r['actual_child_poll_exit_code']==0,'Actual Creator not terminal-completed: '+c)
        require(r['configuration_sha256']=='562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47' and r['training_lineage_binding_sha256']=='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c','Actual006 origin differs: '+c)
    for c in CHUNKS:
        childrun=RUN+'-'+c.replace('_','')
        argv=[sys.executable,'-B',str(SOURCE),'--mode','creator','--chunk',c,'--run',childrun]
        print('CPU actual Creator audit starting '+c,flush=True)
        with (OUTPUT/(c+'-process.log')).open('x',encoding='utf-8') as log:
            child=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
        evidence.append({'chunk':c,'command_argv':argv,'actual_child_exit_code':child.returncode,'process_log_path':str(OUTPUT/(c+'-process.log'))})
        require(child.returncode==0,'Actual Creator CPU child failed: '+c)
        childdir=OUT/childrun;report=read(childdir/'report.json');terminal=read(childdir/'outcome.json')
        require(report['status']=='completed_actual006_Creator_anchor_source_exposure_CPU_measurement_only' and terminal['status']==report['status'],'Actual Creator CPU completion status differs')
        require(sha(childdir/'report.json')==terminal['report_sha256'] and report['source_sha256']==SOURCE_SHA,'Actual report or source provenance differs')
        require(len(report['records'])==1 and report['records'][0]['chunk']==c,'Actual chunk record differs')
        record=report['records'][0]
        require(record['Creator_completed_only'] and not record['formal_post_completed'] and not report['actual_merger_or1404_complete'],'Creator scope incorrectly completed later stages')
        sets=record['actual_anchor_classification']
        require(sha(sets['actual_set_npz_path'])==sets['actual_set_npz_sha256'],'Actual anchor-set file changed')
        evidence[-1].update(report_path=str(childdir/'report.json'),report_sha256=sha(childdir/'report.json'),outcome_path=str(childdir/'outcome.json'),outcome_sha256=sha(childdir/'outcome.json'),anchor_sets_path=sets['actual_set_npz_path'],anchor_sets_sha256=sets['actual_set_npz_sha256'])
        records.append(record)
        print(json.dumps({'chunk':c,'A_intersection_L':sets['A_intersection_L']['count'],'L_minus_A':sets['L_minus_A']['count'],'A_minus_L':sets['A_minus_L']['count'],'CPU_completed':True}),flush=True)
        save(OUTPUT/'progress.json',{'completed_chunks':[r['chunk'] for r in records],'evidence':evidence,'formal_post_completed':False,'actual_1404_complete':False})
    require(sha(SOURCE)==SOURCE_SHA,'Creator ledger source changed during execution')
    require(records[-1]['training_source_kind']=='observed_local_official_training' and records[-1]['actual_Gaussian_Adam_calls']==29951 and not records[-1]['formal_training_run'],'Observed11 lineage was reclassified')
    require(all(r['actual_Gaussian_Adam_calls'] is None for r in records[:3]),'Legacy derived upper bound became measured calls')
    report={'status':'completed_actual_all_four006_Creator_anchor_source_exposure_CPU_ledger_only','records':records,'chunk_order':CHUNKS,'child_evidence':evidence,'ledger_source_path':str(SOURCE),'ledger_source_sha256':SOURCE_SHA,'driver_source_sha256':sha(__file__),'configuration_sha256':'562c24c7bf56d5debfd32b62fc523bce7133dc1d322a29eb76e55508a5440e47','training_lineage_binding_sha256':'1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c','formal_observation_reclassification':False,'formal_post_completed':False,'formal_post_future_only':True,'actual_merger_or_1404_complete':False,'no_all_leaf_freeze_assumption':True,'CUDA_called':False,'Torch_imported':'torch' in sys.modules,'units':'Published SfM aligned units, not GT and not metres','elapsed_CPU_ledger_wall_seconds':time.perf_counter()-start}
    require(not report['Torch_imported'],'CPU controller imported Torch')
    save(OUTPUT/'report.json',report)
    outcome.update(status=report['status'],report_sha256=sha(OUTPUT/'report.json'),elapsed_seconds=time.perf_counter()-start,actual_child_exit_codes=[e['actual_child_exit_code'] for e in evidence]);save(OUTPUT/'outcome.json',outcome)
    print(json.dumps({'status':report['status'],'report_path':str(OUTPUT/'report.json'),'report_sha256':sha(OUTPUT/'report.json')}),flush=True)
except Exception as exc:
    outcome.update(status='failed_preserved_actual_all4_Creator_CPU_ledger',error=str(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start,completed_chunks=[r['chunk'] for r in records],child_evidence=evidence);save(OUTPUT/'outcome.json',outcome);raise
