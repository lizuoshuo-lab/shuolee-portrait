"""Version the immutable005 evaluation interfaces; CPU source generation only."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
DIR=Path(__file__).resolve().parent
OLD=OUT/'city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py'
POSTERIOR=OUT/'city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py'
BINDING_SHA='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)
def swap_function(text,name,replacement):
    tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n'+''.join(lines[node.end_lineno:])
def write(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

require(sha(OLD)=='c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610','Oldadapter changed')
require(sha(POSTERIOR)=='bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133','Oldposterior changed')
DEST=DIR/'evaluate_local_city_hierarchy_006.py'
require(not DEST.exists(),'Never overwrite generated adapter')
text=OLD.read_text(encoding='utf-8')
text=text.replace("ARCHIVES=DRIVER/'archives'", "ARCHIVES=e.OUTROOT/'city-local-hierarchy-evaluation-driver-CPU-001/archives'")
text=text.replace("small-city-hierarchy-005", "small-city-hierarchy-006").replace('workflow-lock005','workflow-lock006').replace('source_lock005','source_lock006').replace('workflow_lock005','workflow_lock006')
text=text.replace("CONFIG_SHA='55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'", "CONFIG_SHA=None\nNATIVE_WORKFLOW_SHA=None\nNATIVE_LOCK=DRIVER/'native-workflow-lock.json'\nBINDING_PATH=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'\nBINDING_SHA='"+BINDING_SHA+"'\nBINDER_SOURCE_SHA='c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e'")
text=text.replace("WORKFLOW_SHA='0e51", "WORKFLOW_SHA='0e51") # retained pure CPU helper v4, never its training gate
text=text.replace("Notauthorpretrained18M reproduction or equal-budget coarse/dense causal comparison.", "Mixed origins: three legacy003 chunks plus observed11 with unchanged official algorithm and read-only hooks; observed11 remains not formal and its extra synchronization/tracing changes timing/memory. Notauthorpretrained18M reproduction or equal-budget coarse/dense causal comparison.")
LOCK_FUNCTION='''def lineage_binding():
    e.require(BINDING_PATH.is_file() and e.digest(BINDING_PATH)==BINDING_SHA,'Actual completed mixed-lineage binding missing/changed')
    binding=e.read_json(BINDING_PATH)
    e.require(binding['status']=='actual_complete_mixed_local_training_lineage_CPU_bound' and binding['binder_source_sha256']==BINDER_SOURCE_SHA and binding['formal_observation_reclassification'] is False,'Wrong completed lineage origin')
    e.require(binding['Creator_post_Merger_or_evaluation_complete'] is False and [r['chunk'] for r in binding['ordered_chunks']]==['0_0','0_1','1_0','1_1'],'Training-only receipt/order scope differs')
    accounting=binding['observed_terminal']
    e.require(accounting['formal_training_run'] is False and accounting['actual_exposure_Adam_calls']==accounting['actual_opacity_grad_condition_visits']==29999 and accounting['actual_Gaussian_Adam_calls']+accounting['actual_None_gradient_skips']==29999 and 30000 in accounting['successful_original_save_iterations'],'Observed actual budget/call/skip accounting differs')
    return binding

def native_configuration():
    path=LOCK/'configuration.json';config=e.read_json(path);plan=e.read_json(LOCK/'plan.json')
    digest=e.digest(path);driver=e.digest(LOCK/'driver-source.py')
    e.require(config['driver_version']>=5 and config['driver_sha256']==driver and plan['configuration_sha256']==digest,'Wrong/changed actual006 driver/config/plan')
    e.require(config['training_lineage_binding_sha256']==BINDING_SHA and Path(config['training_lineage_binding_path']).resolve()==BINDING_PATH.resolve(),'Native config uses another training family')
    e.require(config['input_manifest_sha256']==INPUT_SHA and config['input']=='small-city-training-002' and config['coarse_run']==COARSE_RUN and not config['author_pretrained_model_used'] and config['resolution']==4 and config['pixels']==[256,172],'Actual006 input/coarse/r4 lineage differs')
    binding=lineage_binding();expected={r['chunk']:Path(r['training_directory']).name for r in binding['ordered_chunks']}
    e.require(config['chunk_runs']==expected and list(config['chunk_runs'])==['0_0','0_1','1_0','1_1'],'006 chunk runs/origin order differs')
    e.require(config['post_iterations']==15000 and config['expected_post_optimizer_calls']==14999 and config['post_full_train_defaults']['dimensionless_log_uniform_limit']==[.005,.1],'Official post budget/default range differs')
    e.require(config['official_main_commit']==e.COMMIT and config['official_hierarchy_commit']=='677c8553dc64dfd62c272eca94a291a277733113','Wrong official commits')
    for row in config['source_records']:e.require(e.digest(row['path'])==row['sha256'],'Locked actual006 source changed')
    for row in config['frozen_helper_records']:e.require(e.digest(row['snapshot_path'])==row['sha256'],'Locked actual006 helper changed')
    return config,digest,driver

def freeze_workflow_cpu(output):
    config,digest,driver=native_configuration()
    e.require(not NATIVE_LOCK.exists(),'006 native evaluation lock already exists; preserve it')
    lock=dict(status='actual006_native_source_config_and_completed_training_lineage_CPU_locked_not_trees',adapter_source_sha256=e.digest(__file__),configuration_sha256=digest,driver_source_sha256=driver,configuration_path=str(LOCK/'configuration.json'),driver_source_path=str(LOCK/'driver-source.py'),lineage_binding_path=str(BINDING_PATH),lineage_binding_sha256=BINDING_SHA,CUDA_called=False)
    e.save(NATIVE_LOCK,lock);e.save(output/'006-native-workflow-lock.json',lock)
    return lock

def lock_schema(workflow,output):
    global CONFIG_SHA,NATIVE_WORKFLOW_SHA
    e.require(NATIVE_LOCK.is_file(),'Actual006 native source/config evaluation lock missing; CPU preparation is not real tree completion')
    seal=e.read_json(NATIVE_LOCK)
    e.require(seal['adapter_source_sha256']==e.digest(__file__) and seal['lineage_binding_sha256']==BINDING_SHA,'Wrong immutable006 evaluation lock')
    config,digest,driver=native_configuration()
    e.require(seal['configuration_sha256']==digest and seal['driver_source_sha256']==driver,'Actual006 configuration/source changed after evaluation lock')
    CONFIG_SHA,NATIVE_WORKFLOW_SHA=digest,driver
    inputs=e.TRACK/'official-inputs'/config['input'];manifest=e.read_json(inputs/'manifest.json')
    e.require(e.digest(inputs/'manifest.json')==INPUT_SHA,'Input manifest changed')
    split_audit=workflow.input_split_audit(inputs,manifest)
    e.require(split_audit==config['input_split_audit'],'Actual sparse/0/test Reader partition changed')
    for filename in ('configuration.json','plan.json','driver-source.py'):
        shutil.copyfile(LOCK/filename,output/('workflow-lock006-'+filename))
    shutil.copyfile(NATIVE_LOCK,output/'006-native-workflow-lock.json')
    shutil.copyfile(BINDING_PATH,output/'006-completed-training-lineage-binding.json')
    return config,manifest'''
text=swap_function(text,'lock_schema',LOCK_FUNCTION)
TRAIN_FUNCTION='''def local_training_gate(workflow,config,manifest,output):
    binding=lineage_binding();training=[]
    for source in [binding['coarse']]+binding['ordered_chunks']:
        stage,chunk=source['stage'],source['chunk'];directory=Path(source['training_directory'])
        receipt=e.read_json(directory/'outcome.json');configuration=e.read_json(directory/'configuration.json')
        observed=chunk=='1_1'
        expected_directory=e.TRACK/('official-training-observation' if observed else 'official-training')/directory.name
        e.require(directory.resolve()==expected_directory.resolve(),'Training source directory kind differs')
        e.require(e.digest(directory/'outcome.json')==source['training_receipt']['sha256'] and e.digest(directory/'configuration.json')==source['configuration']['sha256'] and e.digest(directory/'model-audit.json')==source['model_audit_sha256'],'Actual training receipt/config/audit differs from completed binding')
        e.require(receipt['status']==('completed_observation_only_final_finite_not_formal' if observed else 'completed') and receipt['exit_code']==0 and receipt['requested_iterations']==30000,'Incomplete actual mixed-origin local training')
        e.require(source['training_source_kind']==('observed_local_official_training' if observed else 'legacy_local_official_training'),'Training origin sourcekind differs')
        if observed:e.require(source['formal_training_run'] is False and receipt['formal_completed_model_claimed'] is False,'Observed origin cannot be formal-reclassified')
        else:e.require(source['actual_Gaussian_Adam_calls'] is None,'Legacy derived count relabeled as actual')
        parameter=Path(source['model']['path']);e.require(e.digest(parameter)==source['model']['sha256']==receipt['model_parameter_sha256'],'Actual local modelparameter SHA mismatch')
        expected_names=manifest['train_names'] if stage=='coarse' else next(c['train_names'] for c in manifest['chunks'] if c['name']==chunk)
        e.require(configuration['training_names']==expected_names and not set(expected_names)&set(manifest['test_names']),'Actual mixed training image family changed')
        exp=workflow.exposure_audit(directory/'model/exposure.json',expected_names,manifest['test_names'])
        e.require(exp['sha256']==source['exposure']['sha256'] and exp['test_overlap']==0,'Actual trained exposure differs from completed lineage')
        training.append(dict(stage=stage,chunk=chunk,run=directory.name,training_directory=str(directory),training_source_kind=source['training_source_kind'],formal_training_run=source['formal_training_run'],actual_Gaussian_Adam_calls=source['actual_Gaussian_Adam_calls'],derived_Gaussian_call_upper_bound=source['derived_Gaussian_call_upper_bound'],outcome_sha256=e.digest(directory/'outcome.json'),configuration_sha256=e.digest(directory/'configuration.json'),model_path=str(parameter),model_sha256=e.digest(parameter),saved_model_directory=receipt['saved_model_directory'],exposure=exp,source_receipt=receipt))
    result=dict(status='allfive_actual_completed_mixed_local_source_split_model_exposure_receipts_passed',records=training,training_lineage_binding_path=str(BINDING_PATH),training_lineage_binding_sha256=BINDING_SHA,formal_observation_reclassification=False,observed_timing_memory_limit=binding['observed_terminal']['observation_overhead'],CUDA_called=False)
    e.save(output/'actual-local-training-lineage-CPU.json',result)
    return training'''
text=swap_function(text,'local_training_gate',TRAIN_FUNCTION)
text=text.replace("actual_training=next(r for r in lineage if r['chunk']==chunk)", "actual_training=next(r for r in lineage if r['chunk']==chunk)\n        for receipt in (creator,post):e.require(receipt['training_lineage_binding_sha256']==BINDING_SHA,'Native stage training lineage changed')")
text=text.replace("exp['sha256']==creator['copied_exposure_sha256']==post['frozen_exposure']['sha256']", "exp['sha256']==actual_training['exposure']['sha256']==creator['copied_exposure_sha256']==post['frozen_exposure']['sha256']")
text=text.replace("receipt['status']=='completed' and receipt['exit_code']==0 and receipt['source_kind']==kind and receipt['configuration_sha256']==CONFIG_SHA,'Merge incomplete/changed scope'", "receipt['status']=='completed' and receipt['exit_code']==0 and receipt['source_kind']==kind and receipt['configuration_sha256']==CONFIG_SHA and receipt['training_lineage_binding_sha256']==BINDING_SHA,'Merge incomplete/changed scope'")
text=text.replace("configuration_sha256=CONFIG_SHA,lineage=lineage", "configuration_sha256=CONFIG_SHA,training_lineage_binding_sha256=BINDING_SHA,formal_observation_reclassification=False,lineage=lineage")
text=text.replace("hierarchy_workflow_driver_sha256=WORKFLOW_SHA", "hierarchy_workflow_driver_sha256=NATIVE_WORKFLOW_SHA,training_lineage_binding_path=str(BINDING_PATH),training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),formal_observation_reclassification=False")
text=text.replace("mode.add_argument('--gpu',action='store_true');a=p.parse_args()", "mode.add_argument('--gpu',action='store_true');mode.add_argument('--freeze-workflow-cpu',action='store_true');a=p.parse_args()")
text=text.replace("        e.require({k:e.importlib.metadata.version(k) for k in e.FIXED}==e.FIXED", "        if a.freeze_workflow_cpu:\n            freeze_workflow_cpu(output);outcome.update(status='actual006_native_source_and_lineage_CPU_locked_not_trees',driver_sha256=e.digest(__file__),core_evaluator_sha256=e.digest(e.__file__),cuda_used=False);e.save(output/'outcome.json',outcome);print(json.dumps(outcome),flush=True);return\n        e.require({k:e.importlib.metadata.version(k) for k in e.FIXED}==e.FIXED")
text=text.replace("source_lock006_configuration_SHA=CONFIG_SHA,actual_owncoarse_SHA", "source_lock006_configuration_SHA=CONFIG_SHA,training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),formal_observation_reclassification=False,actual_owncoarse_SHA")
text=text.replace("outcome.update(ended_utc=e.baseline.now()", "outcome.update(training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=e.digest(NATIVE_LOCK),hierarchy_configuration_sha256=CONFIG_SHA,hierarchy_workflow_driver_sha256=NATIVE_WORKFLOW_SHA,formal_observation_reclassification=False,ended_utc=e.baseline.now()")
ast.parse(text);DEST.write_text(text,encoding='utf-8')
retained=DIR/'retained-005-sources';retained.mkdir(exist_ok=False)
shutil.copyfile(OLD,retained/'evaluate_local_city_hierarchy_c3.py')
shutil.copyfile(POSTERIOR,retained/'audit_city_1404_bef.py')
write(DIR/'generation-manifest.json',{'status':'CPU_source_versioning_prepared_not_actual_trees_or_GPU','old_adapter_sha256':sha(OLD),'old_posterior_sha256':sha(POSTERIOR),'new_adapter_sha256':sha(DEST),'new_adapter_path':str(DEST),'binding_sha256':BINDING_SHA,'core_sha256':'672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec','settings':['leaves','tau1','tau3','tau5','tau6','tau15'],'views':117,'future_frames':1404,'new_native_source_not_yet_bound':True,'CUDA_called':False})
print(json.dumps({'adapter':str(DEST),'sha256':sha(DEST),'status':'CPU_source_versioned_native_source_and_trees_pending'}))
