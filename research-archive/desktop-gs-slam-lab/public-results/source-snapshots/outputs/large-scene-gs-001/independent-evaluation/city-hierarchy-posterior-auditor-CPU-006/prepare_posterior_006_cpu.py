"""CPU-only versioning of the original immutable posterior auditor."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
DIR=Path(__file__).resolve().parent
OLD=OUT/'city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py'
ADAPTER=OUT/'city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py'
DEST=DIR/'audit_city_1404_006.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)
require(sha(OLD)=='bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133','Oldposterior changed')
require(not DEST.exists(),'Never overwrite generated posterior')
text=OLD.read_text(encoding='utf-8')
text=text.replace('small-city-hierarchy-005','small-city-hierarchy-006')
text=text.replace('city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py','city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py')
text=text.replace("ADAPTER_SHA='c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610'", "ADAPTER_SHA='"+sha(ADAPTER)+"'")
text=text.replace("CONFIG_SHA='55c7dd48fb06b07340fd0a49e611fd6d4334dc98253bcc45aa7ac2261b6fdcea'", "CONFIG_SHA=None\nNATIVE_LOCK=ADAPTER.parent/'native-workflow-lock.json'\nBINDING_PATH=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'\nBINDING_SHA='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'\nBINDER_SHA='c72fea79bb3b083ead1dfa58e9cf7e7d0134842d764dd3c5482162448dc4db9e'")
HELPER='''def mixed_lineage006(source):
    global CONFIG_SHA
    exact_file(BINDING_PATH,BINDING_SHA);binding=read(BINDING_PATH)
    require(binding['status']=='actual_complete_mixed_local_training_lineage_CPU_bound' and binding['formal_observation_reclassification'] is False and binding['binder_source_sha256']==BINDER_SHA,'Wrong actual mixed-training binding')
    exact_file(source/'006-completed-training-lineage-binding.json',BINDING_SHA)
    seal=read(NATIVE_LOCK);CONFIG_SHA=seal['configuration_sha256']
    require(seal['adapter_source_sha256']==ADAPTER_SHA and seal['lineage_binding_sha256']==BINDING_SHA,'006 native source seal belongs to another adapter/lineage')
    exact_file(source/'006-native-workflow-lock.json',digest(NATIVE_LOCK))
    exact_file(LOCK/'configuration.json',CONFIG_SHA);exact_file(LOCK/'driver-source.py',seal['driver_source_sha256'])
    config=read(LOCK/'configuration.json');plan=read(LOCK/'plan.json')
    require(config['driver_version']>=5 and config['driver_sha256']==seal['driver_source_sha256'] and config['training_lineage_binding_sha256']==BINDING_SHA and Path(config['training_lineage_binding_path']).resolve()==BINDING_PATH.resolve() and plan['configuration_sha256']==CONFIG_SHA,'006 actual Native configuration/source/lineage changed')
    observed=binding['observed_terminal'];require(observed['formal_training_run'] is False and observed['actual_exposure_Adam_calls']==observed['actual_opacity_grad_condition_visits']==29999 and observed['actual_Gaussian_Adam_calls']+observed['actual_None_gradient_skips']==29999 and 30000 in observed['successful_original_save_iterations'],'Actual observed full-budget/source-call semantics differ')
    return binding,seal

'''
text=text.replace('def full_audit(source,output,evaluation_run):',HELPER+'def full_audit(source,output,evaluation_run):')
text=text.replace("    exact_file(ADAPTER,ADAPTER_SHA);exact_file(ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py',CORE_SHA);exact_file(LOCK/'configuration.json',CONFIG_SHA)", "    binding,seal=mixed_lineage006(source)\n    exact_file(ADAPTER,ADAPTER_SHA);exact_file(ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py',CORE_SHA);exact_file(LOCK/'configuration.json',CONFIG_SHA)")
text=text.replace("    ownpath=Path(local['independent_input_npz_path'])", "    require(local['training_lineage_binding_sha256']==BINDING_SHA and local['formal_observation_reclassification'] is False and local['native_workflow_lock_sha256']==digest(NATIVE_LOCK) and local['hierarchy_workflow_driver_sha256']==seal['driver_source_sha256'],'Actual006 evaluation source/lineage metadata differs')\n    require(outcome['training_lineage_binding_sha256']==overall['training_lineage_binding_sha256']==BINDING_SHA and outcome['formal_observation_reclassification'] is False and overall['formal_observation_reclassification'] is False and outcome['native_workflow_lock_sha256']==overall['native_workflow_lock_sha256']==digest(NATIVE_LOCK),'006 outcome/metrics source linkage differs')\n    ownpath=Path(local['independent_input_npz_path'])")
text=text.replace("actual['configuration_sha256']==CONFIG_SHA,'Actual workflow CPU binding missing/failed'", "actual['configuration_sha256']==CONFIG_SHA and actual['training_lineage_binding_sha256']==BINDING_SHA and actual['formal_observation_reclassification'] is False,'Actual workflow CPU binding missing/failed'")
old="""        directory=BASE/'official-training'/r['run'];exact_file(directory/'outcome.json',r['outcome_sha256']);exact_file(directory/'configuration.json',r['configuration_sha256']);exact_file(r['model_path'],r['model_sha256']);require(read(directory/'outcome.json')['status']=='completed','Real training incomplete')
        exact_file(r['exposure']['path'],r['exposure']['sha256']);require(r['exposure']['test_overlap']==0,'Actual test exposure fitted')"""
new="""        bound=next(x for x in [binding['coarse']]+binding['ordered_chunks'] if (x['stage'],x['chunk'])==(r['stage'],r['chunk']))
        directory=Path(r['training_directory']);observed=r['chunk']=='1_1'
        require(directory.resolve()==Path(bound['training_directory']).resolve() and directory.name==r['run'] and r['training_source_kind']==bound['training_source_kind'],'Actual training directory/origin kind changed')
        require(r['outcome_sha256']==bound['training_receipt']['sha256'] and r['configuration_sha256']==bound['configuration']['sha256'] and r['model_sha256']==bound['model']['sha256'] and r['actual_Gaussian_Adam_calls']==bound['actual_Gaussian_Adam_calls'],'Actual completed binding lineage no longer matches evaluation')
        exact_file(directory/'outcome.json',r['outcome_sha256']);exact_file(directory/'configuration.json',r['configuration_sha256']);exact_file(r['model_path'],r['model_sha256'])
        train=read(directory/'outcome.json');require(train['status']==('completed_observation_only_final_finite_not_formal' if observed else 'completed') and train['exit_code']==0 and train['requested_iterations']==30000,'Real mixed-origin training incomplete')
        if observed:require(r['formal_training_run'] is False and train['formal_completed_model_claimed'] is False,'Observed source cannot be formal-reclassified')
        else:require(r['actual_Gaussian_Adam_calls'] is None,'Legacy29999 budget upper cannot be claimed actual')
        exact_file(r['exposure']['path'],r['exposure']['sha256']);require(r['exposure']['sha256']==bound['exposure']['sha256'] and r['exposure']['test_overlap']==0,'Actual exposure source/test fitting changed')"""
require(old in text,'Posterior source lineage block changed');text=text.replace(old,new)
text=text.replace("receipt['configuration_sha256']==CONFIG_SHA,'ActualMerger source family invalid'", "receipt['configuration_sha256']==CONFIG_SHA and receipt['training_lineage_binding_sha256']==BINDING_SHA,'ActualMerger source family invalid'")
text=text.replace("actual_owncoarse_SHA=COARSE_SHA,\n        summaries=", "actual_owncoarse_SHA=COARSE_SHA,training_lineage_binding_path=str(BINDING_PATH),training_lineage_binding_sha256=BINDING_SHA,native_workflow_lock_sha256=digest(NATIVE_LOCK),formal_observation_reclassification=False,training_lineage_kind=binding['training_lineage_kind'],observed_training_timing_memory_limit=binding['time_memory_comparison_limit'],\n        summaries=")
ast.parse(text);DEST.write_text(text,encoding='utf-8')
shutil.copyfile(OLD,DIR/'original-bef-source.py')
(DIR/'generation-manifest.json').write_text(json.dumps({'status':'CPU_posterior006_source_prepared_not_actual1404','old_source_sha256':sha(OLD),'new_source_sha256':sha(DEST),'adapter_source_sha256':sha(ADAPTER),'binding_sha256':'1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c','views':117,'settings':12,'future_frames':1404,'Torch_imported':False,'CUDA_called':False},indent=2)+'\n',encoding='utf-8')
print(json.dumps({'posterior':str(DEST),'sha256':sha(DEST),'adapter_sha256':sha(ADAPTER),'status':'prepared_actual1404_missing'}))
