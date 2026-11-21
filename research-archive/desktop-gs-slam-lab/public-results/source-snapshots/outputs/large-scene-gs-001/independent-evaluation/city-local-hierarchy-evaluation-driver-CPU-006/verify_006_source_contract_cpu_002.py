"""Compare versioned source contracts without imports, models or tensors."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'outputs/large-scene-gs-001/independent-evaluation'
DIR=Path(__file__).resolve().parent
OLD=OUT/'city-local-hierarchy-evaluation-driver-CPU-001/evaluate_local_city_hierarchy.py'
NEW=DIR/'evaluate_local_city_hierarchy_006.py'
OLDPOST=OUT/'city-hierarchy-posterior-auditor-CPU-001/audit_city_1404.py'
NEWPOST=OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py'
REPORT=DIR/'CPU-source-contract-review-002.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(c,m):
    if not c:raise ValueError(m)
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8'))
def functions(p):return {n.name:n for n in tree(p).body if isinstance(n,ast.FunctionDef)}
def dump(n):return ast.dump(n,include_attributes=False)
def calls(p,chosen):
    result={name:[] for name in chosen}
    for n in ast.walk(tree(p)):
        if isinstance(n,ast.Call):
            name=ast.unparse(n.func)
            if name in result:result[name].append(dump(n))
    return result
def literal(p,name):
    for n in tree(p).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id==name:return ast.literal_eval(n.value)
    raise ValueError(name)

require(not REPORT.exists(),'Review artifact exists; preserve it')
require(sha(OLD)=='c3d9c5c7547cee632f79ba0e66b6d24bbcd35771295b7f4fd23fc5b93ad40610','Oldadapter changed')
require(sha(OLDPOST)=='bef0c57ed5529c04ab4d2d837d5efbd15e76938787dab72b24527375fe9cd133','Oldposterior changed')
require(sha(ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py')=='672533b6aef0f79d64a77976055ce747d05c3f226646e576cb913b0c405af1ec','Core changed')
chosen=['e.gpu_model','e.metrics','coarse.masked_numpy','e.threshold_scope','e.model_cpu','e.memory_plan','e.tau_to_threshold','topology.reference_counts','np.mean','np.stack','np.array_equal','np.max']
a,b=calls(OLD,chosen),calls(NEW,chosen)
equal={name:a[name]==b[name] for name in chosen}
require(all(equal.values()),'Renderer/input/mask/metric/count/timing call expression changed')
f,g=functions(OLDPOST),functions(NEWPOST)
names=['numpy_mask_metrics','metric_row','canonical_inputs','topology_only','gaussian_storage_finite','selection_audit','timing_audit','coarse_contract_evidence']
same={name:dump(f[name])==dump(g[name]) for name in names}
require(all(same.values()),'Posterior numeric/storage/coverage/timing/terminal function changed')
require(literal(OLD,'SETTINGS')==literal(NEW,'SETTINGS') and literal(OLDPOST,'SETTINGS')==literal(NEWPOST,'SETTINGS'),'LOD grid changed')
require(literal(NEWPOST,'ADAPTER_SHA')==sha(NEW),'Posterior binds another adapter')
binding=ROOT/'outputs/large-scene-gs-001/official-hierarchy/small-city-hierarchy-006/binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
require(sha(binding)=='1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c','Actual mixed binding changed')
receipt=json.loads(binding.read_text(encoding='utf-8'))
require(receipt['status']=='actual_complete_mixed_local_training_lineage_CPU_bound' and receipt['formal_observation_reclassification'] is False,'Actual binding status changed')
require([r['training_source_kind'] for r in receipt['ordered_chunks']]==['legacy_local_official_training']*3+['observed_local_official_training'],'Actual lineage origins changed')
failed_adapter=OUT/'small-city006-actual-missing-tree-CPU-003/outcome.json'
failed_posterior=OUT/'small-city006-posterior-actual-missing1404-CPU-003/outcome.json'
fa=json.loads(failed_adapter.read_text(encoding='utf-8'));fp=json.loads(failed_posterior.read_text(encoding='utf-8'))
require(fa['status']=='failed_preserved_own_local_city_evaluation' and fa['cuda_used'] is False and 'merged trees and receipts missing' in fa['error'],'Missingtree control did not reject')
require(fp['status']=='failed_preserved_CPU_posterior_no_completion' and fp['actual1404_completed'] is False and fp['Torch_imported'] is False,'Missing1404 control did not reject')
require(not (failed_posterior.parent/'completion-audit.json').exists(),'False posterior completion artifact')
result={'status':'CPU_source_contract_and_actual_missing_models_outputs_controls_passed_pending_root_peer_review',
    'old_adapter_sha256':sha(OLD),'new_adapter_sha256':sha(NEW),'old_posterior_sha256':sha(OLDPOST),'new_posterior_sha256':sha(NEWPOST),
    'core_sha256':sha(ROOT/'scripts/large_scene_gs/evaluate_official_hierarchy.py'),'renderer_mask_metric_K_memory_call_AST_exact':equal,
    'posterior_numeric_storage_coverage_timing_function_AST_exact':same,'settings_exact':True,'future_expected_frames':1404,
    'actual_completed_training_binding_sha256':sha(binding),'actual_native_source_lock_and_trees_not_bound':True,
    'actual_missing_tree_outcome':{'path':str(failed_adapter),'sha256':sha(failed_adapter),'status':fa['status']},
    'actual_missing1404_outcome':{'path':str(failed_posterior),'sha256':sha(failed_posterior),'status':fp['status']},
    'actual1404_completed':False,'native_or_renderer_executed':False,'Torch_imported':False,'CUDA_called':False,
    'review_scope':'Executable AST/source identity comparison and real refusal paths. This self-audit is not an independent peer review, actual Native completion, or any rendered result.'}
REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
snap=DIR/'source-lock-PS7-v2';snap.mkdir(exist_ok=False)
shutil.copyfile(NEW,snap/NEW.name);shutil.copyfile(NEWPOST,snap/NEWPOST.name)
print(json.dumps({'status':result['status'],'report':str(REPORT),'report_sha256':sha(REPORT),'adapter_sha256':sha(NEW),'posterior_sha256':sha(NEWPOST)}))
