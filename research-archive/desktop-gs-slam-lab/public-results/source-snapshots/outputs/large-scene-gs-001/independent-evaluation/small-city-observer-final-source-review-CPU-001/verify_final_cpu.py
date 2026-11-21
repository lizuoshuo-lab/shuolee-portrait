"""Final source/actualoracle review and small CPU snapshot gate controls.

NoTorch/CUDA/training/model creation; stubcapture fixtures only test the real
initialized() identity gate. Originalruntimecapture/fullfinite are not replaced.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,shutil,sys
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';OUT=Path(__file__).parent
OLD=BASE/'recovery-audit/small-city-observation-only11-peer-controls-CPU-002'
ORACLE=BASE/'official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json'
PLAN=BASE/'official-training-observation-plans/small-city-chunk11-observation-i30000-r4-001-v2'
EXPECT={
'observe_small_city_train_single_001.py':'0dfb6a869cf4751b76b3e1b263b95cc4fe0f7de3798612a312949ceaab848386',
'run_small_city_observation_001.py':'5f0a3a0073af0c44d6720c9adafa8b96a9f5563fa9c3cf1d946db7a6bc48cba7',
'supervise_small_city_observation_001.py':'d6375412b445c8923498d2130c45aee172afb0a0dbb3f58593b54ee711b9e6de',
'launch_small_city_observation_001.ps1':'69c39a195a7949b09cedb988d80aaf774ad126aceb200e209daf39ad69b63d3a'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def methods(p):
    tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReadOnlyObserver')
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
def main():
    require(not (OUT/'review.json').exists(),'Preserveactualfinalreview')
    sources={}
    for name,value in EXPECT.items():
        path=ROOT/'scripts/large_scene_gs'/name;require(sha(path)==value,'Finalsource differs:'+name);sources[str(path)]=value;shutil.copyfile(path,OUT/name)
    core=ROOT/'scripts/large_scene_gs/observe_small_city_train_single_001.py'
    require(sha(OLD/'observer-core-tested-snapshot.py')=='ef7ff83353ef90213159a36f6faffef3a6c39abe0fc6facc55847f0e098a91ff','Actualoldcontrol snapshot differs')
    require(sha(OLD/'CPU-verification.json')=='e78536a35a16036b458af8732783314de5a58bade85efd40f101f3debd3694c0','OldCPUmath/spawnproof changed')
    old,new=methods(OLD/'observer-core-tested-snapshot.py'),methods(core);require(set(old)==set(new),'Observermethod set changed')
    unchanged=[name for name in old if old[name]==new[name]];changed=[name for name in old if old[name]!=new[name]]
    require(changed==['initialized'],'Previously testedobserver methods unexpectedlychanged')
    oldtree=ast.parse((OLD/'observer-core-tested-snapshot.py').read_text());newtree=ast.parse(core.read_text())
    for name in ('tensor_record','nonfinite_paths','encoded'):
        a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        require(ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),'Oldsnapshot capture helper changed')
    require(sha(ORACLE)=='3fc724de94b8609f443ec63e9cdb6705a2083ca4785298af57d208064899c236','Actualraworacle changed')
    require(sha(PLAN/'configuration.json')=='763552482e02b45312505f9c39021758752117d2542453c8c00fb46d2c604f11','Finalpreparedplan changed')
    independent=BASE/'independent-evaluation/small-city-observer-initial-oracle-independent-review-CPU-001/review.json';r=read(independent)
    require(r['status']=='passed_independent_actual_coarse_raw6groups_uint32_and_SH3_padding_oracle_CPU' and sha(independent)=='47b05c9934d792f74c5ee26c13f46be85d698fed0fddc94f22a32da6d4ba43a1','Independentactualraw6group proof unavailable')
    spec=importlib.util.spec_from_file_location('immutable_final_CPU_identity_gate',core);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);oracle=read(ORACLE)
    snapshot=dict(scaffold_points=1835543,G=2065281,skybox_points=100000,active_SH_degree=0,groups={name:dict(parameter=dict(shape=value['shape'],dtype='float32',uint32_hex=value['uint32_hex'],values=value['values'],all_finite=True)) for name,value in oracle['groups'].items()})
    controls=[]
    for label,corrupt in [('actual_oracle_snapshot',False),('onebit_corrupt_snapshot',True)]:
        directory=OUT/label;directory.mkdir();probe=copy.deepcopy(snapshot)
        if corrupt:probe['groups']['xyz']['parameter']['uint32_hex'][0]=f'{int(probe["groups"]["xyz"]["parameter"]["uint32_hex"][0],16)^1:08x}'
        observer=module.ReadOnlyObserver(directory)
        observer.capture=lambda model:probe
        observer.whole_finite=lambda model:{'CPU_control_only':{'all_finite':True}}
        error=None
        try:observer.initialized(SimpleNamespace())
        except RuntimeError as caught:error=str(caught)
        identity=read(directory/'initial-raw6group-identity.json')
        require(identity['actual_match_passed'] is (not corrupt),'Realinitialized rawidentity gate wrongresult')
        require((error is not None)==corrupt,'Onebit invalidrawsnapshot wasnotrejected')
        controls.append(dict(control=label,original_initialized_method_called=True,identity_match=identity['actual_match_passed'],rejected=bool(error),error=error,
            input_scope='SmallCPUdictionarysnapshot fromactual raworacle. capture/wholefinite stubs localfixture only; no1.3Mpoints/runtimeGPUmodel.',runtimecapture_or_fullmodel_finite_not_executed=True))
    require(all(sha(Path(path))==value for path,value in sources.items()) and 'torch' not in sys.modules,'Sourcechanged orTorch imported')
    result=dict(status='passed_final_immutable_observer_raworacle_stage_supervisor_hidden_source_and_CPU_gate_review',source_hashes=sources,
        actual_oracle_sha256=sha(ORACLE),independent_actual6group_oracle_proof_sha256=sha(independent),final_plan_sha256=sha(PLAN/'configuration.json'),
        old_actual_CPU_Adam_math_and_runpy_spawn_report_sha256=sha(OLD/'CPU-verification.json'),old_control_observer_snapshot_sha256=sha(OLD/'observer-core-tested-snapshot.py'),
        observer_class_method_AST_changed_only=changed,observer_class_method_AST_unchanged=unchanged,read_capture_and_real_Adam_forwarding_helpers_unchanged=True,
        initial_identity_gate_actual_CPU_controls=controls,
        static_contract=dict(initial6rawgroup_uint32_shape_beforeoptimizer=True,SH1_to_SH3_padding_independently_checked=True,original_OurAdam_pinned_forwarded_once_unchanged_arguments=True,currentParameters_observed_afterdensify=True,
            actual_official_argv_equals_failed003_except_isolated_model=True,actual_coarse_fixed005_source_receipt_model_binding=True,fresh_python_native_including_MergerUnoptimized_exclusivity=True,
            Popen_handle_birth_CIM_exact_argv_checked=True,terminal_handle_poll_plus_freshglobalbusy_guard=True,diagnostic86_vsfullbudget0_statuses_distinguished=True,
            hidden_StartProcess_quoted_exact_argv=True,no_existing_output_or_supervision_overwrite_or_retry=True,jobclosure_survival_not_promised=True,
            observation_sync_overhead_not_deterministic_reproduction_timing=True),
        CUDA_called=False,Torch_imported=False,official_training_or_observer_execute_started=False,actual_GPU_initialized_model_identity_not_yet_tested=True,no_formal_training_completion_claim=True)
    write(OUT/'review.json',result);print(json.dumps(dict(status=result['status'],review_sha256=sha(OUT/'review.json'),changedmethods=changed,controls=controls)))
if __name__=='__main__':main()
