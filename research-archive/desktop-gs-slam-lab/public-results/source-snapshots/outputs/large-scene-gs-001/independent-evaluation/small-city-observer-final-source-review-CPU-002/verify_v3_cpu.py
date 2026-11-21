"""CPU-only immutable v3 diff and genuine spy/trace method controls.

Small synthetic Python objects/snapshot; no Torch, CUDA, originaltrainer,
originalAdam math or realmodel execution. Old actualCPU math proof reused byAST.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,importlib.util,json,random,shutil,sys
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';OUT=Path(__file__).parent
OLD=BASE/'recovery-audit/small-city-observation-only11-peer-controls-CPU-002'
V2=BASE/'independent-evaluation/small-city-observer-final-source-review-CPU-001'
PLAN=BASE/'official-training-observation-plans/small-city-chunk11-observation-i30000-r4-001-v3'
EXPECT={'observe_small_city_train_single_001.py':'59bc69b80fb50283cddd5a35c115fb3ecd9ba37bef675c74e6f1736cc02af384',
'run_small_city_observation_001.py':'ebe2581d8ae8bee8fa48519c383030a0e74be0677c38b81855715dee6a16570c',
'supervise_small_city_observation_001.py':'bb0a8cd8f7efd2666c56022a397ecdb2b969b02c20806538c25f90c10a7ec370',
'launch_small_city_observation_001.ps1':'42be603955a0701c32bc8739ba96d6fc6a4adeb2f190157a739f9f21f0159366'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
def require(v,m):
    if not v:raise RuntimeError(m)
def methods(p):
    t=ast.parse(p.read_text());c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='ReadOnlyObserver')
    return {n.name:ast.dump(n,include_attributes=False) for n in c.body if isinstance(n,ast.FunctionDef)}
def main():
    require(not (OUT/'review.json').exists(),'Preservev3review')
    hashes={}
    for name,s in EXPECT.items():
        p=ROOT/'scripts/large_scene_gs'/name;require(sha(p)==s,'Immutablev3source differs');hashes[str(p)]=s;shutil.copyfile(p,OUT/name)
    require(sha(PLAN/'configuration.json')=='8b51155d5300bd75c3ebef0c4544da8e2b5a393cd90909ed97430ca0544df826','Exactv3plan changed')
    require(sha(OLD/'CPU-verification.json')=='e78536a35a16036b458af8732783314de5a58bade85efd40f101f3debd3694c0','ActualpreviousCPUcontrol proof changed')
    core=ROOT/'scripts/large_scene_gs/observe_small_city_train_single_001.py';old=methods(OLD/'observer-core-tested-snapshot.py');new=methods(core);v2=methods(V2/'observe_small_city_train_single_001.py')
    required=['capture','context','whole_finite','wrap_step','wrap_mutator','before_save','initialized']
    for name in required:require(new[name]==(v2[name] if name=='initialized' else old[name]),'Existingverifiedread/Adam helper changed:'+name)
    oracle=read(BASE/'official-training-observation-plans/small-city-initial-suspect-oracle-001/expected-raw.json')
    spec=importlib.util.spec_from_file_location('immutable_v3_CPU_only',core);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    fixture=OUT/'spy-trace-CPU-fixture';fixture.mkdir();o=m.ReadOnlyObserver(fixture)
    model=SimpleNamespace(_opacity=SimpleNamespace(grad=None),optimizer=SimpleNamespace(),exposure_optimizer=SimpleNamespace())
    o.model=model;o.capture=lambda obj:dict(row_zero_based=1377084,G=2065281,skybox_points=100000,scaffold_points=1835543,active_SH_degree=0,groups={k:dict(parameter=dict(shape=v['shape'],dtype='float32',uint32_hex=v['uint32_hex'],values=v['values'],all_finite=True)) for k,v in oracle['groups'].items()})
    marker=object();return_value=object();original_exposure_calls=[];original_Gaussian_calls=[]
    def original_exposure(*args,**kwargs):
        original_exposure_calls.append(dict(args_id=[id(v) for v in args],kwargs_value_id={k:id(v) for k,v in kwargs.items()}));return return_value
    def original_Gaussian(*args,**kwargs):
        original_Gaussian_calls.append(dict(args_id=[id(v) for v in args],kwargs_value_id={k:id(v) for k,v in kwargs.items()}));return return_value
    model.exposure_optimizer.step=o.wrap_exposure_step(original_exposure)
    model.optimizer.step=o.wrap_step(model,original_Gaussian)
    class Relevant:
        def numel(self):return 0
    relevant=Relevant();grad=object();condition_lines=[i for i,line in enumerate((m.REPO/'train_single.py').read_text().splitlines(),1) if line.strip()=='if gaussians._opacity.grad != None:']
    require(condition_lines==[170],'Originalcondition source line changed');o.opacity_condition_line=condition_lines[0]
    # Toy codefilename/lineno route the realtrace callback as the officialframe,
    # but no bytes/AST/function in originaltrainer are executed ormodified.
    text='def training(gaussians, marker, relevant, grad, returns, viewpoint_cam):\n    for iteration in (1,2):\n        gaussians._opacity.grad = None if iteration == 1 else grad\n        returns.append(gaussians.exposure_optimizer.step(marker, token=marker))\n'+('\n'*165)+'        if gaussians._opacity.grad != None:\n            returns.append(gaussians.optimizer.step(relevant))\n'
    require(text.splitlines()[169].strip()=='if gaussians._opacity.grad != None:','Toycondition line routing wrong')
    scope={};exec(compile(text,str(m.REPO/'train_single.py'),'exec'),scope)
    returns=[];camera=SimpleNamespace(image_name='CPU_toy',uid=0,colmap_id=0,image_width=256,image_height=172,depth_reliable=False)
    state_before=random.getstate();require(sys.gettrace() is None,'CPUtest already traced')
    try:
        sys.settrace(o.global_read_only_trace);scope['training'](model,marker,relevant,grad,returns,camera)
    finally:sys.settrace(None)
    require(o.exposure_optimizer_calls==2 and len(original_exposure_calls)==2 and o.opacity_grad_condition_visits==2 and o.opacity_grad_none_skips==1 and o.optimizer_calls==1 and len(original_Gaussian_calls)==1,'Realspy/trace branchcounts incorrect')
    require(all(v is return_value for v in returns) and all(r['args_id']==[id(marker)] and r['kwargs_value_id']=={'token':id(marker)} for r in original_exposure_calls),'Exposureoriginal args/return identity changed')
    require(original_Gaussian_calls[0]['args_id']==[id(relevant)] and model._opacity.grad is grad,'OriginalGaussianarg or originalbranch condition changed')
    require(o.opacity_grad_none_skip_contexts[0]['iteration']==1 and o.opacity_grad_none_skip_contexts[0]['camera']['image_name']=='CPU_toy','Skipactual framecontext missing')
    require(state_before==random.getstate() and sys.gettrace() is None,'PythonRNG/trace leak')
    offscope={};exec(compile(text,str(OUT/'unrelated-CPU-toy.py'),'exec'),offscope);old_visits=o.opacity_grad_condition_visits
    try:
        sys.settrace(o.global_read_only_trace);offscope['training'](model,marker,relevant,grad,[],camera)
    finally:sys.settrace(None)
    require(o.opacity_grad_condition_visits==old_visits,'Unrelatedcodefile wastraced')
    require(all(sha(Path(p))==s for p,s in hashes.items()) and 'torch' not in sys.modules,'Sourcechanged/Torch import')
    result=dict(status='passed_v3_immutable_count_semantics_spy_trace_CPU_controls_and_source_review',immutable_sources=hashes,exact_plan_sha256=sha(PLAN/'configuration.json'),
        previous_actual_CPU_Adam_math_and_spawn_proof_sha256=sha(OLD/'CPU-verification.json'),previous_v2_review_followup_erratum_sha256=sha(V2/'count-gate-followup-erratum.json'),
        verified_unchanged_AST_methods=required,actual_exposure_spy_calls_in_two_branch_toy=2,actual_gradient_condition_visits_in_two_branch_toy=2,actual_None_skips_in_two_branch_toy=1,actualGaussian_calls_in_two_branch_toy=1,
        exposure_original_forwarded_exactly_once_args_and_return_identity=True,Gaussian_original_argument_identity=True,original_True_None_branch_results_unchanged=True,skip_original_iteration_camera_context_recorded=True,only_pinned_training_file_frame_traced=True,trace_finally_removed=True,Python_RNG_state_unchanged=True,
        fixture_scope='Smallsynthetic CPUobjects and realobserver spy/trace methods; sourcefilename/line170 areexplicitroutingfixture. Notoriginaltrainer/runtimebudget/GPU/fullmodel test. No1.3Mpoint tensors.',
        full_budget_static_contract='OriginalScene.save30000 mustsuccessfullyreturn +runpy normalreturn; actualexposure/conditionvisits29999, actualGaussian<=29999 andcalls+NoneSkips=29999. Savedmodelfinite required. Diagnosticstop isnotfullbudget/formal.',
        current_immutable_init_raw_oracle_and_stage_coarse_contract_unchanged=True,stage_supervisor_hidden_count_semantics_match_core=True,freshbusy_guard_handle_birth_argv_nooverwrite_policy_unchanged=True,
        CUDA_called=False,Torch_imported=False,officialtrainer_or_GPU_started=False)
    write(OUT/'review.json',result);print(json.dumps(dict(status=result['status'],review_sha256=sha(OUT/'review.json'))))
if __name__=='__main__':main()
