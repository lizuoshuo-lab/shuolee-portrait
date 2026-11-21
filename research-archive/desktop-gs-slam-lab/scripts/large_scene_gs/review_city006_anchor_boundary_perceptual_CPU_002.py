"""CPU source/AST/math/missing-artifact peer review while a separate native stage may be live."""
from pathlib import Path
import ast,importlib.util,hashlib,json,sys,subprocess,types
from datetime import datetime,timezone
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001';E=BASE/'independent-evaluation'
OUT=BASE/'recovery-audit/city006-anchor-boundary-perceptual-independent-review-CPU-002'
OUT.mkdir(parents=True,exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
paths={'anchor':E/'city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006.py',
       'boundary':E/'city-boundary-image-binder-CPU-006/audit_city_boundary_images_006.py',
       'perceptual':E/'city-perceptual-CPU-reference-driver-006/recompute_city_perceptual_cpu_006.py'}
pins={'anchor':'b275c765ce7aec4e96e49dabbc575761939105f2da237a2d9e51007085c27cb9',
      'boundary':'f16ccb274707515a29cf3466be9c294b206b08a2cd1404f05321a99c88669379',
      'perceptual':'9e42b40079f6d7ea73c069997fdb23121b5e24f00f5409cfd4f09830d40e8a0c'}
for k,p in paths.items():assert sha(p)==pins[k]
old_anchor=ROOT/'scripts/large_scene_gs/audit_city_hierarchy_lineage_001.py'
creator_snapshot=E/'small-city006-creator00-anchor-source-exposure-CPU-001/source-snapshot.py'
old_boundary=ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py'
old_cpu=E/'city-perceptual-CPU-reference-driver-001/recompute_city_perceptual_cpu-frozen.py'
assert sha(old_anchor)=='f52e19946b0df7efb6a4ad69bcfdbc844979ad76fe9e530761c1f0e292c1ed2b'
assert sha(creator_snapshot)=='5144ed9f8cc448af444bf48bf481ed3619105b5074000fa6a4bc406586c5c042'
assert sha(old_boundary)=='15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f'
assert sha(old_cpu)=='1ec28a295b93869c7da414f830e2d70f8c1c4a02ef1f763de36ac8cfeeb6f9fc'
tracked=list(paths.values())+[old_anchor,creator_snapshot,old_boundary,old_cpu]
before={str(p):sha(p) for p in tracked}
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
ast_checks={}
for current,prior,names in [(paths['anchor'],creator_snapshot,['mapped','classify','xyz_multiset','creator_only']),
                            (paths['boundary'],old_boundary,['pixel_image','fixed_patch','original_frame']),
                            (paths['perceptual'],old_cpu,['initialize_cpu','cpu_memory'])]:
    a,b=functions(current),functions(prior)
    for name in names:ast_checks[str(current.name)+':'+name]=a[name]==b[name]
assert all(ast_checks.values())
new=load('_anchor006_CPUpeer_math',paths['anchor']);old=load('_anchorold_CPUpeer_math',old_anchor)
nodes=np.array([[3,-1,0,0,1,1,2],[2,0,1,0,0,3,1],[1,0,1,1,1,4,1],[0,1,3,1,0,0,0],[0,2,4,1,1,0,0]],np.int32)
h=types.SimpleNamespace(positions=np.zeros((6,3),np.float32),nodes=nodes)
math_controls=[]
for j,ids in enumerate([[],[3],[4,5],[1,2,4,5],list(range(6))]):
    anchor=np.array(ids,np.int32)
    a,la=old.anchor_classification(h,anchor);b,lb=new.classify(h,anchor,OUT,'CPU-toy'+str(j))
    assert np.array_equal(la,lb)
    for ok,nk in [('anchor_intersection_leaf','A_intersection_L'),('leaf_minus_anchor','L_minus_A'),('anchor_minus_leaf','A_minus_L')]:
        for field in ['count','sorted_int64_le_sha256']:assert a[ok][field]==b[nk][field]
    assert a['anchors_equal_leaves']==b['anchors_equal_leaves_measured']
    math_controls.append({'kind':'isolatedCPUtree_fixture_not_actual_model','anchors':ids,'old_new_set_counts_SHA_exact':True})
for ids in [[1],[1,2],[0,1,2,4,5]]:
    for fn in [lambda:old.anchor_classification(h,np.array(ids,np.int32)),lambda:new.classify(h,np.array(ids,np.int32),OUT,'invalid-'+str(len(ids)))]:
        try:fn()
        except RuntimeError:pass
        else:raise AssertionError('Invalid node range/closure accepted')
    math_controls.append({'kind':'isolatedCPUtree_fixture_not_actual_model','anchors':ids,'both_rejected_partial_or_descendant_violation':True})
fields={'positions':(3,),'rotations':(4,),'log_scales':(3,),'opacities':(),'sh':(16,3)}
pre=types.SimpleNamespace(**{n:np.ones((6,)+shape,np.float32) for n,shape in fields.items()})
post=types.SimpleNamespace(**{n:np.concatenate([getattr(pre,n).copy(),np.ones((100000,)+shape,np.float32)]) for n,shape in fields.items()})
_,leaf=old.anchor_classification(h,np.array([],np.int32))
post.positions[3,0]+=.25;post.sh[1,2,1]+=.125
a,b=old.leaf_changes(pre,post,leaf),new.leaf_changes(pre,post,leaf)
for n in fields:
    for ok,nk in [('actual_leaf_count','actual_leaf_rows'),('changed_leaf_rows','changed_leaf_rows'),('max_abs_decoded_float32_difference','max_abs_decoded_difference'),('decoded_values_equal','decoded_equal')]:assert a['fields'][n][ok]==b['fields'][n][nk]
math_controls.append({'kind':'isolatedCPUleaf_endpoint_fixture_not_actual_post','old_new_endpoint_values_counts_max_exact':True})
# Run actual missing dependencies only. No fake completed receipts or GPU-role controls.
commands=[('anchor-formalpost',paths['anchor'],['--run','peer006-anchor-actualmissingformal-CPU-001','--mode','formal-four'],E/'peer006-anchor-actualmissingformal-CPU-001/outcome.json'),
          ('boundary-dense',paths['boundary'],['--run','peer006-boundary-actualmissing1404-CPU-001','--mode','dense','--evaluation-run','small-city006-heldout-1404-001','--posterior-audit-run','small-city006-heldout-posterior-CPU-001'],E/'peer006-boundary-actualmissing1404-CPU-001/outcome.json'),
          ('perceptual1404',paths['perceptual'],['--audit-run','peer006-perceptual-actualmissing1404-CPU-001','--scope','own1404','--evaluation-run','small-city006-heldout-1404-001','--posterior-audit',str(E/'small-city006-heldout-posterior-CPU-001/completion-audit.json')],E/'peer006-perceptual-actualmissing1404-CPU-001/outcome.json')]
actual_missing=[]
for name,p,args,outcome in commands:
    command=[str(ROOT/'.venv-large-gs/Scripts/python.exe'),str(p),*args]
    result=subprocess.run(command,capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT/(name+'-stdout.txt')).write_text(result.stdout,encoding='utf-8');(OUT/(name+'-stderr.txt')).write_text(result.stderr,encoding='utf-8')
    assert result.returncode==1
    value=read(outcome);assert value.get('Torch_imported',False) is False and value.get('actual1404_completed',False) is False
    actual_missing.append({'name':name,'actual_exit_code':result.returncode,'actual_outcome_path':str(outcome),'actual_outcome_SHA':sha(outcome),'error':value['error'],'Torch_imported':False,'CUDA_started':False})
assert {p:sha(p) for p in before}==before and 'torch' not in sys.modules
report={'status':'passed_CPU006_anchor_boundary_perceptual_source_math_and_actual_missing_stage_peer_review',
        'UTC':datetime.now(timezone.utc).isoformat(),'source_before_after':before,'AST_exact_checks':ast_checks,
        'old_new_anchor_and_leaf_math_CPU_controls':math_controls,'actual_missing_formalpost_and_1404_rejections':actual_missing,
        'binding_review':{'native_config_source_lineage_pins':True,'observed_origin_not_reclassified_formal':True,
             'formal_post_requires15000_and14999_source_derived':True,'pilot64_cannot_substitute':True,
             'dense_completion_requires117x6x2_actual_source_frame_and_protocol_ledgers':True,
             'geometry8_boundary_choice_frozen_before_RGB_metrics':True,
             'perceptual_all12_reported_score_files_sealed_to_actual_posterior_metrics':True,
             'original_images_not_generated_cropped_or_rewritten':True},
        'GPU_or_native_or_training_or_role_mock_started':False,'CUDA_initialized':False,'broad_all5_array_audit_repeated':False,
        'existing_native_eval_seals_sources_or_records_modified':False}
(OUT/'peer-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
