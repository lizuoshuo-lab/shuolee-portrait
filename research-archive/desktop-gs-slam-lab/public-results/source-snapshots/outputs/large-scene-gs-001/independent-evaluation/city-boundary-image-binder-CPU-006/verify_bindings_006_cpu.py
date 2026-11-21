"""CPU source/schema evidence only; no source images or GPU metric substitute."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'independent-evaluation'
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
old=ROOT/'scripts/large_scene_gs/audit_city_boundary_images_002.py';new=HERE/'audit_city_boundary_images_006.py'
ow=OUT/'city-perceptual-CPU-reference-driver-001/recompute_city_perceptual_cpu.py';nw=OUT/'city-perceptual-CPU-reference-driver-006/recompute_city_perceptual_cpu_006.py'
assert sha(old)=='15cc350108647f47ee7c20c6297ce7930d6af674233bbc3eac2017eeb74fcb1f'
assert sha(ow)=='1ec28a295b93869c7da414f830e2d70f8c1c4a02ef1f763de36ac8cfeeb6f9fc'
f,g=functions(old),functions(new);boundary_same={k:f[k]==g[k] for k in ('pixel_image','fixed_patch','original_frame')};assert all(boundary_same.values())
f,g=functions(ow),functions(nw);perceptual_same={k:f[k]==g[k] for k in ('initialize_cpu','cpu_memory')};assert all(perceptual_same.values())
assert sha(OUT/'city-hierarchy-posterior-auditor-CPU-001/recompute_perceptual_cpu.py')=='4b318c7362c35cce5087605ac7c1902fb0697c67185c039e3150676cfd3a3181'
assert sha(OUT/'city-local-hierarchy-evaluation-driver-CPU-006/evaluate_local_city_hierarchy_006.py')=='260827a71f429fb49f0454c96054cdc3b3aaa08867225a56dafc525c39a75bb3'
assert sha(OUT/'city-hierarchy-posterior-auditor-CPU-006/audit_city_1404_006.py')=='5c9c1b7da3d2bbeeb072a977aa69bac09c81dc5511f71c7c51fe0e35139e4d8c'
assert sha(OUT/'city-local-hierarchy-evaluation-driver-CPU-006/native-workflow-lock.json')=='0fac8a7e861fb05f5b67d3edab7851dacf4d6af99f211d16f1cf78c9b3eebea8'
names=['small-city006-formalpost-ledger-actual-missing-CPU-001','small-city006-boundary-actual-missing1404-CPU-001','small-city006-perceptual-actual-missing1404-CPU-001']
controls=[]
for name in names:
    p=OUT/name/'outcome.json';r=read(p)
    assert r['status'].startswith('failed_preserved') and not (OUT/name/'completion-audit.json').exists()
    assert not r.get('Torch_imported',False) and not r.get('CUDA_called',False) and not r.get('CUDA_initialized',False)
    controls.append({'path':str(p),'sha256':sha(p),'status':r['status'],'error':r['error']})
creator=OUT/'small-city006-creator00-anchor-source-exposure-CPU-001/report.json';actual=read(creator);assert sha(creator)=='63775e5373125819b39dd689a70b2b1053dfc45f47263bafec6d6c7ab1702728'
rec=actual['records'][0];cls=rec['actual_anchor_classification'];proof=Path(cls['actual_set_npz_path']);assert sha(proof)==cls['actual_set_npz_sha256']
assert cls['A_intersection_L']['count']==2978113 and cls['L_minus_A']['count']==0 and cls['A_minus_L']['count']==451 and cls['descendant_closure_violations']==0
assert all(rec['actual_single_source_sky_raw_uint32'].values()) and rec['actual_leaf_source_XYZ']['actual_raw_float32_XYZ_bit_multiset_equal'] and rec['all_read_source_bytes_unchanged']
report={'status':'CPU006_future_binding_sources_and_real_Creator00_evidence_prepared_pending_root_peer_review','boundary006_path':str(new),'boundary006_sha256':sha(new),'perceptual006_path':str(nw),'perceptual006_sha256':sha(nw),'fixed_patch_original_image_function_AST_exact':boundary_same,'fixed_CPU_initialize_memory_function_AST_exact':perceptual_same,'fixed_optional_actual1404_metric_algorithm_sha256':'4b318c7362c35cce5087605ac7c1902fb0697c67185c039e3150676cfd3a3181','actual_Creator00_report':{'path':str(creator),'sha256':sha(creator)},'actual_Creator00_set_proof':{'path':str(proof),'sha256':sha(proof)},'actual_missing_dependencies_controls':controls,'formal_post_budget_required':15000,'pilot64_eligible':False,'actual_1404_complete':False,'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'old005_sources_changed':False,'raw_images_written_or_website_published':False,'scope':'New source binding preparation, actual one Creator tree array measurement and real absence refusals. Self source audit only; no formal post/Merger/GPU/perceptual/dense-boundary results.'}
p=HERE/'CPU-bindings-preparation-review-001.json';p.open('x',encoding='utf-8').write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(p),'sha256':sha(p),'boundary_sha256':sha(new),'perceptual_sha256':sha(nw),'status':report['status']}))
