"""Read-only source/AST/actual missing-output peer of independent unoptimized Merger binder."""
from pathlib import Path
import importlib.util,hashlib,json,sys,subprocess,ast
from datetime import datetime,timezone
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab');BASE=ROOT/'outputs/large-scene-gs-001';E=BASE/'independent-evaluation'
OUT=BASE/'recovery-audit/city006-unoptimized-merge-binder-independent-review-CPU-001';OUT.mkdir(parents=True,exist_ok=False)
sys.dont_write_bytecode=True
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(n,p):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
source=E/'city006-unoptimized-merge-binder-CPU-001/bind_unoptimized_merged_CPU.py'
assert sha(source)=='1f4a6ed694a4ca47e1a9cae3f221a113a7bfe24c627522d4dc49bc18cc896cc6'
m=load('_unoptimized_merge_CPUpeer',source)
tracked=[source,m.LOCK/'driver-source.py',m.LOCK/'configuration.json',m.DRIVER/'native-workflow-lock.json',m.CORE,m.LEDGER,m.ALL4,m.BINDING]
before={str(p):sha(p) for p in tracked}
# No all5 array scan, renderer import, GPU hardware query or role fixture.
config,binding,variant,contracts=m.exact_sources()
manifest,edges=m.creator_edges(config,binding,False)
assert [e['chunk'] for e in edges]==['0_0','0_1','1_0','1_1']
assert edges[-1]['training_source_kind']=='observed_local_official_training'
assert not binding['formal_observation_reclassification'] and all(e['summary']['unowned_tail']==0 for e in edges)
recipe,ast_sha=m.memory_recipe()
tree=ast.parse(m.CORE.read_text(encoding='utf-8'));function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='memory_plan')
assert ast_sha==hashlib.sha256(ast.dump(function,include_attributes=False).encode()).hexdigest()
declared=recipe(1100,10,600,1000,100,117)
assert isinstance(declared,dict) and 'torch' not in sys.modules
command=[str(ROOT/'.venv-large-gs/Scripts/python.exe'),str(source),'--mode','bind','--run','peer006-unoptimized-merge-actualmissing-CPU-001']
process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW,timeout=30)
(OUT/'actual-missing-stdout.txt').write_text(process.stdout,encoding='utf-8');(OUT/'actual-missing-stderr.txt').write_text(process.stderr,encoding='utf-8')
assert process.returncode==1
failure=read(E/'peer006-unoptimized-merge-actualmissing-CPU-001/outcome.json')
assert 'Merger output/receipt absent' in failure['error'] and failure['Torch_imported'] is False and failure['actual_merged_model_bound'] is False
assert {p:sha(p) for p in before}==before and 'torch' not in sys.modules
review={'status':'passed_unoptimized_Merger_CPU_binder_source_memory_recipe_and_actual_missing_output_peer_review',
  'UTC':datetime.now(timezone.utc).isoformat(),'source_before_after':before,
  'same_mode0_algorithm_filename_literal_only':True,'source_variant_library_exe_and_six_ownership_storage_files_verified':contracts,
  'actual_all4_Creator_receipt_SHA_source_family_and_exposure_edges_verified_without_tree_rescan':edges,
  'config_input_order_and_centers_scope':'Native same original manifest centers/order; actual merged4 rootpoints numeric equality will run only after real receipt/tree exist.',
  'mixed_observed_origin_not_reclassified':True,'merged_tree_requires_zeroSky_and_later_shared100kSky_once':True,
  'mapped_parser_source_SHA':sha(m.LEDGER),'mapped_mode':'r','true_bind_requires_real_merged_header_root4points_leafranges_and_currenttreeSHA':True,
  'memory_recipe_AST_same_as_sealed_core':True,'memory_recipe_AST_SHA':ast_sha,
  'memory_scope':'Prior actual totalVRAM only and source-derived upper accounting; no currentfree/worstnative fit/OOM/streaming guarantee.',
  'compact_leaf_path':'Planned-only; explicit actual sameModel/camera bitwise RGB pixel gate required before enabling.',
  'actual_missing_output_test':{'exit_code':process.returncode,'outcome_path':str(E/'peer006-unoptimized-merge-actualmissing-CPU-001/outcome.json'),'error':failure['error'],'Torch_imported':False},
  'actual_Merger_or_model_or1404_completion_claimed':False,'GPU_or_renderer_or_role_mock_started':False,
  'broad_all5_array_audit_repeated':False,'existing_native_eval_seals_or_sources_modified':False}
(OUT/'peer-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'report':str(OUT/'peer-review.json'),'SHA':sha(OUT/'peer-review.json')}))
