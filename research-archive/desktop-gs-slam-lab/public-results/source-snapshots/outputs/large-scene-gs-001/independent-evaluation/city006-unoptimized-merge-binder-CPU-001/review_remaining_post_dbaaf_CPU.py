"""Static remaining post auditor peer; no audit main/models/CIM/Torch/CUDA."""
import ast
from datetime import datetime,timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.dont_write_bytecode=True
LAB=Path(__file__).resolve().parents[4];BASE=LAB/'outputs/large-scene-gs-001';LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
SOURCE=LAB/'scripts/large_scene_gs/audit_city006_remaining_formalpost_completion_CPU_001.py'
OLD=LAB/'scripts/large_scene_gs/audit_city006_formalpost01_completion_CPU_001.py'
OLDER=LAB/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_002.py'
PREP=BASE/'recovery-audit/city006-remaining-formalpost-auditor-preparation-CPU-001/preparation-audit.json'
LEDGER=BASE/'independent-evaluation/small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
BINDING=LOCK/'binding-audits/lineage006-complete-cpu-001/lineage-binding.json'
OUT=BASE/'independent-evaluation/small-city006-remaining-post-auditor-dbaaf-static-peer-CPU-001'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={SOURCE:'dbaafb632319cb98d17d687a7337404550854eb12778ac404a7eeddc4c3363a2',OLD:'0ecd266bbd645d52378914bb763161ff5f1415942b4974a9b98708df0172c375',OLDER:'1fcd7c22871f92bf49de27ec0c21cfb1d4c32a38dc9518a79310f3e795bcec87',PREP:'8357724dae1be08180aa8e69fc884dda50b8ddcc79cfb14ba5ba4199a16f3b87',LEDGER:'17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774',BINDING:'1514402738f6bc6d7c6534224cfabd9412c9f41aaecf1c142601926abacd9f0c'}
assert all(sha(p)==d for p,d in pins.items())
new=ast.parse(SOURCE.read_text(encoding='utf-8'));old=ast.parse(OLD.read_text(encoding='utf-8'))
def funcs(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
n,o=funcs(new),funcs(old)
for name in ['read','sha','require','load']:assert ast.dump(n[name],include_attributes=False)==ast.dump(o[name],include_attributes=False)
def require_conditions(f):return [ast.dump(v.value.args[0],include_attributes=False) for v in f.body if isinstance(v,ast.Expr) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=='require']
assert require_conditions(n['gate'])==require_conditions(o['gate'])
def assign(t,name):return isinstance(t,ast.Assign) and any(isinstance(v,ast.Name) and v.id==name for v in t.targets)
def take_block(main,start,end):
    lo=next(i for i,v in enumerate(main.body) if assign(v,start));hi=next(i for i,v in enumerate(main.body) if i>lo and assign(v,end));return main.body[lo:hi]
old_physical=take_block(o['main'],'fields','leaf_fields');new_physical=take_block(n['main'],'fields','leaf_fields')
# Compare the original numeric and anchor computation before the ID-reading
# block. The new raw node/box comparator is an added, separately reviewed gate.
old_physical=old_physical[:next(i for i,v in enumerate(old_physical) if isinstance(v,ast.With))]
new_physical=[v for v in new_physical[:next(i for i,v in enumerate(new_physical) if isinstance(v,ast.With))] if not assign(v,'node_box_raw_bytes_exact')]
assert ast.dump(ast.Module(body=old_physical,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=new_physical,type_ignores=[]),include_attributes=False)
assert ast.dump(ast.Module(body=take_block(o['main'],'leaf_fields','freeze'),type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=take_block(n['main'],'leaf_fields','freeze'),type_ignores=[]),include_attributes=False)
class RuntimeNames(ast.NodeTransformer):
    def visit_Constant(self,v):
        if isinstance(v.value,str):
            value=v.value.replace('_remaining_formalpost_CPU','_formalpost01_CPU').replace('Selected selectedpost historical process/descendant still live','Selected secondpost historical process/descendant still live')
            return ast.copy_location(ast.Constant(value=value),v)
        return v
assert ast.dump(ast.Module(body=take_block(o['main'],'freeze','result'),type_ignores=[]),include_attributes=False)==ast.dump(RuntimeNames().visit(ast.Module(body=take_block(n['main'],'freeze','result'),type_ignores=[])),include_attributes=False)
spec_node=next(v.value for v in new.body if assign(v,'SPECS'))
specs={ast.literal_eval(k):{kw.arg:ast.literal_eval(kw.value) for kw in v.keywords} for k,v in zip(spec_node.keys,spec_node.values)}
prep=read(PREP);ledger=read(LEDGER);binding=read(BINDING)
assert prep['status']=='remaining_selected_physical_auditor_CPU_source_prepared_no_physical_execution' and prep['source_SHA']==pins[SOURCE]
assert not prep['physical_model_arrays_read'] and not prep['Torch_imported'] and not prep['GPU_or_process_or_lock_control'] and not prep['training_tested']
checked=[]
for chunk in ['1_0','1_1']:
    spec=specs[chunk];record=ledger['records'][spec['index']];origin=binding['ordered_chunks'][spec['index']];creator_path=LOCK/'trained_chunks'/chunk/'creator-outcome.json';creator=read(creator_path)
    assert record['chunk']==origin['chunk']==chunk and creator['training_origin']==origin
    assert sha(creator_path)==spec['creator_SHA'] and creator['hierarchy_sha256']==record['actual_tree_sha256']==spec['before_SHA']
    c=record['actual_anchor_classification'];assert spec['G']==c['G']==creator['hierarchy']['gaussian_count']
    assert spec['A']==c['A']['count'] and spec['L']==c['L']['count'] and spec['A_minus_L']==c['A_minus_L']['count']
    assert spec['npz_SHA']==c['actual_set_npz_sha256']==sha(c['actual_set_npz_path'])
    assert spec['origin_kind']==origin['training_source_kind']==record['training_source_kind']==creator['training_source_kind']
    if chunk=='1_1':assert origin['actual_Gaussian_Adam_calls']==29951 and origin['derived_Gaussian_call_upper_bound']==29999 and origin['formal_training_run'] is False and origin['model']['sha256']=='367bef34e584616daeda938e2e0b5826a8e4bbf30d13b4d987db32c9f72abcb4'
    call=next(v for v in prep['actual_prepare_calls'] if v['chunk']==chunk);result=call['result'];assert call['actual_exit_code']==0 and read(call['prepare_report'])==result
    assert result['audit_source_SHA']==pins[SOURCE] and result['chunk']==chunk and result['original_Creator_expected_SHA']==spec['before_SHA'] and result['expected_training_origin_kind']==spec['origin_kind']
    assert result['expected_stage_run']=='city006-post'+spec['short']+'-i15000-r2-stage-001' and result['arrays_read'] is False and result['GPU_started'] is False
    assert call['command'][-3:]==['--chunk',chunk,'--prepare']
    checked.append({'chunk':chunk,'SPECS':spec,'actual_Creator_receipt_SHA':sha(creator_path),'actual_measured_set_SHA':c['actual_set_npz_sha256'],'recorded_prepare_status_not_current_process_state':result['actual_current_stage_status'],'recorded_prepare_exit_code':0})
main=n['main'];gate_index=next(i for i,v in enumerate(main.body) if isinstance(v,ast.Assign) and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=='gate')
numpy_index=next(i for i,v in enumerate(main.body) if isinstance(v,ast.Import) and any(a.name=='numpy' for a in v.names))
decode_index=next(i for i,v in enumerate(main.body) if assign(v,'before'))
assert gate_index<numpy_index<decode_index
source_text=SOURCE.read_text(encoding='utf-8')
for required in ["final['actual_receipt']==stage","final['receipt_sha256']==sha(DIR/'post-outcome.json')","final['model_sha256']==stage['hierarchy_sha256']","sup['active_stage_lock']['chunk']==CHUNK","sup['active_stage_lock']['supervision_run']==RUN","protected_before={str(f):sha(f) for f in tracked}","tracked=[Path(__file__)","{f:sha(f) for f in protected_before}==protected_before","origin['actual_Gaussian_Adam_calls']==29951","origin['formal_training_run'] is False","'global_GPU_free_claimed':False","'optimizer_call_count_instrumented':False"]:assert required in source_text
raw=n['stored_node_box_bytes_equal'];raw_text=ast.get_source_segment(source_text,raw)
for required in ['remaining=36*len(before.nodes)','4+124*len(before.positions)+4','4+124*len(after.positions)+4','min(8*1024*1024,remaining)','len(a)==len(b)==size and a==b']:assert required in raw_text
assert source_text.count("'float32_bits_equal'")==2
assert "require(np.array_equal(x,y),'Frozen A decoded raw field changed: '" in source_text
assert "require(np.array_equal(x,y),'Actual frozen leaf endpoint field changed: '" in source_text
# No require turns the independently measured bit booleans into mandatory gates.
assert not any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='require' and any(isinstance(k,ast.Constant) and k.value=='float32_bits_equal' for k in ast.walk(v.args[0])) for v in ast.walk(new))
assert all(sha(p)==d for p,d in pins.items())
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
diff=''.join(difflib.unified_diff(OLD.read_text(encoding='utf-8').splitlines(True),SOURCE.read_text(encoding='utf-8').splitlines(True),fromfile='immutable0ecd_post01',tofile='immutable_dbaaf_remaining'))
(OUT/'actual-source-diff.patch').write_text(diff,encoding='utf-8')
report={'status':'passed_static_dbaaf_remaining_post10_11_selected_CPU_auditor_peer_with_explicit_bit_measurement_scope','UTC':datetime.now(timezone.utc).isoformat(),
 'current_pinned_sources_and_preparation_SHA':{str(p):d for p,d in pins.items()},'actual_source_diff_path':str(OUT/'actual-source-diff.patch'),'actual_source_diff_SHA':sha(OUT/'actual-source-diff.patch'),
 'actual_SPECS_and_existing_Creator_ledger_origin_checks':checked,'actual_terminal_gate_before_numpy_and_decoder':True,'no_actual_main_or_prepare_reexecuted_here':True,
 'unchanged_base_finite_scale_nonzero_quaternion_unbounded_alpha_A_numeric_AST_exact':True,'unchanged_leaf_numeric_bits_and_f64_max_computation_AST_exact':True,'unchanged_frozen_Sky_exposure_composite_source_owned_birth_terminal_block_AST_exact':True,
 'new_final_receipt_model_and_exact_same_supervision_chunk_path_checks_reviewed':True,'new_current_auditor_source_before_after_hash_tracking_reviewed':True,'new_compressed_node_box_raw_bytes_gate_reviewed':'Offsets8+124G, length36N, bounded8MiB both files, equality required; no Gaussian endpoint array mutation.',
 'A_L_numeric_value_equality_required':True,'A_L_bit_equality_required_by_passed_status':False,'A_L_bit_equality_measured_boolean_only':True,
 'future_bitexact_claim_requirement':'Parent must inspect every actual_A_raw5fields/actual_L_raw5fields float32_bits_equal boolean in the true completed report; passed status alone does not prove bitexact. Signedzero may pass numerical equality while its separately reported bit flag is false.',
 'observed11_origin_preserved':'Immutable151 + real367bef input, actualGaussian29951, upper29999, formalfalse; post stage does not reclassify the old NaN-failed training run or claim NaN fix.',
 'actual_remaining_post_models_completed_or_audited_here':False,'actual_1404_complete':False,'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'model_arrays_read':False,'process_query_or_control_executed':False,'old_sources_or_reports_modified':False,
 'scope':'Static/source/preparation-evidence peer only. Preparation snapshot records running10/notyet11 at its past timestamp, not current liveness. Real selectedphysical execution requires parent scheduling and true same-stage full15000 native/supervision terminal; no globalGPU-free condition or automatic dispatch.'}
assert not report['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
