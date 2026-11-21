"""Narrow source/report peer; no audit main, raw model arrays, CUDA or CIM."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001';LOCK=BASE/'official-hierarchy/small-city-hierarchy-006'
OUT=BASE/'independent-evaluation/small-city006-post01-physical-source-report-peer-CPU-001'
OLD=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_completion_CPU_002.py'
NEW=ROOT/'scripts/large_scene_gs/audit_city006_formalpost01_completion_CPU_001.py'
REPORT=BASE/'recovery-audit/city006-formalpost01-independent-completion-CPU-001/completion-audit.json'
LEDGER=BASE/'independent-evaluation/small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={OLD:'1fcd7c22871f92bf49de27ec0c21cfb1d4c32a38dc9518a79310f3e795bcec87',NEW:'0ecd266bbd645d52378914bb763161ff5f1415942b4974a9b98708df0172c375',REPORT:'c4e04e9a0d0e1f7799c631eca4dff5338f25801a0407e451c5d6668f2e238af6',LEDGER:'17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774'}
assert all(sha(p)==d for p,d in pins.items())
old=ast.parse(OLD.read_text(encoding='utf-8'));new=ast.parse(NEW.read_text(encoding='utf-8'))
def funcs(tree):return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
of,nf=funcs(old),funcs(new)
for name in ['read','sha','require','load']:assert ast.dump(of[name],include_attributes=False)==ast.dump(nf[name],include_attributes=False)
class AdaptStrings(ast.NodeTransformer):
    def visit_Constant(self,n):
        if isinstance(n.value,str):
            value=n.value.replace('post01','post00').replace('0_1','0_0').replace('Second','First').replace('SECOND','FIRST').replace('second','first')
            return ast.copy_location(ast.Constant(value=value),n)
        return n
assert ast.dump(of['gate'],include_attributes=False)==ast.dump(AdaptStrings().visit(nf['gate']),include_attributes=False)
def target_assignment(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
def block(main,start,end):
    lo=next(i for i,n in enumerate(main.body) if target_assignment(n,start));hi=next(i for i,n in enumerate(main.body) if i>lo and target_assignment(n,end))
    return main.body[lo:hi]
for start,end in [('fields','freeze'),('freeze','result')]:
    a=block(of['main'],start,end);b=block(nf['main'],start,end)
    if start=='fields':
        # The original numeric/anchor block ends immediately before newly added
        # actual-leaf ID and endpoint checks; no base math is removed.
        b=b[:next(i for i,n in enumerate(b) if isinstance(n,ast.With))]
    assert ast.dump(ast.Module(body=a,type_ignores=[]),include_attributes=False)==ast.dump(AdaptStrings().visit(ast.Module(body=b,type_ignores=[])),include_attributes=False)
text=NEW.read_text(encoding='utf-8')
assert "origin=binding['ordered_chunks'][1]" in text and "records'][1]['actual_anchor_classification']" in text
assert "selected=leaves[b:min(len(leaves),b+65536)]" in text
assert "getattr(before,field)[selected]" in text and "getattr(after,field)[selected]" in text
assert "leaves[-1]<len(before.positions)" in text and "np.all(np.isin(leaves,actual_A))" in text
assert "x.astype(np.float64)-y.astype(np.float64)" in text
r=read(REPORT);ledger=read(LEDGER);classification=ledger['records'][1]['actual_anchor_classification'];sets_path=Path(classification['actual_set_npz_path'])
pins[sets_path]=classification['actual_set_npz_sha256'];assert sha(sets_path)==pins[sets_path]=='f289a4e013ebb8c24f38c31c042e5bd690241d9133d35e8f17e39e44bcd570bd'
with np.load(sets_path,allow_pickle=False) as z:
    L=np.asarray(z['leaves_sorted'],dtype='<i8');A=np.asarray(z['anchors_sorted'],dtype='<i8')
    assert len(L)==classification['L']['count']==3032121 and len(A)==classification['A']['count']==3032960
    assert hashlib.sha256(L.tobytes()).hexdigest()==classification['L']['sorted_int64_le_sha256']
    assert hashlib.sha256(A.tobytes()).hexdigest()==classification['A']['sorted_int64_le_sha256']
    assert np.all(L[1:]>L[:-1]) and np.all(A[1:]>A[:-1]) and L[0]>=0 and A[0]>=0 and L[-1]<4411128 and A[-1]<4411128
    assert np.all(np.isin(L,A)) and len(z['anchor_minus_leaf'])==839 and len(z['leaf_minus_anchor'])==0
assert r['status']=='actual_second15000_formalpost01_physical_source_and_owned_terminal_CPU_audit_passed' and r['chunk']=='0_1' and r['source_kind']=='post_optimized'
assert r['requested_iterations']==15000 and r['optimizer_call_count_source_derived']==14999 and not r['optimizer_call_count_instrumented']
assert r['original_hierarchy']['gaussian_count']==r['original_hierarchy']['node_count']==4411128
assert r['actual_full15000_hierarchy']['gaussian_count']==4511128 and r['actual_full15000_hierarchy']['node_count']==4411128 and r['actual_full15000_hierarchy']['unowned_tail']==100000
assert r['actual_anchor_count']==3032960 and r['actual_leaf_count']==3032121 and r['actual_A_minus_L_count']==839
for name in ['actual_A_raw5fields','actual_L_raw5fields']:
    for field,rec in r[name].items():
        assert rec['decoded_float32_values_exact'] and rec['float32_bits_equal'] and rec['max_abs_delta']==0
        if name=='actual_L_raw5fields':assert rec['actual_leaf_rows']==3032121
assert r['nodes_and_stored_boxes_exact'] and not r['global_GPU_free_claimed'] and not r['GPU_or_process_or_lock_control_performed']
assert r['broad_all5_array_audit_repeated'] is False and r['CUDA_initialized'] is False and r['observation_reclassified_formal'] is False
for p,d in r['protected_source_and_output_before_after'].items():
    if not p.endswith(('.hier','.hier_opt')) and not p.endswith('run_small_city_hierarchy_006.py'):
        assert sha(p)==d
for p,d in pins.items():assert sha(p)==d
# Preserve rather than rewrite the actual source/report description mismatch.
assert 'only originalchunk00/coarse/secondpost arrays checked' in r['mixed_origin_scope']
erratum={'field':'mixed_origin_scope','original_value':r['mixed_origin_scope'],'corrected_description':'Same frozen151 binding and actual source/config/receipt bytes verified; only originalchunk01/coarse/secondpost arrays checked. Other4 training arrays not rescanned.','scope':'Description only; actual source DIR/originindex1/argv/SHAs/set IDs and numerical results identify0_1 correctly. Owner notified to append independent metadata errata; old source/report unchanged.'}
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
review={'status':'passed_narrow_post01_0ecd_source_actual_c4e04_physical_report_CPU_peer_with_metadata_erratum','UTC':datetime.now(timezone.utc).isoformat(),
 'protected_evidence_current_SHA':{str(p):d for p,d in pins.items()},'original_gate_after_explicit_chunk_label_substitution_AST_exact':True,'unchanged_base_numeric_anchor_halfSky_exposure_runtime_and_terminal_AST_exact':True,
 'new_actual_leaf_checks_reviewed':'Actual f289 measured IDs, count/hash/order/bounds/subsetA gate; same endpoints selected in safe65536ID windows, decodedf32 values exact plus separate bitbool/f64 max. No all-leaf universal contract; only independently measured actual01 leaves.',
 'actual_measured_A_count':3032960,'actual_measured_L_count':3032121,'actual_measured_A_minus_L_count':839,'actual_report_A_L_5fields_values_and_bits_exact':True,
 'actual_post_G':4511128,'actual_post_N':4411128,'actual_single_Sky_tail':100000,'actual_model_SHA':'a0cd50a0f337ae2fd6ed782100939a4c8512a35b6ef8ebbd9ef79e05f7c3f32f',
 'metadata_erratum':erratum,'models_or_CIM_rescanned_here':False,'physical_audit_main_or_mutator_executed':False,'source_or_old_reports_edited':False,'Torch_imported':'torch' in sys.modules,'CUDA_called':False,
 'scope':'Static math/source and actual bound report/ID evidence review, not a second physical model scan or proof of all4 posts/Merger/1404 completion.14999 is source-derived; endpoint freezing does not imply quality or instrumented per-iteration update count.'}
assert not review['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
