"""Independent narrow source/report/set review; no model rescans or main execution."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4];BASE=ROOT/'outputs/large-scene-gs-001'
OUT=BASE/'independent-evaluation/small-city006-post00-endpoint-source-report-peer-CPU-001'
SOURCE=ROOT/'scripts/large_scene_gs/audit_city006_formalpost00_parameter_changes_CPU_001.py'
REPORT=BASE/'recovery-audit/city006-formalpost00-parameter-change-CPU-001/parameter-change-audit.json'
PHYSICAL=BASE/'recovery-audit/city006-first-formalpost00-independent-completion-CPU-002/completion-audit.json'
LEDGER=BASE/'independent-evaluation/small-city006-all4-creator-anchor-source-exposure-CPU-001/report.json'
AA=BASE/'independent-evaluation/city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006_posebytes_v2.py'
CODEC=BASE/'official-hierarchy/small-city-hierarchy-006/helper-snapshots/read_hierarchy.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={SOURCE:'d720f280c697596050b45884d5ce63f71082373084d57ad60897a62b0cb71ac1',REPORT:'46299290c1c1bd10b8db11005f1bbcfe07157741d22bc3d0b922a568f2327cb2',PHYSICAL:'dd0611011d19665a44bbd9a8aebe9817951246a9614e7fbfab46c2ec6ab7d462',LEDGER:'17be8b334c09f3dfbe91ba5e56d0e8107b674b379bc8977dc7447c12d572f774',AA:'aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0',CODEC:'583c96f50b6f486d73bbbbaf9faf053ed15d486a7c7b48e15ef8e3ebc0964538'}
assert all(sha(p)==d for p,d in pins.items())
s=ast.parse(SOURCE.read_text(encoding='utf-8'));a=ast.parse(AA.read_text(encoding='utf-8'))
mapping=next(n for n in s.body if isinstance(n,ast.FunctionDef) and n.name=='map_fields')
old_mapping=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='mapped')
schema=next(n.value for n in ast.walk(mapping) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='schema' for t in n.targets))
old_schema=next(n.value for n in ast.walk(old_mapping) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='fields' for t in n.targets))
class Names(ast.NodeTransformer):
    def visit_Name(self,n):
        return ast.copy_location(ast.Name(id='g' if n.id=='count' else n.id,ctx=n.ctx),n)
assert ast.dump(Names().visit(schema),include_attributes=False)==ast.dump(old_schema,include_attributes=False)
text=SOURCE.read_text(encoding='utf-8')
assert "end = min(count, start+65536)" in text and "old[start:end]" in text and "new[start:end]" in text and "selected = mask[start:end]" in text
assert "np.isfinite(x).all() and np.isfinite(y).all()" in text and "delta = np.abs(x.astype(np.float64)-y.astype(np.float64))" in text
assert "x.view(np.uint32) != y.view(np.uint32)" in text
r=read(REPORT);physical=read(PHYSICAL);ledger=read(LEDGER);classification=ledger['records'][0]['actual_anchor_classification'];count=r['original_tree_G']
assert r['status']=='actual_post00_full15000_original_tree_parameter_endpoint_changes_measured_CPU'
assert count==classification['G']==4332649 and r['after_G_including_Sky']==count+100000 and r['excluded_shared_Sky_rows']==100000
assert physical['status']=='actual_first15000_formalpost00_physical_source_and_owned_terminal_CPU_audit_passed'
assert r['before_sha256']=='9b3e8ad0901601410e7e73d902cdb448051258d582e41574f6b0fd98f7a30cd7' and r['after_sha256']=='a387578911cc4d630e1105659010de4af7bd7b7add3d21be483e73243cd712a6'
assert r['all_read_bytes_unchanged'] and not r['Torch_imported'] and not r['GPU_control_or_process_query_performed'] and not r['global_GPU_free_claimed']
npz=Path(classification['actual_set_npz_path']);pins[npz]=classification['actual_set_npz_sha256'];assert sha(npz)==pins[npz]
with np.load(npz,allow_pickle=False) as z:
    A=np.zeros(count,np.bool_);L=np.zeros(count,np.bool_)
    for key,record,mask in [('anchors_sorted',classification['A'],A),('leaves_sorted',classification['L'],L)]:
        ids=np.asarray(z[key],dtype='<i8');assert ids.ndim==1 and len(ids)==record['count'] and ids[0]>=0 and ids[-1]<count and np.all(ids[1:]>ids[:-1])
        assert hashlib.sha256(ids.tobytes()).hexdigest()==record['sorted_int64_le_sha256'];mask[ids]=True
    expected={'anchor_intersection_leaf':np.flatnonzero(A&L),'leaf_minus_anchor':np.flatnonzero(L&~A),'anchor_minus_leaf':np.flatnonzero(A&~L)}
    for key,value in expected.items():assert np.array_equal(z[key],value)
groups={'A_all':A,'L_all':L,'anchor_leaf_A_intersection_L':A&L,'anchor_internal_A_minus_L':A&~L,'nonAnchor_leaf_L_minus_A':~A&L,'nonAnchor_internal':~A&~L}
parts=['anchor_leaf_A_intersection_L','anchor_internal_A_minus_L','nonAnchor_leaf_L_minus_A','nonAnchor_internal']
assert np.all(sum(groups[p].astype(np.int8) for p in parts)==1)
widths={'positions':3,'rotations':4,'log_scales':3,'opacities':1,'sh':48}
stats=r['group_statistics'];assert stats.keys()==groups.keys()
for name,mask in groups.items():
    g=stats[name];rows=int(mask.sum());assert g['rows']==rows
    for field,width in widths.items():
        f=g['fields'][field]
        assert f['total_rows']==rows and f['total_scalar_values']==rows*width
        assert 0<=f['changed_rows']<=f['float32_bit_changed_rows']<=rows
        assert 0<=f['changed_scalar_values']<=f['float32_bit_changed_scalar_values']<=rows*width
        assert f['changed_rows']<=f['changed_scalar_values']<=f['changed_rows']*width
        assert f['float32_bit_changed_rows']<=f['float32_bit_changed_scalar_values']<=f['float32_bit_changed_rows']*width
        assert np.isfinite(f['max_abs_decoded_difference']) and f['max_abs_decoded_difference']>=0
        assert (f['changed_scalar_values']==0)==(f['max_abs_decoded_difference']==0)
        ids=f['first_changed_model_local_IDs'];assert len(ids)<=16 and all(mask[x] for x in ids) and all(ids[i]<ids[i+1] for i in range(len(ids)-1))
    for aggregate,key in [('changed_rows_in_any_of_five_fields','changed_rows'),('float32_bit_changed_rows_in_any_of_five_fields','float32_bit_changed_rows')]:
        per=[f[key] for f in g['fields'].values()];assert max(per)<=g[aggregate]<=min(rows,sum(per))
for aggregate,parts2 in [('A_all',['anchor_leaf_A_intersection_L','anchor_internal_A_minus_L']),('L_all',['anchor_leaf_A_intersection_L','nonAnchor_leaf_L_minus_A'])]:
    assert stats[aggregate]['rows']==sum(stats[p]['rows'] for p in parts2)
    for field in widths:
        for key in ['changed_rows','changed_scalar_values','float32_bit_changed_rows','float32_bit_changed_scalar_values']:
            assert stats[aggregate]['fields'][field][key]==sum(stats[p]['fields'][field][key] for p in parts2)
        assert stats[aggregate]['fields'][field]['max_abs_decoded_difference']==max(stats[p]['fields'][field]['max_abs_decoded_difference'] for p in parts2)
assert stats['A_all']['changed_rows_in_any_of_five_fields']==stats['A_all']['float32_bit_changed_rows_in_any_of_five_fields']==0
assert stats['L_all']['changed_rows_in_any_of_five_fields']==stats['L_all']['float32_bit_changed_rows_in_any_of_five_fields']==0
assert stats['nonAnchor_internal']['rows']==1354085 and stats['nonAnchor_internal']['changed_rows_in_any_of_five_fields']==1052901
sh=stats['nonAnchor_internal']['fields']['sh'];assert sh['float32_bit_changed_scalar_values']-sh['changed_scalar_values']==144
for p,d in pins.items():assert sha(p)==d
for p,d in r['protected_files_before_and_after'].items():
    if not p.endswith(('.hier','.hier_opt')):assert sha(p)==d
assert physical['protected_source_and_output_before_after'][str(BASE/'official-hierarchy/small-city-hierarchy-006/trained_chunks/0_0/hierarchy.hier')]==r['before_sha256']
assert physical['protected_source_and_output_before_after'][str(BASE/'official-hierarchy/small-city-hierarchy-006/trained_chunks/0_0/hierarchy.hier_opt')]==r['after_sha256']
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
review={'status':'passed_narrow_post00_endpoint_d720_source_actual462_report_and_measured_group_set_CPU_peer_review',
 'UTC':datetime.now(timezone.utc).isoformat(),'protected_evidence_current_SHA':{str(p):d for p,d in pins.items()},
 'source_schema_AST_matches_aa4_and_codec_contract':True,'actual_four_partitions_disjoint_cover_original_tree':True,'actual_A_L_NPZ_membership_and_sorted_ID_hashes_verified':True,
 'actual_group_counts':{k:stats[k]['rows'] for k in groups},'actual_report_changed_nonAnchor_internal_rows':1052901,'actual_report_frozen_A_and_L_numeric_and_bits_changed_rows':0,
 'actual_report_SH_scalar_numeric_changes':sh['changed_scalar_values'],'actual_report_SH_scalar_bit_changes':sh['float32_bit_changed_scalar_values'],
 'bit_semantics':'Finite decodedfloat32 unequal values imply unequal bits. Extra144 SH bit differences are signed-zero distinctions under the implemented finite comparisons; this peer did not reread their model endpoint scalars.',
 'tail_window_scope':{'original_tree_G':count,'last_window_rows':count%65536,'after_G':count+100000,'shared_Sky_excluded':100000,'source_both_endpoint_and_mask_end_min_count':True},
 'numeric_algorithm_review':'Source compares finite decoded f32 directly, separately uint32 bits; row/scalar counts and OR-any masks are correct, maxima subtract in float64. No exp/sigmoid/quaternion normalization during comparison.',
 'prior_actual_physical_finite_and_model_SHA_evidence':str(PHYSICAL),'model_endpoint_arrays_or_full_model_SHA_rescanned_here':False,
 'numeric_counts_independently_recomputed_from_raw_model_here':False,'actual_report_count_bounds_aggregate_arithmetic_firstIDs_and_set_membership_verified':True,
 'interpretation':'Measured stored endpoints only, not optimizer call instrumentation, per-iteration updates or image/geometry quality improvement.14999 remains source-derived; halfquantized stored endpoints can hide sub-ULP/transient/cancelled updates.',
 'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'source_main_or_old_mutator_executed':False,'process_query_or_control_executed':False,'frozen_sources_or_reports_modified':False}
assert not review['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
