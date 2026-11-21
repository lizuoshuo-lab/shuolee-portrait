"""Preserve003 real tail-mask defect; peer-review separate fixed posebytes_v2 in004."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
src=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_003.py'
fail=ROOT/'outputs/large-scene-gs-001/recovery-audit/city006-anchor-boundary-perceptual-independent-review-CPU-003/failure.json'
assert not fail.exists();fail.write_text(json.dumps({'status':'CPU_peer_detected_real_future_post_ledger_tail_window_defect','actual_exit_code':1,
    'source_SHA':hashlib.sha256(src.read_bytes()).hexdigest(),'tested_anchor_source_SHA':'b275c765ce7aec4e96e49dabbc575761939105f2da237a2d9e51007085c27cb9',
    'error':'IndexError: after[b:b+65536] includes appended100kSky, but final mask has only pre-tree remainingR rows',
    'scope':'Actual CPU math fixture: before6/after100006. Creator-only recorded637/5144 did not execute this futureformalpost branch.',
    'native_or_GPU_algorithm_or_model_changed':False,'boundary_and_perceptual_AST_checks_passed_before_failure':True},indent=2)+'\n',encoding='utf-8')
s=src.read_text(encoding='utf-8').replace('city006-anchor-boundary-perceptual-independent-review-CPU-003','city006-anchor-boundary-perceptual-independent-review-CPU-004')
s=s.replace("'city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006.py'","'city-anchor-lineage-driver-CPU-006/audit_city_hierarchy_lineage_006_posebytes_v2.py'")
s=s.replace("'anchor':'b275c765ce7aec4e96e49dabbc575761939105f2da237a2d9e51007085c27cb9'","'anchor':'aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0'")
s=s.replace("['mapped','classify','xyz_multiset','creator_only']","['mapped','classify','xyz_multiset']")
s=s.replace("peer006-anchor-actualmissingformal-CPU-001","peer006-anchor-actualmissingformal-CPU-004")
s=s.replace("peer006-boundary-actualmissing1404-CPU-001","peer006-boundary-actualmissing1404-CPU-004")
s=s.replace("peer006-perceptual-actualmissing1404-CPU-001","peer006-perceptual-actualmissing1404-CPU-004")
s=s.replace("'binding_review':{", "'narrow_revision_review':{'actual_v2_posebyte_gate_and_tail_window_source_SHA':'aa4d620778231f1b9c405d95b4c6a04cf2af734eab5a8941817a0cfa0b408bb0','original_creator_classification_XYZ_numerical_AST_retained':True,'pose_archive_source_identity_strengthened':True,'before_after_leaf_window_clamped_to_pre_tree':True,'original_b275_source_Creator_evidence_and_prior_failure_preserved':True},'binding_review':{")
target=ROOT/'scripts/large_scene_gs/review_city006_anchor_boundary_perceptual_CPU_004.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
