"""Preserve002's unsupported probability bound; audit official unbounded abs opacity instead."""
from pathlib import Path
import hashlib,json
ROOT=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
source=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_002.py'
s=source.read_text(encoding='utf-8')
failure=ROOT/'outputs/large-scene-gs-001/recovery-audit/small-city-post006-i64-completion-CPU-002/failure.json'
assert not failure.exists()
failure.write_text(json.dumps({'status':'independent_CPU_audit_failed_preserved','actual_exit_code':1,
    'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'error':'Activated alpha outside[0,1] original_Creator',
    'unsupported_audit_assumption':'002 added probability <=1 bound.Official hierarchical post uses abs activation and ClusterMerger parent opacity=weight_sum/ellipseSurface, with no <=1 bound.',
    'official_source_evidence':{'post':'scene/gaussian_model.py:388,393','native':'ClusterMerger.cpp:51,133'},
    'original_finite_or_anchor_or_sky_gate_relaxed':False,'model_or_receipt_modified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=s.replace('small-city-post006-i64-completion-CPU-002','small-city-post006-i64-completion-CPU-003')
s=s.replace("count=len(h.positions);proof={'all_fields_finite':True,'count':count}","count=len(h.positions);proof={'all_fields_finite':True,'count':count}; above_one=0; alpha_min=float('inf');alpha_max=float('-inf')")
s=s.replace("alpha=h.opacities[b:b+65536];require(((alpha>=0)&(alpha<=1)).all(),'Activated alpha outside[0,1] '+hname)",
"alpha=h.opacities[b:b+65536];require((alpha>=0).all(),'Negative stored opacity inconsistent with official abs activation '+hname)\n        above_one+=int(np.count_nonzero(alpha>1));alpha_min=min(alpha_min,float(np.min(alpha)));alpha_max=max(alpha_max,float(np.max(alpha)))")
s=s.replace("proof.update(exp_f32_scales_finite_positive=True,quaternion_norm_nonzero=True,activated_alpha_in_0_1=True)",
"proof.update(exp_f32_scales_finite_positive=True,quaternion_norm_nonzero=True,stored_opacity_finite_nonnegative=True,opacity_above_one_count=above_one,opacity_min=alpha_min,opacity_max=alpha_max,opacity_scope='Official hierarchical abs activation, not a sigmoid probability bound')")
s=s.replace("report={'status':", "opacity_source=ROOT/'repos/hierarchical-3d-gaussians/scene/gaussian_model.py'\nmerger_source=ROOT/'repos/gaussian-hierarchy-large-gs/ClusterMerger.cpp'\ncanonical=subprocess.check_output(['git','-C',str(d.NATIVE_REPO),'show',d.NATIVE_COMMIT+':ClusterMerger.cpp'])\nrequire(merger_source.read_bytes().replace(b'\\r\\n',b'\\n')==canonical.replace(b'\\r\\n',b'\\n'),'Official ClusterMerger source changed')\nrequire(sha(opacity_source)=='7a51ca481fbfa387db6ce85718dee89528369579e9680b2d249e17cf8d87573c','Official post opacity source changed')\nreport={'opacity_semantics':{'post_activation':'abs, unbounded above','post_source_sha256':sha(opacity_source),'native_merged_opacity':'weight_sum / ellipseSurface(merged_scale)','native_source_sha256':sha(merger_source),'no_original_finite_gate_relaxed':True},'status':")
target=ROOT/'scripts/large_scene_gs/audit_small_city_post006_completion_CPU_003.py'
assert not target.exists();compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print(target)
