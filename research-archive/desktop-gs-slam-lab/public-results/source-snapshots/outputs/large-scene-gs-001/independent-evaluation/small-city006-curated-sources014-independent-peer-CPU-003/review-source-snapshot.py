"""Independent readonly014 source/curated-data peer; never run its mutator."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
sys.dont_write_bytecode=True
LAB=Path(__file__).resolve().parents[4];BASE=LAB/'outputs/large-scene-gs-001'
CURATED=BASE/'research/city-first-formalpost-unoptimized-merge-014'
NATIVE=BASE/'official-hierarchy/small-city-hierarchy-006'
OUT=BASE/'independent-evaluation/small-city006-curated-sources014-independent-peer-CPU-003'
CHAPTER=LAB/'knowledge/30_large_scene_gs_sfm_partition_lod.md';MANIFEST=LAB/'manifests/large-scene-gs-work.json'
SYNC=Path(r'D:\Projects\EngineLab\engine-home\scripts\sync-gs-lab.mjs')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={CURATED/'verified-stage-summary.json':'b3ce18255b54dbb41ecf6f53112cbfac9b00a9346a0e9558c74d7d760fed867f',MANIFEST:'572ce2ddd167d7542936fd5825dcdabafc8cb15187ff3b510a8c1a6468ab8beb',CHAPTER:'0cea1744acbcede463aaa788fd867158aebc0ee986edfc7d3c1040479bc14b75',SYNC:'a68d9d14db21137821410fe785785f54532cfe990ace259dfee81eb410320dc5'}
assert all(sha(p)==d for p,d in pins.items())
source=CURATED/'update_verified_sources.py';protected={str(p):d for p,d in pins.items()};protected[str(source)]=sha(source)
summary=read(CURATED/'verified-stage-summary.json');receipt=read(CURATED/'source-update-and-continuation.json')
assert receipt['summary']['sha256']==pins[CURATED/'verified-stage-summary.json'] and receipt['manifest']['sha256']==pins[MANIFEST] and receipt['chapter']['sha256']==pins[CHAPTER] and receipt['sync_sha256']==pins[SYNC]
assert receipt['theory_unchanged'] and not receipt['raw_data_or_models_added'] and not receipt['GPU_started'] and not receipt['next_stage_dispatched']
refs=[]
def refs_current(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            p=(LAB/value['path']).resolve();assert p.is_relative_to(LAB.resolve()) and p.is_file() and sha(p)==value['sha256'];refs.append(value)
        for v in value.values():refs_current(v)
    elif isinstance(value,list):
        for v in value:refs_current(v)
refs_current(summary)
assert summary['creator_completed_chunks']==['0_0','0_1','1_0','1_1'] and summary['formalpost_completed_chunks']==['0_0','0_1'] and summary['formalpost_pending_chunks']==['1_0','1_1']
assert summary['training_lineage_kind']=='three_legacy_local_plus_one_observed_local_official_training' and not summary['formal_observation_reclassification']
assert [x['training_source_kind'] for x in summary['creators']]==['legacy_local_official_training']*3+['observed_local_official_training']
assert not summary['original_NaN_cause_or_fix_proven'] and not summary['City_full_post_merge_or_1404_complete'] and not summary['optimized_global_Merger_completed'] and summary['actual_City1404_evaluation_frames']==0
assert not summary['raw_dataset_or_model_arrays_copied_to_site'] and not summary['post64']['eligible_for_formal_merge']
posts=summary['formalposts'];actual=[];nodes_box_bits=[]
for p in posts:
    chunk=p['chunk'];r=read(LAB/p['receipt']['path']);a=read(LAB/p['independent_completion_CPU']['path']);creator=read(NATIVE/'trained_chunks'/chunk/'creator-outcome.json')
    assert r['status']=='completed' and r['exit_code']==r['actual_child_poll_exit_code']==0 and r['source_kind']=='post_optimized' and r['requested_iterations']==15000
    assert p['Gaussian_calls_source_derived_not_instrumented']==14999 and r['optimizer_call_count_source_derived'] and not a['optimizer_call_count_instrumented']
    assert r['exposure_optimizer_updates']==p['exposure_optimizer_calls']==0 and not r['depth_loss_present_in_unchanged_post']
    assert p['G']==r['hierarchy']['gaussian_count']==a['actual_full15000_hierarchy']['gaussian_count'] and p['nodes']==r['hierarchy']['node_count'] and p['leaves']==r['hierarchy']['leaf_gaussian_count']
    assert p['internal_Gaussians']==r['hierarchy']['merged_gaussian_count'] and p['shared_sky_tail']==r['hierarchy']['unowned_tail']==100000
    assert p['output_SHA256']==r['hierarchy_sha256'] and p['original_Creator_tree_SHA256']==creator['hierarchy_sha256']
    assert p['anchor_count']==a['actual_anchor_count'] and all(x['decoded_float32_values_exact'] and x['float32_bits_equal'] for x in a['actual_A_raw5fields'].values())
    assert p['frozen_trained_exposure_matrix_count']==a['frozen_original_train_exposure']['matrix_count'] and a['frozen_original_train_exposure']['test_overlap']==0
    assert p['actual_native_process_seconds']==r['process_seconds'] and p['logged_Torch_allocated_peak_bytes']==a['official_logged_peak_allocated_bytes']
    assert p['memory_scope']=='Logged PyTorch allocated peak, not native allocator or whole-device instantaneous peak'
    assert not p['image_quality_improvement_or_LOD_speedup_measured'] and not p['pilot64_used_as_input']
    # Independently substantiate the stronger public bit-equality claim by
    # reading only node/box byte tails, not five Gaussian arrays or all5 models.
    paths=[NATIVE/'trained_chunks'/chunk/n for n in ['hierarchy.hier','hierarchy.hier_opt']];headers=[]
    for path in paths:
        with path.open('rb') as f:
            signed=struct.unpack('<i',f.read(4))[0];assert signed<0;g=-signed;f.seek(4+124*g);n=struct.unpack('<i',f.read(4))[0]
        assert path.stat().st_size==8+124*g+36*n;headers.append((g,n,8+124*g,path.stat().st_size,path.stat().st_mtime_ns))
    assert headers[0][1]==headers[1][1]==p['nodes'] and headers[1][0]==headers[0][0]+100000
    hashes=[hashlib.sha256(),hashlib.sha256()];remaining=36*p['nodes']
    with paths[0].open('rb') as x,paths[1].open('rb') as y:
        x.seek(headers[0][2]);y.seek(headers[1][2])
        while remaining:
            size=min(remaining,2*1024**2);xb=x.read(size);yb=y.read(size);assert len(xb)==len(yb)==size and xb==yb
            hashes[0].update(xb);hashes[1].update(yb);remaining-=size
    assert all(path.stat().st_size==h[3] and path.stat().st_mtime_ns==h[4] for path,h in zip(paths,headers))
    nodes_box_bits.append({'chunk':chunk,'raw_node_box_tail_bytes':36*p['nodes'],'before_path':str(paths[0]),'after_path':str(paths[1]),'before_and_after_range_SHA256':hashes[0].hexdigest(),'raw_node_and_box_storage_bits_equal':True,'scope':'Only compressed20Bnode+16Bbox per node, byte-range compared; no fullGaussian payload rescan or new optimizer/quality audit.'})
    actual.append({'chunk':chunk,'G':p['G'],'nodes':p['nodes'],'leaves':p['leaves'],'seconds':p['actual_native_process_seconds'],'torch_allocated_peak_bytes':p['logged_Torch_allocated_peak_bytes']})
assert 'inherited mixed_origin_scope text says originalchunk00' in posts[1]['physical_audit_scope_description_erratum']
assert 'actual audited path, origin index, Creator source, argv and A/L IDs are chunk0_1' in posts[1]['physical_audit_scope_description_erratum']
endpoint=summary['post00_parameter_endpoints'];parameter=read(LAB/endpoint['independent_CPU_report']['path']);assert endpoint['unanchored_internal_G']==parameter['group_statistics']['nonAnchor_internal']['rows']==1354085
assert endpoint['unanchored_internal_changed_any_five_fields']==parameter['group_statistics']['nonAnchor_internal']['changed_rows_in_any_of_five_fields']==1052901
assert parameter['group_statistics']['A_all']['float32_bit_changed_rows_in_any_of_five_fields']==parameter['group_statistics']['L_all']['float32_bit_changed_rows_in_any_of_five_fields']==0
assert 'not optimizer call count' in endpoint['scope'] and 'image quality' in endpoint['scope']
merge=summary['actual_unoptimized_global_merge'];mr=read(LAB/merge['receipt']['path']);binder=read(LAB/merge['independent_CPU_binding']['path'])
assert mr['status']=='completed' and mr['exit_code']==mr['actual_child_poll_exit_code']==0 and mr['source_kind']=='unoptimized'
assert merge['output_SHA256']==mr['hierarchy_sha256']==binder['actual_merged_model_sha256']=='e92827534bef94f0f2e213ce31508570c6f6da5f24a59cf62da20793d2276e78'
for k,h in [('G','gaussian_count'),('nodes','node_count'),('leaves','leaf_gaussian_count'),('internal_Gaussians','merged_gaussian_count'),('root_depth','root_depth')]:assert merge[k]==mr['hierarchy'][h]==binder['actual_merged_counts'][h]
assert merge['shared_sky_tail']==mr['skybox_in_tree']==0 and merge['render_Sky_to_append_exactly_once']==100000 and not merge['post64_or_post15000_used_as_input']
assert merge['known_full_tree_tensor_fixed_resident_bytes']==binder['full_tree_explicit_leaf_memory_accounting']['known_fixed_resident_bytes']==2520680520
assert merge['known_explicit_leaf_Python_tensor_peak_upper_bytes']==binder['full_tree_explicit_leaf_memory_accounting']['conservative_Python_tensor_peak_upper_bytes']==7624584800
assert merge['known_all_storage_cut_Python_tensor_peak_upper_bytes']==binder['full_tree_all_storage_cut_upper_accounting']['conservative_Python_tensor_peak_upper_bytes']==10038300256
assert merge['native_workspace_upper_bound'] is None and not merge['actual_GPU_peak_or_fit_proven'] and not merge['compact_leaf_enabled_or_streaming']
manifest=read(MANIFEST);old_manifest=read(CURATED/'manifest-before.json');city=next(x for x in manifest['milestones'] if x['id']=='public-large-scene-extension');old_city=next(x for x in old_manifest['milestones'] if x['id']=='public-large-scene-extension')
for key in ['local_formal_completed_chunks','local_formal_chunk11_failed','local_observation_only_diagnostic','optimizer_call_count_erratum','local_mixed_training_lineage_CPU_binding','local_mixed_training_valid_chunk_inputs']:assert city[key]==old_city[key]
assert city['local_formal_completed_chunks']==['0_0','0_1','1_0'] and city['local_formal_chunk11_failed']['nonfinite_gaussians']==1 and not city['local_formal_chunk11_failed']['computational_cause_known']
assert city['local_observation_only_diagnostic']['actual_GaussianAdam_calls']==29951 and city['local_observation_only_diagnostic']['actual_opacity_grad_None_skips']==48 and not city['local_observation_only_diagnostic']['original_NaN_cause_or_fix_proven']
assert city['local_hierarchy006_actual_formalposts']==posts and city['local_hierarchy006_formalpost_completed_chunks']==summary['formalpost_completed_chunks'] and city['local_hierarchy006_formalpost_pending_chunks']==summary['formalpost_pending_chunks']
assert city['local_hierarchy006_unoptimized_global_merge']==merge and city['local_hierarchy006_post00_parameter_endpoints']==endpoint and city['local_hierarchy006_actual_City1404_evaluation_frames']==0 and not city['local_hierarchy006_optimized_global_Merger_completed']
assert city['local_hierarchy006_verified_stage_summary']==receipt['summary']
before=(CURATED/'chapter-before.md').read_text(encoding='utf-8-sig');chapter=CHAPTER.read_text(encoding='utf-8-sig')
assert chapter.count('## 实验附录')==before.count('## 实验附录')==1
assert chapter.split('## 实验附录')[0]==before.split('## 实验附录')[0]
raw_main_equal=CHAPTER.read_bytes().split('## 实验附录'.encode())[0]==(CURATED/'chapter-before.md').read_bytes().split('## 实验附录'.encode())[0]
assert not raw_main_equal and CHAPTER.read_bytes().split('## 实验附录'.encode())[0].replace(b'\r\n',b'\n')==(CURATED/'chapter-before.md').read_bytes().split('## 实验附录'.encode())[0]
for phrase in ['原因已查明或问题已修复','29,951','48 次','未逐次观测','不能代替图像质量验证','不能证明实际 GPU 峰值','优化后的全景合并','117 个留出视角','城市合并图像与 LOD 评测尚未生成']:
    assert phrase in chapter
for p in posts:
    assert f"{p['actual_native_process_seconds']/60:.2f} 分钟" in chapter and f"{p['logged_Torch_allocated_peak_bytes']/1024**3:.2f} GiB" in chapter
assert '7.10 GiB、9.35 GiB' in chapter and '2.35 GiB' in chapter
sync=SYNC.read_text(encoding='utf-8');old_sync=(CURATED/'sync-before.mjs').read_text(encoding='utf-8')
line="  ['outputs/large-scene-gs-001/research/city-first-formalpost-unoptimized-merge-014/verified-stage-summary.json', '城市四块层级、前两块完整优化与未优化全景合并的核验摘要'],"
assert sync.count(line)==1 and sync.replace('\n'+line,'')==old_sync
assert all(sha(p)==d for p,d in protected.items())
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
review={'status':'passed_independent_CPU_readonly_014_curated_sources_current_refs_numbers_limits_and_main_unchanged_peer','UTC':datetime.now(timezone.utc).isoformat(),
 'protected_current_source_SHA':protected,'current_summary_ref_checks_count':len(refs),'all_curated_report_ref_SHA_current':True,'actual_completed_formalposts':actual,'actual_unoptimized_G':merge['G'],'actual_unoptimized_N':merge['nodes'],'actual_unoptimized_L':merge['leaves'],
 'actual_node_and_box_storage_bit_equality_supplement':nodes_box_bits,'actual_endpoint_report_and_peer_462_13de_refs_current':True,'post01_description_erratum_appended_only':True,'old_0ecd_c4e04_sources_and_reports_protected':True,
 'main_theory_raw_prefix_bytes_equal_original_snapshot':raw_main_equal,'substantive_content_changed_only_in_appendix':True,'main_theory_content_bytes_equal_after_CRLF_normalization':True,'main_line_ending_note':'Only384main-line LF endings become CRLF; no words/math changed. Rawbyte equality is false, not claimed.','prior_peer_failed_attempts_preserved':['small-city006-curated-sources014-independent-peer-CPU-001','small-city006-curated-sources014-independent-peer-CPU-002'],'sync_diff_only_one_curated_summary_whitelist_entry':True,
 'original_NaN_failure_and_mixed_observation_identity_retained':True,'Gaussian_call_counts_qualified_source_derived_not_observed':True,
 'known_memory_upper_limits_and_native_unknown_qualified':True,'paired_optimized_merge_not_complete':True,'actual_City1404_frames':0,'no_quality_LOD_speedup_or_GPU_fit_claim':True,
 'source_mutator_executed_here':False,'site_or_training_sources_written_here':False,'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'process_control_or_query_performed':False,
 'full_model_parameter_payload_or_allfive_arrays_rescanned':False,'serving_export_sync_build_HTTP_QA_verified_here':False,
 'scope':'Current local curated source evidence only. Root export/sync/build/served QA is separate; this peer neither publishes nor marks unexecuted GPU evaluation complete.'}
assert not review['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':review['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
