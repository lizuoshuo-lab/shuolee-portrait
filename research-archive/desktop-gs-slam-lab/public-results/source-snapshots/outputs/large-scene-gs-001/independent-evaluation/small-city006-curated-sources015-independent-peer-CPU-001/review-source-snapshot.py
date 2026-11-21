"""Read-only015 curated source/reference/count-report peer, no mutator/GPU."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
sys.dont_write_bytecode=True
LAB=Path(__file__).resolve().parents[4];BASE=LAB/'outputs/large-scene-gs-001'
CURATED=BASE/'research/city-third-formalpost-resource-counts-015'
OUT=BASE/'independent-evaluation/small-city006-curated-sources015-independent-peer-CPU-001'
CHAPTER=LAB/'knowledge/30_large_scene_gs_sfm_partition_lod.md';MANIFEST=LAB/'manifests/large-scene-gs-work.json';SYNC=Path(r'D:\Projects\EngineLab\engine-home\scripts\sync-gs-lab.mjs')
CONTENT=Path(r'D:\Projects\EngineLab\engine-home\lib\gs-lab-content.json')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
pins={CURATED/'verified-stage-summary.json':'0810dcb5f742d7e90e8915ab10c324e6e654f3968d4dbf2e44042f892d6be7f8',MANIFEST:'ecfb078888bd80b14d06af50b8950d56a796998169170daaa51ebd965a08a1f0',CHAPTER:'a735bf36ff624391254d21200f85e618bf83ac4f5a09a0f681e0317ab9590f06',SYNC:'55d0d05ab550b57be79ea871933c06663a3221331f2a3b7dac4d32e72208d24d'}
assert all(sha(p)==d for p,d in pins.items())
source=CURATED/'update_verified_sources.py';protected={str(p):d for p,d in pins.items()};protected[str(source)]=sha(source)
summary=read(CURATED/'verified-stage-summary.json');update=read(CURATED/'source-update-and-continuation.json');refs=[]
def current_refs(v):
    if isinstance(v,dict):
        if set(v)=={'path','sha256'}:
            p=(LAB/v['path']).resolve();assert p.is_relative_to(LAB.resolve()) and p.is_file() and sha(p)==v['sha256'];refs.append(v)
        for x in v.values():current_refs(x)
    elif isinstance(v,list):
        for x in v:current_refs(x)
current_refs(summary)
assert update['summary']['sha256']==pins[CURATED/'verified-stage-summary.json'] and update['manifest']['sha256']==pins[MANIFEST] and update['chapter']['sha256']==pins[CHAPTER]
assert update['theory_text_unchanged'] and update['actual_City1404_frames']==0 and not update['GPU_started_by_curator'] and not update['raw_dataset_or_model_arrays_added_to_site']
assert summary['formalpost_completed_chunks']==['0_0','0_1','1_0'] and summary['formalpost_pending_chunks']==['1_1']
assert summary['creator_completed_chunks']==['0_0','0_1','1_0','1_1'] and summary['creators'][3]['training_source_kind']=='observed_local_official_training'
assert summary['training_lineage_kind']=='three_legacy_local_plus_one_observed_local_official_training' and not summary['formal_observation_reclassification']
assert not summary['original_NaN_cause_or_fix_proven'] and summary['actual_City1404_evaluation_frames']==0 and not summary['optimized_global_Merger_completed'] and not summary['City_full_post_merge_or_1404_complete']
assert not summary['raw_dataset_or_model_arrays_copied_to_site'] and not summary['post64']['eligible_for_formal_merge']
posts=summary['formalposts'];assert [p['chunk'] for p in posts]==['0_0','0_1','1_0'];third=posts[2]
audit=read(LAB/third['independent_completion_CPU']['path']);native=read(LAB/third['receipt']['path']);terminal=read(LAB/third['independent_terminal_peer']['path'])
assert third['independent_completion_CPU']['sha256']=='d14a5197745d0552a7fd175a4aa65825a9c2437e8d185d4d49cc8679792e5962'
assert third['independent_terminal_peer']['sha256']=='bd92c7ba9fdc2db6819dfe27fcff9b48e06b5f677f06f319cd0bf0d4ca0e52d2'
assert native['status']=='completed' and native['requested_iterations']==15000 and native['exit_code']==native['actual_child_poll_exit_code']==0
assert third['output_SHA256']==native['hierarchy_sha256']=='1252510e495e94bd2fdc265b534defc2cf3103aba62d5c7b0cd7a98b9f781851'
for k,h in [('G','gaussian_count'),('nodes','node_count'),('leaves','leaf_gaussian_count'),('internal_Gaussians','merged_gaussian_count')]:assert third[k]==native['hierarchy'][h]==audit['actual_full15000_hierarchy'][h]
assert third['G']==4294329 and third['nodes']==4194329 and third['leaves']==2850099 and third['anchor_count']==audit['actual_anchor_count']==2850201 and third['shared_sky_tail']==100000
assert third['Gaussian_calls_source_derived_not_instrumented']==14999 and native['optimizer_call_count_source_derived'] and not audit['optimizer_call_count_instrumented']
assert audit['actual_leaf_count']==2850099 and audit['actual_A_minus_L_count']==102
for key in ['actual_A_raw5fields','actual_L_raw5fields']:
    assert set(audit[key])=={'positions','rotations','log_scales','opacities','sh'}
    assert all(x['decoded_float32_values_exact'] and x['float32_bits_equal'] and x['max_abs_delta']==0 for x in audit[key].values())
assert audit['node_and_box_compressed_raw_bytes_exact'] and third['original_node_and_box_compressed_raw_bytes_exact']
assert all(x['within_allowed_tolerance'] for x in audit['single_shared100kSky_halfwriter_roundtrip']['skybox_fields'].values())
assert 'not blanket bit-exactness' in third['sky_storage_scope'] and audit['frozen_original_train_exposure']['matrix_count']==third['frozen_trained_exposure_matrix_count']==1474 and audit['frozen_original_train_exposure']['test_overlap']==0
assert third['logged_Torch_allocated_peak_bytes']==audit['official_logged_peak_allocated_bytes']==7657206784 and third['actual_native_process_seconds']==native['process_seconds']
assert third['training_source_kind']=='legacy_local_official_training' and not third['pilot64_used_as_input'] and not third['image_quality_improvement_or_LOD_speedup_measured']
resource=summary['unoptimized117_CPU_cut_resource_accounting'];count=read(LAB/resource['independent_CPU_report']['path']);count_peer=read(LAB/resource['independent_source_and_actual_report_peer']['path'])
assert resource['independent_CPU_report']['sha256']=='e58e69fb8f2dde7ace15e31c3273301401b1645d1909a91a796a7ab8db0ea2d1' and resource['independent_source_and_actual_report_peer']['sha256']=='1744c2b2958e9c69c97738bd482fc47e4bc427da2f3623f2f612108762be78d4'
assert count_peer['status']=='passed_narrow_static_source_and_actual702_CPU_count_report_independent_peer_review'
assert resource['camera_count']==count['camera_count']==117 and resource['CPU_predicate_count_records']==len(count['CPU_count_records'])==702
settings=['leaves','tau1','tau3','tau5','tau6','tau15'];assert [s['setting'] for s in resource['settings']]==settings
assert count['actual_rendered_frames']==count['actual_City1404_frames']==resource['actual_rendered_frames']==0 and not count['CUDA_initialized']
assert not resource['native_GPU_cut_counts_verified'] and resource['native_workspace_upper_bound'] is None and not resource['actual_GPU_fit_or_peak_or_quality_or_speed_measured']
assert not count['actual_GPU_memory_peak_or_fit_guaranteed'] and not count['compact_leaf_enabled_or_streaming']
keys=[(row['camera_index'],row['setting']) for row in count['CPU_count_records']];assert len(keys)==len(set(keys))==702 and set(keys)=={(i,s) for i in range(117) for s in settings}
stats=[]
for item in resource['settings']:
    rows=[r for r in count['CPU_count_records'] if r['setting']==item['setting']];values=[r['CPU_predicate_selected_tree_G'] for r in rows]
    assert len(rows)==117 and len({r['name'] for r in rows})==117 and item['minimum_tree_G']==min(values) and item['maximum_tree_G']==max(values)
    original=next(s for s in count['setting_statistics'] if s['setting']==item['setting'])
    assert item['known_Python_tensor_peak_upper_bytes_at_CPU_max']==original['known_tensor_memory_plan_at_actual_CPU_max']['conservative_Python_tensor_peak_upper_bytes']
    assert all(r['one_shared_Sky_G_for_future_render']==100000 and r['CPU_predicted_total_render_G_including_Sky']==r['CPU_predicate_selected_tree_G']+100000 for r in rows)
    stats.append(dict(setting=item['setting'],min=min(values),max=max(values),known_tensor_peak_upper_bytes=item['known_Python_tensor_peak_upper_bytes_at_CPU_max']))
assert [(s['min'],s['max']) for s in stats]==[(4643252,4643252),(1589427,2080489),(689009,1058511),(402380,636796),(331771,514933),(95499,180295)]
assert resource['actual_unoptimized_tree_SHA256']==summary['actual_unoptimized_global_merge']['output_SHA256']==count['protected_source_model_and_input_SHA_before_after'][str(BASE/'official-hierarchy/small-city-hierarchy-006/merged-unoptimized.hier')]
assert resource['canonical117_archive_SHA256']=='60ab56f705c6db347b00f821b08a44e49c0d2dca4f0f06e9a978b9c73c8bce25'
assert 'not native GPU selection validation' in resource['scope']
manifest=read(MANIFEST);old_manifest=read(CURATED/'manifest-before.json');city=next(x for x in manifest['milestones'] if x['id']=='public-large-scene-extension');old_city=next(x for x in old_manifest['milestones'] if x['id']=='public-large-scene-extension')
for key in ['local_formal_completed_chunks','local_formal_chunk11_failed','local_observation_only_diagnostic','optimizer_call_count_erratum','local_mixed_training_lineage_CPU_binding','local_mixed_training_valid_chunk_inputs']:assert city[key]==old_city[key]
assert city['local_formal_completed_chunks']==['0_0','0_1','1_0'] and city['local_formal_chunk11_failed']['nonfinite_gaussians']==1 and not city['local_formal_chunk11_failed']['computational_cause_known']
assert city['local_observation_only_diagnostic']['actual_GaussianAdam_calls']==29951 and city['local_observation_only_diagnostic']['actual_opacity_grad_None_skips']==48 and not city['local_observation_only_diagnostic']['original_NaN_cause_or_fix_proven']
assert city['local_hierarchy006_actual_formalposts']==posts and city['local_hierarchy006_formalpost_completed_chunks']==summary['formalpost_completed_chunks'] and city['local_hierarchy006_formalpost_pending_chunks']==['1_1']
assert city['local_hierarchy006_unoptimized117_CPU_cut_resources']==resource and city['local_hierarchy006_actual_City1404_evaluation_frames']==0 and not city['local_hierarchy006_optimized_global_Merger_completed']
before=(CURATED/'chapter-before.md').read_bytes();now=CHAPTER.read_bytes();marker='## 实验附录'.encode();assert before.split(marker)[0]==now.split(marker)[0]
chapter=now.decode('utf-8-sig');assert chapter.count('## 实验附录')==1
for text in ['前三块现已分别完成完整 15,000 次层级优化','| 1_0 | 4,294,329 | 27.91 分钟 | 7.13 GiB |','689,009—1,058,511','约为 3.68 GiB','还没有核验原生 GPU','不能作为完成城市 LOD 渲染评测的成绩','不能证明实际 GPU 峰值','最后一块完整优化、优化后的全景合并','没有复现原 NaN']:assert text in chapter
old_sync=(CURATED/'sync-before.mjs').read_text(encoding='utf-8');sync=SYNC.read_text(encoding='utf-8')
line="  ['outputs/large-scene-gs-001/research/city-third-formalpost-resource-counts-015/verified-stage-summary.json', '城市前三块完整层级优化与留出视角选点资源核验摘要'],"
assert sync.count(line)==1 and sync.replace('\n'+line,'')==old_sync
# Read actual exported markup and previously written root QA artifacts, without
# executing their updater/HTTP-check scripts or fetching pages ourselves.
content=read(CONTENT);doc=next(d for d in content['documents'] if d['slug']=='large-scene-gs-sfm-partition-lod');main=re.split(r'<h2[^>]*>实验附录</h2>',doc['html'])[0]
math_count=len(re.findall(r'<math\b',main));figure_count=len(re.findall(r'<img\b',main));assert math_count==100 and figure_count==6
old_qa_path=BASE/'research/city-first-formalpost-unoptimized-merge-014/served-update-check.json';new_qa_path=CURATED/'served-update-check.json';old_qa=read(old_qa_path);new_qa=read(new_qa_path)
assert old_qa['mathCount']==new_qa['mathCount']==math_count and old_qa['completePaperFigures']==new_qa['completePaperFigures']==figure_count
assert new_qa['HTTP']==200 and new_qa['selfDrawnFiguresAdded']==0 and not new_qa['allFourFormalpostsOrPairedLODEvaluationClaimed'] and not new_qa['originalNaNCauseOrFixProven'] and not new_qa['rawDatasetsOrModelsCopied']
for item in new_qa['sourcesAndPortableBytesVerified']:
    if item['path'] in ['manifests/large-scene-gs-work.json','outputs/large-scene-gs-001/research/city-third-formalpost-resource-counts-015/verified-stage-summary.json']:assert sha(LAB/item['path'])==item['sourceSHA256']
assert not any('/small-city-hierarchy-006/' in r['path'] or r['path'].endswith(('.hier','.ply','.npz')) for r in content['resources'])
assert all(sha(p)==d for p,d in protected.items())
OUT.mkdir(exist_ok=False);shutil.copyfile(__file__,OUT/'review-source-snapshot.py')
report={'status':'passed_readonly015_three_actual_posts_CPU117_counts_current_refs_limits_and_unchanged_theory_MathML_peer','UTC':datetime.now(timezone.utc).isoformat(),
 'protected_current_source_SHA':protected,'actual_curated_ref_checks_count':len(refs),'all_curated_ref_SHA_current':True,'actual_completed_post_chunks':['0_0','0_1','1_0'],'actual_pending_post_chunks':['1_1'],
 'third_actual_A2850201_L2850099_all5_numeric_and_bits_measured_true':True,'third_original_node_box_raw_bytes_measured_true':True,'third_Sky_roundtrip_tolerance_not_blanket_bitexact':True,
 'CPU_only_expected_counts_117x6':702,'actual_rendered_or1404_frames':0,'independent_recomputed_count_report_min_max':stats,'native_GPU_counts_or_fit_or_peak_or_quality_or_speed_verified':False,
 'original_NaN_failure_and_observed_mixed_identity_retained':True,'actual_source_derived14999_call_qualifier_retained':True,'same_fullresident_model_and_native_workspace_unknown_qualified':True,
 'theory_raw_prefix_bytes_equal_015_before':True,'new_substantive_content_appendix_only':True,'current_exported_main_MathML_count':math_count,'previous014_QA_MathML_count':old_qa['mathCount'],'current_exported_complete_paper_figures':figure_count,
 'MathML_scope':'Theory input bytes unchanged and actual current100 MathML nodes match old014/new015 rootQA counts; prior full MathML payload byte snapshot not independently compared here.',
 'root_actual_served_QA_artifact_path':str(new_qa_path),'root_actual_served_QA_artifact_SHA':sha(new_qa_path),'root_actual_served_QA_HTTP':new_qa['HTTP'],'own_HTTP_QA_script_reexecuted':False,
 'source_mutator_executed_here':False,'training_or_model_or_website_source_edited':False,'Torch_imported':'torch' in sys.modules,'CUDA_called':False,'model_arrays_or_full_model_payloads_rescanned':False,'process_query_or_control_performed':False,
 'scope':'Curated source/current report reference and arithmetic/markup evidence review only. Fourth active post is not complete; no paired post-Merger/City GPU quality completion claimed.'}
assert not report['Torch_imported']
(OUT/'peer-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'report_path':str(OUT/'peer-review.json'),'report_sha256':sha(OUT/'peer-review.json')}))
