"""One-shot local website curation of completed four posts and paired merges."""
from pathlib import Path
from datetime import datetime, timezone
import copy, hashlib, json, shutil

LAB = Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
BASE = LAB / 'outputs/large-scene-gs-001'
OUT = BASE / 'research/city-four-post-paired-merges-016'
OLD = BASE / 'research/city-third-formalpost-resource-counts-015'
LOCK = BASE / 'official-hierarchy/small-city-hierarchy-006'
SUPROOT = BASE / 'official-hierarchy-stage-supervision-006-revision-002'
RECOVERY = BASE / 'recovery-audit'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def ref(p): return dict(path=Path(p).relative_to(LAB).as_posix(), sha256=sha(p))
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists(), 'Never replay this curation'
    prior=OLD/'verified-stage-summary.json'
    assert sha(prior)=='0810dcb5f742d7e90e8915ab10c324e6e654f3968d4dbf2e44042f892d6be7f8'
    summary=copy.deepcopy(read(prior))
    postpath=LOCK/'trained_chunks/1_1/post-outcome.json'
    post=read(postpath)
    physicalpath=RECOVERY/'city006-formalpost11-independent-completion-CPU-001/completion-audit.json'
    physical=read(physicalpath)
    assert sha(physicalpath)=='10c4f705a7d135474e6d9006dcfcdf9bf0d127ab36a8ab206669be149567e9d2'
    assert physical['chunk']=='1_1' and physical['requested_iterations']==15000
    assert all(v['float32_bits_equal'] for v in physical['actual_A_raw5fields'].values())
    assert all(v['float32_bits_equal'] for v in physical['actual_L_raw5fields'].values())
    assert physical['node_and_box_compressed_raw_bytes_exact'] and physical['nodes_and_stored_boxes_exact']
    assert all(physical['independent_physical_numeric']['formalpost15000'][k] for k in
        ('all5_fields_finite','exp_f32_scales_finite_positive','quaternion_norm_nonzero','stored_abs_kappa_finite_nonnegative'))
    assert all(v['within_allowed_tolerance'] for v in physical['single_shared100kSky_halfwriter_roundtrip']['skybox_fields'].values())
    assert not physical['observation_reclassified_formal'] and not physical['pilot64_used_as_input']
    assert physical['frozen_original_train_exposure']['test_overlap']==0
    assert post['training_source_kind']=='observed_local_official_training'
    assert post['status']=='completed' and post['requested_iterations']==15000 and post['optimizer_update_calls']==14999
    assert post['hierarchy_sha256']=='7b0d79d341ace6ef42e15e5e6b9029a1212fb269b8ddddbbb59ffe3a0d271e55'
    post_sup=SUPROOT/'city006-post11-i15000-r2-stage-001/supervisor-outcome.json'
    post_terminal=RECOVERY/'stage006-formalpost11-supervision-terminal-CPU-001/terminal-peer.json'
    assert sha(post_terminal)=='972eca26cb532e9cae93473c8f0f4e9d85fd75d09dc344d475b46ff6200b0d71'
    for path in (post_sup,SUPROOT/'city006-merge-optimized-r2-stage-001/supervisor-outcome.json'):
        s=read(path)
        assert s['status']=='completed_one_actual_stage_only' and s['child_handle_terminal'] and s['child_exit_code']==0
        assert s['GPU_stage_processes_released'] and s['active_stage_lock_released']
    fourth=copy.deepcopy(summary['formalposts'][-1])
    h=post['hierarchy']
    fourth.update(chunk='1_1',training_source_kind=post['training_source_kind'],
        receipt=ref(postpath),supervision_receipt=ref(post_sup),independent_completion_CPU=ref(physicalpath),
        independent_terminal_peer=ref(post_terminal),
        original_Creator_tree_SHA256=read(LOCK/'trained_chunks/1_1/creator-outcome.json')['hierarchy_sha256'],
        output_SHA256=post['hierarchy_sha256'],G=h['gaussian_count'],nodes=h['node_count'],
        leaves=h['leaf_gaussian_count'],internal_Gaussians=h['merged_gaussian_count'],anchor_count=physical['actual_anchor_count'],
        frozen_trained_exposure_matrix_count=physical['frozen_original_train_exposure']['matrix_count'],
        actual_native_process_seconds=post['process_seconds'],logged_Torch_allocated_peak_bytes=physical['official_logged_peak_allocated_bytes'],
        inherited_opacity_symbol_metadata_erratum=ref(RECOVERY/'city006-formalpost11-symbol-metadata-errata-CPU-001/metadata-erratum.json'))
    ledger=BASE/'independent-evaluation/small-city006-four-formalpost-anchor-source-exposure-CPU-001/report.json'
    assert sha(ledger)=='fb001632de408b4fe913a43c3f42c58f9c90bff1f9fa04d23a4fee756f221ef8'
    assert read(ledger)['status']=='completed_actual_four006_I15000_post_anchor_exposure_CPU_ledger'
    mergedpath=LOCK/'merge-outcome.json'
    merged=read(mergedpath)
    terminalpath=RECOVERY/'stage006-optimized-merger-supervision-terminal-CPU-001/terminal-peer.json'
    terminal=read(terminalpath)
    assert sha(terminalpath)=='f76bd9c0482e9ecab080b0cd48a3980753e42be022e23ab338fb79af81d4b1b9'
    assert terminal['status']=='actual_optimized_merger_source_and_owned_terminal_audit_passed'
    assert terminal['all4_actual_I15000_post_inputs_only'] and terminal['selected_owned_lock_released_by_actual_supervisor']
    assert merged['status']=='completed' and merged['source_kind']=='post_optimized'
    assert merged['exit_code']==merged['actual_child_poll_exit_code']==0 and merged['skybox_in_tree']==0
    assert merged['hierarchy_sha256']=='3caa47ee4f0e01a072f217de6701fb7f4ed775eb64708dcb49e39c06d9751f62'
    assert merged['numeric_audit']['all_Gaussian_fields_finite'] and merged['hierarchy']['unowned_tail']==0
    assert terminal['actual_source_input_output_receipts_SHA_before_after_unchanged'][str(mergedpath)]==sha(mergedpath)
    for item in merged['input_hierarchies']:
        assert item['source_kind']=='post_optimized'
        assert item['sha256']==read(LOCK/'trained_chunks'/item['chunk']/'post-outcome.json')['hierarchy_sha256']
    hm=merged['hierarchy']
    merge_summary=dict(receipt=ref(mergedpath),
        supervision_receipt=ref(SUPROOT/'city006-merge-optimized-r2-stage-001/supervisor-outcome.json'),
        independent_terminal_audit=ref(terminalpath),output_SHA256=merged['hierarchy_sha256'],
        G=hm['gaussian_count'],nodes=hm['node_count'],leaves=hm['leaf_gaussian_count'],
        internal_Gaussians=hm['merged_gaussian_count'],shared_sky_tail=0,render_Sky_to_append_exactly_once=100000,
        actual_native_process_seconds=merged['process_seconds'],actual_four_I15000_post_inputs=True,
        observed11_training_origin_preserved=True,leaf_pixel_equivalence_proven=False,
        standalone_paired_model_binding_CPU_completed=False,actual_City1404_evaluation_frames=0,
        scope='Actual native optimized global merge and owned terminal audit complete. Full paired independent CPU model binding is running; no actual City GPU renders or quality/time/memory claims yet.')
    utc=datetime.now(timezone.utc).isoformat()
    summary.pop('City_full_post_merge_or_1404_complete',None)
    summary.update(status='actual_four_full_posts_and_both_global_merges_complete_paired_evaluation_pending',UTC=utc,
        formalposts=[*summary['formalposts'],fourth],formalpost_completed_chunks=['0_0','0_1','1_0','1_1'],
        formalpost_pending_chunks=[],all4_formalpost_independent_anchor_exposure_ledger=ref(ledger),
        actual_optimized_global_merge=merge_summary,optimized_global_Merger_completed=True,
        City_full_pipeline_including_1404_complete=False,
        completion_scope='Four full I15000 posts and both actual global merges complete. Independent paired model CPU binding is running; 117x12 actual rendering, posterior metrics and visual review remain pending.')
    chapter_path=LAB/'knowledge/30_large_scene_gs_sfm_partition_lod.md'
    manifest_path=LAB/'manifests/large-scene-gs-work.json'
    sync_path=Path(r'D:\Projects\EngineLab\engine-home\scripts/sync-gs-lab.mjs')
    assert sha(chapter_path)=='a735bf36ff624391254d21200f85e618bf83ac4f5a09a0f681e0317ab9590f06'
    assert sha(manifest_path)=='ecfb078888bd80b14d06af50b8950d56a796998169170daaa51ebd965a08a1f0'
    chapter=chapter_path.read_text(encoding='utf-8-sig')
    main_before=chapter.split('## 实验附录')[0]
    chapter=chapter.replace('前三块现已分别完成完整 15,000 次层级优化','四块现已分别完成完整 15,000 次层级优化')
    row=next(x for x in chapter.splitlines() if x.startswith('| 1_0 | 4,294,329 |'))
    chapter=chapter.replace(row,row+f"\n| 1_1，观察训练来源 | 4,368,945 | {post['process_seconds']/60:.2f} 分钟 | {physical['official_logged_peak_allocated_bytes']/2**30:.2f} GiB |")
    pending='最后一块完整优化、优化后的全景合并，以及同一组 117 个留出视角的质量、耗时和显存对照仍待完成。'
    assert chapter.count(pending)==1
    chapter=chapter.replace(pending,'优化后的四份层级也已完成全局合并，得到 7,000,388 个高斯，其中叶高斯 4,643,252 个、内部高斯 2,357,136 个。相较未优化合并树，内部高斯减少八个；叶高斯数量一致尚不证明渲染像素相同。两次合并的真实进程均正常退出，共享天空仍由粗模型单独提供。四块完整后优化的锚点、叶参数和冻结曝光已通过独立 CPU 核验；同一组 117 个留出视角的质量、耗时和显存对照正在进行评测前的两份模型检查，尚无城市 LOD 渲染成绩。')
    oldrel='outputs/large-scene-gs-001/research/city-third-formalpost-resource-counts-015/verified-stage-summary.json'
    newrel='outputs/large-scene-gs-001/research/city-four-post-paired-merges-016/verified-stage-summary.json'
    oldlabel='城市前三块完整层级优化与留出视角选点资源核验摘要'
    newlabel='城市四块完整层级优化与两份全景合并核验摘要'
    assert chapter.count(oldrel)==1
    chapter=chapter.replace(f'[{oldlabel}](../{oldrel})',f'[{newlabel}](../{newrel})')
    assert chapter.split('## 实验附录')[0]==main_before
    manifest=read(manifest_path)
    city=next(x for x in manifest['milestones'] if x['id']=='public-large-scene-extension')
    city.update(status='mixed4_valid_local_inputs_four_full_posts_paired_merges_complete_City1404_pending',
        local_hierarchy006_actual_formalposts=summary['formalposts'],
        local_hierarchy006_formalpost_completed_chunks=summary['formalpost_completed_chunks'],
        local_hierarchy006_formalpost_pending_chunks=[],
        local_hierarchy006_native_and_supervision_status=summary['completion_scope'],
        local_hierarchy006_optimized_global_Merger_completed=True,
        local_hierarchy006_optimized_global_merge=merge_summary,
        local_hierarchy006_four_formalpost_anchor_exposure_ledger=ref(ledger))
    manifest['last_verified_utc']=utc
    sync=sync_path.read_text(encoding='utf-8')
    anchor=f"  ['{oldrel}', '{oldlabel}'],"
    assert sync.count(anchor)==1 and newrel not in sync
    sync=sync.replace(anchor,anchor+f"\n  ['{newrel}', '{newlabel}'],")
    OUT.mkdir(parents=True,exist_ok=False)
    for path,name in [(chapter_path,'chapter-before.md'),(manifest_path,'manifest-before.json'),(sync_path,'sync-before.mjs')]:
        shutil.copyfile(path,OUT/name)
    shutil.copyfile(__file__,OUT/'curator-source.py')
    save(OUT/'verified-stage-summary.json',summary)
    city['local_hierarchy006_verified_stage_summary']=ref(OUT/'verified-stage-summary.json')
    save(manifest_path,manifest)
    chapter_path.write_text(chapter,encoding='utf-8')
    sync_path.write_text(sync,encoding='utf-8')
    report=dict(status='source_curation_complete_local_serving_pending',UTC=utc,
        summary=ref(OUT/'verified-stage-summary.json'),chapter=ref(chapter_path),manifest=ref(manifest_path),
        theory_text_unchanged=True,actual_City1404_frames=0,raw_models_or_datasets_copied=False)
    save(OUT/'source-update.json',report)
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
