"""Integrate completed City evidence into the existing local experiment appendix.

Only run after the existing full evaluation, CPU checks, and original image review.
Preserves the theory body and all historical training/failure evidence.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import shutil

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/large-scene-gs-001'
DATA = BASE / 'research/city-final-comparison-020'
OUT = BASE / 'research/city-final-website-021'
CHAPTER = LAB / 'knowledge/30_large_scene_gs_sfm_partition_lod.md'
MANIFEST = LAB / 'manifests/large-scene-gs-work.json'
SYNC = Path(r'D:\Projects\EngineLab\engine-home\scripts\sync-gs-lab.mjs')
SETTINGS = ('leaves', 'tau1', 'tau3', 'tau5', 'tau6', 'tau15')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def theory(s):
    return re.sub(r'(?m)^updated:.*$', 'updated: <metadata>', s.split('## 实验附录')[0]).replace('\r\n', '\n')


def main():
    assert not OUT.exists(), 'Preserve any existing curation; do not replay'
    data = read(DATA / 'comparison.json')
    review = read(DATA / 'visual-review.json')
    assert data['status'] == 'complete_numeric_comparison_manual_visual_review_still_separate'
    assert len(data['rows']) == 12
    assert review['status'] == 'completed_geometry_selected_original_review'
    assert len(review['images']) == 32 and all(r['complete_original_pair_viewed'] for r in review['images'])
    assert review['observations_zh'], 'Actual image observations are required'
    boundary = BASE / 'independent-evaluation/small-city006-heldout-boundary-CPU-001/report.json'
    assert sha(boundary) == review['boundary_report_sha256']
    assert all(sha(Path(p)) == h for p, h in data['source_files_sha256'].items())
    assert all(sha(Path(r['path'])) == r['sha256'] for r in review['images'])
    rows = {(r['model'], r['setting']): r for r in data['rows']}
    assert len(rows) == 12 and all(r['views'] == 117 for r in rows.values())
    before = CHAPTER.read_text(encoding='utf-8')
    chapter = before
    replacements = {
        '优化前后两份全景合并均已完成，完整本地层级渲染对照仍在进行。':
            '优化前后两份全景合并及完整本地层级渲染对照均已完成，结果见本附录末尾。',
        '两份模型的独立 CPU 检查已通过，正在同一组 117 个留出视角上进行实际 GPU 渲染。两份模型各使用完整叶层及五档层次细节，共计 1,404 张图像；完整质量、耗时和显存对照尚未完成。':
            '两份模型已在同一组 117 个留出视角上完成实际 GPU 渲染。每份模型使用完整叶层及五档层次细节，共计 1,404 张图像；完整质量、阶段耗时和显存对照已完成，并通过独立 CPU 复核。',
        '这组 CPU 结果核验固定选择公式与资源估算；原生 GPU 的选择、实际工作区和图像结果正在另一份完整评测中检查。部分设置完成渲染不等于整个城市 LOD 对照已经完成。':
            '这组 CPU 结果核验固定选择公式与资源估算；随后完成的全部 1,404 张图像另核验了实际原生 GPU 选择与插值。下方列出实测阶段耗时及 PyTorch 分配峰值，后者不覆盖全部原生临时工作区。',
        '城市合并模型现已开始生成实际图像，完整 LOD 质量结论仍待全部渲染及独立重算完成。':
            '城市合并模型的全部图像与独立感知指标复核也已完成，其结果与粗模型分开列出。',
    }
    for old, new in replacements.items():
        assert chapter.count(old) == 1, old
        chapter = chapter.replace(old, new)
    label = lambda setting: '完整叶层' if setting == 'leaves' else f'{setting[3:]} 像素'
    lines = ['### 城市完整留出对照：层级优化、画质与资源', '',
             '下表每格依次为优化前 → 优化后，均为相同 117 张留出图像的逐图指标均值。1、3、5 像素位于本次层级优化的训练尺度范围；6、15 像素单列为范围外诊断。', '',
             '| 设置 | PSNR / dB | SSIM | Alex LPIPS ↓ |',
             '| --- | ---: | ---: | ---: |']
    for setting in SETTINGS:
        a, b = (rows[(m, setting)] for m in ('merged-unoptimized', 'merged-post'))
        values = [f"{a['quality'][k]:.{p}f} → {b['quality'][k]:.{p}f}" for k, p in
                  [('psnr_numpy_mean_per_image_db', 4), ('ssim_mean_per_image', 6), ('lpips_alex_mean_per_image', 6)]]
        lines.append('| ' + ' | '.join([label(setting)] + values) + ' |')
    pairs = {r['setting']: r for r in data['paired_merge_stage_comparison']}
    leaf = pairs['leaves']
    leaf_text = ('两份合并模型的叶层在全部 117 张渲染中逐像素一致。' if leaf['all117_RGB_bitwise_equal'] else
                 f"两份合并模型的叶层并非逐像素一致，浮点 RGB 最大绝对差为 {leaf['max_abs_RGB_difference']:.8g}。")
    delta = pairs['tau3']['post_minus_unoptimized']
    lines += ['', leaf_text + f"3 像素粒度的优化后减优化前变化为：PSNR {delta['psnr_numpy_mean_per_image_db']:+.4f} dB，SSIM {delta['ssim_mean_per_image']:+.6f}，Alex LPIPS {delta['lpips_alex_mean_per_image']:+.6f}。这些结果针对同一原生渲染器和本次固定划分；粗模型与密集分块模型的训练预算、表示及监督不同，不构成分块单一因素的因果比较。", '',
              '| 优化后设置 | 选中树高斯均值 | 同步阶段耗时中位数 / ms | PyTorch 分配峰值 / GiB |',
              '| --- | ---: | ---: | ---: |']
    for setting in SETTINGS:
        r = rows[('merged-post', setting)]
        lines.append(f"| {label(setting)} | {r['selected_tree_G']['mean']:,.1f} | {r['stage_p50_ms']['sum_synchronized_GPU_stages_seconds']:.3f} | {r['torch_peak_allocated_GiB']:.3f} |")
    leaf_r = rows[('merged-post', 'leaves')]
    tau3_r = rows[('merged-post', 'tau3')]
    stage_key = 'sum_synchronized_GPU_stages_seconds'
    selected_reduction = 100 * (1 - tau3_r['selected_tree_G']['mean'] / leaf_r['selected_tree_G']['mean'])
    memory_reduction = 100 * (1 - tau3_r['torch_peak_allocated_GiB'] / leaf_r['torch_peak_allocated_GiB'])
    time_ratio = tau3_r['stage_p50_ms'][stage_key] / leaf_r['stage_p50_ms'][stage_key]
    lines += ['', f"相对于同一优化后模型的完整叶层，3 像素粒度平均选中的树高斯减少 {selected_reduction:.1f}%，PyTorch 分配峰值下降 {memory_reduction:.1f}%，但同步阶段耗时中位数为叶层的 {time_ratio:.2f} 倍。数量、内存与速度必须分别判断，不能由选中高斯更少推导出渲染更快。"]
    if all(rows[('merged-post', setting)]['stage_p50_ms'][stage_key] >= leaf_r['stage_p50_ms'][stage_key]
           for setting in SETTINGS if setting != 'leaves'):
        lines += ['', '在本次实现和计时条件下，五档 LOD 的同步阶段耗时中位数均高于叶层，因此没有测得阶段加速。仅凭此表不能把开销单独归因于树遍历、参数插值或光栅化中的某一个环节。']
    lines += ['', '选中数量不含额外的 100,000 个共享天空高斯。阶段时间为分别等待 GPU 完成后的树选择、插值权重与渲染时间之和，不包含文件读写、CPU 检查或指标计算，也不是连续播放帧率。桌面负载未隔离，单次分布不能作为严格硬件排名。整棵树和天空常驻显存；PyTorch 分配峰值、保留内存和整卡使用量定义不同，不能据此宣称实现流式加载。', '',
              '完整指标延续固定公开实现的软掩膜顺序：真实图像应用两次掩膜，预测应用一次掩膜，按完整图像计算；对称掩膜辅助误差另存。不能将这些分数直接与采用未知划分、曝光、掩码或 VGG 感知网络的论文表格比较。', '']
    cpu = data['CPU_perceptual_check']
    lines += [f"1,404 张浮点预测的 SSIM 和 Alex LPIPS 又在 CPU 上独立重算，与 GPU 结果的逐图最大绝对差分别为 {cpu['max_abs_CPU_GPU_SSIM_delta']:.8g} 和 {cpu['max_abs_CPU_GPU_AlexLPIPS_delta']:.8g}。原 GPU 指标保留；这是实际后端数值差异，不定义通用验收容差。", '']
    lines += list(review['observations_zh']) + ['', '边界检查固定使用八个几何预选视角、每个视角的完整四组原始配对图及二十四个预选区域，没有按画质挑图。稀疏几何代理不等于最终可见接缝真值，局部误差也包含动态物体、遮挡和抗锯齿影响，不能据此宣称全部接缝消除。', '']
    summary_label = '城市完整 1404 帧画质、资源与边界核验摘要'
    summary_rel = OUT.relative_to(LAB).as_posix() + '/verified-city-comparison.json'
    lines.append(f'- [{summary_label}](../{summary_rel})')
    marker = '- [原图来源、完整图范围与许可]'
    assert chapter.count(marker) == 1
    chapter = chapter.replace(marker, '\n'.join(lines) + '\n\n' + marker)
    assert theory(chapter) == theory(before)
    assert chapter.count('## 实验附录') == 1
    utc = datetime.now(timezone.utc).isoformat()
    summary = dict(status='complete_actual_City1404_numeric_and_selected_visual_comparison', UTC=utc,
                   actual_verified_frames=1404, models=2, settings_per_model=6, views_per_setting=117,
                   rows=data['rows'], paired_merge_stage_comparison=data['paired_merge_stage_comparison'],
                   CPU_perceptual_check=cpu, fixed_patch_comparisons=data['fixed_patch_comparisons'],
                   manual_dense_boundary_comparison_complete=True, viewed_original_pairs=32,
                   fixed_views=8, fixed_patches=24, observations_zh=review['observations_zh'],
                   limits_zh=data['limits_zh'], formal_observation_reclassification=False,
                   raw_models_or_datasets_copied_to_site=False,
                   local_sources=[dict(path=p.relative_to(LAB).as_posix(), sha256=sha(p))
                                  for p in (DATA / 'comparison.json', DATA / 'visual-review.json')])
    # Portable summary keeps small metrics and source references, not model arrays or training data.
    manifest = read(MANIFEST)
    city = next(r for r in manifest['milestones'] if r['id'] == 'public-large-scene-extension')
    assert all(r['status'].startswith('completed_') for r in manifest['milestones']
               if r['id'] != 'public-large-scene-extension')
    assert city['local_formal_completed_chunks'] == ['0_0', '0_1', '1_0']
    city.update(status='mixed4_four_posts_paired_merges_City1404_and_independent_metrics_visual_review_complete',
                local_hierarchy006_native_and_supervision_status='Both actual merges and all1404 renders complete; independent posterior, CPU SSIM/LPIPS recalculation and32 original pair reviews complete. Mixed training origins retained.',
                local_hierarchy006_actual_City1404_evaluation_frames=1404,
                local_hierarchy006_actual_City1404_frame_count_scope='All1404 independently posterior-verified frames, all12 conditions,117 views each',
                local_hierarchy006_actual_City1404_frame_count_asof_utc=utc,
                local_hierarchy006_actual_City1404_pipeline_complete=True)
    manifest['last_verified_utc'] = utc
    manifest['updated'] = datetime.now().date().isoformat()
    manifest['status'] = 'completed_local_experiment_scope_presentation_delivery_separate'
    manifest['completion_scope'] = 'Local baseline, overlapping chunks, hierarchical LOD and public City comparison complete. Final PPT delivery has separate package and visual acceptance; streaming, new scenes and joint sensor calibration are not claimed.'
    sync = SYNC.read_text(encoding='utf-8')
    old_line = "  ['outputs/large-scene-gs-001/research/city-running-evaluation-status-019/verified-evaluation-progress.json', '城市两份模型 CPU 核验与实际渲染状态摘要'],"
    assert sync.count(old_line) == 1 and summary_rel not in sync
    sync = sync.replace(old_line, old_line + f"\n  ['{summary_rel}', '{summary_label}'],")
    OUT.mkdir(parents=True, exist_ok=False)
    for p, name in ((CHAPTER, 'chapter-before.md'), (MANIFEST, 'manifest-before.json'), (SYNC, 'sync-before.mjs')):
        shutil.copyfile(p, OUT / name)
    shutil.copyfile(__file__, OUT / 'curator-source.py')
    save(OUT / 'verified-city-comparison.json', summary)
    city['local_hierarchy006_final_comparison_summary'] = dict(path=summary_rel, sha256=sha(OUT / 'verified-city-comparison.json'))
    CHAPTER.write_text(chapter, encoding='utf-8')
    save(MANIFEST, manifest)
    SYNC.write_text(sync, encoding='utf-8')
    result = dict(status='final_City_source_update_complete_local_serving_pending', UTC=utc,
                  theory_body_unchanged=True, final_City_frames=1404,
                  chapter_sha256=sha(CHAPTER), manifest_sha256=sha(MANIFEST), summary_sha256=sha(OUT / 'verified-city-comparison.json'))
    save(OUT / 'source-update.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
