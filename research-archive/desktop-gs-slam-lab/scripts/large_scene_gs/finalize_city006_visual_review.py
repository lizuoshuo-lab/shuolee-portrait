"""Bind previously inspected, unmodified originals to completed City evidence.

The observations below record actual full-image inspections performed in this task.
This script checks identity and assembles that review; it does not perform vision.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import struct

LAB = Path(__file__).resolve().parents[2]
BASE = LAB / 'outputs/large-scene-gs-001'
EVALS = BASE / 'independent-evaluation'
RUN = EVALS / 'small-city006-heldout-1404-001'
FOLDER = BASE / 'research/city-final-comparison-020'
PRIOR = BASE / 'research/city-heldout-postprocessing-018'
FILES = ('preliminary-leaf-visual-review.json', 'preliminary-tau3-visual-review.json',
         'preliminary-post-leaf-visual-review.json', 'preliminary-post-tau3-visual-review.json')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    output = FOLDER / 'visual-review.json'
    assert not output.exists(), 'Preserve an existing completed review'
    comparison = read(FOLDER / 'comparison.json')
    assert comparison['status'] == 'complete_numeric_comparison_manual_visual_review_still_separate'
    boundary_path = EVALS / 'small-city006-heldout-boundary-CPU-001/report.json'
    boundary = read(boundary_path)
    assert boundary['status'] == 'completed_actual_city_geometry_selected_original_boundary_image_index'
    assert boundary['actual_original_frame_sets'] == 32 and boundary['fixed_patch_count'] == 24
    assert sha(boundary_path) == comparison['source_files_sha256'][str(boundary_path)]
    protected = {Path(p).resolve(): h for p, h in boundary['protected_original_sources'].items()}
    expected = {}
    for view in boundary['views']:
        stem = f"{view['ordinal']:02d}-{Path(view['name']).stem}"
        for model in ('merged-unoptimized', 'merged-post'):
            for setting in ('leaves', 'tau3'):
                path = (RUN / model / setting / 'paired' / (stem + '.png')).resolve()
                expected[path] = (model, setting, stem, view['name'])
    images, seen, sources = [], set(), []
    for filename in FILES:
        prior_path = PRIOR / filename
        prior = read(prior_path)
        assert len(prior['images']) == 8
        sources.append(dict(path=str(prior_path), sha256=sha(prior_path)))
        for item in prior['images']:
            assert item.get('complete_original_pair_viewed', item.get('inspected_original_complete_pair')) is True
            assert item.get('image_edited', False) is False
            path = Path(item['path']).resolve()
            assert path in expected and path not in seen
            model, setting, stem, camera_name = expected[path]
            assert prior['model'] == model and prior['setting'] == setting
            assert sha(path) == item['sha256'] == protected[path]
            header = path.read_bytes()[:24]
            assert header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR'
            assert struct.unpack('>II', header[16:24]) == (512, 172)
            images.append(dict(model=model, setting=setting, name=stem, camera_name=camera_name,
                               path=str(path), sha256=item['sha256'], complete_original_pair_viewed=True,
                               width=512, height=172, edited=False,
                               observation_zh=item.get('observation_zh', item.get('preliminary_observation_zh',
                                   prior.get('preliminary_observation_zh', '原始完整配对已查看，观察见其他同视角对照记录。'))),
                               prior_review_path=str(prior_path)))
            seen.add(path)
    assert seen == set(expected) and len(images) == 32
    patches = comparison['fixed_patch_comparisons']
    raw = next(r for r in patches if r['setting'] == 'tau3' and r['diagnostic'] == 'raw_RGB_MSE')
    official = next(r for r in patches if r['setting'] == 'tau3' and r['diagnostic'] == 'official_GTm2_predm_patch_MSE')
    symmetric = next(r for r in patches if r['setting'] == 'tau3' and r['diagnostic'] == 'symmetric_single_mask_patch_MSE')
    assert (raw['improved'], raw['worsened'], official['improved'], official['worsened'], symmetric['improved']) == (20, 4, 21, 3, 21)
    assert all(r['equal'] == 24 for r in patches if r['setting'] == 'leaves')
    worsened = []
    for view in boundary['views']:
        for tile in view['fixed_patch_diagnostics']:
            a = tile['RGB_patch_diagnostics']['merged-unoptimized/tau3']
            b = tile['RGB_patch_diagnostics']['merged-post/tau3']
            if b['official_GTm2_predm_patch_MSE'] > a['official_GTm2_predm_patch_MSE']:
                worsened.append(dict(camera_name=view['name'], bbox_xyxy_exclusive=a['bbox_xyxy_exclusive'],
                                     before=a['official_GTm2_predm_patch_MSE'], after=b['official_GTm2_predm_patch_MSE']))
    assert len(worsened) == 3
    first = boundary['views'][0]
    report = dict(status='completed_geometry_selected_original_review', created_utc=datetime.now(timezone.utc).isoformat(),
                  boundary_report_path=str(boundary_path), boundary_report_sha256=sha(boundary_path),
                  comparison_path=str(FOLDER / 'comparison.json'), comparison_sha256=sha(FOLDER / 'comparison.json'),
                  images=images, prior_manual_review_sources=sources,
                  presentation_view_name=f"{first['ordinal']:02d}-{Path(first['name']).stem}",
                  presentation_view_selection='First preselected geometry view; not selected by quality or improvement',
                  slide_caption_zh='本地原图。按原实现掩膜口径，24 个预选区域中 21 个误差下降、3 个上升；近车不一致与细节平滑仍存在。',
                  main_limit_zh='大场景：层级优化改善了部分粗层外观，但阶段耗时未加速；整树常驻，仍需优化执行路径并验证流式加载。',
                  observations_zh=[
                      '逐张检查了八个固定视角的优化前后叶层及 3 像素结果，共三十二幅完整输入—渲染配对图。部分招牌、窗框和车辆边缘看起来更清楚，但有些视角的变化较小，路面与细枝仍然平滑。',
                      '近处车辆与输入不一致、缺失或模糊残影仍可见。背景局部清晰度变化不能写成前景问题已解决；仅凭这些图像不能区分运动、遮挡、观测覆盖或分块近似各自造成的误差。',
                      '二十四个预选区域中，3 像素优化后按原实现掩膜误差有二十一个下降、三个上升；按原始 RGB 均方误差则是二十个下降、四个上升。叶层的二十四个区域全部保持相同。三种口径分别保存，区域均值不等于把不同分母的误差直接合并。',
                      '三个按原实现掩膜口径变差的区域分别位于两个固定视角中。这也表明局部看起来更清楚，不保证每个预选区域的像素均方误差都降低。',
                      '数值对照还显示，范围外的 15 像素设置在优化后 PSNR、SSIM 下降而 LPIPS 降低；不能把这一结果写成三个指标一致改善或一致恶化。这里对该设置仅作全量指标解释，逐张原图检查限定于叶层与 3 像素。'],
                  fixed_patch_comparisons=patches, worsened_official_tau3_patches=worsened,
                  patch_aggregation='Arithmetic mean and direction counts of per-patch MSE, kept separate by denominator; no pooled seam statistic',
                  all32_original_image_hashes_bound_to_completed_audit=True,
                  actual_visual_inspection_complete=True, numerical_patch_review_complete=True,
                  pixel_equality_source='Complete independent posterior plus actual identical leaf patch values; not inferred from visual similarity',
                  all_seams_eliminated=False, causal_attribution_to_seams=False,
                  source_images_modified=False, whole_city_representative_sampling_claimed=False)
    (FOLDER / 'visual-review-source.py').write_bytes(Path(__file__).read_bytes())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], images=len(images), patches=24, output=str(output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
