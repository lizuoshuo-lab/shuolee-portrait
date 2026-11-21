"""Record actual final source-view and36-view inspection; no training edits."""
import shutil
from pathlib import Path
from run_official_gaussians123 import read, save, sha, now
from train_sofa_half8_sh0124 import OUT


def main():
    snapshot = OUT/'gaussian-run/step-018000'
    delivery = OUT/'delivery/step-018000'
    assert read(OUT/'process-exit.json')['exit_code'] == 0
    assert read(OUT/'progress.json')['state'] == 'bounded_control_completed'
    review = read(snapshot/'review/result.json')
    result = read(snapshot/'result.json')
    validation = read(delivery/'validation.json')
    assert len(review['records']) == 8 and len(review['pages']) == 10
    assert validation['native_source_ply_parity']['passed']
    assert validation['native_rotated_render_parity']['passed']
    for row in review['records']:
        for picture in row['images']:
            assert sha(picture['path']) == picture['sha256']
    for page in review['pages']:
        assert sha(page['path']) == page['sha256']
    note = ('18,000步全部8个原视角的原图／当前mesh／高斯，以及脸、双手、脚的10页均已实际检查。'
            '正面和已拍侧面的脸、眼镜、衬衫与裤子细节较完整，坐姿连贯；未发现新增严重肘膝反折。'
            '手指边缘仍软，黑鞋有mask缺口和碎片，第46帧远侧膝腿边缘仍有拉丝。'
            '共同7帧原生像素前景MAE由0.02044降至0.01314，脸由0.03003降至0.01996；'
            '这是训练视角的整流程比较，不是独立留出、几何真值或单纯增加一帧的因果证据。')
    save(snapshot/'visual-review.json', dict(
        updated=now(), actual_visual_review=True,
        inspected_frames=[r['frame'] for r in review['records']],
        inspected_regions=['body', 'face', 'left_hand', 'right_hand', 'feet'],
        inspected_pages=review['pages'], finding_zh=note,
        point_count=result['points'], reference_frame='frame_000026',
        old_seven_comparison=review['common_seven_foreground_mae'],
        causal_scope=review['common_seven_causal_scope'],
        geometry=read(OUT/'gaussian-run/geometry-progress.json'),
        source_camera_physical_calibration_accepted=False,
        novel_view_ground_truth=False, quality_accepted=False,
        decision='Deliver a named exploratory model with full remaining-defect disclosure; no more updates to this candidate.'))
    orbit_records = []
    archive = Path('/mnt/d/00/记录/006')
    for start in (0, 12, 24):
        source = delivery/f'novel/orbit_{start:02d}-{start+11:02d}.png'
        target = archive/f'分析_沙发半圈8帧124_018000_高斯连续视角_{start:02d}-{start+11:02d}.png'
        assert source.is_file() and not target.exists()
        shutil.copyfile(source, target)
        orbit_records.append(dict(path=str(source), sha256=sha(source), record=str(target)))
    save(delivery/'visual-review.json', dict(
        updated=now(), actual_visual_review=True, inspected_pages=orbit_records,
        synthetic_views_reviewed=36, same_reference_pose=True,
        finding_zh=('36个连续转身视角已实际检查。已拍前侧比较连贯，越过原视角覆盖后，'
                    '衣服和后脑逐渐出现斑驳、孔洞及碎片，鞋底和远侧鞋不完整。'
                    'SH0没有高阶球谐颜色项；背面的异常仍来自未观测区域的空间高斯与颜色，不能算恢复可靠。'),
        unobserved_rear_unknown=True, source_views_better_than_unseen_views=True,
        renderer='Original CUDA renderer and exported+Y PLY with serialized camera readback.',
        mesh_motion=read(OUT/'gaussian-run/geometry-progress.json'),
        point_target_band=[170000,230000], point_target_met=170000 <= result['points'] <= 230000,
        quality_accepted=False, end_to_end_15min_accepted=False,
        next='Same pipeline across subjects. Do not claim full360quality or reuse this candidate for unregistered retries.'))
    print('SOFA_HALF8_FINAL_ACTUAL_REVIEW_RECORDED')


if __name__ == '__main__':
    main()
