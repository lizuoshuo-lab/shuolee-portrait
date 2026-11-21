"""Record the completed human visual review; never edit or select fitted frames."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import numpy as np
import audit_gaussianavatars_params123 as a


def local(path):
    return Path(str(path).replace('/mnt/c/', 'C:/').replace('/mnt/d/', 'D:/'))


def main():
    now = datetime.now(timezone.utc).isoformat()
    run = a.REC/'demo-vhap-initialization/fullbody-smplifyx/temporal-root-sequence'
    review_path = run/'review/visual-review.json'
    previous = a.read(review_path)
    result = a.read(run/'result.json')
    objective = a.read(run/'review/metrics-objective.json')
    summary = a.read(run/'review/summary.json')
    assert result['frames'] == summary['frames'] == objective['frames'] == 59
    assert a.sha(run/'fit/fitted.npz') == result['fit_sha256']
    live = subprocess.check_output(['wsl', '-d', 'Ubuntu-20.04', '--', 'bash', '-lc',
        'ps -o pid=,stat=,args= -p 331,8558,8566'], text=True, encoding='utf-8', errors='replace').strip()
    lines = [line.split() for line in live.splitlines() if line.split()]
    baseline = next(parts for parts in lines if parts[0] == '331')
    assert 'T' not in baseline[1] and 'train_demo_body_upstream_offset123.py' in ' '.join(baseline)
    assert not any(parts[0] in ('8558', '8566') for parts in lines)
    notes = {
        13: '新增明显错臂：原图口袋侧的手臂被抬至头顶附近，身体朝向和头部也偏移；不是衣物外扩。',
        16: '抬臂、身体朝向相较独立初始化明显改善，但手指和衣物表面仍不精确。',
        24: '抬臂、肩肘相较基线进一步偏离原图，手腕前臂遮挡关系不合理。',
        40: '抬臂与抓握相较独立初始化改善；不代表整段准入。',
        41: '新增抬腕与前臂方向退化。',
        42: '新增抬腕与前臂方向退化，左肘残差也明显增大。',
        51: '抬臂与抓握相较独立初始化改善；不代表整段准入。',
        54: '新增抬腕与前臂退化，前臂形状与原图不符。',
    }
    rows = []
    for i in range(1, 60):
        frame = f'frame_{i:06d}'
        record = a.read(run/'review/frames'/(frame+'.json'))
        assert record['frame'] == frame and record['unchanged_camera_shape_topology']
        assert len(record['files']) == 5
        for image in record['files']:
            assert a.sha(local(image['path'])) == image['sha256']
            assert a.sha(local(image['record_path'])) == image['sha256']
            assert image['no_resampling'] and image['no_text_added']
        note = notes.get(i, '已审阅原生全身、脸、双手与脚对照；除另列问题外未看到新增明显全身翻转，裸模型与衣物/帽檐/鞋外表面仍有差异。')
        if 18 <= i <= 23:
            note = '仍有明显肩肘/抬臂错误，连续初始化没有修复这段；抓握、口袋手和部分脚部同样不稳定。'
        rows.append(dict(frame=frame, regions_seen=['body', 'face', 'left_hand', 'right_hand', 'feet'],
                         images=record['files'], findings_zh=note))
    sheets = []
    for path in sorted(a.lp(run/'review/native-sheets').glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        # The previously reviewed 27..59 sheets and this turn's 1..26 sheets cover all frames.
        png = run/'review/native-sheets'/path.with_suffix('.png').name
        manifest = run/'review/native-sheets'/path.name
        sheets.append(dict(path=str(png), sha256=a.sha(png), manifest_sha256=a.sha(manifest)))
    assert len(sheets) == 15
    review = dict(updated=now, state='complete_actual_visual_review_quality_rejected',
        actual_visual_review=True, reviewed_frame_count=59, reviewed_image_groups=295, total=59,
        previous_partial_review=dict(updated=previous['updated'], reviewed_frame_count=previous['reviewed_frame_count']),
        frames=rows, native_sheets_seen=sheets,
        new_visible_regression_frames=[f'frame_{i:06d}' for i in (13, 24, 41, 42, 54)],
        persistent_wrong_arm_frames=[f'frame_{i:06d}' for i in range(18, 25)],
        improvements_seen=[f'frame_{i:06d}' for i in (16, 40, 51)],
        quality_accepted=False, full59_quality_passed=False, goal_complete=False,
        model_promoted=False, Gaussian_training_started=False, no_resampling=True, no_text_on_images=True,
        findings_zh=[
            '59帧、295组全身/脸/双手/脚原生对照均已实际审阅；所有帧和原始输出保留。',
            '第16、40、51帧抬臂改善；第13帧新增明显错臂，第24、41、42、54帧退化，第18–24帧整体仍不稳定。',
            '第13帧原版总目标由2953.35升至40843.88，且新增明显姿态跳变；仅传递相邻姿态/朝向不能保证求解落在正确解附近。',
            '第12、43、55帧等能恢复较合理姿态，不能概括为持续累积漂移，也不能由此唯一归因于相机、观测或优化器。',
            '脸部多数帧与基线接近，嘴唇/胡须轮廓和侧脸仍非高精度表面；口袋手与设备遮挡处没有可验证的完整外观。',
            '帽檐、鞋和衣物外表面不等于裸SMPL-X表面；观察残差是拟合数据误差，不是独立物理真值。',
            '固定相机、体型、观测、损失与优化器的控制已核验；不通过删帧或拼接历史最好结果掩盖失败。',
            '本轮拒绝接入新VHAP/高斯训练；只拟合就耗时约37.5分钟，全流程15分钟与质量目标均未完成。'])
    a.save(review_path, review)
    rotation = {}
    for label in ('baseline', 'candidate'):
        rotation[label] = {}
        for key in ('root_deg', 'local_joint_max_deg', 'left_elbow_deg', 'right_elbow_deg'):
            values = np.array([pair[label][key] for pair in objective['adjacent_pairs']])
            maximum = max(objective['adjacent_pairs'], key=lambda p: p[label][key])
            rotation[label][key] = dict(median=float(np.median(values)), maximum=float(values.max()),
                maximum_pair=[maximum['previous'], maximum['frame']])
    assert len(objective['adjacent_pairs']) == 58
    closeout = dict(updated=now, state='complete_quality_rejected', frames=59, actual_reviewed_frames=59,
        actual_reviewed_image_groups=295, adjacent_pairs=58, fit_seconds_monotonic=result['elapsed_seconds'],
        seconds_scope='Full59 control fitter only, excludes preprocessing, review, VHAP, Gaussian training and export.',
        aggregate_fitting_residuals=summary['aggregate'], continuity_so3=rotation,
        lower_original_objective_frames=sum(row['delta_loss'] < 0 for row in objective['rows']),
        max_original_objective_increase=max(objective['rows'], key=lambda row: row['delta_loss']),
        root_pose_scope='Camera-relative rotations; not a recovered physical world camera path.',
        continuous_video_playback_reviewed=False,
        actual_visual_review_path=str(review_path), visual_review_sha256=a.sha(review_path),
        metrics_sha256=a.sha(run/'review/metrics-objective.json'), fit_sha256=result['fit_sha256'],
        plan_sha256=result['plan_sha256'], input_camera_shape_observation_contract_unchanged=True,
        no_frame_selection=True, no_splicing=True, candidate_fit_modified=False,
        original_result_preserved=True, worker_pid=None, supervisor_pid=None, baseline_pid=331,
        baseline_resumed_verified=True, live_processes=live,
        quality_accepted=False, goal_complete=False, Gaussian_training_started=False,
        main_model_preserved='local-surface-123-demo-body-smplx-offset-s120000',
        next='Keep current120k export and the original running Gaussian baseline. Review the original solver multi-start/winner contract before any new uniform intervention; no automatic fit or Gaussian launch from this rejected candidate.')
    a.save(run/'review/closeout.json', closeout)
    print(json.dumps({k: closeout[k] for k in ('state', 'frames', 'actual_reviewed_image_groups', 'fit_seconds_monotonic', 'continuity_so3', 'baseline_resumed_verified')}))


if __name__ == '__main__':
    main()
