"""Close the completed warm-start control without changing its fit or admission rules."""
from datetime import datetime, timezone
import json
import subprocess
import numpy as np
import audit_gaussianavatars_params123 as a


def local(path):
    return str(path).replace('/mnt/c/', 'C:/').replace('/mnt/d/', 'D:/')


def main():
    body = a.REC / 'demo-vhap-initialization/fullbody-smplifyx'
    run = body / 'warmtail-guarded-sequence'
    plan_path = body / 'warmtail-guarded-sequence-plan.json'
    result = a.read(run / 'result.json')
    visual = a.read(run / 'review/visual-review.json')
    criteria = a.read(run / 'review/criteria-audit.json')
    selection = a.read(run / 'selection-audit.json')
    seeds = a.read(run / 'seed-path-audit.json')
    stages = a.read(run / 'audit-scope.json')
    handoff = a.read(body / 'warmtail-guarded-sequence-gpu-handoff.json')
    assert result['frames'] == visual['reviewed_frame_count'] == criteria['frames'] == selection['checked_frames'] == 59
    assert seeds['frames'] == stages['frames'] == 59
    assert a.sha(plan_path) == '64b7d85c2881f3f5d4475eb03f57ed32b80c642b28b3e2aa34580c2061126e62'
    expected = [f'frame_{i:06d}' for i in range(1, 60)]
    assert [row['frame'] for row in visual['frames']] == expected
    assert visual['actual_visual_review'] and visual['reviewed_image_groups'] == 295
    assert len(criteria['adjacent_pairs']) == 58
    assert a.sha(run / 'fit/fitted.npz') == result['fit_sha256']
    for row in visual['frames']:
        assert set(row['regions_seen']) == {'body', 'face', 'left_hand', 'right_hand', 'feet'}
        assert len(row['images']) == 5
        for image in row['images']:
            assert a.sha(local(image['path'])) == a.sha(local(image['record_path'])) == image['sha256']
            assert image['no_resampling'] and image['no_text_added']
    plan_linux = str(plan_path).replace('\\', '/').replace('C:/', '/mnt/c/')
    verifier = '''import json,hashlib,sys
from pathlib import Path
plan=json.loads(Path(sys.argv[1]).read_text())
hashes={**plan['source_hashes'],**plan['input_hashes']}
bad=[p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
assert not bad,bad
assert not Path('/proc/10662').exists() and not Path('/proc/10654').exists()
stat=Path('/proc/331/stat').read_text().split(') ',1)[1].split()
assert stat[0]!='T' and stat[19]=='9229',stat[:20]
cmd=Path('/proc/331/cmdline').read_bytes().replace(b'\\0',b' ').decode()
assert 'train_demo_body_upstream_offset123.py train' in cmd
print(json.dumps(dict(pinned_hashes_checked=len(hashes),baseline_pid=331,baseline_state=stat[0],baseline_identity=stat[19],baseline_command=cmd)))'''
    live = json.loads(subprocess.check_output(['wsl', '-d', 'Ubuntu-20.04', '--',
        '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python', '-B', '-c', verifier, plan_linux],
        text=True, encoding='utf-8', errors='strict'))
    assert handoff['state'] == 'resumed'
    # This control has observed regional failures; do not relabel it as admitted.
    assert criteria['protection_failures']
    now = datetime.now(timezone.utc).isoformat()
    visual.update(updated=now, state='complete_actual_visual_review_quality_rejected',
        full59_quality_passed=False, quality_accepted=False)
    a.save(run / 'review/visual-review.json', visual)
    continuity = {}
    for label in ('baseline', 'candidate'):
        continuity[label] = {}
        for key in ('root_deg', 'local_joint_max_deg', 'left_elbow_deg', 'right_elbow_deg'):
            pairs = criteria['adjacent_pairs']
            largest = max(pairs, key=lambda row: row[label][key])
            continuity[label][key] = dict(median=float(np.median([p[label][key] for p in pairs])),
                maximum=largest[label][key], maximum_pair=[largest['previous'], largest['frame']])
    next_action = ('Inspect the existing two-candidate selection against regional and temporal quality evidence. '
        'Any filtering proposal must be uniform, declared before application, and audited on all59 frames; '
        'do not manually replace failed frames, relax the completed control criteria, or rerun the same fit. '
        'Keep the published120k model and original Gaussian training unchanged.')
    closeout = dict(updated=now, state='complete_quality_rejected', frames=59,
        actual_reviewed_frames=59, actual_reviewed_image_groups=295, adjacent_pairs=58,
        incremental_fit_seconds=result['elapsed_seconds'], independent_cache_fit_seconds=criteria['independent_cache_fit_seconds'],
        timing_scope='Fit only. Separate independent baseline cache; excludes input preparation, VHAP, Gaussian training, review, and export.',
        fresh_video_to_export_timing=False, target_15min_accepted=False,
        retained_independent_frames=[row['frame'] for row in selection['rows'] if row['selected_index']==1],
        selected_original_objective_never_increased=True, seed_paths_verified=59, stage_contracts_verified=59,
        all_six_elbows_halved=criteria['all_six_elbows_halved'],
        six_elbows=[dict(frame=row['frame'], **row['left_elbow']) for row in criteria['six_elbows']],
        protection_failures=criteria['protection_failures'], continuity_so3=continuity,
        root_pose_scope='Camera-relative rotations, not a physical world-camera path.',
        continuous_video_playback_reviewed=False,
        visual_review_sha256=a.sha(run/'review/visual-review.json'), criteria_sha256=a.sha(run/'review/criteria-audit.json'),
        selection_sha256=a.sha(run/'selection-audit.json'), fit_sha256=result['fit_sha256'], plan_sha256=a.sha(plan_path),
        pinned_sources_and_inputs_unchanged=True, immutable_inputs_and_sources=live,
        no_manual_frame_selection=True, no_frame_omission=True, candidate_fit_modified=False,
        baseline_resumed_verified=True, baseline_pid=331,
        quality_accepted=False, goal_complete=False, Gaussian_training_started=False, model_promoted=False,
        main_model_preserved='local-surface-123-demo-body-smplx-offset-s120000',
        findings_zh=[
            '59帧、295组原生图像已实际审阅，保存参数、原目标选择、初值来源和原阶段索引均逐帧核验。',
            '18至23帧六个严重折臂均明显改善，肘部检测残差约52至90像素降到10至14像素。',
            '仍有逐区域保护失败；原版总目标更低不能保证每个部位不退化，未放宽门槛或人工换帧。',
            '姿态初始化控制尚未证明衣物、帽子、鞋面与遮挡手的物理几何，未接入新的VHAP或高斯。',
            '独占拟合结束后原高斯进程已恢复；现有12万步模型及同坐标相机保持可查看。'],
        next=next_action)
    a.save(run/'review/closeout.json',closeout)
    print(json.dumps({key:closeout[key] for key in ('state','frames','incremental_fit_seconds',
        'independent_cache_fit_seconds','all_six_elbows_halved','protection_failures','baseline_resumed_verified')}))


if __name__ == '__main__':
    main()
