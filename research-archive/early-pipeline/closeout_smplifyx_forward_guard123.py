"""Close the chronological control using the existing full59 audit contract."""
from datetime import datetime,timezone
from pathlib import Path
import json,subprocess
import numpy as np
import audit_gaussianavatars_params123 as a
from closeout_smplifyx_guarded_seed123 import local

def main():
    body = a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    run = body/'forward-guarded-sequence'
    result = a.read(run/'result.json')
    review = a.read(run/'review/visual-review.json')
    criteria = a.read(run/'review/criteria-audit.json')
    selection = a.read(run/'selection-audit.json')
    handoff = a.read(body/'forward-guarded-sequence-gpu-handoff.json')
    plan = a.read(body/'forward-guarded-sequence-plan.json')
    expected = [f'frame_{i:06d}' for i in range(1, 60)]
    assert result['frames'] == review['reviewed_frame_count'] == criteria['frames'] == selection['checked_frames'] == 59
    assert [row['frame'] for row in review['frames']] == expected
    assert review['actual_visual_review'] and review['reviewed_image_groups'] == 295
    assert len(criteria['adjacent_pairs']) == 58
    assert a.sha(run/'fit/fitted.npz') == result['fit_sha256']
    plan_linux = str(body/'forward-guarded-sequence-plan.json').replace('\\', '/').replace('C:/', '/mnt/c/')
    verifier = ('import json,hashlib,sys;from pathlib import Path;'
        'p=json.loads(Path(sys.argv[1]).read_text());'
        'bad=[x for x,h in {**p["source_hashes"],**p["input_hashes"]}.items() if hashlib.sha256(Path(x).read_bytes()).hexdigest()!=h];'
        'assert not bad,bad;print("PINNED_SOURCES_AND_INPUTS_UNCHANGED")')
    subprocess.run(['wsl','-d','Ubuntu-20.04','--',
        '/opt/gs-slam-lab/shuolee-portrait-clean-001/env/bin/python','-B','-c',verifier,plan_linux],check=True)
    for row in review['frames']:
        assert set(row['regions_seen']) == {'body', 'face', 'left_hand', 'right_hand', 'feet'}
        assert len(row['images']) == 5
        for item in row['images']:
            assert a.sha(local(item['path'])) == a.sha(local(item['record_path'])) == item['sha256']
            assert item['no_resampling'] and item['no_text_added']
    for item in review['native_sheets_seen']:
        assert a.sha(local(item['path'])) == item['sha256']
    live = subprocess.check_output(['wsl', '-d', 'Ubuntu-20.04', '--', 'bash', '-lc',
        'ps -o pid=,stat=,args= -p 331,9672,9673'], text=True, encoding='utf-8', errors='replace').strip()
    processes = {int(line.split()[0]): line.split() for line in live.splitlines() if line.split()}
    assert 9672 not in processes and 9673 not in processes
    assert 331 in processes and 'T' not in processes[331][1]
    assert 'train_demo_body_upstream_offset123.py' in ' '.join(processes[331])
    assert handoff['state'] == 'resumed'
    # This closeout records the observed rejection, never turns a numerical pass into quality acceptance.
    assert not criteria['all_six_elbows_halved'] or criteria['protection_failures']
    now = datetime.now(timezone.utc).isoformat()
    review.update(updated=now, state='complete_actual_visual_review_quality_rejected',
        quality_accepted=False, full59_quality_passed=False,
        persistent_wrong_arm_frames=[f'frame_{i:06d}' for i in [19,20,21,22,23,40,51]])
    a.save(run/'review/visual-review.json', review)
    rotation = {}
    for label in ('baseline', 'candidate'):
        rotation[label] = {}
        for key in ('root_deg', 'local_joint_max_deg', 'left_elbow_deg', 'right_elbow_deg'):
            rows = criteria['adjacent_pairs']
            largest = max(rows, key=lambda p: p[label][key])
            rotation[label][key] = dict(median=float(np.median([p[label][key] for p in rows])),
                maximum=largest[label][key], maximum_pair=[largest['previous'], largest['frame']])
    closeout = dict(updated=now, state='complete_quality_rejected', frames=59,
        actual_reviewed_frames=59, actual_reviewed_image_groups=295, adjacent_pairs=58,
        incremental_fit_seconds=result['elapsed_seconds'],
        independent_cache_fit_seconds=criteria['independent_cache_fit_seconds'],
        seconds_scope='Fitting only. Incremental control reuses a separately computed independent cache; excludes preprocessing, VHAP, Gaussian training, review and export.',
        fresh_video_to_export_timing=False,
        retained_independent_frames=[r['frame'] for r in selection['rows'] if r['selected_index']==1],
        selected_original_objective_never_increased=True,
        all_six_elbows_halved=criteria['all_six_elbows_halved'],
        six_elbows=[dict(frame=r['frame'], **r['left_elbow']) for r in criteria['six_elbows']],
        protection_failures=criteria['protection_failures'], continuity_so3=rotation,
        root_pose_scope='Camera-relative rotations, not a physical world camera path.',
        continuous_video_playback_reviewed=False,
        visual_review_sha256=a.sha(run/'review/visual-review.json'),
        criteria_sha256=a.sha(run/'review/criteria-audit.json'),
        selection_sha256=a.sha(run/'selection-audit.json'),
        fit_sha256=result['fit_sha256'], plan_sha256=a.sha(body/'forward-guarded-sequence-plan.json'),
        pinned_sources_and_inputs_unchanged=True, original_result_preserved=True,
        no_manual_frame_selection=True, no_frame_omission=True, candidate_fit_modified=False,
        baseline_resumed_verified=True, baseline_pid=331, live_processes=live,
        quality_accepted=False, goal_complete=False, Gaussian_training_started=False,
        main_model_preserved='local-surface-123-demo-body-smplx-offset-s120000',
        findings_zh=[
            '完整59帧、295组原生图像已审阅，候选选择与实际保存参数逐帧核验通过。',
            '正向身体/根朝向初值改善了第16和18帧，但第19–23帧未达到减半要求，逐区域保护仍有失败。',
            '低总目标不等于人体正确。保留全部失败证据，不把本轮拟合接入新VHAP或高斯。',
            '第19帧只读重算显示实际身体初值左肘约19.5px，求解后约84.9px；原版总目标仍降低，不能只靠更低总损失保证肢体正确。',
            '既有120k高斯和配套相机继续供查看，原高斯进程已由监督进程恢复。'],
        next='Inspect the actual stage trajectory of the original frame19 objective before proposing a loss or observation change. Register a bounded diagnostic first; do not repeat the rejected full59 fit or admit it to VHAP/Gaussians.')
    a.save(run/'review/closeout.json', closeout)
    seeds=a.read(run/'seed-path-audit.json')
    assert seeds['frames']==59 and seeds['state']=='all59_seed_paths_verified'
    closeout.update(seed_paths_verified=59,stage19_readonly_replay_sha256=a.sha(run/'review/stage19-replay-result.json'))
    a.save(run/'review/closeout.json',closeout)
    print(json.dumps({k:closeout[k] for k in ('state','frames','incremental_fit_seconds','independent_cache_fit_seconds','all_six_elbows_halved','protection_failures','baseline_resumed_verified')}))


if __name__=='__main__':main()
