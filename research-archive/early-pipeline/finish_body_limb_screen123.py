"""Record completed read-only screening; preserve the delivered model and trainer."""
from datetime import datetime, timezone
from pathlib import Path
import audit_gaussianavatars_params123 as a


def win(path):
    p=str(path)
    return Path(p[5].upper()+':/'+p[7:]) if p.startswith('/mnt/') else Path(p)


def main():
    run=a.REC/'demo-vhap-initialization/body-gaussian-upstream-offset'
    out=run/'limb-screen-v2';result=a.read(out/'result.json');manifest=a.read(out/'image-manifest.json')
    assert manifest['result_sha256']==a.sha(out/'result.json') and result['frames']==manifest['frames']==59
    for item in manifest['pages']+manifest['examples']: assert a.sha(win(item['path']))==item['sha256']
    fit=a.read(run/'gaussian-run/step-001000/quality-screen/result.json')
    union=sorted(set(fit['flagged_frames'])|set(result['warning_frames'])|set(result['unknown_frames']))
    now=datetime.now(timezone.utc).isoformat()
    visual=dict(updated=now,actual_visual_review=True,frames=59,contact_pages_reviewed=10,
        pages=manifest['pages'],result_sha256=a.sha(out/'result.json'),
        findings=[
            'Frames16,18-23 show visible RGB arm/shoulder defects; frame16 also has missing alpha connection.',
            'Frame18 whole-arm soft missing13.18% is below15%, but distal segment18.78% correctly prompts review.',
            'Frame51 proximal-arm21.51% is hidden by whole-arm13.88%.',
            'Frame23 still has poor arm appearance despite sufficient alpha; mask/opacity alone cannot establish correct body/colour/depth.',
            'Frames35-38 target mask removes portions of wrist/hand; undefined target support must remain unknown.',
            'Hands40/51/58/59 remain fragmented, and feet/face details remain imperfect. Limb screen is not a hand/facial-quality acceptance test.'
        ],previous_review_preserved=True,quality_accepted=False,auto_excluded_frames=[],current_training_changed=False)
    a.save(out/'visual-review.json',visual)
    captions={
        'frame_000016':'第16帧：原图／高斯／mask／真实透明度。前臂连接缺损，触发复核。',
        'frame_000018':'第18帧：整臂平均会掩盖局部问题，拆分前臂后触发复核。',
        'frame_000023':'第23帧：透明度连续仍不能证明肢体与颜色正确；保留这个漏检例，不放宽结论。',
        'frame_000033':'第33帧：参考姿态的对照，未触发配置内告警不等于质量验收。'}
    images=[dict(item,caption=captions[item['frame']]) for item in manifest['examples'] if item['frame'] in captions]
    report=dict(updated=now,step=60000,frames=59,warning_count=len(result['warning_frames']),
        unknown_count=len(result['unknown_frames']),fit_flags_count=len(fit['flagged_frames']),
        combined_review_count=len(union),combined_review_frames=union,
        conclusion='修正筛选分数接口，并拆分肢体检查：13帧覆盖预警，14帧存在不可靠观测或遮罩支持。两组有重合，均不自动删帧。',
        limits='这是同一6万步模型的只读筛选，不是新训练。分数为相对可靠性，未标定可见性；透明度连接仍会漏掉错误姿态、颜色或深度。原始拟合29帧预警继续保留，合并后需复核'+str(len(union))+'帧。',
        score_defect_scope='Only the older render-coverage screening used raw RTMW peaks; actual VHAP fitter already used relative_confidence. No claim this caused the trained distortion.',
        display_images=images,actual_visual_review=True,quality_accepted=False,model_promoted=False,
        source_result_sha256=a.sha(out/'result.json'),visual_review_sha256=a.sha(out/'visual-review.json'),
        source_fit_flags_sha256=a.sha(run/'gaussian-run/step-001000/quality-screen/result.json'),
        native_replay_sha256=a.sha(run/'gaussian-run/step-060000/render-coverage/native-replay.json'),
        auto_excluded_frames=[],current_training_changed=False,goal_complete=False)
    a.save(out/'report.json',report)
    summary=dict(updated=now,state='all59_screen_reviewed',step=60000,frames=59,
        warning_count=report['warning_count'],unknown_count=report['unknown_count'],combined_review_count=len(union),
        report=str(out/'report.json'),quality_accepted=False,model_promoted=False,current_training_changed=False,
        baseline_process_pid=331,baseline_progress=a.read(run/'progress.json'))
    for path in (a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json'):
        data=a.read(path);data['body_limb_screen_v2']=summary
        data['next_action']='Keep the live original-parameter trainer unchanged. The all59 60000-step screening correction is complete; do not repeat. Review the next immutable120k snapshot when saved. No automatic frame exclusion: combine fit reprojection/temporal flags, true-alpha segment coverage and actual RGB/geometry review. The next quality intervention must target fitted limb ambiguity under one registered causal change; do not use more Gaussian steps or alpha pass as a geometry fix.'
        a.save(path,data)
    for path in (a.R/'evidence-ledger.json',a.REC/'evidence-ledger.json'):
        data=a.read(path);data['body_limb_screen_v2']=dict(summary,score_interface_scope=report['score_defect_scope'],limitations=report['limits'])
        a.save(path,data)
    log=a.lp(a.R/'long-term-plan.md')
    marker='## 2026-11-02 60000-step limb screening correction'
    text=log.read_text(encoding='utf-8')
    if marker not in text:
        log.write_text(text+'\n\n'+marker+'\n\nRaw RTMW peaks were incorrectly gated as confidence only in the old opacity review; the actual fitter already used the normalized score contract. Reused that exact contract, split each limb, and checked true-alpha connection on all59 immutable60000-step views. 13 coverage warnings and14 unknown-support frames; combined fit/screen review='+str(len(union))+' frames. All10 contact pages reviewed; frame23 remains an alpha-screen appearance miss. No auto-exclusion, training change or model replacement. Native replay serialized GPU use and resumed PID331. Goal and15-minute end-to-end acceptance remain incomplete.\n',encoding='utf-8')
    print('LIMB_SCREEN_CLOSEOUT',len(union),'review; goal incomplete')


def record_ui():
    run=a.REC/'demo-vhap-initialization/body-gaussian-upstream-offset';out=run/'limb-screen-v2'
    screenshot=a.H/'body-60000-limb-screen-ui.png'
    report=a.read(out/'report.json');report['report_ui_verified']=True;a.save(out/'report.json',report)
    now=datetime.now(timezone.utc).isoformat()
    for folder in (a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer'):
        target=folder/'demo-body-gaussian-upstream/limb-screen-report.json'
        assert a.lp(target).exists();a.lp(target).write_bytes(a.lp(out/'report.json').read_bytes())
    expected=('local-surface-123-demo-body-smplx-offset-s060000',45958,'31f90191caea01ed5e5a25719716b654e96e3ce13a239f895b75896928371099')
    for folder in (a.V,a.S):
        path=folder/'latest-result.json';latest=a.read(path)
        assert tuple(latest[k] for k in ('modelId','points','sha256'))==expected
        latest['latestDiagnostic']=dict(updated=now,round='123-body-limb-screen-v2',title='6万步模型：肢体筛选复核',
            summary='修正分数接口、拆分肢体检查：13帧覆盖预警、14帧支持不足。未删训练帧，主模型仍为6万步。',
            reportUrl='diagnostics/video10-global-initializer/index.html#limb-screen',status='reviewed_screen_not_quality_acceptance',qualityAccepted=False,goalComplete=False)
        a.save(path,latest)
        assert tuple(a.read(path)[k] for k in ('modelId','points','sha256'))==expected
    a.save(out/'report-ui-review.json',dict(updated=now,actual_browser_review=True,tab='36',
        url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#limb-screen',
        screenshot=str(screenshot),screenshot_sha256=a.sha(screenshot),
        visible_counts={'coverage_warning_frames':13,'unknown_frames':14,'combined_review_frames':35},
        latest_model_preserved=True,quality_accepted=False))
    publication=a.read(run/'publication.json');publication.update(report_ui_verified=True,model_preserved_in_this_update=True)
    a.save(run/'publication.json',publication)
    pending=run/'gaussian-run/step-120000/result.json'
    if a.lp(pending).exists():
        snap=a.read(pending);assert snap['step']==120000
        for path in (a.R/'queue-state.json',a.REC/'queue-state.json',a.C/'local-research-status.json'):
            state=a.read(path)
            state['body_limb_screen_v2']['baseline_progress']=a.read(run/'progress.json')
            state['pending_body_checkpoint_review']=dict(step=120000,points=snap['points'],
                path=str(pending),source_sha256=snap['world_ply_sha256'],state='saved_not_reviewed_not_promoted')
            a.save(path,state)
    print('LIMB_SCREEN_UI_VERIFIED_MODEL_PRESERVED')


if __name__=='__main__':
    import sys
    record_ui() if '--ui-reviewed' in sys.argv else main()
