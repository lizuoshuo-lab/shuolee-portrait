"""Report actual officewoman127 phases; preserve history and the football model.

This module never starts a worker, changes observations, fits a person, trains,
or publishes a model. A missing PID/start identity is never treated as live.
"""
from datetime import datetime, timezone
import hashlib
import html
import os
from pathlib import Path

import audit_gaussianavatars_params123 as a
from publish_indoor_circle16_124 import atomic
from update_new_person126_state import optional, worker
from officewoman_fit_recovery_contract127 import verify_recovery

ROOT = a.REC/'cross-subject-sh0-124'
NATIVE = ROOT/'officewoman-native127'
SOURCE = ROOT/'officewoman-circle24-input127'
FIT = ROOT/'officewoman-circle24-pose127'
GS = ROOT/'officewoman-circle24-paired18k127'
ANCHOR = 'new-person127'
BEGIN,END = '<!-- NEW_PERSON127_BEGIN -->','<!-- NEW_PERSON127_END -->'
PLAN = a.T/'shuolee/long-term-plan.md'
PLAN_BEGIN,PLAN_END = '<!-- NEW_PERSON127_STATE_BEGIN -->','<!-- NEW_PERSON127_STATE_END -->'
VIDEO_SHA256 = '71d993f2f1e2c1e2ad1f5e3b58c4f351cd375e172f92c3526818dfb4ab380edb'


def native_path(value):
    value = str(value).replace('\\','/')
    return Path('//wsl.localhost/Ubuntu-20.04'+value) if value.startswith('/opt/') else Path(value)


def text_atomic(path,text):
    temporary = path.with_suffix(path.suffix+'.tmp')
    a.lp(temporary).write_text(text,encoding='utf-8')
    os.replace(a.lp(temporary),a.lp(path))


def outside_section(page):
    if BEGIN not in page:
        return page
    assert page.count(BEGIN)==page.count(END)==1
    left = page.index(BEGIN);right=page.index(END,left)+len(END)
    return page[:left]+page[right:]


def plan_update(previous,stamp,stage,phase,source_reviewed,training_frames,heldout_frames,reference,latest_model):
    visibility=('人工源图审阅已冻结面部及左右膝／脚四组支持标记；在scores副本上屏蔽不支持项后使用原q90，原始XY／NPZ、人体先验不改。'
        if source_reviewed else '面部及左右膝／脚四组源支持标记尚未冻结，不把检测器分数当真实可见性。')
    route=('本轮只保存1000步／28,901点；600秒审阅等待超时后进程退出1，没有完成18000步，没有补写存活续训决定或自动重启；下一步先修审阅生命周期，再在新人物做一次有界对照。'
        if stage=='new_person127_gaussian_review_timeout_snapshot_retained' else
        '沿原生mask→官方SMPLify-X共享体型→SH0表面绑定高斯既有路线；条件完整K、空个体初始化，先1000步实际审阅、最多18000步。')
    body=(f'### 新人物127 · 办公室女子（当前状态）\n\n'
        f'更新时间：{stamp}；阶段：`{stage}`。{phase}\n\n'
        f'video19原生1520×2704、3fps真实PTS共107帧；本轮训练名单{len(training_frames)}/24、独立留出{len(heldout_frames)}/8，参考帧{reference or "未冻结"}。'
        '上半身源图呈定性完整环绕，部分侧／背面腿脚出画保持未知，不称精确360°、完整人体覆盖或物理标定通过。'
        f'{visibility}这是本批人工可观测性处理，尚非验证通过的通用自动可见性方案。\n\n'
        f'{route}'
        f'当前默认登记模型：`{latest_model}`；足球126与历史结果保留，本报告不发布或替换点云。'
        '拟合、重建质量和15分钟端到端目标均未验收；不重启旧人物、不建立定时任务。'
        '\n\n[当前证据与下一步](http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#new-person127)\n')
    wrapped=PLAN_BEGIN+'\n'+body+PLAN_END
    if PLAN_BEGIN in previous or PLAN_END in previous:
        assert previous.count(PLAN_BEGIN)==previous.count(PLAN_END)==1
        left=previous.index(PLAN_BEGIN);right=previous.index(PLAN_END,left)+len(PLAN_END)
        changed=previous[:left]+wrapped+previous[right:]
        assert changed[:left]==previous[:left] and changed[left+len(wrapped):]==previous[right:]
    else:
        changed=previous+('' if previous.endswith('\n') else '\n')+'\n'+wrapped+'\n'
        assert changed.startswith(previous)
    assert changed.count(PLAN_BEGIN)==changed.count(PLAN_END)==1
    return changed


def main():
    stamp = datetime.now(timezone.utc).isoformat()
    progress,result = a.read(NATIVE/'progress.json'),a.read(NATIVE/'result.json')
    assert result['video_sha256']==VIDEO_SHA256 and result['frames']==107
    native_worker = worker(NATIVE,progress,'prepare_newperson127.py','observe')
    ready = result.get('observations_ready',False)
    if ready:
        assert result['state']=='observations_complete_review_pending'
        data = native_path(result['data_root'])
        # a.lp is for drive paths; applying its drive-prefix spelling to a
        # WSL UNC path creates an invalid path. Read this short UNC directly.
        assert hashlib.sha256((data/'data/inputs.json').read_bytes()).hexdigest()==result['observation_manifest_sha256']
        assert hashlib.sha256((data/'data/observations.npz').read_bytes()).hexdigest()==result['observations_sha256']
        assert len(__import__('json').loads((data/'data/inputs.json').read_text(encoding='utf-8-sig'))['records'])==107
    source = optional(SOURCE/'plan.json')
    fit = {name:optional(FIT/(name+'.json')) for name in
        ('plan','launch','progress','result','process-exit','conclusion','failure','run-failure')}
    recovery=None
    try:
        recovery=verify_recovery(FIT,Path(__file__).with_name('run_officewoman_circle24_fit127.py'))
    except (AssertionError,ValueError,KeyError,OSError,TypeError,IndexError) as error:
        fit['metadata-recovery-verification-error']=str(error) or type(error).__name__
    if recovery:
        fit['metadata-postprocess-recovery']=recovery
    gs = {name:optional(GS/(name+'.json')) for name in
        ('plan','launch','progress','process-exit','train-failure')}
    gs['result'] = optional(GS/'gaussian-run/result.json')
    gs['publication'] = optional(GS/'delivery/publication.json')
    gs['review-1000-decision']=optional(GS/'review-1000-decision.json')
    diagnostic=None
    delivery=GS/'delivery/step-001000'
    if a.lp(delivery/'viewer-ui-review.json').exists():
        ui=a.read(delivery/'viewer-ui-review.json');package=a.read(delivery/'package-result.json')
        assert all(ui[k] is True for k in ('actual_ui_verified','open_file_zip','gaussian_mode','point_mode','imported_camera_selected_in_UI','derived_full_K_preserved'))
        assert ui['step']==package['step']==1000 and ui['optimizer_updates']==package['optimizerUpdates']==999
        assert ui['points']==package['points']==28901 and ui['camera_count']==package['cameraCount']==1
        assert ui['reference_frame']==package['referenceFrameId']=='frame_000094'
        assert package['diagnosticOnly'] and not package['planned18000Completed'] and not ui['quality_accepted']
        assert a.sha(delivery/'scene.zip')==package['sha256']
        for proof in ui['proofs']:
            assert a.sha(Path(proof))==ui['proof_sha256'][Path(proof).name]
        diagnostic=dict(step=1000,optimizer_updates=999,points=28901,reference_frame='frame_000094',
            camera_count=1,package_sha256=package['sha256'],package_bytes=a.lp(delivery/'scene.zip').stat().st_size,
            local_OpenFile_Gaussian_points_camera_UI_verified=True,viewer_ui_receipt_sha256=a.sha(delivery/'viewer-ui-review.json'),
            quality_accepted=False,completed18000=False,company00_verified=False,latest_model_changed=False,
            report_relative='new-person127/officewoman_circle24_diagnostic_SH0_001000_with_camera.zip')
        gs['diagnostic-download']=diagnostic
    fit_worker = worker(FIT,fit['progress'],'run_officewoman_circle24_fit127.py','run')
    gs_worker = worker(GS,gs['progress'],'train_officewoman_circle24_sh0127.py','train')
    preprocessing_live,fitting_live,training_live = native_worker['live'],fit_worker['live'],gs_worker['live']
    active = gs_worker if training_live else fit_worker if fitting_live else native_worker if preprocessing_live else {}
    stage = ('new_person127_observations_ready' if ready else
        'new_person127_observations_running' if preprocessing_live else 'new_person127_observations_process_unverified')
    next_action = ('Review the107new masks/body-face-foot observations; register24quality/diverse training candidates and8separate heldouts, then actually review source visibility. No old-person reruns.' if ready else
        'Inspect the same registered observation worker and terminal files; do not launch a duplicate or infer liveness from stale progress.')
    phase = '107个原生3fps真实PTS帧的mask及RTMW133观测已完成，待源观测审阅。' if ready else '新观测阶段尚未记录完整完成。'
    training_frames = source.get('training_frames',[])
    heldout_frames = source.get('heldout_frames',[])
    reference = source.get('reference_frame') or fit['plan'].get('reference_frame')
    source_frozen = bool(source.get('training_roster_ready'))
    source_observability_reviewed = bool(source_frozen and source.get('actual_source_visual_review') is True)
    if source_observability_reviewed:
        groups=source['joint_group_visibility']
        expected={'left_knee','right_knee','left_foot','right_foot'}
        assert set(groups)==set(training_frames+heldout_frames)
        assert all(set(flags)==expected and all(type(v) is bool for v in flags.values()) for flags in groups.values())
        assert all(row['source_joint_group_supported']==groups[row['frame']] for row in source['rows'])
    if training_frames:
        assert len(set(training_frames))==len(training_frames)==24
        assert len(set(heldout_frames))==len(heldout_frames)==8
        assert not set(training_frames)&set(heldout_frames)
        assert reference in training_frames
        stage = 'new_person127_source24_frozen_review_pending' if source_frozen else 'new_person127_source24_candidates_registered'
        if source_frozen and source.get('actual_source_visual_review'):
            phase = '107个原生帧的mask及关键点已完成；24个训练视角和8个独立留出的源图与可见性已审阅并冻结。拟合与重建质量须另行审阅。'
        else:
            phase += ' 已登记24个训练候选及8个独立留出；源可见性与拟合质量须另行审阅。'
        next_action = 'Actually review the registered24training+8heldout source observations and mask visibility. Fit only after saved source admission; keep heldout coordinates/scores out of training statistics.'
    if fit['plan']:
        stage = 'new_person127_fitting_prepared'
        phase += ' 新人物共享体型拟合接口已准备。'
    if (fit['process-exit'].get('exit_code',0)!=0 or fit['failure'] or fit['run-failure']) and not recovery:
        stage='new_person127_fitting_failed'
        next_action='Inspect the saved current fitting failure and terminal evidence; no automatic retry, old-model initialization or Gaussian admission from loss alone.'
    elif fit['result']:
        stage='new_person127_fit_metadata_recovered_review_pending' if recovery else 'new_person127_fit_completed_review_pending'
        phase += (' 原拟合五阶段已执行并保存selected-fit；原封装因空支持组的指标汇总退出1，原错误和退出码保留。CPU仅重建指标／元数据，已核验同一selected-fit及原日志哈希；不视为拟合质量通过。' if recovery else
            f' 本轮拟合记录完成，用时{fit["result"].get("seconds",0):.1f}秒。')
        next_action='Actually review the24source-frame body/face/feet and mesh pose depth before admitting one bounded Gaussian candidate.'
        conclusion=fit['conclusion']
        if conclusion.get('actual_visual_review'):
            admitted=conclusion.get('exploratory_gaussian_admitted',False)
            stage='new_person127_fit_reviewed_gaussian_ready' if admitted else 'new_person127_fit_review_not_admitted'
            next_action=('Use only this newly reviewed fit for the registered SH0/ROI2x/paired-background candidate; actual1000review before further steps.' if admitted else
                'Keep this explicit fit-quality review and defects; this report does not authorize an automatic Gaussian launch.')
            phase += ' '+(conclusion.get('finding_zh') or conclusion.get('finding') or '实际拟合审阅已记录，质量仍未验收。')
    elif fitting_live:
        stage='new_person127_fitting_running'
        phase += ' 已核验当前拟合进程的完整命令与PID启动身份，正在运行。'
        next_action='Inspect this same new-person fitter; after its actual terminal completion perform source and spatial mesh review. Do not duplicate it.'
    elif fit['launch'] and not fit['result']:
        stage='new_person127_fitting_process_unverified'
        next_action='Reconcile the saved fitter PID/start identity/exact argv with terminal artifacts; stale running metadata is not a live process.'
    gaussian_started=bool(gs['launch'] or gs['result'] or a.lp(GS/'gaussian-run').exists())
    review_timeout=False
    if gs['plan']:
        step=gs['progress'].get('step',gs['result'].get('snapshot_step',0))
        phase += f' 高斯记录到{step:,}步，{gs["progress"].get("points",gs["result"].get("points",20908)):,}点。'
        stage='new_person127_gaussian_prepared'
        next_action='Inspect this prepared new-person candidate only; do not duplicate its launch.'
        if training_live:
            stage='new_person127_gaussian_review_waiting' if gs['progress'].get('optimizer_waiting') else 'new_person127_gaussian_training'
            next_action='Inspect the same live candidate; actually review its1000step source/unknown views before any continuation. No silent LR/loss/opacity changes.'
        elif gs['process-exit'].get('exit_code',0)!=0 or gs['train-failure']:
            stage='new_person127_gaussian_failed_checkpoint_retained'
            next_action='Inspect current failure/checkpoint evidence without automatic retry or threshold relaxation.'
        elif gs['result'].get('state')=='bounded_control_completed':
            stage='new_person127_gaussian_completed_review_export_pending'
            next_action='Actually review all24source and continuous unknown views, then perform native/PLY/camera package and local viewer checks; completion is not quality acceptance.'
        elif gs['result']:
            stage='new_person127_gaussian_stopped_review_required'
            next_action='Read the saved terminal Gaussian state and stop reason. Do not describe stopped computation as successful completion.'
        elif gs['launch']:
            stage='new_person127_gaussian_process_unverified'
            next_action='Reconcile the recorded Gaussian PID/start identity/exact command and actual artifacts before any follow-up.'
        if gs['publication']:
            stage='new_person127_delivered_quality_open' if gs['publication'].get('ui_checked') else 'new_person127_exported_ui_pending'
            next_action='Retain the named new-person result and its quality limitations; do not infer broad robustness or15minute acceptance.' if gs['publication'].get('ui_checked') else 'Complete actual local preset/OpenFileGaussian/Points/same-coordinate-camera checks for this new-person package.'
    if (not training_live and gs['process-exit'].get('exit_code')==1 and
        gs['train-failure'].get('error')=="AssertionError('Review timeout; checkpoint retained')"):
        review_timeout=True
        gs['snapshot1000']=a.read(GS/'gaussian-run/step-001000/result.json')
        gs['alpha-summary127']=a.read(GS/'gaussian-run/step-001000/alpha-summary127.json')
        assert gs['snapshot1000']['step']==1000 and gs['snapshot1000']['points']==28901
        assert gs['alpha-summary127']['step']==1000 and gs['alpha-summary127']['frames']==24
        assert gs['alpha-summary127']['training_changed'] is False
        alpha=gs['alpha-summary127']['regions'];ma=alpha['mask'];er=alpha['mask_eroded_5_derived_pixels']
        stage='new_person127_gaussian_review_timeout_snapshot_retained'
        phase+=(' 1000步审阅等待超过既定600秒，实际进程退出1；这是本轮审阅协调失误，原失败记录保留。'
            '停止决定在进程退出后补记，不是存活期间的续训决定；只保存28,901点的1000步结果，没有完成18000步，也没有自动重训。'
            f'24视角按像素加权的累计alpha：mask内低于0.5为{ma["fraction_below_0_5"]*100:.5f}%，'
            f'诊断性腐蚀5个派生像素后低于0.5为{er["fraction_below_0_5"]*100:.5f}%，低于0.95为{er["fraction_below_0_95"]*100:.5f}%。'
            '这是原生渲染器沿整条射线、所有层的累计1−T，不是仅前表面透明度；诊断腐蚀没有改训练mask。'
            '76／80／82帧裤侧有源轮廓外的浅色凸出块，衣缘碎片、模糊面部与手指仍在，质量不通过。')
        next_action='Retain the1000snapshot and actualexit1/lateSTOP evidence; no continuation of the dead worker, automatic retry, old-person rerun,18k claim or model publication.'
        next_action+=' Repair the review lifecycle before one bounded comparison on the next new person; the candidate remains unimplemented.'
    entry=dict(updated=stamp,subject='officewoman_video19',native_root=str(NATIVE),record_root='D:/00/记录/019',
        native_progress=progress,native_result=result,native_process=native_worker,fit_root=str(FIT),fit=fit,
        gaussian_root=str(GS),gaussian=gs,fit_process=fit_worker,gaussian_process=gs_worker,
        actual_pid=active.get('pid'),actual_command=active.get('observed_command',''),
        preprocessing_running=preprocessing_live,fitting_running=fitting_live,training_running=training_live,
        native_terminal_result_observed=ready,native_observe_exit_code=0 if ready else None,
        native_exit_evidence='Root observed original tool session68898 complete exit0; terminal result107/107, no invented launch PID.' if ready else None,
        training_frame_target=24,heldout_frame_target=8,source_roster_frozen=source_frozen,
        source_frames=training_frames,heldout_frames=heldout_frames,reference_frame=reference,
        full_circle_source_contact_only=True,physical_angles_accepted=False,conditional_K=True,
        full_circle_coverage_scope='Qualitative upper-body source contact only; not legs/feet observed from every angle.',
        source_observability_manually_reviewed=source_observability_reviewed,
        partial_side_rear_legs_and_feet_unknown=True,
        generic_automatic_visibility_validated=False,
        physical_calibration_accepted=False,old_person_retraining=False,new_gaussian_started=gaussian_started,
        review_timeout_600s=review_timeout,root_review_coordination_failed=review_timeout,
        completed_18000=False if review_timeout else None,
        research_followup_paths=[str(GS/'diagnostics/opacity-followup-plan127.json'),str(Path(__file__).with_name('checkpoint-lifecycle-readonly-audit127.json'))],
        diagnostic_download=diagnostic,
        quality_accepted=False,end_to_end_15min_accepted=False,company00_deployed=False)
    reports=[a.R/'video10-global-initializer/index.html',a.V/'diagnostics/video10-global-initializer/index.html',a.S/'diagnostics/video10-global-initializer/index.html']
    originals=[a.lp(p).read_text(encoding='utf-8') for p in reports]
    assert originals[0]==originals[1]==originals[2], 'Report copies diverged; preserve history instead of overwriting.'
    baseline_latest={str(folder):{k:v for k,v in a.read(folder/'latest-result.json').items() if k!='latestDiagnostic'} for folder in (a.V,a.S)}
    viewer_hashes={str(folder):a.sha(folder/'viewer.js') for folder in (a.V,a.S)}
    assert baseline_latest[str(a.V)]['modelId']==baseline_latest[str(a.S)]['modelId']
    original_plan=a.lp(PLAN).read_text(encoding='utf-8')
    updated_plan=plan_update(original_plan,stamp,stage,phase,source_observability_reviewed,
        training_frames,heldout_frames,reference,baseline_latest[str(a.V)]['modelId'])
    roster_text='<p>目标24个训练视角＋8个独立留出；名单与优选参考帧尚未冻结。每帧可见性、mask及姿态质量不由环绕外观自动保证。</p>'
    if training_frames:
        ids='、'.join(str(int(x.split('_')[-1])) for x in training_frames)
        held='、'.join(str(int(x.split('_')[-1])) for x in heldout_frames)
        roster_text=(f'<p>训练候选{ids}；独立留出{held}；参考{html.escape(str(reference))}。'
            '源名单登记与最终拟合／重建质量验收分开；留出坐标和分数不参与训练归一化。</p>')
    publication_text=''
    timeout_text=''
    report_assets=[]
    if review_timeout:
        ui_image=(GS/'ui-review/portable001000-camera94-gaussian.png' if diagnostic else GS/'ui-review/step001000-gray-front.png')
        ui_name='step001000-package-camera94-gaussian.png' if diagnostic else 'step001000-ui-gray-front.png'
        assets=[(GS/'gaussian-run/step-001000/review/body-05.png','step001000-source-body05.png'),(ui_image,ui_name)]
        if diagnostic:
            assets.append((delivery/'scene.zip','officewoman_circle24_diagnostic_SH0_001000_with_camera.zip'))
        for source,name in assets:
            data=a.lp(source).read_bytes();digest=hashlib.sha256(data).hexdigest()
            for report in reports:
                dest=report.parent/'new-person127'/name
                a.lp(dest.parent).mkdir(exist_ok=True)
                if a.lp(dest).exists():
                    assert a.sha(dest)==digest
                else:
                    a.lp(dest).write_bytes(data)
                assert a.sha(dest)==digest
            report_assets.append(dict(source=str(source),sha256=digest,report_relative='new-person127/'+name))
        timeout_text='''<p><strong>实际停止：600秒审阅等待超时，进程退出1。</strong>本轮审阅协调未在时限内完成；退出后补记STOP，不是存活续训。保留1000步快照及原失败，没有18000步结果，没有自动重启。侧面76／80／82裤侧的浅色凸出块超出对应源轮廓；衣物边缘、头发和手部仍有碎片／模糊，质量未通过。</p>
<figure><img loading="lazy" src="new-person127/step001000-source-body05.png" style="max-width:100%;height:auto" alt="1000步四个源视角的原图、mask、当前mesh、高斯和累计alpha对照"><figcaption>76、80、82、89帧：原图／原mask／当前mesh／1000步高斯／累计alpha。各组按现有CPU审阅图展示，不作为独立几何真值。</figcaption></figure>
<figure><img loading="lazy" src="new-person127/step001000-ui-gray-front.png" style="max-width:100%;height:auto" alt="实际查看器1000步灰色背景正面截图"><figcaption>实际OpenFile交互检查截图，灰色背景用于观察边缘与透明缺口；不是00部署或正式点云相机包发布。</figcaption></figure>'''
        if diagnostic:
            timeout_text=timeout_text.replace('new-person127/step001000-ui-gray-front.png','new-person127/'+ui_name).replace('实际查看器1000步灰色背景正面截图','实际OpenFile ZIP与参考94相机的1000步高斯截图').replace('实际OpenFile交互检查截图，灰色背景用于观察边缘与透明缺口；不是00部署或正式点云相机包发布。','实际OpenFile ZIP已核验Gaussian、points和配套参考94相机；仅1000步／999次更新的诊断交付，质量未通过，未完成18000步，未验证公司00。')
            timeout_text+='<p><a download href="'+diagnostic['report_relative']+'">下载1000步SH0诊断点云＋参考94相机ZIP</a>（28,901点，2,425,488字节）。未注册新模型预设，默认latest仍保留足球126；透明度缺口尚未修复。下一步先修审阅生命周期，再在新人物做一次有界对照，候选目前只作计划。</p>'
    source_notice=''
    if source_observability_reviewed:
        source_notice='<p>源可见性更正：部分侧面／背面帧只拍到大腿，腿脚在画外，保持未知；不能把检测器在裤子上猜出的膝、踝、脚趾当作实际观测。完整环绕仅指上半身的定性源图覆盖，不表示所有方向都看到了完整腿脚。本轮按人工审阅的逐帧面部、左右膝和左右脚可见性，在分数副本上屏蔽不支持的组，再使用原q90流程；原始像素坐标、NPZ及人体先验不改。这是本批源图的人工可观测性处理，尚未验证为通用自动可见性方法。</p>'
    if gs['publication']:
        pub=gs['publication']
        publication_text=f'<p><a href="{html.escape(pub["viewerUrl"])}">打开本轮已登记高斯与配套相机</a>。模型仍按实际审阅记录其局限，不称00已部署。</p>'
    block=f'''<section id="{ANCHOR}"><h2>新人物019 · 办公室短发黑衣女子</h2>
<p><strong>{html.escape(phase)}</strong></p>
<p>video19.mp4，1520×2704，35.707秒，858个编码原帧；已有3fps真实PTS取样107帧。原生像素不旋转、不再次去畸变。10bit BT709 SDR沿既有入口转RGB24 8bit，不声称保留10bit。</p>
<p>六个实际源图显示正面、双侧、双侧后方及回到正面，支持按完整环绕选约24帧；这是源图覆盖判断，不是精确360°、外参标定或完整人体质量通过。</p>
{source_notice}
<p>沿现有YOLOv8n／SAM2 tiny生成mask，RTMW133在CPU执行并进行双翻转观测。第一组YAML K以4000×3000为条件基准，按高度2704半像素缩放后理想居中裁宽1520；中间宽度与裁剪整数舍入仍未独立核实，完整K不是物理标定验收。</p>
{roster_text}<p>当前阶段：{html.escape(stage)}。{'实际worker PID：'+str(active['pid'])+'。' if active else '当前没有已核验的本轮GPU worker；不将历史running记录当作存活。'}不读取旧人物体型、优化姿态或高斯，不调用GVHMR模板替代官方SMPLify-X。</p>
{timeout_text}{publication_text}<p>研究资料留在工程；本人物各阶段图片按规则放D:/00/记录/019，仅图片。既有足球126模型与旧人物结果保留；本节仅更新当前研究状态，不替换已有高斯。质量、广泛鲁棒性与视频到导出15分钟均未验收。</p></section>'''
    wrapped=BEGIN+block+END
    page=originals[0]
    before_outside=outside_section(page)
    if BEGIN in page:
        left=page.index(BEGIN);right=page.index(END,left)+len(END)
        page=page[:left]+wrapped+page[right:]
    else:
        marker='<!-- NEW_PERSON126_END -->'
        assert page.count(marker)==1
        page=page.replace(marker,marker+wrapped,1)
    assert outside_section(page)==before_outside
    assert page.count(f'id="{ANCHOR}"')==1
    for path in reports:
        text_atomic(path,page)
    text_atomic(PLAN,updated_plan)
    for folder in (a.V,a.S):
        doc=a.read(folder/'latest-result.json')
        doc['latestDiagnostic']=dict(updated=stamp,round='127',title='新人物019 · 完整环绕24帧路线',
            summary=phase,reportUrl='diagnostics/video10-global-initializer/index.html#'+ANCHOR,
            status=stage,qualityAccepted=False,goalComplete=False)
        atomic(folder/'latest-result.json',doc)
        assert {k:v for k,v in a.read(folder/'latest-result.json').items() if k!='latestDiagnostic'}==baseline_latest[str(folder)]
        assert a.sha(folder/'viewer.js')==viewer_hashes[str(folder)]
    for path in (a.R/'protocol.json',a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json'):
        doc=a.read(path)
        doc.update(updated=stamp,current_stage=stage,new_person127=entry,preprocessing_running=preprocessing_live,
            fitting_running=fitting_live,training_running=training_live,own_gpu_task_running=bool(active),
            current_worker_pid_wsl=active.get('pid'),next_action=next_action,quality_accepted=False,
            end_to_end_15min_accepted=False,goal_complete=False,goal_turn_classification='progress',goal_turn_blocked_streak=0)
        if path==a.C/'local-research-status.json':
            doc.update(updated_at=stamp,state=stage,next=next_action)
        atomic(path,doc)
    receipt=dict(updated=stamp,stage=stage,actual_process_live=bool(active),native_terminal_result_observed=ready,
        native_process=native_worker,fit_process=fit_worker,gaussian_process=gs_worker,
        report_url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#'+ANCHOR,
        report_outside_section_preserved=True,all_three_report_copies_equal=True,
        long_term_plan_section_updated=True,long_term_plan_history_preserved=True,
        long_term_plan_sha256=a.sha(PLAN),
        report_assets=report_assets,review_timeout_600s=review_timeout,
        previous_latest_model_preserved=True,viewer_code_unchanged=True,ui_verified=False,
        helper_sha256=a.sha(Path(__file__)),quality_accepted=False)
    atomic(NATIVE/'state-publication.json',receipt)
    assert all(a.lp(p).read_text(encoding='utf-8')==page for p in reports)
    print(stage,phase,flush=True)


if __name__=='__main__':
    main()
