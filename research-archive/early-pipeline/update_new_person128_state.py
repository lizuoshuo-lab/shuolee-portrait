"""Publish compact, evidence-bound128 progress; never start a worker/model."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
H=Path(__file__).resolve().parent
NATIVE=ROOT/'graywoman-native128'
INPUT=ROOT/'graywoman-half12-input128'
FIT=ROOT/'graywoman-half12-pose128'
GS=ROOT/'graywoman-half12-paired18k128'
CONTROL=R/'checkpoint-lifecycle128'
V=T/'viewer'
S=Path('D:/00/工作/查看器')
C=H.parent/'shuolee-coordination'
BEGIN,END='<!-- NEW_PERSON128_BEGIN -->','<!-- NEW_PERSON128_END -->'
PBEGIN,PEND='<!-- NEW_PERSON128_STATE_BEGIN -->','<!-- NEW_PERSON128_STATE_END -->'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def optional(p): return read(p) if Path(p).exists() else {}
def atomic(p,data):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(data if isinstance(data,str) else json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,p)
def outside(text,begin,end):
    if begin not in text: return text
    assert text.count(begin)==text.count(end)==1
    a=text.index(begin);b=text.index(end,a)+len(end)
    return text[:a]+text[b:]


def main():
    stamp=datetime.now(timezone.utc).isoformat()
    n=read(NATIVE/'result.json');exit=read(NATIVE/'observe-process-exit.json')
    assert n['frames']==64 and n['observations_ready'] and exit['exit_code']==0
    assert n['video_sha256']=='965befa346fdb0b5e5140745ab5651ef6ed7f8a87ae32777e1d88017bc45e547'
    cpu=read(H/'staged_training_lifecycle128_cpu_audit.json')
    assert cpu['passed'] and sha(H/'staged_training_lifecycle128.py')==cpu['helper_sha256']
    strict=read(CONTROL/'strict-result.json')
    numerical=optional(CONTROL/'numerical-diagnosis128.json')
    zero=optional(CONTROL/'nonzero-boundary9/zero-update-restore9/result.json')
    summary=('新人物video15已完成64个原生3fps帧的mask与关键点。六源图显示一侧经正面到另一侧，按半圈12训练视角准备；双手在身后，遮挡手指保持未知。'
        '已补完整检查点保存／恢复接口，16项CPU检查通过。原生GPU连续与分段训练的逐位浮点比较未通过，差异在暂停前已出现；相机和随机抽样流一致，仍保留原失败。'
        '本轮人物尚未拟合或高斯训练，透明度修复及15分钟全流程均未验收。')
    if zero:
        summary+=' 同一非零动量检查点的零更新回读结果已另行登记，范围限工程恢复验证。'
    state=dict(updated=stamp,state='new_person128_observations_review_pending',subject='graywoman_video15',record_id='015',
        native_frames=64,native_result_path=str(NATIVE/'result.json'),native_result_sha256=sha(NATIVE/'result.json'),
        observations_sha256=n['observations_sha256'],conditional_K=True,second_undistortion=False,
        source_arc='qualitative side/front/opposite-side; no rear observed',training_view_target=12,
        source_visibility='hands behind lower back, concealed wrists/fingers unknown; cuff-covered ankle detail unknown',
        CPU_lifecycle_check_count=cpu['check_count'],native_strict_bitwise_passed=False,
        control_paths=dict(plan=str(CONTROL/'plan.json'),strict=str(CONTROL/'strict-result.json'),
            numerical=str(CONTROL/'numerical-diagnosis128.json'),zero_update=str(CONTROL/'nonzero-boundary9/zero-update-restore9/result.json')),
        source_binding='Current sourceSHA/intake images differ from trained people; video2 exact duplicate was rejected before person processing.',
        preprocessing_running=False,fitting_running=False,training_running=False,
        model_published=False,quality_accepted=False,end_to_end_15min_accepted=False,company00_verified=False,
        next='Finish independent checkpoint-native replay verification; review native masks/source support, freeze12/holdouts/reference, fresh official SMPLify-X then source/Gaussian admission.')
    source_plan=optional(INPUT/'plan.json')
    fit_result=optional(FIT/'result.json')
    if fit_result:
        fit_exit=read(FIT/'process-exit.json')
        assert fit_exit['exit_code']==0 and len(fit_result['frames'])==12
        assert source_plan['reference_frame']=='frame_000023'
        state.update(state='new_person128_fit_completed_review_pending',source_roster_frozen=True,
            training_frames=source_plan['training_frames'],heldout_frames=source_plan['heldout_frames'],
            reference_frame=source_plan['reference_frame'],source_plan_sha256=sha(INPUT/'plan.json'),
            fit_result_path=str(FIT/'result.json'),fit_result_sha256=sha(FIT/'result.json'),
            fit_process_exit_code=0,fit_seconds=fit_result['seconds'],fitting_running=False,
            next='Review all12 actual native/side meshes and source-supported face/body evidence before the new-person Gaussian admission.')
        summary=('新人物video15已完成64个原生3fps帧的mask与关键点，源图审阅后冻结12训练帧、4独立留出，参考帧23。'
            '全新neutral初始化的官方SMPLify-X五阶段联合拟合已实际完成，拟合计时154.28秒；这不是视频到模型全流程计时。'
            '目前正在审核全部12帧实际网格及侧面，尚未启动本轮高斯训练。双手在身后，腕和手指保持未知；面部检测点误差不作为物理标定真值。'
            '透明度修复、人物质量及15分钟全流程仍未验收。')
    snapshot=GS/'gaussian-run/step-001000'
    gs_exit=optional(GS/'process-exit.json')
    extra_assets=[]
    if gs_exit:
        shot=read(snapshot/'result.json');accounting=read(GS/'compute-phase-train.json')
        assert gs_exit['exit_code']==0 and shot['step']==shot['optimizer_updates']==1000
        assert shot['points']==34027 and accounting['successful']
        alphas=[read(snapshot/(f+'_cumulative-alpha.json')) for f in source_plan['training_frames']]
        regions={}
        for name in ('mask','mask_eroded_5_derived_pixels'):
            pixels=sum(a['regions'][name]['pixels'] for a in alphas)
            regions[name]=dict(pixels=pixels,**{k:sum(a['regions'][name]['pixels']*a['regions'][name][k] for a in alphas)/pixels
                for k in ('fraction_below_0_5','fraction_below_0_95')})
        alpha_summary=dict(step=1000,optimizer_updates=1000,frames=12,regions=regions,
            source_json_sha256={f:sha(snapshot/(f+'_cumulative-alpha.json')) for f in source_plan['training_frames']},
            measurement='Pixel-weighted all-layer native cumulative1-T; not front-surface opacity or physical clothing-layer truth.',
            training_changed=False,quality_accepted=False)
        atomic(snapshot/'alpha-summary128.json',alpha_summary)
        state.update(state='new_person128_diagnostic1000_offline_review',gaussian_started=True,
            training_running=False,actual_pid=None,actual_process_exit_code=0,step=1000,optimizer_updates=1000,
            points=34027,gaussian_compute_seconds=accounting['cumulative_compute_seconds'],
            snapshot_path=str(snapshot),snapshot_result_sha256=sha(snapshot/'result.json'),
            alpha_summary_path=str(snapshot/'alpha-summary128.json'),known_shared_arm_pose_failure=True,
            completed18000=False,continue_to18000_allowed=False,opacity_candidate_allowed=False,
            next='Actually review all12 Gaussian/mesh/alpha outputs; stop this known-pose-failure diagnostic. No18k or opacity causal claim.')
        er=regions['mask_eroded_5_derived_pixels']
        summary=('新人物video15的12训练／4留出已冻结，官方五阶段拟合实际完成。全部12帧及原生局部审阅发现：源图双手在背后，mesh前臂和手却垂在两侧大腿前，姿态质量未通过。'
            '为检查同一可复现流程，仅运行原配置1000步诊断，实际1000次更新、34,027点、计算约54.56秒后正常退出，已保存完整检查点，取消了600秒等待。'
            f'12视角诊断性腐蚀5个派生像素的mask内部，累计alpha低于0.5占{er["fraction_below_0_5"]*100:.4f}%，低于0.95占{er["fraction_below_0_95"]*100:.4f}%。'
            '这是整条射线所有层的1−T，不能证明前表面已封闭。当前审阅高斯图像；不进入18k、不做透明度因果候选，质量与15分钟全流程仍未验收。')
        extra_assets=[(FIT/'review/page-02.png','fit-page02.png','23、28、34、38帧实际网格投影及四个合成方向；前臂下垂与背后手臂源姿态不符。')]
    delivery=GS/'delivery/step-001000'
    opacity_proxy=optional(snapshot/'point-alpha-attribution-summary128.json')
    ui=optional(GS/'ui-review/portable-package-ui-review.json')
    decision=optional(GS/'review-1000-decision.json')
    download=''
    if ui:
        assert ui['actual_UI_checked'] and ui['OpenFile_ZIP_import'] and ui['points']==34027
        assert decision['decision']=='stop' and decision['actual_visual_review']
        assert sha(delivery/'scene.zip')==ui['package_sha256']
        state.update(state='new_person128_diagnostic1000_stopped_delivered',actual_visual_review=True,
            review_decision='stop',review_decision_sha256=sha(GS/'review-1000-decision.json'),
            native_36_synthetic_views_reviewed=True,local_OpenFile_Gaussian_points_camera_UI_verified=True,
            ui_receipt_path=str(GS/'ui-review/portable-package-ui-review.json'),
            package_path=str(delivery/'scene.zip'),package_sha256=ui['package_sha256'],package_bytes=ui['package_bytes'],
            model_published=False,diagnostic_package_delivered=True,
            next='Preserve128 diagnostic and STOP. Next new-capture intake129 has source motion/partial coverage; select source-supported stable views before fitting, keep cropped limbs unknown. No old-person rerun or automatic opacity candidate.')
        summary=summary.replace('当前审阅高斯图像；','已实际审阅全部12源视角、36个合成未知视角，并在本机查看器通过ZIP打开Gaussian、points和参考23配套相机。')
        summary+=' points模式仍显示灰色垂手点团，高斯模式中这些部位被隐藏；暂不能直接强制alpha=1。默认latest未替换，仅交付未通过质量验收的诊断包。'
        if opacity_proxy:
            state['opacity_point_proxy_evidence']=dict(path=str(snapshot/'point-alpha-attribution-summary128.json'),
                sha256=sha(snapshot/'point-alpha-attribution-summary128.json'),points_mode_ignores_per_point_alpha=True,
                hand_wrist_finger_binding_proxy_count=3118,reference23_all_proxy_centers_outside_mask=True,
                proxy_alpha_median=0.003921435699870748,forced_alpha_render_tested=False)
            summary+=' 只读核验进一步确认：3118个手腕／手指绑定代理点全部投在参考23的mask外，alpha中位0.0039214；points代码直接忽略逐点alpha。错位灰色几何仍存在，被低alpha压住；绑定代理与单相机mask归属不等于真实前后表面。'
        extra_assets.extend([
            (snapshot/'review/body-02.png','gs1000-body02.png','23、28、34、38帧：原图／mask／当前mesh／1000步高斯／累计alpha。'),
            (GS/'ui-review/portable001000-camera23-gaussian.png','gs1000-ui-gaussian.png','本机实际OpenFile ZIP、高斯渲染与配套参考23相机；仅1000步诊断。'),
            (GS/'ui-review/portable001000-camera23-points.png','gs1000-ui-points.png','同一相机的points模式仍显示灰色垂手点团；points显示不等于可见表面透明度。'),
            (delivery/'novel/orbit_12-23.png','gs1000-native-novel12-23.png','原生后端连续合成视角12—23；未拍背面的斑块与几何仍不可靠。'),
            (delivery/'scene.zip','graywoman_half12_diagnostic_SH0_001000_with_camera.zip',None)])
        download='<p><a download href="new-person128/graywoman_half12_diagnostic_SH0_001000_with_camera.zip">下载1000步SH0诊断点云＋参考23相机ZIP</a>（34,027点；2,853,189字节）。完整K及同坐标相机已配套；公司00、官方SIBR和独立Spark页面仍未实测。</p>'
    for p in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',C/'local-research-status.json']:
        j=read(p);j['new_person128']=state
        if p.name in ['queue-state.json','local-research-status.json']:
            j['state']=state['state'];j['next']=state['next'];j['updated_at']=stamp
        j['last_goal_turn_audit']=dict(updated=stamp,previous_turn='progress',current_progress=state['state'],goal_complete=False)
        next_result=optional(ROOT/'leafwoman-native129/result.json')
        next_exit=optional(ROOT/'leafwoman-native129/observe-process-exit.json')
        if next_exit:
            assert next_exit['exit_code']==0 and next_result['observations_ready'] and next_result['frames']==57
            next_state=dict(updated=stamp,state='new_capture129_observations_ready_source_pose_review_pending',
                subject='leafwoman_video22_new_capture_outfit_instance',physical_actor_identity_independently_verified=False,
                native_frames=57,native_wh=[2704,1520],training_view_target=12,
                source_video_sha256=next_result['video_sha256'],observations_sha256=next_result['observations_sha256'],
                native_result_path=str(ROOT/'leafwoman-native129/result.json'),
                actual_observe_exit_code=0,second_undistortion=False,conditional_K=True,
                source_only_stable_pose_group_review_required=True,training_roster_frozen=False,
                cropped_feet_unknown_user_authorized=True,preprocessing_running=False,fitting_started=False,gaussian_started=False,
                GPU_worker_running=False,quality_accepted=False,source_upload=False,
                next='Review source-only masks/native arm/face motion and freeze a stable pose group and12+independent holdouts; do not mix raised-arm sources into one shared body pose.')
            j['new_capture129']=next_state
            if p.name in ['queue-state.json','local-research-status.json']:
                j['state']=next_state['state'];j['next']=next_state['next']
        atomic(p,j)
    reports=[R/'video10-global-initializer/index.html',V/'diagnostics/video10-global-initializer/index.html',S/'diagnostics/video10-global-initializer/index.html']
    originals=[p.read_text(encoding='utf-8') for p in reports]
    assert originals[0]==originals[1]==originals[2]
    source=ROOT/'new-person-intake128/video15-arc/source-contact-six.png'
    asset='new-person128/source-contact-six.png'
    for p in reports:
        target=p.parent/asset;target.parent.mkdir(exist_ok=True)
        if not target.exists():target.write_bytes(source.read_bytes())
        assert sha(target)==sha(source)
    figures=''
    for source,name,caption in extra_assets:
        for p in reports:
            dest=p.parent/'new-person128'/name
            if not dest.exists():dest.write_bytes(source.read_bytes())
            assert sha(dest)==sha(source)
        if caption:
            figures+='<figure><img loading="lazy" src="new-person128/'+name+'" style="max-width:100%;height:auto" alt="'+caption+'"><figcaption>'+caption+'</figcaption></figure>'
    section=(BEGIN+'<section id="new-person128"><h2>新人物015 · 灰衣女性 · 半圈12视角</h2><p>'+summary+'</p>'
        '<figure><img loading="lazy" src="'+asset+'" style="max-width:100%;height:auto" alt="video15六张实际源图显示一侧、正面及另一侧"><figcaption>实际源图覆盖；手在背后，未拍背面不作为可靠监督。</figcaption></figure>'
        '<p>原生1520×2704，64帧真实PTS；原视频已经去畸变、下采样和裁剪，不再次去畸变。第一组YAML经既有变换得到的完整K仍是条件内参。源视频8bit，颜色标记缺失，不声称已独立确认BT709。</p>'
        +download+figures+'<p>原版求解器、损失和18k时钟保持；本次诊断预算只允许1000步。恢复接口只保存完整状态，审阅改为离线完成，禁止把127的旧不完整快照当作精确续训。没有发布新模型预设，默认模型保留足球126。</p>'
        '<p>证据存工程；后续各阶段图片按video15放D:/00/记录/015，仅图片。质量和公司00运行仍未验证。</p></section>'+END)
    page=outside(originals[0],BEGIN,END)
    marker='<!-- NEW_PERSON127_BEGIN -->';assert page.count(marker)==1
    page=page.replace(marker,section+marker,1)
    for p,old in zip(reports,originals):
        assert outside(page,BEGIN,END)==outside(old,BEGIN,END)
        atomic(p,page)
    plan=T/'shuolee/long-term-plan.md';old=plan.read_text(encoding='utf-8')
    text=outside(old,PBEGIN,PEND).rstrip()+'\n\n'+PBEGIN+'\n### 新人物128 · 灰衣女性\n\n'+summary+'\n\n原证据不覆盖：`checkpoint-lifecycle128/strict-result.json`逐位失败仍保留；同检查点非零动量零更新回读和原生forward另行通过。既有127停止、不恢复。128已完整保存post-update1000并实际STOP，不续18k，不作为opacity单变量结果。下一新片段129脚被裁掉可按用户授权保留未知；须识别手臂动作变化，不能混成已验证的同一共享姿态。\n'+PEND+'\n'
    atomic(plan,text)
    for folder in [V,S]:
        p=folder/'latest-result.json';j=read(p);before={k:v for k,v in j.items() if k!='latestDiagnostic'}
        j['latestDiagnostic']=dict(updated=stamp,round='128',title='新人物015 · 1000步诊断与透明度限制',summary=summary,reportUrl='diagnostics/video10-global-initializer/index.html#new-person128',status=state['state'],qualityAccepted=False,goalComplete=False)
        atomic(p,j);assert {k:v for k,v in read(p).items() if k!='latestDiagnostic'}==before
    atomic(NATIVE/'state-publication.json',dict(updated=stamp,all_three_report_copies_equal=True,report_outside_128_preserved=True,latest_model_preserved=True,quality_accepted=False,helper_sha256=sha(__file__)))
    print('NEW_PERSON128_COMPACT_STATE_PUBLISHED')


if __name__=='__main__':main()
