"""Publish compact129 evidence. Never mutate a scientific input or latest model."""
import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

H=Path(__file__).resolve().parent
C=H.parent/'shuolee-coordination'
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'
ROOT=T/'shuolee/current/local-clothed-surface-123/cross-subject-sh0-124'
NATIVE=ROOT/'leafwoman-native129'
INPUT=ROOT/'leafwoman-half12-input129'
FIT=ROOT/'leafwoman-half12-pose129'
GS=ROOT/'leafwoman-half12-paired18k129'
V=T/'viewer'; S=Path('D:/00/工作/查看器')
BEGIN='<!-- BEGIN NEW-PERSON129 -->';END='<!-- END NEW-PERSON129 -->'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    b=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    t=p.with_name(p.name+'.tmp129');t.write_text(b,encoding='utf-8');t.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['fit_running','fit_complete','fit_reviewed','gaussian1000','gaussian_running'])
    a=parser.parse_args();stamp=datetime.now(timezone.utc).isoformat()
    p=read(INPUT/'plan.json');groups=NATIVE/'source-pose-review/pose-groups.json'
    assert sha(INPUT/'plan.json')=='0115b08fed63b9c93042ee00f985dedf81f2634ead26fac0c79ea78044ff2373'
    assert len(p['training_frames'])==24 and len(p['heldout_frames'])==4
    state=dict(updated=stamp,state='new_capture129_'+a.phase,subject='leafwoman_video22_capture_instance',
        physical_actor_identity_independently_verified=False,source_video_sha256=p['video_sha256'],
        native_frames=57,native_wh=[2704,1520],source_review_complete=True,
        source_groups_path=str(groups),source_groups_sha256=sha(groups),stable_source_episode='early_chest_leaf1..40',
        motion_onset_and_other_episodes_excluded=True,official_fit_candidate_count=24,Gaussian_target_count=12,
        training_roster_is_fit_pool_not_Gaussian_final_roster=True,heldout_count=4,all8_reserves_excluded=True,
        training_frames=p['training_frames'],heldout_frames=p['heldout_frames'],reference_frame=p['reference_frame'],
        input_source_plan_sha256=sha(INPUT/'plan.json'),conditional_K=True,second_undistortion=False,
        exact_static_3D_pose_guaranteed=False,cropped_feet_unknown_user_authorized=True,
        source_pixels_masks_XY_NPZ_modified=False,preprocessing_running=False,fitting_started=True,
        fitting_running=a.phase=='fit_running',gaussian_started=False,GPU_worker_running=a.phase=='fit_running',
        quality_accepted=False,end_to_end_15min_accepted=False,company00_verified=False,source_upload=False)
    if a.phase!='fit_running':
        e=read(FIT/'process-exit.json');assert e['exit_code']==0
        r=read(FIT/'result.json');assert r['frames_count']==24 and sha(FIT/'selected-fit.npz')==r['fit_sha256']
        state.update(fit_process_exit_code=0,fit_sha256=r['fit_sha256'],fit_result_path=str(FIT/'result.json'),
            fit_result_sha256=sha(FIT/'result.json'),fit_seconds=r.get('seconds'),fitting_running=False,GPU_worker_running=False)
    if (FIT/'root-visual-review.json').exists():
        review=read(FIT/'root-visual-review.json');state['fit_visual_review']=dict(path=str(FIT/'root-visual-review.json'),
            sha256=sha(FIT/'root-visual-review.json'),all24_actual_review=True,
            exploratory_Gaussian_max_step=1000,quality_accepted=False)
    if (FIT/'gs-selection/selection.json').exists():
        sel=read(FIT/'gs-selection/selection.json')
        state.update(Gaussian_selected_frames=sel['training_frames'],Gaussian_selected_count=len(sel['training_frames']),
            Gaussian_reference_frame=sel['reference_frame'],Gaussian_selection_sha256=sha(FIT/'gs-selection/selection.json'))
    summary='新video22：57帧原生观测与源图审阅完成；只使用前40帧连续胸前持叶动作，一次拟合24候选，预留4帧及4个备份；再按登记质量与时间分布选约12帧训练。'
    if a.phase=='fit_running':summary+=' 当前正在GPU执行原版五阶段拟合，尚无新高斯。'
    else:summary+=' 24候选拟合已退出0，正在核验真实mesh与侧视；拟合完成不等于质量通过。'
    if a.phase in ('fit_reviewed','gaussian1000'):
        summary+=' 全24个实际mesh与四方向已审阅；已从同一次拟合选出12帧，只准入1000步可见区域探索。参考第4帧睁眼，但直接支持的身体关节更少，代理低分不代表手部更准。'
    if a.phase in ('gaussian1000','gaussian_running'):
        shot=read(GS/'gaussian-run/step-001000/result.json');decision=read(GS/'review-1000-decision.json')
        assert shot['optimizer_updates']==1000 and decision['actual_visual_review'] is True
        state.update(gaussian_started=True,first_Gaussian_checkpoint_step=1000,first_checkpoint_optimizer_updates=1000,
            first_checkpoint_points=shot['points'],first_stage_compute_seconds=read(GS/'compute-phase-train.json')['cumulative_compute_seconds'],
            first_checkpoint_world_sha256=sha(GS/'gaussian-run/step-001000/world.ply'),
            actual_all12_Gaussian_review=True,actual_1000_decision=decision['decision'],
            continuation_maximum_total_steps=18000,quality_accepted=False)
        if a.phase=='gaussian_running':
            process=read(GS/'resume-process-started.json');live=read(GS/'progress.json')
            assert live['pid']==process['actual_pid'] and live['state']=='training'
            state.update(GPU_worker_running=True,training_running=True,actual_pid=process['actual_pid'],
                last_verified_step=live['step'],last_verified_points=live['points'],last_verified_progress_updated=live['updated'])
        alpha=read(GS/'gaussian-run/step-001000/alpha-summary129.json')
        state['alpha_summary_path']=str(GS/'gaussian-run/step-001000/alpha-summary129.json')
        state['alpha_summary_sha256']=sha(GS/'gaussian-run/step-001000/alpha-summary129.json')
        summary='新video22已完成一次24候选共享拟合并筛出12帧，SH0高斯实际完成1000次更新、33,767点、训练计算约75.35秒。原图/mask/当前mesh/高斯/累计alpha的12源视角及36个合成方向已审阅，并在本机实际打开点云＋参考4训练裁片相机ZIP，检查Gaussian、points及灰背景。'
        summary+=' 衣物、帽子和披肩轮廓已覆盖，但衣裙斑块、孔隙、融合手指和第18帧脸部重影仍明显。原生mask内部（诊断内缩5个派生像素）alpha<0.5占7.59%，<0.95占19.44%；是全层累计1−T，不证明最前表面闭合。1000步尚未触发1800周期opacity重置。'
        summary+=' 已实际决定沿同一检查点、同一名单和参数继续最多18,000步，验证原增密能否补足早期覆盖；不强制所有点不透明，不改变alpha/reset或其他科学参数，质量仍未通过。'
        if a.phase=='gaussian_running':summary+=' 当前同候选续训进程已核验启动。'
    state['next']='Review all24 native and synthetic-side fitted meshes; choose only supported fitted candidates by preregistered rules, then fresh1000-step SH0 Gaussian offline review.'
    if a.phase in ('gaussian1000','gaussian_running'):
        state['next']='Continue the same complete1000 checkpoint from1001 to at most18000 under original profile/budget; actual final RGB/alpha, surface/unknown-view and paired-camera viewer review. No old-person rerun or automatic opacity candidate.'
    for q in (R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',C/'local-research-status.json'):
        j=read(q);j['new_capture129']=state
        if q.name in ('queue-state.json','local-research-status.json'):
            j.update(updated_at=stamp,state=state['state'],execution_mode='goal',goal_status='active',next=state['next'])
        atomic(q,j)
    atomic(NATIVE/'root-progress.json',state)
    reports=[R/'video10-global-initializer/index.html',V/'diagnostics/video10-global-initializer/index.html',S/'diagnostics/video10-global-initializer/index.html']
    prior=[q.read_text(encoding='utf-8-sig') for q in reports];assert len(set(prior))==1,'Report copies diverged; no blind overwrite.'
    old=prior[0]
    if BEGIN in old:
        assert old.count(BEGIN)==old.count(END)==1
        start=old.index(BEGIN);finish=old.index(END)+len(END);outside=old[:start]+old[finish:]
    else:outside=old
    figure=''
    sources=[(NATIVE/'source-pose-review'/name,name) for name in ('pose-contact-04.jpg','pose-contact-11.jpg')]
    if a.phase in ('fit_reviewed','gaussian1000','gaussian_running'):
        sources += [(FIT/'review'/name,'fit-'+name) for name in ('page-02.png','page-04.png')]
    if a.phase in ('gaussian1000','gaussian_running'):
        sources += [(GS/'gaussian-run/step-001000/review/body-02.png','gaussian1000-body02.png'),
                    (GS/'ui-review/portable001000-camera4-gaussian-gray.png','gaussian1000-gray-UI.png')]
    for source,name in sources:
        for q in reports:
            dest=q.parent/'new-person129'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest);assert sha(dest)==sha(source)
        figure+='<figure><img loading="lazy" style="max-width:100%;height:auto" src="new-person129/'+name+'" alt="原生源图动作与mask审阅"></figure>'
    section=BEGIN+'<section id="new-person129"><h2>新片段022 · 胸前持叶人物</h2><p>'+html.escape(summary)+'</p>'
    section+='<p>24是拟合候选池，约12才是后续高斯视角；全部24仍影响共享体型和身体姿态。只在同一单次拟合中筛选，不重拟合，不把留出用于排名。时间分布不等于精确180度覆盖。</p>'
    section+='<p>遮挡或画外的膝脚保留未知。原mask不改，叶片道具包含不一致、头部视线变化及细边缘误差仍是限制。K仍是条件K，未通过独立物理标定。</p>'
    section+='<p>透明度：已有诊断中，错位手臂点可通过降低alpha被隐藏；所有层的累计alpha也不能证明前表面已闭合。不能把所有点强制不透明当几何修复。新模型将分别审阅原生累计alpha、衣物覆盖及灰背景透漏；尚未执行新透明度参数对照。</p>'+figure+'</section>'+END
    if a.phase in ('gaussian1000','gaussian_running'):
        package=GS/'delivery/step-001000/scene.zip';name='leafwoman_half12_SH0_001000_with_camera.zip'
        for q in reports:
            dest=q.parent/'new-person129'/name;shutil.copyfile(package,dest);assert sha(package)==sha(dest)
        for folder in (V,S):
            dest=folder/'exports'/name;shutil.copyfile(package,dest);assert sha(package)==sha(dest)
        link='<p><a download href="new-person129/'+name+'">下载1000步SH0诊断点云＋参考4相机ZIP</a>（33,767点，2,834,705字节；早期诊断，非最终质量交付）。只配同一静态姿态的一台完整K训练裁片相机，不将12个姿态伪装成12台静态相机。公司00、官方SIBR及独立Spark GUI未验收。</p>'
        section=section.replace('</section>'+END,link+'</section>'+END)
    assert '</body>' in outside
    fresh=outside.replace('</body>',section+'</body>',1)
    for q in reports:atomic(q,fresh)
    assert len({sha(q) for q in reports})==1
    plan=T/'shuolee/long-term-plan.md';text=plan.read_text(encoding='utf-8-sig')
    if BEGIN in text:
        s=text.index(BEGIN);e=text.index(END)+len(END);text=text[:s]+text[e:]
    atomic(plan,text.rstrip()+'\n\n'+BEGIN+'\n'+summary+'\n\n新人物源分组及24→12筛选是工程输入变化；原版求解器/科学训练profile保留。128已STOP，不恢复。透明度和错误姿态分开验收。\n'+END+'\n')
    for folder in (V,S):
        q=folder/'latest-result.json';j=read(q);keep={k:v for k,v in j.items() if k!='latestDiagnostic'}
        j['latestDiagnostic']=dict(updated=stamp,round='129',title='新片段022 · 24候选到12高斯视角',summary=summary,reportUrl='diagnostics/video10-global-initializer/index.html#new-person129',status=state['state'],qualityAccepted=False,goalComplete=False)
        atomic(q,j);assert {k:v for k,v in read(q).items() if k!='latestDiagnostic'}==keep
    atomic(NATIVE/'state-publication.json',dict(updated=stamp,phase=a.phase,all_three_report_copies_equal=True,
        latest_model_preserved=True,helper_sha256=sha(__file__),quality_accepted=False))
    print('LEAFWOMAN129_STATE_PUBLISHED',a.phase,sha(reports[0]))


if __name__=='__main__':main()
