"""Publish measured new-person phases and the preserved read-only opacity audit."""
from datetime import datetime, timezone
import html
import shlex
import subprocess
import audit_gaussianavatars_params123 as a
from publish_indoor_circle16_124 import atomic


def optional(path):
    return a.read(path) if a.lp(path).is_file() else {}


def worker(folder, progress, script, mode):
    """Read the real Linux argv/start time; a recycled PID is not this worker."""
    launch = optional(folder/'launch.json')
    pid = launch.get('pid') or progress.get('pid')
    observed = {}
    if pid:
        pid = int(pid)
        code = '''import pathlib,json,sys
p=pathlib.Path('/proc')/str(int(sys.argv[1]))
try:
 stat=(p/'stat').read_text(); argv=(p/'cmdline').read_bytes().decode(errors='replace').strip('\\0').split('\\0')
 print(json.dumps(dict(argv=argv,start_ticks=stat[stat.rfind(')')+2:].split()[19])))
except (FileNotFoundError,ProcessLookupError): print('{}')
'''
        proc = subprocess.run(['wsl','-d','Ubuntu-20.04','--','python3','-c',code,str(pid)],capture_output=True)
        if proc.returncode == 0:
            try:
                observed = __import__('json').loads(proc.stdout.decode('utf-8',errors='replace').strip())
            except ValueError:
                observed = {}
    argv = observed.get('argv',[])
    expected = launch.get('command',[])
    if isinstance(expected,str): expected=[]
    expected_script = next((part for part in expected if str(part).endswith('/'+script)),None)
    matches = any(str(part).endswith('/'+script) and index+1<len(argv) and argv[index+1]==mode
                  and (expected_script is None or part==expected_script)
                  for index,part in enumerate(argv))
    registered_start = launch.get('proc_start_ticks',launch.get('proc_start_identity'))
    identity = registered_start is not None and str(registered_start)==str(observed.get('start_ticks'))
    # A missing start identity is reported as unverified, never inferred from argv alone.
    return dict(pid=pid,registered_launch=launch,observed_argv=argv,
                registered_command=shlex.join(expected),observed_command=shlex.join(argv),observed_start_ticks=observed.get('start_ticks'),
                command_matches=matches,start_identity_matches=identity,live=bool(matches and identity))


def main():
    root = a.REC / 'cross-subject-sh0-124'
    native = root / 'football-native126'
    progress, result = a.read(native/'progress.json'), a.read(native/'result.json')
    alpha = a.read(root/'winterhat-half8-paired18k124/opacity-readonly-audit126.json')
    now = datetime.now(timezone.utc).isoformat()
    native_worker = worker(native,progress,'prepare_newperson126.py','observe')
    live = native_worker['live']
    assert result['video_sha256'] == '4e1c1e52818362dcab22bb3ff9548a2d1a8c8b3f6affb8bd13dacfcc12a706a0'
    ready = result.get('observations_ready',False)
    stage = 'new_person126_observations_ready' if ready else 'new_person126_observations_running' if live else 'new_person126_observations_pending'
    next_action = ('Review this new person masks/body2D/face and freeze approximately12 quality/diverse source frames plus separate holdouts; confidence statistics train-only, fresh neutral SMPLify-X fit. Do not restart old people.' if ready else 'Inspect the existing new-person observation process only; no duplicate extraction/detection and no old-person retraining.')
    source = optional(root/'football-half12-input126/plan.json')
    fitting = root/'football-half12-pose126'
    gaussian = root/'football-half12-paired18k126'
    fit = {name:optional(fitting/(name+'.json')) for name in
           ('plan','preflight','launch','progress','result','process-exit','conclusion')}
    gs = {name:optional(gaussian/(name+'.json')) for name in
          ('plan','validity-cuda-preflight','preflight-failure','launch','progress','process-exit','train-failure')}
    gs['result']=optional(gaussian/'gaussian-run/result.json')
    gs['publication']=optional(gaussian/'delivery/publication.json')
    facing_path=gaussian/'diagnostics/ref26-facing-proxy018000/result.json'
    facing=optional(facing_path)
    gs['facing_proxy_diagnostic']=facing
    alpha_summaries=sorted((gaussian/'gaussian-run').glob('step-*/alpha-summary126.json'))
    alpha_text=''
    if alpha_summaries:
        alpha_path=alpha_summaries[-1]
        measured=a.read(alpha_path)
        assert measured['frames_count']==12 and measured['raw_float32_arrays_verified']==12
        assert measured['all_npy_png_mask_hashes_verified'] and measured['all_raw_fullmask_and_eroded_metrics_recomputed']
        assert {row['frame'] for row in measured['frames']}==set(source['training_frames'])
        gs['cumulative_alpha_summary']=dict(path=str(alpha_path),sha256=a.sha(alpha_path),measurement=measured)
        full=measured['totals']['mask']; interior=measured['totals']['mask_eroded_5_derived_pixels']
        face=measured['review_region_ranges']['face']; alpha_step=measured['step']
        alpha_text=(f'<h4>新足球人物{alpha_step}步 · 12个源视角的累计alpha</h4>'
            f'<p>仅此模型在{alpha_step}步保存的12个训练相机：全mask逐帧中位数'
            f'{full["perframe_median_range"][0]:.6f}–{full["perframe_median_range"][1]:.6f}；'
            f'像素加权alpha&lt;0.5占{100*full["pixel_weighted_fraction_below_0_5"]:.3f}%，'
            f'alpha&lt;0.95占{100*full["pixel_weighted_fraction_below_0_95"]:.3f}%。'
            f'内缩5个训练裁片像素后，分别为{100*interior["pixel_weighted_fraction_below_0_5"]:.3f}%和'
            f'{100*interior["pixel_weighted_fraction_below_0_95"]:.3f}%；脸部中位数范围'
            f'{face["median_range"][0]:.6f}–{face["median_range"][1]:.6f}。12份原始float32数组、'
            'npy/PNG/mask哈希已核验，全mask与内缩统计由原始数组重算一致。</p>'
            '<p>这是沿每条相机射线、所有高斯层累积的1−T，含原渲染器排序与截断，'
            '不是单点opacity或最前一层衣物表面的alpha。此数据不支持“新足球模型训练视角整体过透明”；'
            '后方层可能填满射线，不能据此证明前表面几何正确，也未测量未知方向或背面的alpha。'
            '源mask孔洞及真实空隙保持原样；mask遗漏的衣物不进入前景统计，内缩还会排除边缘和孔洞。'
            '这些统计只作诊断，没有改mask、训练损失或透明度。</p>')
    fit_worker = worker(fitting,fit['progress'],'run_football_half12_fit126.py','run')
    gs_worker = worker(gaussian,gs['progress'],'train_football_half12_sh0126.py','train')
    fitting_live,training_live=fit_worker['live'],gs_worker['live']
    gaussian_started=bool(gs['launch'] or gs['result'] or a.lp(gaussian/'gaussian-run').exists())
    phase_message=''
    if source:
        assert source['training_roster_ready'] and len(source['training_frames'])==12
        assert len(source['heldout_frames'])==4 and not set(source['training_frames'])&set(source['heldout_frames'])
        assert source['reference_frame']=='frame_000026'
        stage='new_person126_source12_frozen'
        next_action='Use the frozen new-person12training+4heldout source plan for fresh neutral fitting; heldout coordinates/scores must not enter training normalization.'
        phase_message='新人物12个训练源帧、4个留出已冻结，参考第26帧。'
    if fit['plan']:
        stage='new_person126_fitting_prepared'
        phase_message+=' 原版五阶段共享体型拟合接口已准备。'
    if fit['process-exit'].get('exit_code',0)!=0:
        stage='new_person126_fitting_failed'
        next_action='Inspect the saved football fitting failure/exit and retained artifacts; no duplicate launch or Gaussian admission from loss alone.'
    elif fit['result']:
        stage='new_person126_fit_completed_review_pending'
        phase_message+=f' 联合拟合已记录完成，用时{fit["result"].get("seconds",0):.1f}秒。'
        next_action='Actually review this new12frame mesh/body/face/feet and supported pose depth. No newGaussian training before a saved exploratory admission.'
        conclusion=fit['conclusion']
        if conclusion.get('actual_visual_review'):
            admitted=conclusion.get('exploratory_gaussian_admitted',False)
            stage='new_person126_fit_reviewed_gaussian_ready' if admitted else 'new_person126_fit_review_not_admitted'
            next_action=('Use only this reviewed fresh12frame fit for the registered SH0/ROI2x/paired-background18k candidate, with actual1000review.' if admitted else 'Preserve the reviewed fit and its explicit observed defects; no automatic Gaussian launch from this report.')
            phase_message+=' '+(conclusion.get('finding_zh') or conclusion.get('finding') or '实际拟合审阅已记录，质量仍未验收。')
    elif fitting_live:
        stage='new_person126_fitting_running'
        phase_message+=' 当前真实进程正在执行原版五阶段拟合；尚无完成结果。'
        next_action='Inspect this same football fitting process only; when complete perform actual12frame mesh review before any Gaussian launch. Do not duplicate the worker.'
    elif fit['launch'] and not fit['result']:
        stage='new_person126_fitting_process_unverified'
        phase_message+=' 已登记拟合进程当前身份未确认，不能将旧running记录当作存活。'
        next_action='Reconcile the registered PID/start identity, exact command and terminal artifacts; an observation failure alone does not authorize restart.'
    if gs['plan']:
        step=gs['progress'].get('step',gs['result'].get('snapshot_step',0))
        phase_message+=f' 高斯记录到{step:,}步、{gs["progress"].get("points",gs["result"].get("points",20908)):,}点。'
        stage='new_person126_gaussian_prepared'
        next_action='Use only the saved reviewed football fit; inspect the existing prepared Gaussian candidate, do not duplicate its launch.'
        if training_live:
            stage='new_person126_gaussian_review_waiting' if gs['progress'].get('optimizer_waiting') else 'new_person126_gaussian_training'
            next_action='Inspect this same Gaussian process only; actually review the saved1000step source/unknown views before continuation; maximum registered18000steps, no silent parameter changes.'
        elif gs['process-exit'].get('exit_code',0)!=0 or gs['train-failure']:
            stage='new_person126_gaussian_failed_checkpoint_retained'
            next_action='Inspect the retained current Gaussian failure/checkpoint without automatic retry, threshold relaxation or old-person training.'
        elif gs['result']:
            stage='new_person126_gaussian_completed_review_export_pending'
            next_action='Actually review all12source and continuous unknown views, then verify native/PLY/camera package and local viewer before named delivery; training completion is not quality acceptance.'
        if gs['publication']:
            stage='new_person126_delivered_quality_open' if gs['publication'].get('ui_checked') else 'new_person126_exported_ui_pending'
            next_action='Keep the named football delivery and actual remaining-quality defects; reconcile the full goal without retraining old people or claiming15min acceptance.' if gs['publication'].get('ui_checked') else 'Complete actual local preset and OpenFileGaussian/Points/camera checks for this football package.'
    active=gs_worker if training_live else fit_worker if fitting_live else native_worker if live else {}
    entry = dict(updated=now,subject='football_video11',native_root=str(native),record_root='D:/00/记录/011',
        progress=progress,result=result,actual_pid=active.get('pid'),actual_command=active.get('observed_command',''),
        native_process=native_worker,fit_process=fit_worker,gaussian_process=gs_worker,
        fit_root=str(fitting),fit=fit,gaussian_root=str(gaussian),gaussian=gs,
        source_frames=source.get('training_frames',[]),heldout_frames=source.get('heldout_frames',[]),reference_frame=source.get('reference_frame'),
        preprocessing_running=live,fitting_running=fitting_live,training_running=training_live,
        half_circle_target=12,full_circle_target=24,old_person_retraining=False,
        new_gaussian_started=gaussian_started,quality_accepted=False,end_to_end_15min_accepted=False,company00_deployed=False)
    for path in (a.R/'protocol.json',a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json'):
        doc = a.read(path)
        doc.update(updated=now,current_stage=stage,new_person126=entry,
            preprocessing_running=live,fitting_running=fitting_live,training_running=training_live,
            own_gpu_task_running=bool(live or fitting_live or training_live),current_worker_pid_wsl=active.get('pid'),
            next_action=next_action,quality_accepted=False,end_to_end_15min_accepted=False,goal_complete=False,
            goal_turn_classification='progress',goal_turn_blocked_streak=0)
        if path == a.C/'local-research-status.json': doc.update(state=stage,next=next_action)
        atomic(path,doc)
    message = ('新人物video11：59个原生3fps真实PTS帧；'+('mask和133点观测已生成。' if ready else '正在生成新的mask和133点观测。'))
    phase_text=''
    if source:
        ids='、'.join(str(int(frame.split('_')[-1])) for frame in source['training_frames'])
        held='、'.join(str(int(frame.split('_')[-1])) for frame in source['heldout_frames'])
        phase_text=(f'<h3>本轮12帧联合拟合与高斯阶段</h3><p>{html.escape(phase_message)}</p>'
                    f'<p>训练帧{ids}；留出{held}，参考26。新体型从中性全零重新可学习拟合，'
                    'VPoser全零、各视角统一正面身体初值；不读取旧个体优化姿态或体型。'
                    '置信度统计仅用训练12帧，留出分数与坐标不参与统计。手指插袋与侧面远腕遮挡仍未知；'
                    '统一停用原手部2D数据项，保留身体腕肩肘、脸脚和手部姿态先验。'
                    '本轮留出只对当前拟合独立，不是新视频泛化或物理3D真值。</p>'
                    f'<p>当前阶段：{html.escape(stage)}。实际worker PID：{active.get("pid") or "无已验证进程"}。'
                    '完成后须实际审阅源视角及未见方向；拟合、训练或格式通过均不代表质量验收。</p>')
        last_command=(active.get('observed_command') or gs_worker.get('registered_command')
                      or fit_worker.get('registered_command'))
        if last_command:
            phase_text+=f'<p>{"当前实际命令" if active else "已登记命令（当前未验证存活）"}：<code>{html.escape(last_command)}</code></p>'
        if gs['preflight-failure'] and gs['validity-cuda-preflight'].get('passed'):
            phase_text+='<p>早期接口预检失败记录保留；后续CUDA预检已通过。历史失败文件不表示当前训练失败，训练配置保持登记值。</p>'
        if gs['publication']:
            pub=gs['publication']
            phase_text+=(f'<p><a href="{html.escape(pub["viewerUrl"])}">打开本轮新人物高斯与参考26相机</a> · '
                         '<a href="../../exports/football-half12-paired18k126_SH0_018000_with_camera.zip">下载点云＋相机ZIP</a>。'
                         '18,000步、134,101点；约20万点目标未达到。源视角和36个连续新视角已审阅，正面更清晰，'
                         '未拍背面仍有明显杂色与孔洞；SH0不能消除这些空间误差。质量及视频到导出15分钟仍未验收。</p>')
    if facing:
        assert facing['state']=='diagnostic_completed' and facing['reference_frame']=='frame_000026'
        front=facing['groups']['front']['mask_eroded_5_derived_pixels']
        full=facing['groups']['full']['mask_eroded_5_derived_pixels']
        alpha_text+=(f'<h4>最终18,000步 · 参考26前后绑定面点分组</h4><p>按当前拟合网格官方三角绕序，'
                     f'只渲染朝向相机绑定面上的高斯时，内部alpha&lt;0.95占{100*front["fraction_below_0_95"]:.2f}%；'
                     f'加入背向绑定面点后的全体为{100*full["fraction_below_0_95"]:.2f}%。背向组确实可填补部分射线覆盖，'
                     '但这是网格朝向代理，不是测量得到的衣物前后层，也不是逐像素合成贡献分解。'
                     '网格有32条边界，未补洞；不据此直接删背向点或强制全部不透明。</p>'
                     '<figure><img src="new-person126/front-back-rgb.png" style="max-width:100%;height:auto" alt="全体、前向绑定面点、背向绑定面点的同相机RGB"/>'
                     '<figcaption>左：全体；中：朝向相机的绑定面点；右：背向点。相同参数及相机，未重新训练。</figcaption></figure>'
                     '<figure><img src="new-person126/front-back-alpha.png" style="max-width:100%;height:auto" alt="相同三组的累计alpha"/>'
                     '<figcaption>对应累计alpha；白色更不透明。各组独立渲染，不能相加。</figcaption></figure>')
        for folder in (a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer'):
            dest=a.lp(folder/'new-person126');dest.mkdir(exist_ok=True)
            for source,name in [('rgb_full_front_back.png','front-back-rgb.png'),('alpha_full_front_back.png','front-back-alpha.png')]:
                (dest/name).write_bytes(a.lp(facing_path.parent/source).read_bytes())
    block = f'''<section id="new-person126"><h2>新人物011 · 足球场黑衣男子</h2><p><strong>{html.escape(message)}</strong></p>
<p>2704×1520，19.601秒。六个源视角显示一侧经过正面到另一侧，未见背面，采用半圈约12帧；不声称精确180°。一圈目标更新为24帧。现有草地8/12草案停止，旧人物不再追加实验。</p>
<p>沿原入口进行3fps真实PTS采样、YOLO/SAM2 mask和RTMW133观测；不复用旧个体体型、姿态或高斯。10bit BT709 SDR视频按已有流程转RGB24 8bit，未进行第二次去畸变。K是原YAML经尺寸与居中裁剪形成的条件值，物理标定仍未独立确认。</p>
{phase_text}
<h3 id="opacity126">用户指出的透明问题：只读检查</h3>{alpha_text}<h4>保留的旧冬帽模型单点检查</h4><p>以下0.234与31.67%来自旧冬帽模型，不是新足球模型的结果或原因证明。导出opacity逐字节保留；实际Spark本机CPU解码正确执行sigmoid。单点峰值alpha中位数0.234，31.67%低于0.1。单点值不能替代累计表面alpha，不能据此认定所有表面均半透明。</p>
<p>原版opacity reset与每三角形保点允许低alpha点存留；既有mask孔洞会把本应保留的衣物区域监督成背景。新人物将检查源mask孔洞、覆盖及累计alpha，不整体强制opacity=1，不以这次诊断为由重训旧人。数据及分析文件存工程，用户图片存D:/00/记录/011；不称公司00已部署或质量通过。</p></section>'''
    begin,end='<!-- NEW_PERSON126_BEGIN -->','<!-- NEW_PERSON126_END -->'
    page=a.lp(a.R/'video10-global-initializer/index.html').read_text(encoding='utf-8')
    wrapped=begin+block+end
    if begin in page:
        left=page.index(begin);right=page.index(end,left)+len(end);page=page[:left]+wrapped+page[right:]
    else:
        marker='<!-- CURRENT_GAUSSIAN_RESULTS124_END -->';assert page.count(marker)==1
        page=page.replace(marker,marker+wrapped)
    for folder in (a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer'):
        a.lp(folder/'index.html').write_text(page,encoding='utf-8')
    for folder in (a.V,a.S):
        doc=a.read(folder/'latest-result.json')
        brief=message+(' '+phase_message.split('；')[0] if phase_message else '')
        doc['latestDiagnostic']=dict(updated=now,round='126',title='新人物011 · 12帧联合拟合',summary=brief,
            reportUrl='diagnostics/video10-global-initializer/index.html#new-person126',status=stage,qualityAccepted=False,goalComplete=False)
        atomic(folder/'latest-result.json',doc)
    atomic(native/'state-publication.json',dict(updated=now,stage=stage,actual_process_live=bool(active),
        fit_process=fit_worker,gaussian_process=gs_worker,opacity_audit_readonly=True,new_gaussian_started=gaussian_started,
        report_url='http://127.0.0.1:8870/diagnostics/video10-global-initializer/index.html#new-person126',ui_verified=False))
    print(stage,message,phase_message)


if __name__=='__main__': main()
