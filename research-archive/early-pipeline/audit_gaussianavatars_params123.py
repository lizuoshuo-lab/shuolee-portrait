"""Read-only upstream parameter audit and a reviewable, unexecuted ADC proposal."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import difflib
import hashlib
import json
import os

H = Path(__file__).parent
T = Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R = T/'shuolee/current/robust-portrait-122'
REC = T/'shuolee/current/local-clothed-surface-123'
O = REC/'smplifyx-reference-control/official-first-cycle'
A = O/'official-gaussianavatars-parameters'
P = Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
V = T/'viewer'
S = Path('D:/00/工作/查看器')
C = H.parent/'shuolee-coordination'
URL = 'diagnostics/video10-global-initializer/index.html#official-parameter-audit'


def lp(p):
    p = Path(p).resolve()
    return Path('\\\\?\\'+str(p)) if os.name == 'nt' else p


def read(p):
    return json.loads(lp(p).read_text(encoding='utf-8-sig'))


def save(p, value):
    p = lp(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(p):
    return hashlib.sha256(lp(p).read_bytes()).hexdigest()


def defaults(path):
    tree = ast.parse(lp(path).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OptimizationParams')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    return {n.targets[0].attr: ast.literal_eval(n.value) for n in init.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute)}


def events(start, interval, until, budget):
    return [i for i in range(1, budget+1) if start < i < until and i % interval == 0]


def main():
    now = datetime.now(timezone.utc).isoformat()
    provenance = read(A/'source-provenance.json')
    for f in provenance['files']:
        assert sha(A/'upstream'/f['file']) == f['sha256']
    upstream = defaults(A/'upstream/arguments/__init__.py')
    local = defaults(P/'src/avatar_core/arguments/__init__.py')
    assert local == upstream
    old = read(O/'shared-offset-lr1e-5-plan.json')
    source = O/'shared-vertex-offset-1000'
    for name, digest in old['input_hashes'].items():
        assert sha(source/name) == digest
    for name, digest in old['source_hashes'].items():
        assert sha(P/name) == digest

    schedule = dict(
        official_600k=events(10000, 2000, 600000, 600000),
        current_18k=events(10000, 2000, 18000, 18000),
        proposed_18k=events(300, 60, 18000, 18000),
        current_first1000=events(10000, 2000, 18000, 1000),
        proposed_first1000=events(300, 60, 18000, 1000))
    assert len(schedule['official_600k']) == len(schedule['proposed_18k']) == 294
    assert schedule['current_18k'] == [12000, 14000, 16000]
    assert schedule['proposed_first1000'] == list(range(360, 961, 60))

    # Prepare a small optional interface diff. Do not edit the executable source.
    path = P/'src/shuolee_portrait/training.py'
    before = lp(path).read_text(encoding='utf-8')
    needle = '    options.densify_until_iter = schedule_steps; model.training_setup(options)\n'
    assert before.count(needle) == 1
    replacement = '''    options.densify_until_iter = schedule_steps
    for key in ('densify_from_iter', 'densification_interval'):
        if key in cfg:
            value = cfg[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if key == 'densify_from_iter' else 1):
                raise ValueError('Invalid density-control timing: ' + key)
            setattr(options, key, value)
    if options.densify_from_iter >= schedule_steps:
        raise ValueError('Density-control start must precede the training schedule end.')
    rt.save(out/'effective-optimization.json', dict(vars(options)))
    model.training_setup(options)
'''
    after = before.replace(needle, replacement, 1)
    ast.parse(after)
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
        fromfile='a/src/shuolee_portrait/training.py', tofile='b/src/shuolee_portrait/training.py'))
    lp(A/'optional-timing-interface.patch').write_text(diff, encoding='utf-8', newline='\n')
    assert sha(path) == old['source_hashes']['src/shuolee_portrait/training.py']

    config = dict(old['config'], shared_static_offset_lr=1.e-6,
                  densify_from_iter=300, densification_interval=60)
    proposal = dict(created=now, state='prepared_awaiting_risk_approval', user_authorized_to_execute=False,
        experiment='Only ADC timing changed relative to shared-vertex-offset-1000 baseline',
        human_scope='按照gaussianavrtars的参数调试一下',
        hypothesis='The inherited 600k ADC timing delays all density adaptation beyond the short review; earlier split/clone/prune may improve clothing support, but this is unproven.',
        config=config, actual_step_budget=1000, no_automatic_extension=True,
        baseline=str(source), proposed_output=str(O/'shared-vertex-offset-adc-1000'),
        fresh_initial_gaussians=200000, fresh_optimizer=True, zero_shared_offset=True,
        fit_reused_not_refitted=True, shared_offset_and_triangle_binding_preserved=True,
        pose_camera_mask_resolution_loss_seed_unchanged=True,
        baseline_static_offset_lr=1.e-6, rejected_lr1e5_not_used=True,
        schedule_scale=18000/600000,
        expected_ADC_events_first1000=schedule['proposed_first1000'],
        opacity_reset_or_SH_schedule_changed=False,
        reason_not_full_official_schedule='Only density adaptation timing is a controlled variable. Initial200k fullbody points, fixed poses and shared offset are existing local adaptations. No claim of paper reproduction.',
        existing_point_stop=1200000,
        stop_conditions=['First1000 steps is a hard maximum; no automatic extension.',
            'Existing point-count or face-coverage guard failure: preserve evidence and stop.',
            'Nonfinite loss/gradients or native PLY parity failure: preserve evidence and stop.',
            'No threshold sweep or retry with altered settings.'],
        risks=['Earlier and more frequent splits/clones can increase memory and wall time.',
            'More points can fit pose/calibration errors and produce floaters instead of improving clothes.',
            'ADC alone does not constrain shared vertex-offset folds or fix inaccurate poses.'],
        review=['same seven cameras with native face/body/feet crops', '36 continuous unseen views',
            'point count, offset distribution and canonical triangle orientation',
            'world PLY full covariance/SH plus camera package native readback and actual Gaussian/points UI'],
        source_hashes=old['source_hashes'], input_hashes=old['input_hashes'],
        patch=str(A/'optional-timing-interface.patch'), source_modified=False,
        training_started=False, quality_accepted=False, end_to_end_benchmark=False)
    save(A/'candidate-proposal.json', proposal)
    save(A/'candidate-config.json', config)
    save(A/'event-schedule.json', schedule)

    result = dict(updated=now, state='upstream_audit_complete_candidate_not_executed',
        source=provenance, upstream_OptimizationParams=upstream,
        all_local_OptimizationParams_defaults_match=True,
        local_runtime_overrides=dict(iterations=18000, position_lr_max_steps=18000, densify_until_iter=18000),
        ADC_counts={k:len(v) for k,v in schedule.items()},
        parameter_consumption=dict(
            foreground_photo='0.8 L1 + 0.2 DSSIM; foreground support retained, matches official coefficients',
            xyz='0.01 relu(norm(local_xyz)-1); same local-coordinate regularizer',
            scale='1.0 norm(relu(exp(local_scale)-0.6)); same local-coordinate regularizer',
            opacity_reset='No call in local loop; upstream60k reset also outside current18k horizon',
            SH='Both increment each1000 steps; no change proposed',
            optimizer_order='Local optimizer step before ADC; upstream ADC before step. Identified, not changed in this timing control.'),
        shared_offset=dict(upstream_static_offset_optimizer_commented_out=True,
            upstream_static_offset_loaded_from_tracking=True,
            upstream_laplacian_default_weight=0,
            upstream_laplacian_active_expression='dynamic_offset only, not static_offset',
            local_static_offset_is_explicit_adaptation=True,
            offset_inserted_before_joint_regression_and_LBS_in_both=True,
            mesh_binding_preserved=True, original_default_static_regularizer_missing=False,
            meaning='There is no enabled official shared-static mesh smoothing term that can simply be restored by a parameter.'),
        prior_turn_classification='no_progress',
        prior_turn_reason='Answered binding question from existing evidence; no implementation or experiment advanced. This audit verifies current source and changes next action.',
        gpu_process_revalidated=False, training_running=False,
        proposed_control=str(A/'candidate-proposal.json'), quality_accepted=False,
        end_to_end_15min_accepted=False)
    save(A/'parameter-audit.json', result)
    summary='原版优化参数已一致；60万步增密时序未随1.8万步训练缩短，前1000步没有增密。共享offset是显式适配，原版默认无可直接恢复的静态网格平滑约束。'
    section='<section id="official-parameter-audit"><h2>GaussianAvatars 原版参数核对与下一轮对照</h2>'
    section+='<p><strong>'+summary+'</strong></p>'
    section+='<p>已核对官方固定提交 <code>'+provenance['commit']+'</code> 的 arguments、train 和 FLAME 模型源码；所有本地 OptimizationParams 默认值一致。源码实际日程为600000步；README中的通用30000步文字不能替代可执行配置。</p>'
    section+='<table><thead><tr><th>项目</th><th>官方实际源码</th><th>当前18k</th><th>待确认的1000步对照</th></tr></thead><tbody>'
    for row in [('位置学习率起止','0.005 → 0.00005','相同，衰减至18k','不变'),
                ('特征／透明度／尺度／旋转学习率','0.0025／0.05／0.017／0.001','相同','不变'),
                ('位置／尺度正则','0.01，阈值1／1.0，阈值0.6','相同','不变'),
                ('增密起点／间隔','10000／2000','10000／2000','300／60'),
                ('全日程实际增密次数','294次／600k','3次／18k','294次／18k；本次仅跑至1k'),
                ('前1000步增密次数','0','0','11次：360…960步'),
                ('共享vertexoffset','默认未开启训练','显式适配，默认LR1e-6','保留1e-6和三角形绑定')]:
        section+='<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>'
    section+='</tbody></table><p>局部RGB损失已经是0.8 L1＋0.2 DSSIM，并限制在mask前景；无需重复改权重。原版lambda_laplacian默认0，而且实现只选择dynamic_offset；不能把新加静态平滑项称为“恢复原版默认”。</p>'
    section+='<p>共享vertexoffset改变网格，再通过对应三角形的位置、方向与尺度驱动高斯；不会自动解除绑定。历史沙发119则由mask生成独立外壳，binding=None，两种表示有明确区别。</p>'
    section+='<p>待确认方案：复用同94帧拟合、相机和mask；原生2704×1520，seed6060，20万新高斯、零共享偏移、空优化器。只将增密起点和间隔按600k→18k比例缩短，最多1000步。保留已有损失、绑定、1e-6偏移学习率与其他策略。它不是完整论文复现，也不保证修复拟合或重影。</p>'
    section+='<p><strong>风险与停止：</strong>点数、显存和耗时可能上升；增密也可能把错误姿态拟合成漂浮点。1000步必停，沿用120万点／三角形覆盖／非有限数／PLY回读保护；不自动扩展或扫阈值。草案与接口补丁已准备，尚未修改执行源码或启动训练。</p>'
    section+='<p>15分钟目标尚未通过：此前94帧拟合约64.69分钟，18k高斯训练循环约25.77分钟（含审阅等待）；这些旧计时不是本次端到端性能测试。</p>'
    section+='<p><a href="'+provenance['files'][0]['url']+'">官方参数源码</a> · <a href="'+provenance['files'][2]['url']+'">官方FLAME偏移与正则源码</a></p></section>'
    bases=(R/'video10-global-initializer', V/'diagnostics/video10-global-initializer', S/'diagnostics/video10-global-initializer')
    for base in bases:
        page=base/'index.html'; text=lp(page).read_text(encoding='utf-8')
        marker='<section id="official-parameter-audit">'
        if marker in text:
            start=text.index(marker);end=text.index('</section>',start)+len('</section>')
            text=text[:start]+section+text[end:]
        else:
            anchor='<section id="official-shared-offset-lr">'
            assert anchor in text
            text=text.replace(anchor,section+anchor,1)
        lp(page).write_text(text,encoding='utf-8')
    for base in (V,S):
        path=base/'latest-result.json'; state=read(path)
        state['latestDiagnostic']=dict(updated=now,round='123-official-parameters',
            title='原版参数核对完成：增密时序对照待确认',summary=summary,status='诊断完成，训练未启动',
            url=URL,report=URL,reportUrl=URL,qualityAccepted=False)
        save(path,state)
    brief=dict(updated=now,result=str(A/'parameter-audit.json'),proposal=str(A/'candidate-proposal.json'),
        new_training_started=False,source_modified=False,approval_pending=True,
        baseline='shared-vertex-offset-1000',quality_accepted=False)
    for path in (R/'queue-state.json',REC/'queue-state.json',C/'local-research-status.json'):
        state=read(path)
        state.update(updated_at=now,state='official_parameter_audit_complete_timing_proposal_pending',
            latest_parameter_audit=brief,training_running=False,fitting_running=False,
            local_GPU_task_active=False,GPU_task_running=False,goal_complete=False,
            next='Ask once for concrete ADC-timing risk approval; no new training until answered. Other Gaussian rates already match official.')
        save(path,state)
    ledger=read(R/'evidence-ledger.json');ledger['official_parameter_audit']=brief;save(R/'evidence-ledger.json',ledger)
    plan=T/'shuolee/long-term-plan.md'
    with lp(plan).open('a',encoding='utf-8') as f:
        f.write('\n2026-10-30 原版固定提交参数核对完成：所有OptimizationParams默认一致；600k增密起点10k/间隔2k保留到18k，实际仅12k/14k/16k三次。准备单因素增密时序300/60的1000步草案，原1e-6共享offset和网格绑定保留；尚未修改执行源码或启动训练，按用户风险确认要求待答复。官方static_offset默认不优化，laplacian默认0且仅动态偏移，不能宣称恢复原版静态约束。目标未完成。\n')
    print(json.dumps(dict(result=str(A/'parameter-audit.json'),draft=str(A/'candidate-proposal.json'),
        ADC_counts=result['ADC_counts'],source_unchanged=True,report=URL),ensure_ascii=False))


if __name__ == '__main__':
    main()
