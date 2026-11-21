"""CPU-only same-pixel review and existing viewer-page update for the long run."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import shutil
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from plyfile import PlyData
from review_shared_vertex_offset123 import join, clip_box, stats
from review_full_sequence123 import read, sha, relocate

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/smplifyx-reference-control/official-first-cycle'
ROOT=O/'shared-vertex-offset-18000'
SHORT=O/'shared-vertex-offset-1000/gaussian-run/step-001000'
R=T/'shuolee/current/robust-portrait-122'
REC=T/'shuolee/current/local-clothed-surface-123'
V=T/'viewer'; S=Path('D:/00/工作/查看器')
C=Path(__file__).parent.parent/'shuolee-coordination'
BASES=(R/'video10-global-initializer',V/'diagnostics/video10-global-initializer',S/'diagnostics/video10-global-initializer')
URL='diagnostics/video10-global-initializer/index.html#official-shared-vertex-offset-long'
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)

def comparison(step):
    snap=ROOT/f'gaussian-run/step-{step:06d}'
    assert read(snap/'native-ply-parity.json')['passed']
    out=ROOT/f'c/s{step:06d}';out.mkdir(parents=True,exist_ok=False)
    with np.load(ROOT/'fit/fitted.npz',allow_pickle=False) as f:fit=dict(f)
    rows=read(ROOT/'data/training-inputs.json')['records'];results=[]
    for i in (0,18,37,50,55,74,93):
        row=rows[i];fid=row['frame']
        ims=[Image.open(relocate(row['image'])).convert('RGB'),
             Image.open(SHORT/f'{fid}_render.png').convert('RGB'),
             Image.open(snap/f'{fid}_render.png').convert('RGB')]
        assert all(im.size==(2704,1520) for im in ims)
        mask=np.asarray(Image.open(relocate(row['mask'])).convert('L'))>252
        y,x=np.nonzero(mask)
        boxes={'body':clip_box(np.c_[x,y],30,2704,1520),
               'face':clip_box(fit['observed_native_keypoints'][i,67:118,:2],40,2704,1520),
               'feet':clip_box(fit['observed_native_keypoints'][i,19:25,:2],45,2704,1520)}
        arrays=[np.asarray(im,dtype=float)/255 for im in ims];metrics={}
        for label,box in boxes.items():
            path=out/f'f{i+1:06d}_{label}.png'
            join([im.crop(box) for im in ims]).save(path)
            x0,y0,x1,y1=box;valid=mask[y0:y1,x0:x1]
            error=lambda a:float(abs(a[y0:y1,x0:x1]-arrays[0][y0:y1,x0:x1])[valid].mean())
            metrics[label]=dict(crop=list(box),short1000_l1=error(arrays[1]),long_l1=error(arrays[2]),image=path.name)
            dest=Path('D:/00/记录/005')/('3DGS' if label=='body' else '分析')
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest/f'20261030_共享顶点偏移长训练{step}步_左原图_中1000步_右本轮_{fid}_{label}.png')
        results.append(dict(frame=fid,regions=metrics))
    v=PlyData.read(snap/'world.ply')['vertex'].data
    xyz=np.column_stack([v[k] for k in ('x','y','z')])
    distance=cKDTree(fit['vertices_world'][50]).query(xyz,k=1,workers=4)[0]
    result=dict(step=step,updated=datetime.now(timezone.utc).isoformat(),points=len(v),
        world_sha256=sha(snap/'world.ply'),offset=read(snap/'shared-static-offset.json'),
        distance_to_bare_vertices_m=stats(distance),comparisons=results,
        metric_scope='Same-pose native8bit foreground L1, not physical error; nearest bare vertex is not clothing ground truth.',
        actual_visual_review=False,quality_accepted=False)
    save(out/'comparison.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons','offset')}))

def final_contacts():
    """Preserve aspect ratios and no words on images, following record005 rules."""
    out=ROOT/'c/s018000'
    paths=sorted((ROOT/'export/review').glob('novel_*.png'))
    assert len(paths)==36 and (out/'comparison.json').exists()
    sheet=Image.new('RGB',(960,1440),(242,242,242))
    for i,path in enumerate(paths):
        im=Image.open(path).convert('RGB');im.thumbnail((160,240),Image.Resampling.LANCZOS)
        sheet.paste(im,((i%6)*160,(i//6)*240))
    sheet.save(out/'novel36.png')
    for label,size in [('body',(440,450)),('face',(440,180)),('feet',(440,180))]:
        source=sorted(out.glob(f'f*_{label}.png'));assert len(source)==7
        w,h=size;panel=Image.new('RGB',(w*2,h*4),(242,242,242))
        for i,path in enumerate(source):
            im=Image.open(path).convert('RGB');im.thumbnail(size,Image.Resampling.LANCZOS)
            panel.paste(im,((i%2)*w,(i//2)*h))
        panel.save(out/f'{label}7.png')
    shutil.copy2(out/'novel36.png',Path('D:/00/记录/005/分析/20261030_共享顶点偏移18000步_连续未知视角000至035.png'))
    print('FINAL_CONTACTS_READY_FOR_ACTUAL_REVIEW')

def publish():
    now=datetime.now(timezone.utc).isoformat()
    progress=read(ROOT/'progress.json'); launch=read(ROOT/'gaussian-launch/progress.json')
    running=launch['state'] in ('training','review_pending','exporting')
    summaries=[read(p) for p in sorted((ROOT/'c').glob('s*/comparison.json'))]
    section='<section id="official-shared-vertex-offset-long"><h2>共享顶点偏移：18000步长训练对照</h2>'
    section+='<p>按“做长训练试一下”推进。相机、94帧官方SMPLify-X拟合、mask、2704×1520输入、共享偏移学习率10⁻⁶、原版高斯损失与增密设置均不改，只延长实际训练。前轮没有完整优化器检查点，本轮重新初始化20万高斯与零顶点偏移，空优化器从头训练。</p>'
    section+=f'<p>已记录进度：{progress["step"]} / 18000 步，{progress["points"]:,} 个高斯；状态：{launch["state"]}。更新时间：{now}。复用现有拟合，本轮不是完整视频到模型的15分钟验收。</p>'
    section+='<p>每组图片：左原图 / 中前轮共享偏移1000步 / 右本轮。所有面部、衣物和脚部使用同一相机、同一姿态与相同原生裁剪；不变形、不在图片上写字。</p>'
    section+='<table><tr><th>步数</th><th>高斯数</th><th>共享位移中位 / 最大（毫米）</th><th>第51帧面部 / 全身 RGB误差</th></tr>'
    for result in summaries:
        regions=next(x for x in result['comparisons'] if x['frame']=='frame_000051')['regions']
        offset=result['offset']
        section+=f'<tr><td>{result["step"]}</td><td>{result["points"]:,}</td><td>{offset["median_m"]*1000:.3f} / {offset["max_m"]*1000:.3f}</td><td>{regions["face"]["long_l1"]:.5f} / {regions["body"]["long_l1"]:.5f}</td></tr>'
    section+='</table><p>RGB误差来自训练帧mask内的保存图像，不能替代独立几何精度。位移幅度不能单独证明衣物表面正确。</p>'
    for result in reversed(summaries):
        step=result['step'];part=f'shared-vertex-offset-long/step-{step:06d}'
        reviewed=result.get('actual_visual_review',False)
        section+=f'<h3>{step}步：{result["points"]:,}点 · '+('已实际审阅' if reviewed else '待实际审阅')+'</h3>'
        section+='<p>'+result.get('review_summary','检查图已生成，质量尚未验收。')+'</p>'
        for fid,region in (('frame_000051','body'),('frame_000051','face'),('frame_000051','feet'),('frame_000094','body')):
            section+=f'<figure><img src="{part}/f{int(fid.split("_")[1]):06d}_{region}.png" alt="同帧{region}对照"></figure>'
        if (ROOT/f'c/s{step:06d}/novel36.png').exists():
            section+=f'<figure><img src="{part}/novel36.png" alt="连续未知视角000至035"></figure>'
    section+='<p>高斯中心可随共享偏移和原版局部参数训练；是否贴衣、是否消除重影由实际图片和连续未知视角验收决定。新阶段完成前，主查看器保留上一份已检查的真实点云。</p></section>'
    for base in BASES:
        for result in summaries:
            src=ROOT/f'c/s{result["step"]:06d}'
            dest=base/f'shared-vertex-offset-long/step-{result["step"]:06d}';dest.mkdir(parents=True,exist_ok=True)
            for p in src.glob('*.png'):
                if not (dest/p.name).exists():shutil.copy2(p,dest/p.name)
        page=base/'index.html';html=page.read_text(encoding='utf-8')
        marker='<section id="official-shared-vertex-offset-long">'
        if marker in html:
            start=html.index(marker);end=html.index('</section>',start)+len('</section>')
            html=html[:start]+section+html[end:]
        else:html=html.replace('<section id="official-shared-vertex-offset">',section+'<section id="official-shared-vertex-offset">',1)
        page.write_text(html,encoding='utf-8')
    for base in (V,S):
        p=base/'latest-result.json';d=read(p)
        d['latestDiagnostic']=dict(updated=now,round='123-shared-vertex-offset-long',
            title=f'共享顶点偏移长训练：{progress["step"]}/18000步',url=URL,report=URL,reportUrl=URL,
            summary='仅延长训练；原版方法、相机、mask和拟合保持不变。',status=launch['state'],qualityAccepted=False)
        save(p,d)
    for p in (R/'queue-state.json',REC/'queue-state.json',C/'local-research-status.json'):
        d=read(p);d.update(updated=now,updated_at=now,state='shared_vertex_offset_long_'+launch['state'],
            current_stage='duration_only_long_training',training_running=running,fitting_running=False,
            local_GPU_task_active=running,GPU_task_running=running,
            current_worker_pid_wsl=launch['pid'] if running else None,
            worker_command='run_shared_vertex_offset_long123.py --plan shared-vertex-offset-long-plan.json' if running else None,
            next='Review1k/6k/12k/18k exact snapshots, finish18k export and compare quality; do not restart this run.',
            latest_shared_vertex_offset_long=dict(root=str(ROOT),progress=progress,launch=launch,
                reviews=[dict(step=x['step'],actual_visual_review=x.get('actual_visual_review',False)) for x in summaries],
                quality_accepted=False,report=URL),goal_active=True,goal_complete=False)
        save(p,d)
    print(json.dumps(dict(updated=now,step=progress['step'],state=launch['state'],report=URL)))

def publish_final():
    """Publish only an actually reviewed terminal export and actual OpenFile UI."""
    quality=read(ROOT/'export/quality-review.json')
    ui=read(ROOT/'export/viewer-ui-review.json')
    receipt=read(ROOT/'export/staged-delivery.json')
    result=read(ROOT/'gaussian-run/result.json')
    assert result['steps']==18000 and read(ROOT/'gaussian-launch/progress.json')['state']=='exported_quality_pending'
    assert quality['actual_same_camera_review'] and quality['continuous_novel_views_reviewed']==36
    assert ui['file_opened_through_UI'] and set(ui['modes'])=={'Gaussian','points'}
    assert quality['source_sha256']==ui['source_sha256']==receipt['source_sha256']==sha(ROOT/'export/point_cloud.ply')
    assert receipt['local_UI_verified'] and receipt['steps']==18000
    assert ui['observed_count']==receipt['points'] and ui['camera_count']==94
    for item in ui['screenshots']:assert sha(Path(item['path']))==item['sha256']
    publish()
    ident=receipt['modelId'];summary=quality['summary'];now=datetime.now(timezone.utc).isoformat()
    end='<p>高斯中心可随共享偏移和原版局部参数训练；是否贴衣、是否消除重影由实际图片和连续未知视角验收决定。新阶段完成前，主查看器保留上一份已检查的真实点云。</p>'
    final=f'<p><strong>18000步完成：{summary}</strong></p>'
    final+=f'<p><a href="../../?model={ident}">查看18000步高斯与点云 ↗</a> · <a href="../../models/{ident}/scene.zip?download=1">下载点云＋94相机</a></p>'
    final+='<p>最终静态动作沿用已审阅的睁眼第51帧／视频原帧250。完整协方差、SH、同坐标完整K与相机通过回读；本机实际OpenFile ZIP核验Gaussian与points。格式与读取通过不代表质量通过，未在公司00、官方SIBR或独立Spark页验收。</p>'
    final+=f'<p>本轮高斯训练循环记录{quality["training_loop_wall_seconds"]/60:.2f}分钟，包含1000步人工检查等待和中间快照；此前94帧官方拟合耗时{quality["existing_fit_seconds"]/60:.2f}分钟。当前复用拟合，不是一次从视频开始的15分钟验收。</p>'
    for base in BASES:
        page=base/'index.html';html=page.read_text(encoding='utf-8');assert end in html
        html=html.replace('状态：exported_quality_pending。','状态：已导出并实际审阅，质量'+('通过' if quality['quality_accepted'] else '未通过')+'。',1)
        page.write_text(html.replace(end,final,1),encoding='utf-8')
    for base in (V,S):
        p=base/'latest-result.json';d=read(p);assert d['modelId']==ident
        d['latestDiagnostic']=dict(updated=now,round='123-shared-vertex-offset-long',
            title='共享顶点偏移18000步已完成并审阅',url=URL,report=URL,reportUrl=URL,
            summary=summary,status='质量通过' if quality['quality_accepted'] else '质量未通过',
            qualityAccepted=quality['quality_accepted']);save(p,d)
    final_state=dict(root=str(ROOT),steps=18000,quality=quality,viewer_UI=ui,receipt=receipt,
        sources_unchanged=True,end_to_end_benchmark=False,goal_complete=False)
    for p in (R/'queue-state.json',REC/'queue-state.json',C/'local-research-status.json'):
        d=read(p);d.update(updated=now,updated_at=now,state='shared_vertex_offset_18000_reviewed',
            training_running=False,fitting_running=False,local_GPU_task_active=False,GPU_task_running=False,
            worker_pid=None,current_worker_pid_wsl=None,worker_command=None,running=[],
            latest_shared_vertex_offset_long=final_state,quality_accepted=quality['quality_accepted'],
            current_stage='long_training_quality_review_complete',goal_active=True,goal_complete=False,
            next='Use the completed18k evidence to identify the minimum original-source-compatible quality correction. Do not repeat this run or increase steps automatically.')
        save(p,d)
    ledger=read(R/'evidence-ledger.json');ledger['shared_vertex_offset_long_final']=final_state;save(R/'evidence-ledger.json',ledger)
    with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as stream:
        stream.write('\n2026-10-30 共享vertexoffset18000步已结束并实际审阅：'+summary+' 本机OpenFile Gaussian/points与94相机已核验，完整源/协方差/SH保留。原版相机/拟合/lr/损失未改。全流程15分钟目标尚未完成；不重跑此长训或自动继续加步数，先基于最终图与位移证据决定最小必要修正。\n')
    print(json.dumps(dict(state='final_actual_review_published',model=ident,quality_accepted=quality['quality_accepted'])))

def publish_runtime():
    """Record an independent no-update runtime check, preserving the LR decision."""
    diag=ROOT/'offset-diagnostic';cost=diag/'runtime-cost-evidence-repair'
    result=read(cost/'result.json');proposal=read(O/'shared-offset-rate-proposal.json')
    assert result['all_learned_parameters_unchanged'] and result['inputs_unchanged']
    assert not result['optimizer_created'] and result['optimizer_updates']==0
    assert not proposal['user_decision_received'] and not proposal['training_started']
    rows=[]
    for frame in result['records']:
        a=[r['median_seconds'] for r in frame['modes'] if r['mode'].startswith('cuda')]
        b=[r['median_seconds'] for r in frame['modes'] if r['mode'].startswith('cpu')]
        rows.append(dict(frame=frame['frame'],cuda_ms=1000*float(np.mean(a)),host_ms=1000*float(np.mean(b)),
            reduction_fraction=1-float(np.mean(b)/np.mean(a)),
            vertices_images_losses_equal=all(r['vertex_max_abs_delta']==r['render_max_abs_delta']==r['loss_abs_delta']==0 for r in frame['modes']),
            baseline_repeat_max_gradient_relative_L2=max(v['relative_l2'] for v in frame['modes'][-1]['gradient_delta'].values()),
            host_max_gradient_relative_L2=max(v['relative_l2'] for r in frame['modes'][1:3] for v in r['gradient_delta'].values())))
    fit=read(O/'full94-shared-shape/fit/official-fit-result.json')
    fit_total=fit['elapsed_seconds'];fit_sum=sum(r['elapsed_seconds'] for r in fit['frame_results'])
    now=datetime.now(timezone.utc).isoformat()
    summary=dict(updated=now,source_result=str(cost/'result.json'),source_sha256=sha(cost/'result.json'),rows=rows,
        initial_failure_preserved=str(diag/'runtime-cost/failure.json'),
        official_fit=dict(total_seconds=fit_total,frame_sum_seconds=fit_sum,outside_frame_seconds=fit_total-fit_sum,
            reused_body_and_vposer=True,all94_fresh_official_fit_not_repeated=True),
        stream_synchronizations_per_profile=dict(cuda_indices=128,cpu_indices=24),
        learned_parameters_changed=False,production_source_changed=False,quality_change_measured=False,
        gradient_tolerance_unchanged=1e-5,gradient_parity_passed=result['gradient_parity_passed'],
        decision='Do not deploy this speed candidate. Strict gradient parity failed, including repeated unchanged baseline. Existing LR proposal remains unanswered; do not combine speed and rate changes.',
        goal_complete=False)
    save(Path('\\\\?\\'+str((cost/'review.json').resolve())),summary)
    block='<div id="official-shared-offset-runtime"><h3>共享偏移计算开销：只读GPU对照</h3>'
    block+=f'<p>已完成的94帧官方拟合记录{fit_total/60:.2f}分钟，逐帧拟合占{100*fit_sum/fit_total:.1f}%；外层读图、校验和导出合计{fit_total-fit_sum:.2f}秒。SMPL-X与VPoser已经在帧间复用，重复加载不是主要瓶颈。</p>'
    block+='<p>本次仅将SMPL-X的55个固定整数父节点索引在GPU／CPU间对照；所有浮点参数与计算保留在GPU。固定18k模型，两视角ABBA重复，不创建优化器、不更新参数。</p>'
    block+='<table><thead><tr><th>训练帧</th><th>索引在GPU</th><th>索引在CPU</th><th>采样耗时下降</th></tr></thead><tbody>'
    for row in rows:
        block+=f'<tr><td>{row["frame"]}</td><td>{row["cuda_ms"]:.2f} ms</td><td>{row["host_ms"]:.2f} ms</td><td>{100*row["reduction_fraction"]:.1f}%</td></tr>'
    block+='</tbody></table><p>计时只含两次渲染、损失和反向传播，排除Adam、读图、增密、快照和拟合；不能直接当成全流程加速。第一视角单次分析中CUDA流同步128→24次。</p>'
    block+='<p><strong>暂不合入训练：</strong>顶点、图像和损失完全相同，但梯度未通过预登记1e-5相对L2容差。原模式重复计算也有差异：第51帧最大约0.161%，CPU索引模式最高约0.678%。说明需要区分已有GPU反向波动与索引改动影响，当前不能宣称梯度等价。未放宽门槛、未改学习率，初次未保存数值就退出的诊断失败记录保留。</p>'
    block+='<p>18k模型仍是当前查看器结果，质量未通过。共享偏移学习率1e-5／1000步的单变量试验仍待用户答复；此次速度核查没有替代该决定。</p></div>'
    for base in BASES:
        page=base/'index.html';html=page.read_text(encoding='utf-8')
        assert 'id="official-shared-offset-runtime"' not in html
        anchor='<section id="official-shared-offset-diagnostic">'
        start=html.index(anchor);end=html.index('</section>',start)
        html=html[:end]+block+html[end:];page.write_text(html,encoding='utf-8')
    url='diagnostics/video10-global-initializer/index.html#official-shared-offset-runtime'
    for base in (V,S):
        path=base/'latest-result.json';d=read(path);model=d['modelId']
        assert model=='local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-s18000'
        d['latestDiagnostic']=dict(updated=now,round='123-shared-offset-runtime',
            title='共享偏移耗时对照已审阅；速度修改暂不合入',url=url,report=url,reportUrl=url,
            summary='固定模型速度样本约下降7%–13%，但严格梯度等价未通过；无训练或参数更新，18k质量仍未通过。',status='诊断完成，未合入',qualityAccepted=False)
        save(path,d)
    for path in (R/'queue-state.json',REC/'queue-state.json',C/'local-research-status.json'):
        d=read(path);d.update(updated=now,updated_at=now,latest_shared_offset_runtime_review=summary,
            state='shared_offset_runtime_reviewed_no_production_change',current_stage='awaiting_shared_offset_rate_decision',
            training_started=False,training_running=False,fitting_running=False,local_GPU_task_active=False,GPU_task_running=False,
            current_worker_pid_wsl=None,worker_pid=None,worker_command=None,running=[],goal_active=True,goal_complete=False,
            next='Wait for the existing rate-only1000-step decision. Read-only runtime check finished; do not repeat, relax parity tolerance, change production speed code, or silently start training.')
        save(path,d)
    ledger=read(R/'evidence-ledger.json');ledger['shared_offset_runtime_review']=summary;save(R/'evidence-ledger.json',ledger)
    with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as stream:
        stream.write('\n2026-10-30 共享偏移只读速度核查：94帧官方拟合外围开销仅25.51秒；现有SMPL-X/VPoser已经复用。55整数父索引GPU/CPU的两视角ABBA对照显示渲染+损失+反向样本下降7%–13%，图像与几何相同，但包含原模式重复在内的梯度未通过1e-5门槛，不合入生产。无优化器更新，不改学习率，等待原单变量1000步问题答复。\n')
    print(json.dumps(dict(state='runtime_diagnostic_published',rows=rows,production_changed=False,report=url)))


def publish_runtime_adoption():
    import difflib
    diag=ROOT/'offset-diagnostic';adopt=diag/'host-topology-adoption'
    boundary=read(diag/'gradient-boundary/result.json');verified=read(adopt/'result.json')
    assert boundary['geometric_VJP_max_relative_gradient_difference']==0
    assert verified['all_vertices_and_fixed_adjoint_gradients_identical'] and verified['poses']==94
    source=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/avatar_core/scene/smplx_gaussian_model.py')
    assert sha(source)==verified['production_source_hashes']['src/avatar_core/scene/smplx_gaussian_model.py']
    before=Path('\\\\?\\'+str((adopt/'source-before.py').resolve())).read_text(encoding='utf-8')
    after=source.read_text(encoding='utf-8')
    added='        # Official LBS uses these immutable integers as Python list indices.\n        # Host indices avoid per-joint CUDA syncs; all floating tensors stay CUDA.\n        self.offset_body.parents = self.offset_body.parents.cpu()\n'
    assert after.replace(added,'',1)==before
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='a/src/avatar_core/scene/smplx_gaussian_model.py',tofile='b/src/avatar_core/scene/smplx_gaussian_model.py'))
    Path('\\\\?\\'+str((adopt/'host-topology.patch').resolve())).write_text(patch,encoding='utf-8')
    now=datetime.now(timezone.utc).isoformat()
    summary=dict(updated=now,state='minimal_host_integer_cache_adopted',source=str(source),source_sha256=sha(source),
        change='One functional line plus two explanatory comments; immutable integer parents on host, floats remain CUDA.',
        isolated_native_raster_gradient_difference=boundary['native_raster_max_relative_gradient_difference'],
        isolated_geometric_gradient_difference=boundary['geometric_VJP_max_relative_gradient_difference'],
        complete_pose_count=94,exact_vertex_and_fixed_adjoint_gradient_parity=True,
        patch=str(adopt/'host-topology.patch'),verification=str(adopt/'result.json'),
        no_new_fitting_or_training=True,learned_parameters_changed=False,learning_rate_or_loss_changed=False,
        pending_learning_rate_decision_unchanged=True,quality_accepted=False,end_to_end_15min_accepted=False,
        performance_scope='Prior two cached-view render+loss+backward samples,7%-13% lower wall time; no new full-run benchmark.')
    save(Path('\\\\?\\'+str((adopt/'review.json').resolve())),summary)
    added_html='<div id="official-shared-offset-runtime-adoption"><h3>分离反向环节后：已接入固定索引缓存</h3>'
    added_html+='<p>固定全部世界高斯和像素梯度后，原版渲染器反向重复计算仍有最高约0.256%的相对梯度差；完全绕开渲染器，用固定世界梯度检查SMPL-X／三角绑定时，GPU与CPU父节点索引的梯度逐值一致。原版源码存在atomicAdd像素累加；该证据不能证明它是可见重影的根因。</p>'
    added_html+='<p><strong>最小改动已接入：</strong>仅一行把55个固定整数父节点索引缓存到CPU，另加两行注释；全部浮点参数、SMPL-X计算和高斯训练仍在GPU。对现有94个姿态逐帧验证，顶点与固定输入梯度的最大差均为0，检查约13.8秒完成。</p>'
    added_html+='<p>此前完整渲染梯度的严格容差失败记录继续保留，没有改成通过或放宽门槛。本次采纳依据是分离后对实际改动环节的精确等价验证，原版渲染器的反向波动仍存在。尚未重新长训，不能把样本7%–13%加速当作全流程15分钟成绩。</p>'
    added_html+='<p>现有18k模型及94相机保留，质量仍未通过；共享偏移学习率和损失未改，1000步学习率试验仍待答复。</p></div>'
    old_end='<p>18k模型仍是当前查看器结果，质量未通过。共享偏移学习率1e-5／1000步的单变量试验仍待用户答复；此次速度核查没有替代该决定。</p>'
    for base in BASES:
        path=base/'index.html';html=path.read_text(encoding='utf-8');assert old_end in html
        html=html.replace('<strong>暂不合入训练：</strong>','<strong>首轮历史判断：暂不合入。</strong>',1)
        path.write_text(html.replace(old_end,added_html,1),encoding='utf-8')
    url='diagnostics/video10-global-initializer/index.html#official-shared-offset-runtime-adoption'
    for base in (V,S):
        path=base/'latest-result.json';d=read(path)
        assert d['modelId']=='local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-s18000'
        d['latestDiagnostic']=dict(updated=now,round='123-shared-offset-runtime-adoption',
            title='固定骨架索引缓存已接入；94姿态顶点与梯度一致',url=url,report=url,reportUrl=url,
            summary='只改整数索引存放位置，GPU浮点训练、学习率与损失不变；没有新训练，18k质量仍未通过。',status='接口等价通过，质量未通过',qualityAccepted=False);save(path,d)
    for path in (R/'queue-state.json',REC/'queue-state.json',C/'local-research-status.json'):
        d=read(path);d.update(updated=now,updated_at=now,latest_shared_offset_runtime_adoption=summary,
            state='shared_offset_host_topology_cache_verified',current_stage='awaiting_shared_offset_rate_decision',
            training_started=False,training_running=False,fitting_running=False,local_GPU_task_active=False,GPU_task_running=False,
            current_worker_pid_wsl=None,worker_pid=None,worker_command=None,running=[],goal_active=True,goal_complete=False,
            next='Runtime parent cache is verified on94 poses. Keep the existing1000-step learning-rate decision pending; no new training authorized by this numerical parity check. Do not rerun these diagnostics.')
        save(path,d)
    ledger=read(R/'evidence-ledger.json');ledger['shared_offset_host_topology_adoption']=summary;save(R/'evidence-ledger.json',ledger)
    with (T/'shuolee/long-term-plan.md').open('a',encoding='utf-8') as stream:
        stream.write('\n2026-10-30 反向环节隔离：固定世界高斯与像素梯度，原版光栅器反向重复差最高约0.256%；固定世界梯度后SMPL-X/绑定VJP逐值一致。仅接入55个固定整数父索引CPU缓存一行，94姿态顶点与固定梯度最大差均0。浮点仍GPU，未改LR/损失/参数，未启动训练；15分钟与质量均未通过，待原1000步单变量选择。\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--step',type=int);p.add_argument('--publish',action='store_true');p.add_argument('--final-contacts',action='store_true');p.add_argument('--publish-final',action='store_true');p.add_argument('--publish-runtime',action='store_true');p.add_argument('--publish-runtime-adoption',action='store_true');a=p.parse_args()
    if a.step:comparison(a.step)
    if a.final_contacts:final_contacts()
    if a.publish:publish()
    if a.publish_final:publish_final()
    if a.publish_runtime:publish_runtime()
    if a.publish_runtime_adoption:publish_runtime_adoption()
