"""Publish the last observed PGSR state without claiming a process is live from a stale file."""
import argparse
import html
import json
from pathlib import Path
import re
import time

LAB = Path(__file__).resolve().parents[2]
HOME = LAB / 'outputs/mini-splatting2-mip-depth-001/shuolee'
RUN = HOME / 'current/pgsr-surface/train'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live-pid', type=int)
    args = p.parse_args()
    manifest = json.loads((RUN/'manifest.json').read_text(encoding='utf-8'))
    progress = json.loads((RUN/'model/progress.json').read_text(encoding='utf-8'))
    now = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    running = args.live_pid is not None and args.live_pid == manifest.get('pid') and manifest['status'] == 'running'
    work = json.loads((HOME/'work-status.json').read_text(encoding='utf-8'))
    work.update(updated=now, local_training_running=running,
        current_phase='PGSR full-resolution surface reconstruction training' if running else 'PGSR training status requires evaluation or verification',
        current_experiment=dict(name='shuolee-pgsr-surface', manifest='current/pgsr-surface/train/manifest.json',
            protocol='current/pgsr-surface/train/protocol.json', observed_at=now,
            process_verified=running, pid=manifest.get('pid'), progress=progress,
            accepted_as_best=False), latest_report='research-progress.html')
    follow=HOME/'current/pgsr-surface/followthrough.json'
    if follow.exists():work['current_experiment']['followthrough']=json.loads(follow.read_text(encoding='utf-8'))
    public=HOME/'data/public-validation/diffuman4d-example/elevation-benchmark/status.json'
    if public.exists():
        data=json.loads(public.read_text(encoding='utf-8'))
        work['public_elevation_validation']=dict(status=data['status'],scene=data['scene'],frame=data['frame'],
            roles=data['roles'],training_started=False,test_images_inspected=False,
            protocol='data/public-validation/diffuman4d-example/elevation-benchmark/protocol.json')
    (HOME/'work-status.json').write_text(json.dumps(work, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    ledger_path=HOME/'results-ledger.json'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    identifier='shuolee-pgsr-surface-20261019'
    entry=dict(id=identifier,updated=now,status='in_progress' if running else manifest['status'],
        question='Can a surface-oriented representation improve true native3D rendering on the existing limited-view capture?',
        method='Official PGSR with recorded static-mask integration; no learned depth or diffusion',
        run='current/pgsr-surface/train/manifest.json',evaluation='current/pgsr-surface/evaluation/protocol.json',
        last_verified_iteration=progress.get('iteration'),accepted_as_best=False,
        conclusion='Pending complete training and visual/evidence review; runtime success is not quality acceptance.')
    ledger['decisions']=[d for d in ledger['decisions'] if d.get('id')!=identifier]+[entry]
    ledger['updated']=now
    ledger['final_goal_accepted']=False
    ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state = f"PGSR 完整训练进行中：最近核对到 {progress.get('iteration',0):,} / 30,000 步。" if running else f"PGSR 状态：{manifest['status']}，尚未完成效果验收。"
    section = f'''<!-- pgsr-live-begin --><section><h2>当前实验：用表面约束检验几何问题</h2>
<p><strong>{html.escape(state)}</strong> 核对时间：{now}。这是最近一次进程检查，页面本身不代表实时监控。</p>
<p><a href="https://github.com/zju3dv/PGSR">PGSR</a> 使用平面化高斯、法线和跨视图约束。本次保留相同的 51 个训练视图、8 个开发视图及相机，使用 1514 × 2694 原始分辨率，从稀疏点重新训练。暂不使用单目深度、Mip 或扩散先验。</p>
<p>RTX 5080 已完成 CUDA 扩展编译与四步接口测试，三项几何损失均实际参与计算；短测峰值显存约 2.7 GiB。正式训练的几何约束从第 7,000 步后启用，计划 30,000 步。当前仍未晋级为最佳模型。</p>
<p>这属于不同方法的候选比较，表示、渲染器、增密策略和预算均有不同，不能将全部差异归因于单个模块。</p>
<p><a href="current/pgsr-surface/train/protocol.json">训练协议</a> · <a href="current/pgsr-surface/train/model/progress.json">最近训练记录</a> · <a href="current/pgsr-surface/adaptation.json">输入与代码核对</a></p>
<p><strong>验证数据的新发现：</strong>已有 ENeRF-Outdoor 三个序列的 18 台相机，绕拟合拍摄中心的高度角跨度均约 1.3°。该统计相对于平均相机向上方向，不能替代真实重力标定；它说明这些数据不能用于检验大幅度俯仰外推。三个序列此前均已使用，也不能直接命名为独立新场景。</p>
<p><a href="current/public-validation-inventory.json">ENeRF 相机元数据审计</a>。</p>
<p><strong>已补充真实俯视验证数据：</strong><a href="https://huggingface.co/datasets/krahets/diffuman4d_example">Diffuman4D 作者样例</a>的标定显示三层相机覆盖约 36° 高度角。已冻结一个时刻：16 台水平相机训练，16 台中层相机开发，16 台高层相机保留测试。照片和遮罩已按该划分接入；高层图片不参与初始化、训练、生成指导或调参。样例为作者处理的 1024 × 1024 WebP，仍须检查清晰度、遮罩与同步。</p>
<p><a href="data/public-validation/diffuman4d-example/metadata-audit.json">新样例标定审计</a> · <a href="data/public-validation/diffuman4d-example/elevation-benchmark/protocol.json">俯仰验证协议</a>。当前未启动这个数据集的训练，先完成工作视频 PGSR 候选。</p>
<p><strong>替代机制也继续调研：</strong><a href="https://github.com/sisyphm/StudioRecon">StudioRecon</a> 提供人体／背景分离建模的参考；<a href="https://github.com/humansplat/humansplat">HumanSplat</a> 提供人体结构先验思路；<a href="https://github.com/zju3dv/Diffuman4D">Diffuman4D</a> 侧重人物时空一致生成。三者当前只做适用性分析，尚未复现。二维增强结果仍与直接三维渲染分开验收。</p></section><!-- pgsr-live-end -->'''
    report = HOME/'research-progress.html'
    body = report.read_text(encoding='utf-8')
    body = body.replace('DA3 对照已完成，未晋级；当前没有后台训练在运行。', 'DA3 短程对照未晋级；PGSR 表面建模实验已启动，实际状态见下文。')
    body = body.replace('RTX 5080 完成 DA3 推理及两组训练；原始模型未修改。小试结束，GPU 训练进程已退出。', 'RTX 5080 完成 DA3 对照，目前推进 PGSR 表面建模。原始模型与照片保持不变，只有当前候选在独立目录写入。')
    if '<!-- pgsr-live-begin -->' in body:
        body = re.sub(r'<!-- pgsr-live-begin -->.*?<!-- pgsr-live-end -->',lambda _:section,body,flags=re.S)
    else:
        body = body.replace('</nav>', '</nav>\n'+section, 1)
    report.write_text(body,encoding='utf-8')
    local = [s for s in re.findall(r'(?:src|href)="([^"]+)"',body) if not s.startswith(('https:','http:','#'))]
    assert all((HOME/s).is_file() for s in local),[s for s in local if not (HOME/s).is_file()]
    print(json.dumps(dict(report=str(report),process_verified=running,iteration=progress.get('iteration'),local_links=len(local))))


if __name__ == '__main__': main()
