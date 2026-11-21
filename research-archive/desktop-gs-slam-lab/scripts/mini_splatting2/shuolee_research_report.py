"""Refresh one local research report and truthful execution status."""
import html
import json
from pathlib import Path
import time

from shuolee_geometry_diagnosis import ROOT, write_json


def main():
    home=ROOT/'shuolee';run=home/'current/da3-refinement'
    cleanup=json.loads((run/'cleanup.json').read_text(encoding='utf-8-sig'))
    cleanup.update(status='blocked_by_policy',deleted=False,reason_from_tool='blocked by policy; no more specific reason returned')
    write_json(run/'cleanup.json',cleanup)
    status=json.loads((run/'status.json').read_text(encoding='utf-8'))
    status.update(cleanup_pending=True,cleanup_status='blocked_by_policy',models_still_present=True)
    write_json(run/'status.json',status)
    work=json.loads((home/'work-status.json').read_text(encoding='utf-8'))
    work.update(updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),current_pilot_cleanup=dict(status='blocked_by_policy',deleted=False,
        files=len(cleanup['files']),MiB=cleanup['total_bytes']/2**20),latest_report='research-progress.html')
    write_json(home/'work-status.json',work)
    code='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人物重建：问题诊断与路线选择</title><style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f2f4f5;color:#1b2c37;font:17px/1.85 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:48px 32px 80px}h1{font-size:34px;line-height:1.4}h2{font-size:23px;margin-top:0}a{color:#006277;text-underline-offset:3px}section,.lead{background:white;border:1px solid #dce3e6;border-radius:12px;padding:28px;margin:24px 0}.eyebrow,small{color:#546976}.lead{border-left:5px solid #00798c}.status{color:#8b4711;font-weight:650}figure{margin:24px 0}img{width:100%;height:auto;display:block;border:1px solid #e1e5e7}figcaption{font-size:14px;color:#536371;margin-top:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid p{margin:0}.linkline{display:flex;gap:24px;flex-wrap:wrap}li{margin:12px 0}details{margin:16px 0}summary{cursor:pointer}p{margin:14px 0}.notice{color:#7b471b;font-size:15px}video{width:100%;max-height:620px;background:#111}@media(max-width:700px){main{padding:24px 16px}section,.lead{padding:20px}.grid{grid-template-columns:1fr}h1{font-size:28px}}
</style><main>
<div class="eyebrow">shuolee · 2026 年 10 月 19 日 · 长期目标进行中</div>
<h1>先找模糊的原因，再决定采用哪种方法</h1>
<div class="lead"><p><strong>本轮结论：深度先验得到改善，但短程三维优化没有带来明显清晰度收益。</strong></p>
<p>Mip、扩散和深度指导都是待验证方案。优先修正能被实际证据支持的问题，必要时更换表示或训练思路。</p>
<p class="status">DA3 对照已完成，未晋级；当前没有后台训练在运行。</p></div>
<nav class="linkline"><a href="long-term-plan.md">完整长期规划</a><a href="work-status.json">实际执行状态</a><a href="current/da3-refinement/decision.json">本轮结论与口径</a></nav>
<section><h2>先验偏差是本轮找到的具体问题</h2>
<p>同一组高置信度头部 SfM 特征点上，原单目先验的深度偏差中位数约为 <strong>23.2%</strong>。已知相机条件下，DA3-BASE 对 51 个训练视图的对应值约为 <strong>1.3%</strong>。这些点参与过 SfM 和原模型初始化，因此结果表示开发数据中的几何一致性，不能称为独立深度精度。</p>
<p>继续剔除遮挡、深度边缘与局部冲突后，约 <strong>34.7%</strong> 的头部框内像素用于新深度监督。其余区域保持无深度标签；低分辨率深度只约束粗几何。</p>
<figure><img src="current/geometry-diagnosis/dense-consistency-full-depth-000014.png" alt="真实输入、原单目深度与 DA3 多视图深度，使用相同色标"><figcaption>实际输入与本地推理结果。色标对应 1 至 10 个任意 SfM 单位，更远的值截断显示；不是米制测量。</figcaption></figure></section>
<section><h2>实际训练仍然没有解决新视角模糊</h2>
<p>两组都从同一个全分辨率模型出发，各优化 510 步，保持 316,044 个高斯、相同视图顺序和学习率。A 添加经过筛选的 DA3 深度；B 只用真实图像继续优化。两组均使用原生渲染器与相同 Mip 语义。</p>
<figure><img src="current/da3-refinement/review/key-comparison.png" alt="相同俯视相机的 DA3 候选与真实图像对照，人物和背景仍有相似的模糊与破损"><figcaption>相同俯视相机。A 为 DA3 候选，B 为真实图像对照。头部细节与背景表面仍有明显问题，本候选不替换已有模型。</figcaption></figure>
<p>8 个开发视图中，头部 PSNR 差仅约 +0.0012 dB，SSIM 差约 +0.000029；抽查俯视、仰视、近距离和真实视角也没有清晰改善。三维参数确实更新，但更新本身不代表几何已修复。</p>
<details><summary>查看未删除的连续路径和完整协议</summary><p>两条路径各 60 帧；目前只完成程序生成与若干关键帧检查，尚无完整人类主观评分。</p><p><a href="current/da3-refinement/review/elevation.mp4">上下观察路径</a> · <a href="current/da3-refinement/review/distance.mp4">远近路径</a> · <a href="current/da3-refinement/protocol.json">冻结协议</a></p></details>
<p>该结果只否定本次短程候选的晋级，不足以否定 DA3 或可靠深度参与完整训练的价值。</p></section>
<section><h2>下一步比较不同机制</h2>
<ul>
<li><strong>表面表示：</strong><a href="https://github.com/hbb1/2d-gaussian-splatting">2DGS</a>、<a href="https://github.com/zju3dv/PGSR">PGSR</a>。验证表面约束能否减轻多层高斯、漂浮物与遮挡冲突，列为下一优先候选。</li>
<li><strong>稀疏视角正则化：</strong><a href="https://github.com/jiaw-z/CoR-GS">CoR-GS</a>、<a href="https://github.com/DCVL-3D/DropGaussian_release">DropGaussian</a>。若不同训练种子的未见视角明显不稳定，再开展有控制的对照。</li>
<li><strong>相机与多视图几何：</strong><a href="https://github.com/ByteDance-Seed/Depth-Anything-3">DA3</a>、<a href="https://github.com/facebookresearch/vggt">VGGT</a>。保留已验证的深度输入；仅在诊断支持时更换相机与 SfM 链路。</li>
<li><strong>尺度与生成：</strong><a href="https://github.com/autonomousvision/mip-splatting">Mip-Splatting</a> 处理采样尺度；<a href="https://github.com/Kunhao-Liu/ViewExtrapolator">ViewExtrapolator</a> 探索缺失视角外观。生成画面变清晰后，仍需验证关闭生成模块时的直接三维渲染。</li>
</ul>
<p>每次只启动一个有区分能力的小试。没有清晰收益不扩大算力，也不把多个模块同时堆入工程。完整训练、独立场景与主观验收的条件记录在长期规划中。</p></section>
<section><h2>工程状态</h2><div class="grid"><p><strong>本地</strong><br>RTX 5080 完成 DA3 推理及两组训练；原始模型未修改。小试结束，GPU 训练进程已退出。</p><p><strong>公司服务器</strong><br>本轮未重新验证连接，也未启动服务器训练。新目录、脚本与试验继续使用 shuolee 命名。</p></div>
<p class="notice">清理本轮 66 个失败产物、约 530 MiB 的操作被自动审批阻止，文件实际仍在。保留精简记录和一张关键对比的规则不变，清理不能标记为完成。</p>
<p><small>验证边界：单段工作视频、单个种子、短程固定拓扑；开发视图已重复使用；没有独立高俯仰 GT，没有新视角几何正确性的完成声明。</small></p></section>
</main></html>'''
    report=home/'research-progress.html';report.write_text(code,encoding='utf-8')
    # Check every local resource so the report does not reference removed files.
    import re
    links=re.findall(r'(?:src|href)="([^"]+)"',code)
    local=[u for u in links if not u.startswith(('https:','http:','#'))]
    assert all((home/u).is_file() for u in local),[u for u in local if not (home/u).is_file()]
    print(json.dumps(dict(report=str(report),local_links_verified=len(local),cleanup_deleted=False),ensure_ascii=False))


if __name__=='__main__':main()
