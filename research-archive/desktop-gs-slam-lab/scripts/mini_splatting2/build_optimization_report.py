"""Small visual report from frozen evidence, no invented scores or GT."""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001';VIEW=ROOT/'viewer'


def main():
    assets=VIEW/'report-assets';assets.mkdir(exist_ok=True)
    aggregate={}
    for kind in ['classic','mip','depth','mip-depth']:
        names=[f'{kind}-refine-s{i}'+('-r1' if kind=='classic' and i==0 else '') for i in range(3)]
        reports=[json.loads((ROOT/name/'evaluation/metrics.json').read_text()) for name in names]
        aggregate[kind]={str(d):dict(full_mean=float(np.mean([r['aggregate'][str(d)]['full_psnr'] for r in reports])),
                                    full_std=float(np.std([r['aggregate'][str(d)]['full_psnr'] for r in reports],ddof=1)),
                                    crop_mean=float(np.mean([r['aggregate'][str(d)]['subject_crop_psnr'] for r in reports]))) for d in [1,2,4,8]}
        dst=assets/kind;dst.mkdir(exist_ok=True)
        for im in (ROOT/names[0]/'evaluation').glob('*.png'):shutil.copy2(im,dst/im.name)
    gt=assets/'gt';gt.mkdir(exist_ok=True)
    import cv2
    for frame in range(0,59,8):
        im=Image.open(BASE/'hover-sfm-001/dataset/images'/f'{frame:06d}.jpg').convert('RGB')
        for d in [1,2,4,8]:
            arr=cv2.resize(np.asarray(im),(round(im.width/d),round(im.height/d)),interpolation=cv2.INTER_AREA)
            Image.fromarray(arr).save(gt/f'{frame:06d}-s{d}.png')
    fig,ax=plt.subplots(figsize=(7.4,4.1),constrained_layout=True)
    for kind,label,col in [('classic','Control refinement','#6b7280'),('mip','Mip filter + refinement','#0072B2'),('depth','Depth prior + refinement','#D55E00'),('mip-depth','Mip + depth + refinement','#009E73')]:
        xs=np.arange(4);ys=[aggregate[kind][str(d)]['full_mean'] for d in [1,2,4,8]];err=[aggregate[kind][str(d)]['full_std'] for d in [1,2,4,8]]
        ax.errorbar(xs,ys,yerr=err,fmt='o-',label=label,color=col,capsize=3,lw=1.7,ms=5)
    ax.set_xticks(range(4),['1x','2x','4x','8x']);ax.set_xlabel('Image downsampling factor (same held-out cameras)');ax.set_ylabel('PSNR (dB), higher is better')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
    fig.savefig(assets/'scale-evaluation.png',dpi=180);plt.close(fig)
    geo=json.loads((ROOT/'geometry-evaluation/sfm-depth-agreement.json').read_text())
    difix=json.loads((ROOT/'difix-appearance-test-r1/manifest.json').read_text())
    verification=json.loads((ROOT/'delivery-verification.json').read_text())
    difix_assets=assets/'difix';difix_assets.mkdir(exist_ok=True)
    for im in (ROOT/'difix-appearance-test-r1').glob('*.png'):shutil.copy2(im,difix_assets/im.name)
    summary=dict(repeated_refinement_runs=12,seeds=[0,1,2],steps_per_run=2000,metrics=aggregate,geometry=geo,
                 overall_goal_complete=False,scale_sampling_result='improved in 2x/4x/8x same-camera downsampling',
                 physical_zoom_and_pitch='not independently validated; current capture has limited coverage',
                 training='fixed topology local gsplat refinement of official Mini-Splatting2 PLY, not full algorithm retraining')
    summary['diffusion']=dict(difix,perceptual=verification['difix_summary'],per_frame_perceptual=verification['difix'])
    summary['verification']=verification
    (ROOT/'stage-summary.json').write_text(json.dumps(summary,indent=2))
    (assets/'summary.json').write_text(json.dumps(summary,indent=2))
    ffmpeg=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
    for kind in ['distance','elevation']:
        video=assets/f'{kind}-comparison.mp4'
        if not video.exists():
            cmd=[str(ffmpeg),'-hide_banner','-loglevel','error','-n']
            for model in ['classic-refine-s0-r1','mip-refine-s0','mip-depth-refine-s0']:
                cmd+=['-framerate','20','-i',str(ROOT/'geometry-evaluation'/model/kind/'%03d.png')]
            cmd+=['-filter_complex','[0:v][1:v][2:v]hstack=inputs=3[v]','-map','[v]','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
            subprocess.run(cmd,check=True)
    g=geo['models'];base=g['classic-refine-s0-r1']['mean_median_relative_z_error']*100;md=g['mip-depth-refine-s0']['mean_median_relative_z_error']*100
    a=aggregate
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mini-Splatting2 · 优化实验效果</title>
<style>body{margin:0;background:#101819;color:#eef4ef;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1220px;margin:auto;padding:36px 24px 80px}h1{font-size:34px;line-height:1.35;margin:16px 0}h2{margin-top:48px;font-size:25px}a{color:#a3dfc2}p{max-width:950px;color:#d4dcd7}.eyebrow,small{color:#94aea0}.cards,.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{padding:20px;border:1px solid #345047;border-radius:10px;background:#192624}.card strong{display:block;font-size:26px;color:#c2f0d7}.card span{display:block;color:#a7bab0}figure{margin:0}figcaption{font-size:14px;color:#bad1c3;margin:8px 0}img{display:block;max-width:100%;height:auto;border-radius:6px}figure img{width:100%}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0}select{padding:10px;background:#233a32;color:white;border:1px solid #4f7060;border-radius:5px}video{width:100%;background:#000;border-radius:6px}.note{padding:16px 20px;border-left:3px solid #dcb267;background:#2a2921}details{margin:24px 0;padding:18px;border:1px solid #345047}table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #345047;text-align:left}.chart{max-width:800px;background:white}.refs{font-size:14px}@media(max-width:700px){.panels{grid-template-columns:1fr}.cards{grid-template-columns:1fr}h1{font-size:27px}}</style>
<main><div class="eyebrow">工作视频 · 12 次局部训练对照 · 2026-10-18</div><h1>先稳定尺度变化，再约束几何</h1><p>基于电脑00的 HOVER VERSA 0102 视频，保留原始 Mini-Splatting2 模型，完成四种设置、三个随机种子、每次2000步的局部微调。相机、训练图像与高斯数量保持一致。</p><p><a href="index.html?model=mip-depth">打开五模型三维查看器 ↗</a> · <a href="report-assets/summary.json">查看完整评测口径</a></p>
<div class="cards"><div class="card"><span>原分辨率 · 基线续训 → Mip＋深度</span><strong>__NATIVE__ dB</strong><span>8个外观留出视角，3次重复均值</span></div><div class="card"><span>8倍缩小 · 基线续训 → Mip＋深度</span><strong>__SMALL__ dB</strong><span>同一相机下的采样尺度测试</span></div><div class="card"><span>与稀疏 SfM 深度的相对误差</span><strong>__DEPTH__</strong><span>每帧中位数的平均；并非实测深度GT</span></div></div>
<h2>1. 同视角看效果</h2><p>左侧是真实留出照片，中间为等预算的基线续训，右侧为选择的改进方案。显示完整图像；可以点击图像查看原始尺寸。人物区域指标使用手工矩形，矩形包含少量背景，不是人物分割指标。</p>
<div class="controls"><label>视角 <select id="frame"></select></label><label>缩小倍数 <select id="scale"><option>1</option><option>2</option><option>4</option><option selected>8</option></select></label><label>右侧方案 <select id="variant"><option value="mip">Mip 滤波</option><option value="depth">深度约束</option><option value="mip-depth" selected>Mip＋深度约束</option></select></label></div>
<div class="panels"><figure><figcaption>真实留出照片</figcaption><a id="gt-link" target="_blank"><img id="gt"></a></figure><figure><figcaption>基线续训 · 同样2000步</figcaption><a id="control-link" target="_blank"><img id="control"></a></figure><figure><figcaption id="candidate-label">Mip＋深度约束</figcaption><a id="candidate-link" target="_blank"><img id="candidate"></a></figure></div>
<h2>2. Mip 解决采样尺度变化</h2><p>三维平滑限制小于训练视图采样能力的高斯结构；二维滤波在投影覆盖像素变化时补偿不透明度。实验中，直接给旧模型加滤波会降低原尺寸画质，加入同预算微调后能基本恢复，同时改善缩小渲染。</p><img class="chart" src="report-assets/scale-evaluation.png" alt="四种设置在1、2、4、8倍缩小测试下的PSNR"><small>本地实验：51张训练图、8张留出图，3个随机种子；误差线为重复间标准差。采样尺度变化不等同于移动相机距离。</small>
<h2>3. 距离与俯仰：用同一条路径检查</h2><p>以下三列依次为：基线续训、Mip、Mip＋深度。距离轨迹覆盖参照位置的0.5至1.5倍半径；俯仰轨迹在参照角度上改变约±25°。三种模型使用完全相同的相机。</p><figure><figcaption>相机距离变化 · 左：基线 / 中：Mip / 右：Mip＋深度</figcaption><video controls loop muted playsinline preload="metadata" src="report-assets/distance-comparison.mp4"></video></figure><figure><figcaption>相机俯仰变化 · 左：基线 / 中：Mip / 右：Mip＋深度</figcaption><video controls loop muted playsinline preload="metadata" src="report-assets/elevation-comparison.mp4"></video></figure>
<p class="note">这些轨迹没有配对真实照片，只能用于检查视觉退化，不能计算新视角真实性。原视频约7.4秒，径向距离跨度约1.31倍，轨迹高度变化很小；俯仰覆盖不足仍是当前限制。</p>
<h2>4. 深度指导改善局部几何一致性</h2><p>深度分支使用 Depth Anything V2 Small 预测的相对深度，经可靠 SfM 点拟合到逆深度尺度，再用翻转一致性、深度边界及训练视图之间的一致性剔除部分不可靠区域。稀疏深度误差下降提供了局部几何证据，但尚不能据此宣称漂浮物已全部消失。该深度对照只评测随机种子0，SfM点也不等于独立测量的深度真值。</p>
<h2>5. 扩散修复：残影改善与细节变化同时存在</h2><p>已完成8个留出视角的 Difix 参考图修复，输入为 Mip＋深度模型渲染；参考只从51张训练照片中选取。每张图在576×1024分辨率运行一次去噪，未更新三维模型。</p><p>平均 PSNR 从 <strong>__DIFIX_PSNR__ dB</strong>，SSIM 从 <strong>__DIFIX_SSIM__</strong>，LPIPS 从 <strong>__DIFIX_LPIPS__</strong>。LPIPS越低越好：感知距离有所改善，但像素与结构一致性下降。第000008帧的强残影明显减轻；其他视角会改变原本较好的细节，暂不整体写回三维训练。</p>
<div class="controls"><label>扩散对照视角 <select id="diff-frame"></select></label></div><div class="panels"><figure><figcaption>真实留出照片 · 仅用于评测</figcaption><a id="diff-gt-link" target="_blank"><img id="diff-gt"></a></figure><figure><figcaption>Mip＋深度 · 修复前</figcaption><a id="diff-input-link" target="_blank"><img id="diff-input"></a></figure><figure><figcaption>Difix · 二维修复结果</figcaption><a id="diff-difix-link" target="_blank"><img id="diff-difix"></a></figure></div><p id="diff-score"></p><details><summary>查看实际使用的训练参考照片</summary><img id="diff-reference" style="max-width:480px" alt="参考照片来自训练集，不是本视角真实留出照片"></details><p>下一步先筛选需要修复的区域，检查跨视角与连续帧一致性，再决定是否引入训练。生成的纹理不作为真实几何的证明。</p>
<details><summary>已完成与仍需验证的边界</summary><p>已完成：工作视频一致性核验、本地渲染与官方结果比对、四组×三个种子的固定高斯微调、四种采样尺度评测、稀疏深度一致性评测、同路径新视角渲染。</p><p>仍需验证：大幅俯仰和真实远近移动的配对数据、漂浮物的空间分布与消除程度、扩散修复的跨视角和视频一致性、将有效模块接入 Mini-Splatting2 的完整训练过程。当前局部微调没有重跑其增密、简化及可见性剔除过程。</p><p>所有59帧参与过SfM；8帧仅在外观训练中留出，且已用于开发比较，不能再称最终独立测试集。当前人体大致静止，背景有人移动；本实验不构成动态人物重建达标。</p></details>
<p class="refs">方法来源：<a href="https://github.com/fatPeter/mini-splatting2">Mini-Splatting2</a> · <a href="https://github.com/autonomousvision/mip-splatting">Mip-Splatting</a> · <a href="https://github.com/DepthAnything/Depth-Anything-V2">Depth Anything V2</a> · <a href="https://github.com/nv-tlabs/Difix3D">Difix3D+</a>。所有效果图均为本地视频或真实模型渲染。</p></main>
<script>const $=id=>document.getElementById(id);for(let i=0;i<59;i+=8){const o=document.createElement('option');o.value=String(i).padStart(6,'0');o.textContent=o.value;$('frame').append(o);}$('frame').value='000024';function update(){const f=$('frame').value,s=$('scale').value,v=$('variant').value;for(const [id,dir] of [['gt','gt'],['control','classic'],['candidate',v]]){const src=`report-assets/${dir}/${f}-s${s}.png`;$(id).src=src;$(id+'-link').href=src;}$('candidate-label').textContent=$('variant').selectedOptions[0].textContent;}for(const id of ['frame','scale','variant'])$(id).onchange=update;update();</script></html>'''
    html=html.replace('__NATIVE__',f'{a["classic"]["1"]["full_mean"]:.2f} → {a["mip-depth"]["1"]["full_mean"]:.2f}').replace('__SMALL__',f'{a["classic"]["8"]["full_mean"]:.2f} → {a["mip-depth"]["8"]["full_mean"]:.2f}').replace('__DEPTH__',f'{base:.2f}% → {md:.2f}%')
    ds=verification['difix_summary']
    html=html.replace('每帧中位数的平均；并非实测深度GT','种子0；每帧中位数的平均；非实测GT')
    html=html.replace('__DIFIX_PSNR__',f'{difix["mean_input_psnr"]:.2f} → {difix["mean_output_psnr"]:.2f}')
    html=html.replace('__DIFIX_SSIM__',f'{ds["input"]["ssim"]:.3f} → {ds["difix"]["ssim"]:.3f}')
    html=html.replace('__DIFIX_LPIPS__',f'{ds["input"]["lpips"]:.3f} → {ds["difix"]["lpips"]:.3f}')
    diffusion_js='''const diffRows=__ROWS__;for(const r of diffRows){const o=document.createElement('option');o.value=r.frame;o.textContent=r.frame;$('diff-frame').append(o);}$('diff-frame').value='000008';function updateDiff(){const f=$('diff-frame').value,r=diffRows.find(x=>x.frame===f);for(const kind of ['gt','input','difix']){const src=`report-assets/difix/${f}-${kind}.png`;$('diff-'+kind).src=src;$('diff-'+kind+'-link').href=src;}$('diff-reference').src=`report-assets/difix/${f}-reference.png`;$('diff-score').textContent=`本帧 PSNR：${r.input_psnr.toFixed(2)} → ${r.output_psnr.toFixed(2)} dB；使用训练参考帧 ${r.reference}。`;} $('diff-frame').onchange=updateDiff;updateDiff();'''.replace('__ROWS__',json.dumps(difix['rows']))
    html=html.replace('</script>',diffusion_js+'</script>')
    (VIEW/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
