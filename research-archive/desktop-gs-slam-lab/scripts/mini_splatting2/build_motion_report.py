"""Publish full-frame native motion-mask evidence, separate from filter ablations."""
import json
from pathlib import Path
import shutil
import statistics

LAB=Path(__file__).resolve().parents[2];ROOT=LAB/'outputs/mini-splatting2-mip-depth-001';VIEW=ROOT/'viewer'

def main():
    names=['native-baseline-18000-s0','native-motion-rgb-18000-s0']
    results=[json.loads((ROOT/n/'evaluation-summary.json').read_text()) for n in names]
    assets=VIEW/'motion-assets';assets.mkdir(exist_ok=True)
    for key,name in zip(['control','masked'],names):
        d=assets/key;d.mkdir(exist_ok=True)
        for p in (ROOT/name/'model/test/ours_18000/renders').glob('*.png'):shutil.copy2(p,d/p.name)
    shutil.copytree(ROOT/names[0]/'model/test/ours_18000/gt',assets/'gt',dirs_exist_ok=True)
    shutil.copy2(ROOT/'motion-masks-001/000011-review.jpg',assets/'training-mask.jpg')
    pairs=[]
    for seed in [0,1,2]:
        paths=[ROOT/f'native-{kind}-18000-s{seed}'/'evaluation-summary.json' for kind in ['baseline','motion-rgb']]
        if all(p.exists() for p in paths):
            values=[json.loads(p.read_text()) for p in paths]
            pairs.append(dict(seed=seed,baseline=values[0]['native_mean'],masked=values[1]['native_mean'],
                gaussian_counts=[v['gaussians'] for v in values]))
    aggregate={}
    for kind in ['baseline','masked']:
        aggregate[kind]={}
        for metric in ['full_psnr','crop_psnr','ssim','lpips_alex']:
            values=[p[kind][metric] for p in pairs]
            aggregate[kind][metric]=dict(mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else None,min=min(values),max=max(values))
    summary=dict(status='three_native_seed_pairs_complete' if len(pairs)==3 else 'native_repeats_in_progress',baseline=results[0],masked=results[1],
        native_training_steps=18000,seeds=[p['seed'] for p in pairs],repeats_pending=len(pairs)<3,
        pairs=pairs,aggregate=aggregate,
        intervention='RGB supervision excludes one walking background person on 13 reviewed training frames; no image generation',
        mask_frame_count=13,mask_test_frames=0,depth_reinitialization_masked=False,
        differences='same seed, initialization, schedule, native method; topology trajectory and final point count differ',
        scope='static reconstruction of foreground and background; excluded moving person not reconstructed; no dynamic scene completion claim')
    (ROOT/'native-motion-summary.json').write_text(json.dumps(summary,indent=2))
    (assets/'summary.json').write_text(json.dumps(summary,indent=2))
    src=(VIEW/'report.html').read_text(encoding='utf-8');style=src.split('<style>',1)[1].split('</style>',1)[0]
    a,b=[{k:v['mean'] for k,v in aggregate[t].items()} for t in ['baseline','masked']]
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mini-Splatting2 · 运动残影对照</title><style>__STYLE__</style><main>
<div class="eyebrow">原版完整训练 · 同一工作视频 · 2026-10-18</div><h1>背景运动会被错误地拟合成漂浮结构</h1>
<p>沿用 Mini-Splatting2 的增密、深度重初始化、简化与可见性剔除，两组都从相同稀疏点开始训练18000步。改动只作用于RGB损失：将走动背景人物的像素排除，保留红衣主体与其余场景的监督。</p>
<p><a href="index.html?model=native-motion&view=8">直接查看新三维模型 ↗</a> · <a href="report.html">Mip、深度和扩散对照</a> · <a href="motion-assets/summary.json">评测口径</a></p>
<div class="cards"><div class="card"><span>全图 PSNR · 原版 → 运动遮罩</span><strong>__FULL__ dB</strong><span>8个开发留出视角，随机种子0</span></div><div class="card"><span>人物区域 PSNR · 原版 → 运动遮罩</span><strong>__CROP__ dB</strong><span>预先固定矩形，含少量背景</span></div><div class="card"><span>全图 LPIPS · 越低越好</span><strong>__LPIPS__</strong><span>相同 AlexNet 度量，完整图像</span></div></div>
<h2>同一相机下，直接比较残影</h2><p>第000008帧中，原版模型将跨帧变化解释成错误的静态高斯，残影覆盖了主体。排除该背景人物的训练监督后，主体前方的强残影明显减轻。模型没有使用扩散修复，也没有使用留出照片进行训练。</p>
<div class="controls"><label for="motion-frame">留出视角</label><select id="motion-frame"></select></div>
<div class="panels"><figure><figcaption>真实留出照片</figcaption><a id="gt-link" target="_blank"><img id="gt"></a></figure><figure><figcaption>原版完整重训 · 18000步</figcaption><a id="control-link" target="_blank"><img id="control"></a></figure><figure><figcaption>运动遮罩完整重训 · 18000步</figcaption><a id="masked-link" target="_blank"><img id="masked"></a></figure></div><p id="score"></p>
<h2>训练改了什么</h2><p>先用光流与相机的极线约束定位不符合静态场景假设的区域，再用SAM2.1生成走动背景人物的完整遮罩，并逐张核验。13张训练照片使用遮罩，另外38张训练照片保持原监督。遮罩只控制哪些像素参与光度损失，原始RGB和全部留出照片都保留。</p><img src="motion-assets/training-mask.jpg" alt="第11张训练照片与走动背景人物的遮罩"><small>图源：工作视频训练帧000011；红色为SAM2.1遮罩及4像素边界扩张，经人工核验。右图是训练监督说明，不是重建结果。</small>
<p>这组结果支持先处理输入中的运动不一致，再研究抗混叠与深度约束。第一轮Mip实验针对采样尺度；本轮针对错误的静态拟合，两者作用不同。</p>
<details><summary>实验边界与待验证项</summary><p>目前仅完成一个随机种子，重复实验仍待完成。高斯数量分别为__COUNT__；两组迭代预算相同，拓扑演化和计算量并不完全相同。运行环境为RTX5080、PyTorch2.7.1与CUDA12.8。</p><p>当前版本不重建被排除的走动背景人物；局部未观测背景仍可能有残留。深度重初始化仍采用原版流程，本轮只改变RGB损失。红衣主体在该视频中大致静止，因此这不是动态人物重建的完成证明。</p><p>大幅俯仰、真实远近移动的配对验证及Mip与深度模块的完整原版接入仍在进行。8个视角是开发用外观留出集，全部相机参与过SfM，不构成最终独立测试。</p></details></main>
<script>const rows=__ROWS__;const $=id=>document.getElementById(id);for(let j=0;j<8;j++){const o=document.createElement('option');o.value=j;o.textContent=rows[0][j].frame;$('motion-frame').append(o);}$('motion-frame').value='1';function update(){const j=Number($('motion-frame').value),file=String(j).padStart(5,'0')+'.png';for(const k of ['gt','control','masked']){const src='motion-assets/'+k+'/'+file;$(k).src=src;$(k+'-link').href=src;}const a=rows[0][j],b=rows[1][j];$('score').textContent=`本帧全图 PSNR：${a.full.psnr.toFixed(2)} → ${b.full.psnr.toFixed(2)} dB；人物区域：${a.subject_crop.psnr.toFixed(2)} → ${b.subject_crop.psnr.toFixed(2)} dB。`;} $('motion-frame').onchange=update;update();</script></html>'''
    html=html.replace('__STYLE__',style).replace('__FULL__',f'{a["full_psnr"]:.2f} → {b["full_psnr"]:.2f}').replace('__CROP__',f'{a["crop_psnr"]:.2f} → {b["crop_psnr"]:.2f}').replace('__LPIPS__',f'{a["lpips_alex"]:.3f} → {b["lpips_alex"]:.3f}').replace('__COUNT__',f'{results[0]["gaussians"]:,} 与 {results[1]["gaussians"]:,}').replace('__ROWS__',json.dumps([x['native_rows'] for x in results]))
    html=html.replace('8个开发留出视角，随机种子0',f'8个开发留出视角，{len(pairs)}组随机种子均值')
    html=html.replace('同一相机下，直接比较残影</h2>','同一相机下，直接比较残影</h2><small>下方完整照片固定展示随机种子0；上方指标统计所有已完成重复实验。</small>')
    html=html.replace('目前仅完成一个随机种子，重复实验仍待完成。',
        f'已完成{len(pairs)}组匹配随机种子。每组全图PSNR改善分别为'+ '、'.join(f'{p["masked"]["full_psnr"]-p["baseline"]["full_psnr"]:.2f} dB' for p in pairs)+'。')
    (VIEW/'motion.html').write_text(html,encoding='utf-8')
    if 'motion.html' not in src:
        src=src.replace('</h1>','</h1><p><a href="motion.html">新进展：运动残影与原版完整训练对照 ↗</a></p>',1)
        (VIEW/'report.html').write_text(src,encoding='utf-8')

if __name__=='__main__':main()
