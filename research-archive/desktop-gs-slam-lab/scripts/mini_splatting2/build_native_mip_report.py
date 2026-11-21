"""Four native full-training candidates, shown with original heldout photographs."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001';VIEW=ROOT/'viewer'
NAMES={'base':'native-baseline-18000-s0','motion':'native-motion-rgb-18000-s0',
       'mip':'native-mip-18000-s0','combined':'native-motion-mip-18000-s0',
       'depthmask':'native-motion-mip-depthmask-18000-s0'}

def main():
    results={k:json.loads((ROOT/n/'evaluation-summary.json').read_text()) for k,n in NAMES.items()}
    assets=VIEW/'native-mip-assets';assets.mkdir(exist_ok=True)
    for k,name in NAMES.items():
        d=assets/k;d.mkdir(exist_ok=True)
        shutil.copytree(ROOT/name/'model/test/ours_18000/renders',d/'native',dirs_exist_ok=True)
        for p in (ROOT/name/'gsplat-evaluation').glob('*.png'):shutil.copy2(p,d/p.name)
        for p in (ROOT/name).glob('distance-*.png'):shutil.copy2(p,d/p.name)
        for p in (ROOT/name).glob('elevation-*.png'):shutil.copy2(p,d/p.name)
    shutil.copytree(ROOT/NAMES['base']/'model/test/ours_18000/gt',assets/'gt',dirs_exist_ok=True)
    summary=dict(status='five_full_native_variants_evaluated',steps=18000,seeds=[0],results=results,
        mask_repeats=json.loads((ROOT/'native-motion-summary.json').read_text())['aggregate'],
        comparisons='baseline vs motion; baseline vs Mip; motion vs motion+Mip',
        training_only_views=51,development_holdouts=8,
        mip_implementation='3D smoothing from training-camera sample footprint; projected covariance +0.1I and differentiable determinant opacity compensation',
        topology='native cloning, depth reinitialization, simplification, culling and sparse Adam retained',
        kernel_validation='native-mip-runtime/validation.json',
        limitations=['This page compares seed0; later matched repeats are reported separately','same-camera image downsampling is not physical camera zoom validation',
            'unobserved distance/elevation paths have no GT','motion-masked walking background person intentionally excluded',
            '3D filter export baked into PLY; 2D compensation still required'])
    (ROOT/'native-mip-summary.json').write_text(json.dumps(summary,indent=2));(assets/'summary.json').write_text(json.dumps(summary,indent=2))
    src=(VIEW/'report.html').read_text(encoding='utf-8');style=src.split('<style>',1)[1].split('</style>',1)[0]
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Mini-Splatting2 · 完整训练与Mip滤波</title><style>__STYLE__</style><main>
<div class="eyebrow">同一工作视频 · 四组原生完整训练</div><h1>先处理运动残影，再保持尺度变化下的清晰度</h1>
<p>Mini-Splatting2 保留完整的增密、深度重初始化、简化与可见性剔除。每组从相同稀疏点训练18000步：原版、运动遮罩、Mip、运动遮罩＋Mip。分别观察两个改动的作用，再比较组合效果。</p>
<p><a href="index.html?model=native-motion-mip&view=8">打开组合模型，直接转动观察 ↗</a> · <a href="motion.html">运动遮罩的三次重复</a> · <a href="report.html">深度和扩散的初步对照</a></p>
<div class="cards"><div class="card"><span>原尺寸全图 PSNR · 遮罩 → 遮罩＋Mip</span><strong>__FULL__ dB</strong><span>8个开发留出视角，完整原生渲染</span></div><div class="card"><span>缩小8倍 · 遮罩 → 遮罩＋Mip</span><strong>__SMALL__ dB</strong><span>同一相机，图像面积缩采样</span></div><div class="card"><span>原尺寸人物区域 PSNR</span><strong>__CROP__ dB</strong><span>固定人物矩形，含部分背景</span></div></div>
<h2>原尺寸：检查人物细节与残影</h2><p>左侧始终是真实留出照片。中间与右侧使用同一相机和同一帧，模型没有使用该照片优化。可切换全部8个留出视角，查看改进是否只出现在局部。</p>
<div class="controls"><label>留出视角 <select id="frame"></select></label><label>中间模型 <select id="left"></select></label><label>右侧模型 <select id="right"></select></label></div>
<div class="panels"><figure><figcaption>真实照片</figcaption><a id="gt-link" target="_blank"><img id="gt"></a></figure><figure><figcaption id="left-label"></figcaption><a id="left-link" target="_blank"><img id="left-image"></a></figure><figure><figcaption id="right-label"></figcaption><a id="right-link" target="_blank"><img id="right-image"></a></figure></div><p id="native-score"></p>
<h2>缩小渲染：检查细线、头发与背景纹理</h2><p>Mip-Splatting 用训练视角的采样密度限制三维高斯的最高频率；投影后根据像素面积滤波，并补偿不透明度，避免单纯膨胀高斯造成的亮度和纹理变化。下面保持相机位置不动，只降低渲染分辨率。</p>
<div class="controls"><label>缩小倍数 <select id="scale"><option value="2">2倍</option><option value="4">4倍</option><option value="8" selected>8倍</option></select></label></div>
<div class="panels"><figure><figcaption>中间模型 · 缩小后放大显示</figcaption><a id="small-left-link" target="_blank"><img id="small-left"></a></figure><figure><figcaption>右侧模型 · 缩小后放大显示</figcaption><a id="small-right-link" target="_blank"><img id="small-right"></a></figure></div><p id="scale-score"></p>
<h2>接入保留了什么</h2><p>运动遮罩只改变走动背景人物区域的RGB损失。Mip在所有原生渲染分支中使用一致的三维平滑与二维补偿；原始尺度仍参与增密和简化，滤波半径随拓扑变化和训练位置更新。导出的PLY保留完整三阶球谐颜色，查看器使用匹配的抗混叠设置。</p>
<p>来源：<a href="https://github.com/fatPeter/mini-splatting2">Mini-Splatting2 官方代码</a> · <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting 论文与项目</a> · <a href="native-mip-assets/summary.json">本次评测和配置</a></p>
<details><summary>实验边界与剩余问题</summary><p>运动遮罩已做三个匹配种子；Mip两组当前各一个种子。所有组的迭代预算相同，但高斯数量和运算量会不同。本轮不是速度比较。图像缩小不能替代真实远近移动的配对拍摄。</p><p>原视频的相机半径范围约1.31倍，相机高度变化较小。超出拍摄范围的大幅俯仰仍缺少真实观察；自由视角的结果没有对应GT，不能据此证明恢复了真实遮挡面。</p><p>排除的走动背景人物没有被重建；视频中的红衣主体大致静止。当前工作不代表动态人体重建完成。8个视角是反复使用的开发留出集，全部相机参与过SfM，最终独立数据验证仍需补充。</p></details>
</main><script>const data=__DATA__,names={base:'原版',motion:'运动遮罩',mip:'Mip',combined:'运动遮罩＋Mip'},$=id=>document.getElementById(id);for(let i=0;i<8;i++){const o=document.createElement('option');o.value=i;o.textContent=data.base.native_rows[i].frame;$('frame').append(o);}for(const id of ['left','right'])for(const [k,n] of Object.entries(names)){const o=document.createElement('option');o.value=k;o.textContent=n;$(id).append(o);}$('frame').value=1;$('left').value='motion';$('right').value='combined';function img(id,src){$(id).src=src;$(id+'-link').href=src;}function update(){const i=Number($('frame').value),f=data.base.native_rows[i].frame,png=String(i).padStart(5,'0')+'.png',l=$('left').value,r=$('right').value,s=$('scale').value;img('gt','native-mip-assets/gt/'+png);for(const [side,key] of [['left',l],['right',r]]){$(side+'-label').textContent=names[key]+' · 18000步';$(side+'-image').src='native-mip-assets/'+key+'/native/'+png;$(side+'-link').href=$(side+'-image').src;img('small-'+side,'native-mip-assets/'+key+'/'+f+'-s'+s+'.png');}const a=data[l].native_rows[i],b=data[r].native_rows[i];$('native-score').textContent=`本帧全图PSNR ${a.full.psnr.toFixed(2)} → ${b.full.psnr.toFixed(2)} dB；人物区域 ${a.subject_crop.psnr.toFixed(2)} → ${b.subject_crop.psnr.toFixed(2)} dB。`;$('scale-score').textContent=`${s}倍缩小，8视角平均PSNR：${data[l].scale_gsplat[s].full_psnr.toFixed(2)} → ${data[r].scale_gsplat[s].full_psnr.toFixed(2)} dB。`; }for(const id of ['frame','left','right','scale'])$(id).onchange=update;update();</script></html>'''
    html=html.replace('四组原生完整训练','五组原生完整训练')
    html=html.replace('原版、运动遮罩、Mip、运动遮罩＋Mip。','原版、运动遮罩、Mip、运动遮罩＋Mip，以及进一步在深度补点时排除运动区域的组合。')
    html=html.replace('native-motion-mip&view=8','native-motion-mip-depthmask&view=8')
    html=html.replace('遮罩 → 遮罩＋Mip','遮罩 → Mip＋RGB/补点遮罩')
    html=html.replace("combined:'运动遮罩＋Mip'", "combined:'运动遮罩＋Mip',depthmask:'Mip＋RGB/补点遮罩'")
    html=html.replace("$('right').value='combined'", "$('right').value='depthmask'")
    html=html.replace('原始尺度仍参与增密和简化，','额外的补点遮罩阻止模型在深度重初始化时从已排除的运动像素重新生成高斯；这一步没有添加单目深度损失。原始尺度仍参与增密和简化，')
    html=html.replace('Mip两组当前各一个种子','本页固定展示种子0；组合模型后续的匹配重复见深度指导页')
    extra=json.loads((ROOT/'additional-frame-evaluation-001/summary.json').read_text());assert extra['status']=='complete'
    extra_assets=VIEW/'additional-assets';extra_assets.mkdir(exist_ok=True)
    for key in ['gt',*NAMES]:
        shutil.copytree(ROOT/'additional-frame-evaluation-001'/key,extra_assets/key,dirs_exist_ok=True)
    shutil.copy2(ROOT/'additional-frame-evaluation-001/summary.json',extra_assets/'summary.json')
    shutil.copy2(ROOT/'additional-frame-evaluation-001/frozen-protocol.json',extra_assets/'frozen-protocol.json')
    a0=extra['models']['base']['aggregate'];b0=extra['models']['depthmask']['aggregate']
    extra_section=f'''<h2>额外24帧：验证没有参加训练的时间帧</h2><p>从原视频中均匀选出24张未用于原始SfM和外观训练的照片，仅用固定地图估计其相机姿态，原地图、内参和模型参数保持不变。所有候选模型在计算这些指标前冻结。这里仍是同一视频内的时间插值验证。</p><p>原版 → Mip＋RGB/补点遮罩：原尺寸全图PSNR <strong>{a0['1']['full_psnr']:.2f} → {b0['1']['full_psnr']:.2f} dB</strong>；人物区域 <strong>{a0['1']['crop_psnr']:.2f} → {b0['1']['crop_psnr']:.2f} dB</strong>；缩小8倍 <strong>{a0['8']['full_psnr']:.2f} → {b0['8']['full_psnr']:.2f} dB</strong>。遮罩对全图均值的提升小于原8帧开发集，不能把局部强残影的改善推广到所有视角。</p><div class="controls"><label>额外时间帧 <select id="extra-frame"></select></label></div><p>中间与右侧模型沿用上方选择。</p><div class="panels"><figure><figcaption>额外真实照片</figcaption><a id="extra-gt-link" target="_blank"><img id="extra-gt"></a></figure><figure><figcaption id="extra-left-label"></figcaption><a id="extra-left-link" target="_blank"><img id="extra-left"></a></figure><figure><figcaption id="extra-right-label"></figcaption><a id="extra-right-link" target="_blank"><img id="extra-right"></a></figure></div><p id="extra-score"></p><small>人物区域沿用原有矩形随时间插值，含部分背景；全图指标为主要口径。<a href="additional-assets/frozen-protocol.json">冻结的评测协议</a> · <a href="additional-assets/summary.json">全部24帧指标</a></small>'''
    html=html.replace('<h2>接入保留了什么</h2>',extra_section+'<h2>接入保留了什么</h2>')
    extra_script='''const extraData=__EXTRA__,extraRows=Object.fromEntries(Object.entries(extraData).map(([k,v])=>[k,v.rows.filter(r=>r.divisor===1)]));extraRows.base.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=String(r.source_index).padStart(6,'0');$('extra-frame').append(o);});function extraUpdate(){const j=Number($('extra-frame').value),l=$('left').value,r=$('right').value,f=extraRows.base[j].frame;img('extra-gt','additional-assets/gt/'+f);for(const [side,key] of [['left',l],['right',r]]){$('extra-'+side+'-label').textContent=names[key];img('extra-'+side,'additional-assets/'+key+'/'+f.replace('.png','-s1.png'));}const a=extraRows[l][j],b=extraRows[r][j];$('extra-score').textContent=`本帧全图PSNR ${a.full.psnr.toFixed(2)} → ${b.full.psnr.toFixed(2)} dB；人物区域 ${a.subject_crop.psnr.toFixed(2)} → ${b.subject_crop.psnr.toFixed(2)} dB。`;}$('extra-frame').onchange=extraUpdate;for(const id of ['left','right'])$(id).addEventListener('change',extraUpdate);extraUpdate();'''.replace('__EXTRA__',json.dumps(extra['models']))
    html=html.replace('</script>',extra_script+'</script>')
    summary['additional_frame_validation']=dict(frames=24,source='additional-frame-evaluation-001/summary.json',all_models_frozen_before_evaluation=True)
    (ROOT/'native-mip-summary.json').write_text(json.dumps(summary,indent=2));(assets/'summary.json').write_text(json.dumps(summary,indent=2))
    a,b=results['motion'],results['depthmask']
    html=html.replace('__STYLE__',style).replace('__DATA__',json.dumps(results))
    for key,first,last in [('__FULL__',a['native_mean']['full_psnr'],b['native_mean']['full_psnr']),('__CROP__',a['native_mean']['crop_psnr'],b['native_mean']['crop_psnr']),('__SMALL__',a['scale_gsplat']['8']['full_psnr'],b['scale_gsplat']['8']['full_psnr'])]:html=html.replace(key,f'{first:.2f} → {last:.2f}')
    (VIEW/'native-mip.html').write_text(html,encoding='utf-8')
    for file in ['index.html','motion.html','report.html']:
        p=VIEW/file;text=p.read_text(encoding='utf-8')
        marker='<p><a href="native-mip.html">完整训练：运动遮罩与Mip组合 ↗</a></p>'
        if file=='index.html':
            text=text.replace(marker,'')
            if 'native-mip.html' not in text:text=text.replace('<details>','<a class="report-link" href="native-mip.html">完整训练：运动遮罩与Mip组合 ↗</a><details>',1)
        elif 'native-mip.html' not in text:text=text.replace('</h1>','</h1>'+marker,1)
        p.write_text(text,encoding='utf-8')

if __name__=='__main__':main()
