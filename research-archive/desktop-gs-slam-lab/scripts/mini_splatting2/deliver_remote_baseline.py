from pathlib import Path
import json, shutil, datetime
lab=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
base=lab/'outputs/mini-splatting2-computer00-001'
review=base/'remote-review-001'
run=review/'hover-msv2-fast-001'
dest=Path(r'C:\Users\Administrator\Downloads\04_mini_splatting2_work\RESULTS')
dest.mkdir(exist_ok=True)
for name in ['gt','renders']:
    (dest/name).mkdir(exist_ok=True)
    for p in (run/'test/ours_18000'/name).glob('*.png'):shutil.copy2(p,dest/name/p.name)
summary=json.loads((run/'validated-summary.json').read_text())
summary.update({
 'status':'baseline_complete_numeric_and_visual_review_done',
 'training_function_seconds':float((run/'time.txt').read_text()),
 'timing_scope':'author training function, including initialization/save; excludes SfM, installation, separate rendering and metrics',
 'visual_review':{
  'all_eight_full_pairs_reviewed':True,
  'native_detail_views':['00001.png','00003.png'],
  'observation':'Mostly stationary red-jacket subject retains clothing structure in most views. Test01/source000008 has strong translucent ghosts from moving scene content. Face and hand fine detail is not accepted as recovered.',
  'worst_test_view':{'output':'00001.png','source':'000008.jpg','psnr_db':19.14965057373047},
  'dynamic_human_goal_complete':False,
  'scope':'short handheld static-scene baseline; no synchronized moving-human or large-scene completion claim'},
 'environment_note':'Python3.8.18, torch1.10.0+cu113, CUDA compiler11.4, gcc/g++10.5; differs from author-tested torch1.12.1+cu116/CUDA11.6.',
 'source_compatibility_change':'Only VGG16 weight-loading API fallback for old torchvision in LPIPS evaluation; same ImageNet weights. No training-method source changes.',
 'model_server_path':'/home/zhouhaoyi/codex-mini-splatting2-20261016/hover-msv2-fast-001/point_cloud/iteration_18000/point_cloud.ply',
 'source_commit':'5ec9202c2db7d900728bfc09733e85467cff885e',
 'next_step':'Keep this baseline fixed; evaluate dynamic-region handling separately. Static masking can remove moving clutter but cannot demonstrate reconstruction of moving people.'})
(base/'final-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for name in ['results.json','per_view.json','cfg_args','time.txt']:shutil.copy2(run/name,dest/name)
shutil.copy2(base/'visual-review/independent-checks.json',dest/'independent-checks.json')
per=json.loads((run/'per_view.json').read_text())['ours_18000']
frames=json.loads((base/'hover-sfm-001/state.json').read_text())['appearance_split']['test']
views=[{'file':f'{i:05d}.png','source':f,'psnr':per['PSNR'][f'{i:05d}.png']} for i,f in enumerate(frames)]
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mini-Splatting2 · 电脑00基础复现</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f1f4f7;color:#14283c;font:16px/1.75 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1500px;margin:auto;padding:36px 30px}h1{font-size:34px;line-height:1.3;margin:8px 0 15px}h2{font-size:24px;margin-top:36px}.kicker{color:#52718c;font-size:13px;letter-spacing:1px}.lead{max-width:1000px}.metrics{display:flex;flex-wrap:wrap;gap:16px;margin:24px 0}.metric{background:#fff;border:1px solid #dce3eb;border-radius:10px;padding:16px 22px;min-width:150px}.metric b{display:block;font-size:27px}.metric span{color:#52677a;font-size:14px}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:14px;padding:18px;background:white;border-radius:10px 10px 0 0}select,button{font:inherit;padding:7px 12px;border:1px solid #a4b3c1;border-radius:6px;background:white;color:#14283c;cursor:pointer}button{border-color:#aa643f;color:#7b4226}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;background:#fff;padding:8px}figure{margin:0;min-width:0}figcaption{padding:10px;text-align:center;font-weight:600}figure img{display:block;width:100%;height:auto}.note{color:#53687b;font-size:14px}.reading{background:white;border-radius:10px;padding:20px 25px;margin:18px 0}.reading p{margin:8px 0}details{border-top:1px solid #dce3eb;padding:16px 0}code{overflow-wrap:anywhere;font-size:13px}a{color:#145ca1}footer{padding:28px 0;font-size:13px;color:#53687b}@media(max-width:680px){main{padding:20px 12px}h1{font-size:27px}.metrics{gap:8px}.metric{min-width:calc(50% - 4px);padding:12px}.toolbar{padding:12px;font-size:14px}}
</style><main><div class="kicker">MINI-SPLATTING2 / 基础复现 / 2026-10-16</div><h1>从桌面视频到可检验的高斯重建</h1>
<p class="lead">已在 RTX 3090 上完成官方稀疏版本的 18,000 次迭代。输入来自电脑00的 <b>HOVER_VERSA_0102.mp4</b>：59张标定图像，51张用于外观训练，8张用于留出视图评测。下方展示全部留出视图的原图与重建图，点击图片可查看原尺寸。</p>
<div class="metrics"><div class="metric"><b>29.40 dB</b><span>平均 PSNR</span></div><div class="metric"><b>0.9443</b><span>平均 SSIM</span></div><div class="metric"><b>0.1066</b><span>平均 LPIPS ↓</span></div><div class="metric"><b>149,413</b><span>最终高斯数量</span></div><div class="metric"><b>134.1 s</b><span>训练函数计时</span></div></div>
<p class="note">分辨率757 × 1347。训练计时不含标定、安装和后续评测；SfM单独耗时约138.9秒。全部59张图像参与相机估计，留出的是外观训练图像。指标为8张完整图像的均值。</p>
<h2>完整图像对照</h2><div class="toolbar"><label for="view">留出视图</label><select id="view"></select><span id="score"></span><button id="worst">查看明显残影视图</button></div>
<div class="pair"><figure><figcaption id="gtlabel"></figcaption><a id="gtlink" target="_blank"><img id="gt" alt="原始留出图像"></a></figure><figure><figcaption>Mini-Splatting2 重建</figcaption><a id="renderlink" target="_blank"><img id="render" alt="对应相机的高斯重建图像"></a></figure></div>
<h2>本次复现的方法</h2><div class="reading"><p><b>优化对象：</b>三维高斯的位置、尺度、旋转、不透明度和球谐颜色系数。利用已标定相机投影并合成图像，以图像重建损失更新参数。</p><p><b>快速建立表达能力：</b>Mini-Splatting2在训练早期进行激进的高斯增密，以更快形成覆盖场景的表示；深度重初始化及重要性采样用于后续几何调整和高斯简化。</p><p><b>本次参数：</b>使用作者fast配置；从第500次迭代开始，每250次进行激进克隆；第2,000次进行深度重初始化，第3,000和8,000次进行简化，最终训练到18,000次。采用indoor重要性配置和0.6采样因子。</p><p>这次跑通了“相机估计 → 高斯训练 → 留出视图渲染 → 指标核验”全流程。尚未进行同预算的传统3DGS对照，因此这里不声称测得相对加速或精度提升。</p></div>
<h2>效果边界与下一步</h2><div class="reading"><p>多数视角中，较静止主体的衣服轮廓、纹理和整体形状得到保留。但测试视图01（原图000008.jpg）出现明显半透明残影，PSNR仅19.15 dB；整体平均指标不能掩盖这一失败。</p><p>这段手持视频里有走动与遮挡，静态表示会把不同时刻的内容融合到同一个场景中。这与当前残影现象相符，但这一次训练本身不能单独证明全部误差原因。面部和手部精细结构尚未通过清晰度验收。</p><p><b>基础复现已经完成，移动人物清晰重建尚未完成。</b>接下来应保留这份基线，对动态区域的处理做独立对照。去除走动干扰能验证静态场景质量，但不能代替重建移动人物本身。</p></div>
<details><summary>复现条件、文件位置与核验记录</summary><p>运行环境：RTX 3090 24GB，Ubuntu22.04，Python3.8.18，PyTorch1.10.0+cu113，CUDA编译器11.4，gcc/g++10.5。它与作者测试环境PyTorch1.12.1+cu116/CUDA11.6不同；本结果属于官方方法在现有环境下的适配复现。训练方法源码未改；LPIPS评测增加了旧版torchvision的等价VGG16权重加载接口兼容。</p><p>已独立验证8张GT与原始留出图逐像素一致，8张保存图像重新计算的PSNR与官方结果一致。已查看全部8组整图，以及失败视图和正面主体的原尺寸对照。</p><p>服务器模型：<code>/home/zhouhaoyi/codex-mini-splatting2-20261016/hover-msv2-fast-001/point_cloud/iteration_18000/point_cloud.ply</code></p><p>官方代码版本：<code>5ec9202c2db7d900728bfc09733e85467cff885e</code>。Windows原始资料和服务器原有项目均保留。</p><p><a href="summary.json">本次结果摘要</a> · <a href="per_view.json">逐视图指标</a> · <a href="independent-checks.json">图像独立核验</a> · <a href="https://github.com/fatPeter/mini-splatting2">官方源码</a> · <a href="https://arxiv.org/abs/2411.12788">论文</a></p></details>
<footer>图像为本次实际留出视图与训练结果，未作修复、锐化或生成式补全。页面可离线打开。</footer></main><script>const views=VIEWDATA;const sel=document.getElementById('view');views.forEach((v,i)=>sel.add(new Option(`视图 ${String(i).padStart(2,'0')} · ${v.source}`,i)));function show(){const v=views[Number(sel.value)];document.getElementById('gt').src='gt/'+v.file;document.getElementById('gtlink').href='gt/'+v.file;document.getElementById('render').src='renders/'+v.file;document.getElementById('renderlink').href='renders/'+v.file;document.getElementById('gtlabel').textContent='原图 · '+v.source;document.getElementById('score').textContent='该视图 PSNR '+v.psnr.toFixed(2)+' dB'}sel.value='3';sel.onchange=show;document.getElementById('worst').onclick=()=>{sel.value='1';show()};show();</script></html>'''
(dest/'index.html').write_text(page.replace('VIEWDATA',json.dumps(views)),encoding='utf-8')
shutil.copytree(dest,base/'delivery',dirs_exist_ok=True)
archive=base/'reproduction-scripts';archive.mkdir(exist_ok=True)
for p in dest.parent.iterdir():
    if p.suffix in ['.sh','.cs']:shutil.copy2(p,archive/p.name)
(archive/'version-note.txt').write_text('03-setup.sh here is the prepared offline retry variant; the successful first run used the earlier script with setuptools65.5.0/wheel0.41.3/ninja1.11.1.1 pip install and without --no-index. LPIPS API fallback was applied by 04-train.sh. See actual archived setup and train logs.\n',encoding='utf-8')
manifest=lab/'manifests/mini-splatting2-computer00.json';d=json.loads(manifest.read_text(encoding='utf-8'))
d.update({'status':'baseline_complete_numeric_and_visual_review_done','reproduction_started':True,'completion_claimed':False,'completion_scope':'Single-sequence Mini-Splatting2 baseline complete; official benchmark reproduction and moving-human goal not complete.','baseline_result':summary,'local_delivery':str(base/'delivery/index.html'),'last_recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
d['remote_environment']['status']='four_extensions_compiled_gpu_checks_and_full_training_render_passed'
d['remote_computer']['remote_training_started']=True
manifest.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=lab/'manifests/dynamic-human-work.json';d=json.loads(p.read_text(encoding='utf-8'))
d['status']='actor23_107_preserved_remote00_mini_splatting2_baseline_complete_dynamic_clarity_unaccepted'
d['current_remote_task'].update({'training_started':True,'baseline_complete':True,'dynamic_human_goal_complete':False})
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'delivery':str(dest),'summary':str(base/'final-summary.json'),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())}))
