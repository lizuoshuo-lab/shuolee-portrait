"""Publish only a reviewed static research result, preserving the062 reference."""
from pathlib import Path
import datetime,json,shutil,re
from publish_round_delivery import main as publish
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-pgsr-fresh-064';V=B.parent/'viewer';P=V/'body-static-pgsr-current'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
result=read(R/'gaussian-run/result.json');audit=read(R/'gaussian-run/export-audit/audit.json')
review=read(R/'visual-review.json');assert review['actually_inspected'] and result['steps']==18000 and audit['state']=='passed'
publish(R,'064','whole-body-static-pgsr','064 · 静态表面对照 · 1.8万步')
d=read(V/'models/whole-body-static-pgsr/delivery.json');P.mkdir(exist_ok=True)
for src,dst in {'gaussian-run/review-current.jpg':'review-current.jpg',
    'gaussian-run/export-audit/same-camera-062-064.jpg':'same-camera-062-064.jpg',
    'gaussian-run/export-audit/constant-color-comparison.jpg':'constant-color-comparison.jpg',
    'gaussian-run/export-audit/center-comparison.jpg':'center-comparison.jpg',
    'source-final/manifest.json':'method-source.json'}.items():shutil.copy2(R/src,P/dst)
state=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
    state='static_pgsr_training_and_export_completed_pending_full_quality_acceptance',
    steps=18000,initial_points=200000,points=result['points'],frames_used=result['frames_used'],
    frame_updates_min=min(result['per_frame_updates']),frame_updates_max=max(result['per_frame_updates']),
    from_scratch=True,representation='shared_static_world',body_deformation_used=False,
    compatibility_passed=True,geometry_accepted=False,quality_accepted=False,pc00_opened_this_round=False,
    visual_review=review,download_url=d['download_url'],local_ply=d['download_path'],viewer_url=d['viewer_url'],
    comparison_scope='062 vs064 changes losses, resolution and densification jointly; not a single-module ablation',
    license_scope='PGSR local research only; commercial authorization not confirmed')
save(R/'phase-conclusion.json',state);save(P/'conclusion.json',state)
content=f'''<section id="pose-clothing-current">
<h2>整体静态：让所有观察方向约束同一个穿衣表面</h2>
<p>整个穿衣人物使用同一组高斯位置、尺度、旋转、透明度与颜色。175个有效实拍帧只切换相机，不再逐帧改变身体或衣物几何。视角不同导致的可见性变化，通过投影和遮挡处理。</p>
<p>062是普通静态3DGS对照；064加入PGSR的平面高斯深度、表面法向一致性与相邻实拍视角的几何和局部纹理一致性。新增约束让不同方向看到的衣物表面相互支持；未拍摄的俯仰图像没有作为训练目标。</p>
<p class="notice">两轮均从原始观测重新生成20万个初始点，没有加载旧高斯或旧优化器。064完成18,000步，最终{result['points']:,}个高斯；175帧每帧参与{state['frame_updates_min']}—{state['frame_updates_max']}次更新。当前只重建穿衣人物，背景尚未合入。</p>
<h3>实拍视角的训练结果</h3>
<figure><a href="body-static-pgsr-current/review-current.jpg"><img src="body-static-pgsr-current/review-current.jpg" alt="064六个真实相机下实拍与静态表面渲染对照"></a><figcaption>每对左侧实拍，右侧064静态渲染；这些图像参与了训练。</figcaption></figure>
<h3>固定几何，改变观察方向</h3>
<figure><a href="body-static-pgsr-current/same-camera-062-064.jpg"><img src="body-static-pgsr-current/same-camera-062-064.jpg" alt="相同相机下062普通静态高斯和064静态表面约束的比较"></a><figcaption>每对左062、右064。统一世界坐标和相机，比较正面、斜侧、侧面及俯仰方向。没有这些新视角的实拍真值。</figcaption></figure>
<p>{review['summary_zh']}</p>
<details><summary>查看点中心分布和去掉颜色纹理后的覆盖</summary><figure><a href="body-static-pgsr-current/center-comparison.jpg"><img src="body-static-pgsr-current/center-comparison.jpg" alt="两轮静态模型实际高斯中心位置"></a><figcaption>所有高斯中心每点一个像素；点数不同，不能把密集程度直接当作几何精度。</figcaption></figure><figure><a href="body-static-pgsr-current/constant-color-comparison.jpg"><img src="body-static-pgsr-current/constant-color-comparison.jpg" alt="两轮模型统一灰色的实际渲染"></a><figcaption>去掉颜色纹理以检查覆盖，不是测量得到的真实表面。</figcaption></figure></details>
<div id="clothing-delivery"><h3>每轮可直接加载的标准高斯PLY</h3>
<p><a href="./?model=whole-body-static-pgsr">打开064静态表面对照</a> · <a href="{d['download_url']}">下载064兼容版PLY</a> · <a href="./?model=whole-body-static-fresh">打开062普通静态对照</a> · <a href="models/whole-body-static-fresh/shuolee_062_s18000_f0100.ply?download=1">下载062</a></p>
<p>在00的shuolee查看器点击“打开文件”，选择PLY，竖直方向使用默认+Y。文件包含完整世界坐标、协方差与球谐颜色，不需要人体模板、骨架或额外蒙皮。文件名f0100仅作查看相机参考，不是动态姿态。</p>
<details><summary>064完整路径与核验范围</summary><p style="overflow-wrap:anywhere">{d['download_path']}</p><p>原生PGSR与标准3DGS渲染一致性、标准PLY重载以及+Y坐标与球谐旋转分别核验。本轮尚未在00远端打开。格式通过与重建质量通过分别判定。</p><a href="models/whole-body-static-pgsr/delivery.json">完整导出核验</a></details></div>
<h3>当前结论与继续方向</h3><p>{review['limitations_zh']}</p><p>{review['next_zh']}</p>
<p><small>方法来源：<a href="https://github.com/zju3dv/PGSR">PGSR官方源码</a> · <a href="https://zju3dv.github.io/pgsr/paper/pgsr.pdf">原论文</a>。本轮为本地研究适配，使用全部真实帧、人物遮罩及有界点数；与062同时改变损失、分辨率和增密方式，不称单项消融。PGSR的商业授权尚未确认，本轮未部署00。</small></p>
</section>'''
page=V/'body-smplx.html';text=page.read_text(encoding='utf8')
begin='<!-- CLOTHING_PIPELINE_CURRENT_BEGIN -->';end='<!-- CLOTHING_PIPELINE_CURRENT_END -->'
assert text.count(begin)==1 and text.count(end)==1
text=text[:text.index(begin)+len(begin)]+'\n'+content+'\n'+text[text.index(end):]
page.write_text(text,encoding='utf8')
meta=read(V/'model-whole-body-static-pgsr.json');meta.update(method='PGSR local static research adaptation',
    licenseScope='local research',qualityAccepted=False,report='body-smplx.html#pose-clothing-current');save(V/'model-whole-body-static-pgsr.json',meta)
status=read(B/'work-status.json');status['local_training_running']=False
status['latest_local_execution']=dict(kind='static_pgsr_fresh_064',state=state['state'],running=False,directory=str(R),
    steps=18000,points=result['points'],from_scratch=True,representation='shared_static_world',
    quality_accepted=False,compatibility_passed=True,local_viewer_passed=False,
    review_url='http://127.0.0.1:8870/body-smplx.html#pose-clothing-current',download_url=d['download_url'],local_ply=d['download_path'])
save(B/'work-status.json',status)
p=B/'long-term-plan.md';txt=p.read_text(encoding='utf-8-sig')
start='<!-- GVHMR_059_ACTIVE -->';endp='<!-- END_GVHMR_059_ACTIVE -->'
active=f'''\n## 当前执行：整个穿衣人物先做静态\n\n用户最新要求优先：所有175个有效实拍帧监督同一静态高斯场；无逐帧身体/衣物变形。眼镜专项停止，不同步E盘。所有新实验从原始观测与全新初始化开始，旧模型只作比较。\n\n062普通静态模型完成20万点初始化、18000步，最终150692点；064PGSR静态表面对照同样从原始数据重建20万初始点（与062初始数组散列一致），18000步完成，最终{result['points']}点。175帧各更新{state['frame_updates_min']}—{state['frame_updates_max']}次。\n\n063相机核验仅显示插值相机的背景中位残差1.010→0.691px，人物1.345→1.313px；选择集参与筛选，不是独立测试。原相机未修改，064继续使用59标定+116插值相机。\n\n{review['summary_zh']} {review['limitations_zh']}\n\n064导出通过原生/标准、PLY重载及+Y/协方差/全部SH转换核验；本地浏览器实测状态另记viewer-validation.json。00未打开或部署。PGSR只做本地研究；该项目商业授权未确认。\n\n下载：{d['download_url']}\n本地：{d['download_path']}\n查看：{d['viewer_url']}\n对照页面：http://127.0.0.1:8870/body-smplx.html#pose-clothing-current\n\n{review['next_zh']}\n\n保留原始视频、图像、相机、许可模板、必要对照和当前候选。每个完成的高斯轮次都必须给兼容00查看器的独立标准PLY、路径与核验；质量和格式分别验收。完成有实质结论后更新既有方法PPT，不新增一堆重复报告。\n'''
txt=txt[:txt.index(start)+len(start)]+active+txt[txt.index(endp):];p.write_text(txt,encoding='utf8')
print(json.dumps(dict(download=d['download_url'],points=result['points'],quality_accepted=False),ensure_ascii=False))
