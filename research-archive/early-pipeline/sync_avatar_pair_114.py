from pathlib import Path
import json,html,shutil,re,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer';P=B/'current/avatar-repository-pair-114';D=V/'body-avatar114'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for arm in ['gaussianavatar','smplx']:
    r=P/arm;d=read(r/'fit/visual-review.json')
    d.update(state='coarse_prior_reviewed',usable_as_coarse_static_initialization=True,geometry_accepted=False,review='Six same-camera overlays inspected: global position and limb ordering usable; face, hands and shoes retain residual mismatch. Bare template is only initialization, not clothing truth. Two independent fits are numerically identical; shared static pose has no frame-dependent correction.')
    save(r/'fit/visual-review.json',d);(D/arm).mkdir(parents=True,exist_ok=True);shutil.copy2(r/'fit/overlays.jpg',D/arm/'initial-mesh.jpg')
    if not (D/arm/'progress.json').exists():save(D/arm/'progress.json',dict(arm=arm,state='preflight',step=0,target_steps=18000,quality_accepted=False))
previous=B/'current/person-static-native-align-fresh-113/decision.json';shutil.copy2(previous,V/'body-native113/decision.json')
page=V/'body-smplx.html';s=page.read_text(encoding='utf-8')
old='按用户要求准备两个从头训练的受控分支：固定相机的训练期几何对齐，以及受静止背景和相机先验约束的训练期相机/几何联合校正。两者共用新初始化、175图像、18k日程和固定评测相机；先核验相机梯度与输入映射。'
new='用户已确认两条人像仓库路线：GaussianAvatar（全身）与 GaussianAvatars_SMPLX。两组分别从头拟合静态人体并训练，使用相同175图像、20万初始点、18,000步和固定评测相机。'
assert old in s;s=s.replace(old,new)
section='''<section id="avatar114-pair"><h2>两条人像方法：GaussianAvatar 与 GaussianAvatars_SMPLX</h2>
<p class="notice">两组分别从头训练。175 张真实注册图像，同一套固定相机，20 万初始点，18,000 步；静态穿衣人物。重影是否改善尚待实际对照。</p>
<p>GaussianAvatar 使用官方第一阶段的 UV 特征网络、表面偏移与蒙皮变换；GaussianAvatars_SMPLX 使用公开网格绑定高斯及明示的本地 SMPL-X 适配。后者公开仓库缺少全身入口，不能称作者完整复现。两组共享前景监督与评测条件，保留各方法的参数化差异。</p>
<p>两次独立静态拟合已完成，六个相机叠图已检查。粗人体位置可作初始化，但脸、手、鞋仍有残差；最终高斯需学习衣物表面，拟合完成不代表几何正确。</p>
<h3>GaussianAvatar（全身）</h3><p id="avatar114-gaussianavatar">预检中，尚未开始高斯优化。</p>
<figure><a href="body-avatar114/gaussianavatar/initial-mesh.jpg"><img src="body-avatar114/gaussianavatar/initial-mesh.jpg" alt="GaussianAvatar本轮新拟合的六相机叠图"></a><figcaption>原图与本轮新拟合静态人体；裸模板只作为初始化。</figcaption></figure>
<div id="avatar114-gaussianavatar-images"></div><h3>GaussianAvatars_SMPLX</h3><p id="avatar114-smplx">预检中，尚未开始高斯优化。</p><div id="avatar114-smplx-images"></div>
<p>完成后用同相机、同裁剪检查鼻嘴眼镜、肩袖衣摆、腿脚及连续新视角；两条路线各自提供世界坐标标准 PLY 与适配 00 查看器的独立导入版本。最新结果与质量验收分开标记。</p>
<script>(()=>{async function update(){for(const arm of ['gaussianavatar','smplx']){try{const p=await(await fetch('body-avatar114/'+arm+'/progress.json?t='+Date.now(),{cache:'no-store'})).json();let step=p.step??p.steps??0;document.getElementById('avatar114-'+arm).textContent=(arm==='gaussianavatar'?'GaussianAvatar':'GaussianAvatars_SMPLX')+'：'+step+' / 18000 步；'+(p.state==='training'?'训练中':p.state==='preflight'?'预检中':'待同相机验收')+'。质量未验收。';if(step>=1000){let node=document.getElementById('avatar114-'+arm+'-images');if(!node.children.length)node.innerHTML='<figure><img data-role="body" alt="本轮同相机全身对照"><figcaption>输入、渲染与绝对误差。</figcaption></figure><figure><img data-role="head" alt="本轮同相机面部对照"><figcaption>固定面部裁剪，输入、渲染与绝对误差。</figcaption></figure>';for(const role of ['body','head'])node.querySelector('[data-role="'+role+'"]').src='body-avatar114/'+arm+'/'+(role==='body'?'review':'heads')+'-current.jpg?step='+step;}}catch(e){}}}update();setInterval(update,30000);})();</script></section>'''
assert '<section id="avatar114-pair">' not in s
s=s.replace('<section id="native113-result">',section+'\n<section id="native113-result">',1)
page.write_text(s,encoding='utf-8');print('Updated current pair review section')
