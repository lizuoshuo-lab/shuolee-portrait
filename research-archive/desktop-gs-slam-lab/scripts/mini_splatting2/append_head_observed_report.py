"""Add real24-frame observations to the same image-first resolution report."""
import hashlib
import json
from pathlib import Path
import shutil
import time
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    stage=ROOT/'native-head-observed-publication-001';stage.mkdir(exist_ok=False)
    extra=json.loads((ROOT/'additional-frame-head-007/summary.json').read_text())
    groups=json.loads((ROOT/'native-head-observed-groups-001.json').read_text());assert extra['status']==groups['status']=='complete'
    viewer=ROOT/'viewer';target=viewer/'native-head.html';old=target.read_text(encoding='utf-8')
    assert 'id="extraGroup"' not in old;shutil.copy2(target,stage/'before-native-head.html')
    assets=viewer/'native-head-assets/extra';assets.mkdir(exist_ok=False)
    data=dict(models=extra['models'],groups=groups,frames=[r['frame'] for r in extra['models']['low']['rows'] if r['divisor']==1])
    assert len(data['frames'])==24
    for arm in ['low','full']:
        dest=assets/arm;dest.mkdir()
        source=ROOT/'additional-frame-head-007'/arm
        for name in data['frames']:
            stem=Path(name).stem
            for suffix in ['-s1.png','-head.png']:shutil.copy2(source/(stem+suffix),dest/(stem+suffix))
    (assets/'gt').mkdir()
    for row in extra['models']['low']['rows']:
        if row['divisor']!=1:continue
        name=row['frame'];other=next(r for r in extra['models']['full']['rows'] if r['frame']==name and r['divisor']==1)
        assert row['head_box']==other['head_box']
        source=ROOT/'additional-frame-head-007/gt'/name;shutil.copy2(source,assets/'gt'/name)
        Image.open(source).crop(tuple(row['head_box'])).save(assets/'gt'/(Path(name).stem+'-head.png'))
    (assets/'summary.json').write_text(json.dumps(data,indent=2))
    section='''<section id="extra"><h2>3. 同一视频的另外 24 帧</h2><p>两组都在相同的 757 × 1347 输出尺寸渲染；相机位置由固定地图定位，模型不使用这些照片训练。远近与俯仰分组沿用此前固定界限。</p><div class="controls"><label>已拍视角分组 <select id="extraGroup" aria-label="额外照片分组"><option value="all">全部 24 帧</option><option value="radius_sfm_lower">近距离组</option><option value="radius_sfm_upper">远距离组</option><option value="optical_pitch_deg_lower">俯仰角较低组</option><option value="optical_pitch_deg_upper">俯仰角较高组</option></select></label><label>视频帧 <select id="extraFrame" aria-label="额外视频帧"></select></label><label>范围 <select id="extraRegion" aria-label="额外照片范围"><option value="head">头肩局部</option><option value="full">完整照片</option></select></label></div><div class="photos"><figure><figcaption>真实照片</figcaption><a id="extraGtLink"><img id="extraGt" alt="额外照片真实参考"></a></figure><figure><figcaption>原始 RGB L1</figcaption><a id="extraLowLink"><img id="extraLow" alt="额外视角原始损失模型渲染"></a></figure><figure><figcaption>头部权重 ×2</figcaption><a id="extraFullLink"><img id="extraFull" alt="额外视角头部加权模型渲染"></a></figure></div><p id="extraMetric"></p><p id="extraGroupMetric" class="muted"></p><p class="muted">这是已用于开发的同源视频，不是独立场景或大幅俯仰真值。头肩框由原有矩形随视频时刻插值，不是精确分割；距离单位为任意 SfM 尺度，角度参考平均相机向上方向。这里使用 gsplat 渲染和编码前浮点指标，主实验使用原生内核和 8 位图像，两节分数分别比较。</p></section>'''
    old=old.replace('<details><summary>实验口径与剩余验证</summary>',section+'<details><summary>实验口径与剩余验证</summary>',1)
    script='''<script>const EXTRA=__EXTRA__;(()=>{const e=id=>document.getElementById(id),base='native-head-assets/extra/';function rows(arm,name){return EXTRA.models[arm].rows.find(r=>r.frame===name&&r.divisor===1)}function picture(id,path){e(id).src=path;e(id+'Link').href=path}function populate(){const group=e('extraGroup').value,previous=e('extraFrame').value,allowed=group==='all'?EXTRA.frames:EXTRA.groups.groups.groups[group].frames;e('extraFrame').replaceChildren();for(const name of EXTRA.frames)if(allowed.includes(name))e('extraFrame').add(new Option(String(rows('low',name).source_index).padStart(6,'0'),name));if([...e('extraFrame').options].some(x=>x.value===previous))e('extraFrame').value=previous}function update(){const name=e('extraFrame').value,stem=name.replace(/\\.[^.]+$/,''),head=e('extraRegion').value==='head',g=e('extraGroup').value;picture('extraGt',base+'gt/'+(head?stem+'-head.png':name));picture('extraLow',base+'low/'+stem+(head?'-head.png':'-s1.png'));picture('extraFull',base+'full/'+stem+(head?'-head.png':'-s1.png'));const a=rows('low',name),b=rows('full',name),m=head?'head':'full';e('extraMetric').textContent='本帧'+(head?'头肩':'全图')+' PSNR：'+a[m].psnr.toFixed(2)+' → '+b[m].psnr.toFixed(2)+' dB。'+(head?'梯度保真误差：'+a.head.normalized_gradient_error.toFixed(3)+' → '+b.head.normalized_gradient_error.toFixed(3)+'。':'');if(g==='all'){const lo=EXTRA.models.low.aggregate['1'],hi=EXTRA.models.full.aggregate['1'];e('extraGroupMetric').textContent='全部24帧平均头肩 PSNR：'+lo.head.psnr.toFixed(2)+' → '+hi.head.psnr.toFixed(2)+' dB；全图：'+lo.full_psnr.toFixed(2)+' → '+hi.full_psnr.toFixed(2)+' dB。'}else{const spec=EXTRA.groups.groups.groups[g],lo=EXTRA.groups.models.low[g],hi=EXTRA.groups.models.full[g];e('extraGroupMetric').textContent='本组'+spec.count+'帧，范围 '+spec.actual_range.map(x=>x.toFixed(2)).join('–')+(spec.axis==='radius_sfm'?' SfM单位':'°')+'；头肩 PSNR：'+lo.head_psnr.toFixed(2)+' → '+hi.head_psnr.toFixed(2)+' dB。'}}e('extraGroup').addEventListener('change',()=>{populate();update()});for(const id of ['extraFrame','extraRegion'])e(id).addEventListener('change',update);populate();update()})();</script>'''
    old=old.replace('</body>',script.replace('__EXTRA__',json.dumps(data).replace('</','<\\/'))+'</body>',1)
    target.write_text(old,encoding='utf-8')
    result=dict(status='published_pending_browser_review',created=time.time(),frames=24,report=str(target),
        report_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),original_page_backup='before-native-head.html',
        same_video_development_only=True)
    (stage/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
