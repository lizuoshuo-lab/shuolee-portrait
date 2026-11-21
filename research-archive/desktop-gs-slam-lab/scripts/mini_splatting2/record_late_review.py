"""Store the completed image review and add bounded findings to the report."""
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    output=ROOT/'native-late-final-review-001.json';assert not output.exists()
    primary=json.loads((ROOT/'native-late-review-001/summary.json').read_text())
    extra=json.loads((ROOT/'additional-frame-late-006/summary.json').read_text())
    fit=json.loads((ROOT/'native-training-fit-diagnostic-001/summary.json').read_text())
    manifests={k:json.loads((ROOT/n/'manifest.json').read_text()) for k,n in {
        'base':'native-resolution-cache-full-18000-s0','late':'native-late-full-18000-s0'}.items()}
    a=extra['models']['low']['aggregate']['1'];b=extra['models']['full']['aggregate']['1']
    review=dict(status='reviewed',created=time.time(),accepted=False,goal_complete=False,
        primary_summary='native-late-review-001/summary.json',
        primary_delta={key:value['delta'] for key,value in primary['compared_to_low'].items()},
        additional24_delta=dict(full_psnr=b['full_psnr']-a['full_psnr'],head_psnr=b['head']['psnr']-a['head']['psnr'],
            head_gradient_error=b['head']['normalized_gradient_error']-a['head']['normalized_gradient_error']),
        training51_delta=fit['late_minus_base'],training_fit_is_not_generalization=True,
        visual=dict(primary_all8_both_targets=True,additional_all24=True,inspected_native_full_crop='000024',
            inspected_path_stills=['distance-00','distance-15','distance-45','elevation-00','elevation-15','elevation-45'],
            findings=['Hair and clothing edges sometimes improve; eyes and eyeglasses remain doubled or blurred.',
                'The fixed farther-distance frame shows more dark detached streaks around the head in the late candidate.',
                'Large pitch excursions still deform faces and stretch surfaces. No photo or geometry GT exists for these paths.']),
        cost=dict(base_gaussians=primary['models']['low']['gaussians'],late_gaussians=primary['models']['full']['gaussians'],
            base_seconds=manifests['base']['elapsed_seconds'],late_seconds=manifests['late']['elapsed_seconds']),
        browser=dict(report='http://127.0.0.1:8820/native-late.html',primary_controls=True,extra_groups=['all','far','higher_pitch'],
            extra_full_and_head=True,no_console_errors=True,no_horizontal_overflow=True,
            video_metadata=[dict(kind=k,duration=6,width=1514,height=1380,ready=4) for k in ['distance','elevation']],
            final_findings_section='pending_reload_verification'),
        decision='Do not replace the default model. Limited heldout appearance benefit with observed out-of-coverage artifacts; not accepted as a geometry or human-clarity solution.',
        next='Validate training-only head RGB weighting separately on the original schedule; do not automatically combine with late densification.',
        limits=['Single seed and shared-video development views','Manual head/shoulder rectangles include background',
            'Primary native8bit and extra gsplat float metrics compared only within each protocol','No independent wide-pitch or metric-depth GT'])
    write(output,review)
    viewer=ROOT/'viewer';page=viewer/'native-late.html';old=page.read_text(encoding='utf-8')
    backup=ROOT/'native-late-observed-publication-001/before-final-findings.html';shutil.copy2(page,backup)
    assert 'id="findings"' not in old
    section='''<section id="findings"><h2>4. 本轮结论：局部细节改善，换视角伪影仍在</h2>
<p>原增密时机与延长窗口相比，8 个开发视角在 1514 × 2694 输出下的头肩平均 PSNR 从 26.81 提高到 27.10 dB，边缘误差下降；额外 24 帧头肩平均提高约 0.15 dB，但全图平均下降约 0.12 dB。逐图检查可见部分头发、衣物边缘更清楚，正面眼镜和眼部重影仍然明显。</p>
<p>下面是固定拉远路径的第 15 帧：右侧候选在人物头部附近出现更多黑色悬浮条纹。大幅俯仰时，两组仍有面部变形和表面拉伸。这些路径没有对应真值，因此此处只描述实际可见的伪影。</p>
<figure><figcaption>固定拉远路径：左为原增密时机，右为延长增密窗口。点击查看完整图像。</figcaption><a href="native-late-assets/paths/distance-15.png"><img src="native-late-assets/paths/distance-15.png" alt="固定拉远路径上的两组实际三维渲染，右侧头部周围可见悬浮条纹"></a></figure>
<p>51 张训练照片的头肩平均 PSNR 提高约 0.45 dB，说明已见照片的拟合有所改善；它不能单独证明泛化。模型从 316,044 增至 404,128 个高斯，训练从约 21.6 分钟增至 25.3 分钟。本轮未通过人物清晰度与几何改善验收，不替换默认模型；下一项局部损失试验将回到原增密时机，单独比较。</p>
<p class="muted">单种子结果。训练照片、8 个开发视角、额外 24 帧具有不同口径，数值不直接混合。完整模型、对照与失败记录均保留。</p></section>'''
    old=old.replace('<details><summary>实验口径与剩余验证</summary>',section+'<details><summary>实验口径与剩余验证</summary>',1)
    page.write_text(old,encoding='utf-8')
    review['report_sha256']=hashlib.sha256(page.read_bytes()).hexdigest();write(output,review)
    for folder in ['native-late-visual-review-001','native-late-extra-visual-review-001']:
        path=ROOT/folder/'manifest.json';data=json.loads(path.read_text());data.update(status='reviewed',review=output.name);write(path,data)
    for filename in ['native-late-queue.json','native-training-fit-queue.json']:
        path=ROOT/filename;data=json.loads(path.read_text());assert data['status']=='complete_pending_review'
        data.update(status='complete',review=output.name,accepted=False);write(path,data)
    print(json.dumps(dict(status='reviewed',accepted=False,report=str(page))))


if __name__=='__main__':main()
