"""Publish reviewed selected-view chest-depth progress without model promotion."""
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
import audit_gaussianavatars_params123 as a
from publish_grass_selected_views124 import OUT,state

def win(p):return Path(str(p).replace('/mnt/d/','D:/').replace('/mnt/c/','C:/'))

def main():
    assert not a.lp(OUT/'publication.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    c=a.read(OUT/'conclusion.json');s=a.read(OUT/'selection-reviewed.json')
    review=a.read(OUT/'review/result.json');r=next(r for r in review['records'] if r['frame']=='frame_000030')
    assert c['selected_frames']==['frame_000030','frame_000046']
    raw=Image.open(win(r['images']['body']['path'])).convert('RGB')
    sides=Image.open(win(r['images']['sides']['path'])).convert('RGB').crop((0,480,680,960))
    # Preserve native crop pixels; add padding around the saved same-camera
    # side comparison. No stretched anatomy and no labels on the picture.
    panel=Image.new('RGB',(raw.width+sides.width,max(raw.height,sides.height)))
    panel.paste(raw,(0,0));panel.paste(sides,(raw.width,(panel.height-sides.height)//2))
    picture=Path('D:/00/记录/005/分析_草地八视角联合124_frame_000030_原图_旧新叠加_旧新侧面.png')
    assert not a.lp(picture).exists();panel.save(a.lp(picture))
    next_action=('The one eight-view shared-local-pose candidate is completed; do not rerun it. '
        'First reconcile the existing numeric screen with the actual original SMPLify-X observation weights: hip anchors9/12 are excluded by the fit but currently enter the hard maximum screen. '
        'Keep both metrics and physical review visible; do not simply suppress errors or change the3percent threshold to fill slots. '
        'Then prepare the smallest reproducible per-view refinement initialized from this jointly supported chest pose, using the existing upstream seed hook and optimizer, '
        'rather than choosing neighbour seeds manually or introducing a new sequence solver. Register its fixed candidate scope and compare native overlays and side depth before execution. '
        'The shared body assumption must not force the floating85feet or all94frames to pass. Target remains about8 good frames distributed over recorded views; no GS with the current incomplete set. '
        'Keep demo120kSH0 viewer unchanged; source data, fixed K, old fits and failed diagnostics remain preserved.')
    summary=('8视角联合拟合111.26秒，草地30／46等帧的抬手已由向前伸出改善到胸前附近；'
        '第85帧脚悬空仍拒绝。按现有数值与分箱规则仅选出30／46两帧，未准入高斯；'
        '部分数值标记来自原版损失忽略的髋点，验收口径待核对。')
    state('selected_views124_chest_improved_selection_incomplete',summary,next_action)
    a.save(OUT/'next-interface-plan.json',dict(updated=stamp,state='design_only_not_started',
        method='Retain the eight-view common-pose result and reuse upstream per-view seed/refinement interfaces only after audit.',
        prerequisites=['Audit fitting-versus-screening observation contract without changing thresholds based on outcome.',
            'Verify frozen shared result and original seed-hook hashes; register exact parameter freedom and bounded runtime.',
            'Keep wrist depth review and all supported face/body/feet observations; do not claim precise3D ground truth.'],
        selected_target_about8=True,all94_frames_required=False,quality_accepted=False,
        new_fit_calls=0,new_gaussian_steps=0))
    begin='<!-- QUALITY_CHEST_SHARED124_BEGIN -->';end='<!-- QUALITY_CHEST_SHARED124_END -->'
    block='''<section id="quality-chest-shared124"><h3>8视角联合拟合：手臂回到胸前附近</h3>
<p><strong>新结果有改善，但还不是可发布模型。</strong>8帧共同拟合局部身体姿态，沿用原SMPLify-X五阶段和优化器，用时111.26秒。体型和K固定；整体朝向、位置、脸手参数仍逐帧拟合。没有把手强制贴胸，也没有使用已失败的三角化关节作3D真值。</p>
<figure id="quality-chest-shared30"><img src="smplify-temporal-124/quality-chest-shared30.png" alt="草地30帧原图旧新网格叠加和相同侧面比较"><figcaption>第30帧：左侧依次为原图、旧连续约束、新8视角共同拟合；右侧依次为旧／新网格的相同侧视角。新手臂明显靠近胸前。原图裁片保持像素比例，右侧为合成观察方向，不能当独立3D真值。</figcaption></figure>
<p>这是对单视角深度歧义的实质改善。原拟合在正面能把手投到胸前，却允许它在3D里向前伸；时间平滑只约束相邻姿态变化，无法独自测出手腕离胸口的距离。共同使用不同方向的观测后，第30、46、49等帧的前伸明显减少。实际变化还包括共享局部姿态和整体朝向初始化，不能全部归因于某一个权重或相机校准。</p>
<p>全部8帧的原图、面部、脚部和四方向网格已审阅。第85帧脚明显悬空，继续拒绝。当前旧数值筛选和均匀分箱只留下第30、46帧，尚不足约8帧训练集；部分标记来自原版拟合主动忽略的髋点，接下来核对筛选口径，不按本轮结果放宽阈值。</p>
<p>下一步保留这份统一姿态作为可复现初值，在原接口内准备小范围逐帧精修，保住胸前深度并允许细微动作变化。仍然只要求选出的好帧合格，不要求94帧全部通过。当前没有新高斯训练或模型替换。</p>
<details><summary>查看完整8帧证据与范围</summary><p>原图／旧拟合／新拟合；侧面左旧右新。页面总览做等比例缩小，原生裁片保存在本机 D:/00/记录/005。</p>
<img loading="lazy" src="smplify-temporal-124/quality-chest-shared-page1.png" alt="8视角共同拟合前4帧全部对比">
<img loading="lazy" src="smplify-temporal-124/quality-chest-shared-page2.png" alt="8视角共同拟合后4帧全部对比">
<p>单视角损失／梯度等价、多视角平均和梯度流、完整K投影回读已检查。它们证明接口执行，不证明物理标定、背面重建、面部逐点亚像素或15分钟全流程。未核验公司00部署及WPS云端同步。</p></details></section>'''
    source=a.R/'video10-global-initializer/index.html';content=a.lp(source).read_text(encoding='utf-8')
    assert begin not in content and content.count('<!-- QUALITY_CHEST_DEPTH124_END -->')==1
    content=content.replace('<!-- QUALITY_CHEST_DEPTH124_END -->','<!-- QUALITY_CHEST_DEPTH124_END -->'+begin+block+end)
    artifacts={ 'quality-chest-shared30.png':picture,
        'quality-chest-shared-page1.png':OUT/'review/page-01.png',
        'quality-chest-shared-page2.png':OUT/'review/page-02.png'}
    for base in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        a.lp(base/'index.html').write_text(content,encoding='utf-8')
        for name,path in artifacts.items():a.lp(base/'smplify-temporal-124'/name).write_bytes(a.lp(path).read_bytes())
    url='diagnostics/video10-global-initializer/index.html#quality-chest-shared124';models=[]
    for base in [a.V,a.S]:
        d=a.read(base/'latest-result.json');model=d['modelId'];models.append(model)
        d['latestDiagnostic']=dict(updated=stamp,round='124',title='8视角联合：胸前手臂改善',summary=summary,
            status='partial_pose_improvement_not_admitted',reportUrl=url,qualityAccepted=False,goalComplete=False)
        a.save(base/'latest-result.json',d);assert a.read(base/'latest-result.json')['modelId']==model
    p=a.R/'long-term-plan.md';text=a.lp(p).read_text(encoding='utf-8')
    text+='\n## 2026-11-03 Selected-view shared pose: chest depth improved\n\n'+summary+'\n\n'+next_action+'\n'
    a.lp(p).write_text(text,encoding='utf-8')
    a.save(OUT/'publication.json',dict(updated=stamp,summary=summary,reportUrl=url,
        html_sha256=a.sha(source),picture=str(picture),picture_sha256=a.sha(picture),
        models_preserved=models,actual_UI_verified=False,company00_verified=False,
        WPS_cloud_sync_verified=False,goal_complete=False))
    print(summary)

if __name__=='__main__':main()
