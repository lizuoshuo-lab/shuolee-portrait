"""Close the reviewed eight-view refinement without promoting regressed poses."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import audit_gaussianavatars_params123 as a
from select_quality_frames124 import POLICY
from quality_budget124 import select_budget
from publish_selected_refine124 import OUT, update


def win(path):
    return Path(str(path).replace('/mnt/d/', 'D:/').replace('/mnt/c/', 'C:/'))


def main():
    assert not a.lp(OUT/'conclusion.json').exists()
    stamp = datetime.now(timezone.utc).isoformat()
    root = OUT.parent
    result = a.read(OUT/'result.json')
    plan = a.read(OUT/'plan.json')
    review = a.read(OUT/'review/result.json')
    evidence = {r['frame']: r for r in review['records']}
    rows = {r['frame']: r for r in a.read(root/'grass-temporal-refinement/official-fit/data/training-inputs.json')['records']}
    scores = {r['frame']: r for r in a.read(root/'grass/selection.json')['records']}
    digest = a.sha(OUT/'selected-fit.npz')
    assert digest == result['fit_sha256'] and len(evidence) == 8
    for r in review['records']:
        for im in r['images'].values():
            assert a.sha(win(im['path'])) == im['sha256']
    with np.load(a.lp(OUT/'selected-fit.npz'), allow_pickle=False) as z:
        ids = z['frame_ids'].astype(str).tolist()
        obs = z['observed_native_keypoints'].copy()
        proj = z['joints_projected_native'].copy()
        rots = Rotation.from_rotvec(z['global_orient']).as_matrix()
    assert ids == plan['frames']
    yaw = np.unwrap(np.arctan2(rots[:,0,2], rots[:,2,2]))
    body = np.array([0,16,15,18,17,5,2,6,3,7,4,12,9,13,10,14,11])
    reasons = {
        'frame_000004': 'Chest/hair proximity remains plausible in the displayed view. Shoe/toe alignment remains offset and the unchanged body screen fails; no whole-body quality acceptance.',
        'frame_000012': 'Raised hand stays relatively near the chest; face and toe alignment remain approximate. The unchanged body screen fails.',
        'frame_000030': 'Rejected: front projection remains close but the same side view again shows the raised forearm extending away from the chest. Numerically small 2D residual cannot override this depth failure.',
        'frame_000046': 'Rejected: the shared-pose chest proximity regresses to a forward-reaching forearm, despite close front projection.',
        'frame_000049': 'Rejected: forward-reaching raised arm reappears and body lean changes. Face/shoe reprojection gains do not validate limb depth.',
        'frame_000058': 'Rejected: raised wrist moves away from the chest in the saved side comparison. Foot projections improve but do not compensate for arm-depth regression.',
        'frame_000070': 'Rejected: raised wrist extends away from chest again; feet remain offset. Both visual depth and numeric screening fail.',
        'frame_000085': 'Rejected: the formerly floating feet move nearer the observed shoes, but the other arm is also raised across the chest and head direction remains wrong. The input does not support this new two-raised-arm pose.'
    }
    records, decisions = [], {}
    for i, frame in enumerate(ids):
        err = np.linalg.norm(proj[i]-obs[i,:,:2], axis=1)
        h = rows[frame]['initial_box_xyxy'][3]-rows[frame]['initial_box_xyxy'][1]
        metrics, flags = {}, []
        for name, idx in [('body',body), ('face',np.arange(67,118)), ('feet',np.arange(19,25))]:
            w = obs[i,idx,2]**2
            assert w.sum() > 0
            metrics[name+'_weighted_rms_px'] = float(np.sqrt(np.sum(w*err[idx]**2)/w.sum()))
            maximum = float(err[idx][w>0].max())
            metrics[name+'_supported_max_px'] = maximum
            metrics[name+'_supported_max_over_height'] = maximum/h
            if name in ('body','feet') and maximum/h > POLICY[name+'_max_over_height']:
                flags.append(name+'_reprojection_over_existing3percent_height')
        records.append(dict(frame=frame,time_seconds=rows[frame]['time_seconds'],source_fit_sha256=digest,
            yaw_proxy_deg=float(np.degrees(yaw[i])),face_sharpness=scores[frame]['face_sharpness'],
            body_confidence=float(obs[i,body,2].mean()),screen_flags=flags,person_box_height_px=h,**metrics))
        decisions[frame] = dict(status='uncertain' if frame in ('frame_000004','frame_000012') else 'rejected',
            reason=reasons[frame],depth_reviewed=True,evidence=evidence[frame]['images'],
            evidence_sha256=evidence[frame]['images']['sides']['sha256'],
            review_scope='Both eight-frame body/side overview pages and all eight native face/feet crops were visually inspected. Native body crops were retained; overview body panels are scaled and synthetic sides are not measured 3D truth.')
    selection = select_budget(records, decisions)
    selection.update(updated=stamp,source_fit_sha256=digest,policy=POLICY,thresholds_unchanged=True,
        no_framewise_fit_mixing=True,actual_visual_review=True)
    assert not selection['selected_frames']
    comparison = evidence['frame_000030']['metrics']
    conclusion = dict(updated=stamp,state='completed_rejected_depth_regression',
        frames_reviewed=ids,decisions=decisions,selected_frames=[],seconds=result['seconds'],
        finding='Free per-view SMPLify-X stage3/4 refinement undoes the multi-view chest-depth improvement. The soft shared-pose penalty does not preserve that depth in these observations.',
        native_face_feet_reviewed=ids,body_and_side_overview_reviewed=ids,
        complete_unscaled_body_review_claimed=False,
        illustrative_frame30_metrics=comparison,
        cause_scope='This controlled comparison shows a failure of this per-view refinement setting. It does not prove unique camera error, detector error, or a measured true hand-chest distance.',
        screen_audit='Original SMPLify-X excludes neck/hips1/9/12 from fitting; broad validation still records hips. This round did not remove those validation points or relax the3percent threshold.',
        budget_selection='Quality and explicit depth review first; angular diversity is a preference, then eligible budget fill. Rejected or uncertain frames never fill slots.',
        baseline_preserved=str(root/'grass-selected-view-pose124'),
        no_best_parts_stitching=True,no_chest_contact_target=True,
        quality_accepted=False,gaussian_admitted=False,new_gaussian_steps=0,goal_complete=False)
    a.save(OUT/'visual-review.json',conclusion)
    a.save(OUT/'selection-reviewed.json',selection)
    a.save(OUT/'conclusion.json',conclusion)
    a.save(OUT/'progress.json',dict(updated=stamp,state=conclusion['state'],pid=None,done=8,total=8))

    # Preserve native source-crop pixels and the same side-camera pair.
    r = evidence['frame_000030']
    raw = Image.open(win(r['images']['body']['path'])).convert('RGB')
    sides = Image.open(win(r['images']['sides']['path'])).convert('RGB').crop((0,480,680,960))
    panel = Image.new('RGB',(raw.width+sides.width,max(raw.height,sides.height)))
    panel.paste(raw,(0,0));panel.paste(sides,(raw.width,(panel.height-sides.height)//2))
    picture = Path('D:/00/记录/005/分析_草地八视角精修124_frame_000030_原图_联合初值_逐帧回退_相同侧面.png')
    assert not a.lp(picture).exists();panel.save(a.lp(picture))

    summary = ('8帧逐帧精修223.76秒，部分脚部投影改善，但30／46／49／58／70的胸前手臂再次向前伸，85出现双手抬胸。'
        '本候选质量拒绝，筛选0帧，未训练高斯；保留此前111.26秒联合拟合的胸前改善结果。')
    next_action = ('The one eight-frame per-view refinement is closed and rejected; do not rerun or sweep anchor weights. '
        'Retain the reviewed shared-pose archive and about8 quality-first selection scope. '
        'Before further fitting, prepare the minimum upstream parameter interface that retains multi-view arm/torso information while allowing necessary local corrections; '
        'test zero-step replay and protected-joint invariance before a separately registered bounded candidate. '
        'Never stitch prior arms with new legs or mark side-depth review passed from a frontal projection. '
        'Do not replace the demo120kSH0 viewer or launch GS from this rejected candidate. Continue other subjects independently under the same evidence rules.')
    update('selected_refine124_rejected_depth_regression',summary,next_action)
    source = a.R/'video10-global-initializer/index.html'
    text = a.lp(source).read_text(encoding='utf-8')
    begin='<!-- QUALITY_CHEST_REFINE124_BEGIN -->';end='<!-- QUALITY_CHEST_REFINE124_END -->'
    assert begin not in text and text.count('<!-- QUALITY_CHEST_SHARED124_END -->')==1
    block='''<section id="quality-chest-refine124"><h3>胸前手臂复核：逐帧精修回退，已拒绝</h3>
<p><strong>正面叠得上，不等于手在胸前。</strong>这次从同一份8视角联合结果出发，沿用SMPLify-X末两阶段、原迭代上限和优化器，放开逐帧身体姿态，并保留共享姿态软约束。8帧用时223.76秒。</p>
<figure><img src="smplify-temporal-124/quality-chest-refine30.png" alt="第30帧联合姿态与逐帧精修的正面和同一侧面比较"><figcaption>第30帧：左侧依次为原图、联合初值、逐帧精修；右侧为相同侧视角下的联合初值／逐帧精修。正面变化不大，侧面却重新向前伸手。原图裁片未拉伸，合成侧视图不等于实测3D真值。</figcaption></figure>
<p>第30、46、49、58、70帧均出现抬手离胸的回退。第85帧脚部投影改善，但另一只手也被错误抬到胸前。因此本候选已结束并拒绝，没有用它训练高斯，也没有替换现有查看器模型。</p>
<p>原因范围已缩小：单帧2D项对手腕前后深度约束不足；连续性或软姿态约束可以平滑错误，也可能被2D拟合重新拉偏。本轮结果支持保留不同视角共同确定的手臂信息，不能据此断言相机或关键点是唯一原因。下一步先检查最小的受保护关节接口，不扫描权重，不拼接不同实验的最好部位。</p>
<p>选帧按用户要求改为约8帧：先排除数值或姿态不合格者，再优先覆盖不同角度，最后从其余合格帧补足预算。分箱不再无故限制数量；不要求全序列都通过，也不会用坏帧凑8帧。新增选择器8项检查通过。本轮仍保留原3%数值阈值和髋点误差，不以原版损失忽略髋点为由自动通过。</p>
<details><summary>全部8帧对照与审核范围</summary><p>两页包含全部候选原图／联合初值／精修及相同侧面；已查看这两页，以及8帧原生面部、脚部裁片。完整原生身体裁片亦已保存，本轮不宣称逐一完成其原尺寸审核。此处可见的肢体错误已足以拒绝。</p><img loading="lazy" src="smplify-temporal-124/quality-chest-refine-page1.png" alt="精修前4帧对照"><img loading="lazy" src="smplify-temporal-124/quality-chest-refine-page2.png" alt="精修后4帧对照"></details>
<p>图片在 D:/00/记录/005；未验证WPS云端同步或公司00部署。训练计时目标和高质量模型验收均未完成。</p></section>'''
    text=text.replace('<!-- QUALITY_CHEST_SHARED124_END -->','<!-- QUALITY_CHEST_SHARED124_END -->'+begin+block+end)
    artifacts={'quality-chest-refine30.png':picture,'quality-chest-refine-page1.png':OUT/'review/page-01.png',
        'quality-chest-refine-page2.png':OUT/'review/page-02.png'}
    for base in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        a.lp(base/'index.html').write_text(text,encoding='utf-8')
        for name,path in artifacts.items():a.lp(base/'smplify-temporal-124'/name).write_bytes(a.lp(path).read_bytes())
    url='diagnostics/video10-global-initializer/index.html#quality-chest-refine124'
    models=[]
    for base in [a.V,a.S]:
        data=a.read(base/'latest-result.json');model=data['modelId'];models.append(model)
        data['latestDiagnostic']=dict(updated=stamp,round='124',title='胸前手臂：精修回退已拒绝',summary=summary,
            status='rejected_pose_depth_regression',reportUrl=url,qualityAccepted=False,goalComplete=False)
        a.save(base/'latest-result.json',data);assert a.read(base/'latest-result.json')['modelId']==model
    path=a.R/'long-term-plan.md'
    text=a.lp(path).read_text(encoding='utf-8')
    text+='\n## 2026-11-03 Selected-view per-frame refinement rejected\n\n'+summary+'\n\n'+next_action+'\n'
    a.lp(path).write_text(text,encoding='utf-8')
    a.save(OUT/'publication.json',dict(updated=stamp,summary=summary,reportUrl=url,
        html_sha256=a.sha(source),picture=str(picture),picture_sha256=a.sha(picture),
        models_preserved=models,actual_UI_verified=False,company00_verified=False,
        WPS_cloud_sync_verified=False,goal_complete=False))
    print(dict(state=conclusion['state'],selected=selection['selected_frames'],
        metrics=[{k:v for k,v in r.items() if k in ('frame','body_supported_max_over_height','feet_supported_max_over_height','screen_flags')} for r in records]))


if __name__=='__main__':main()
