"""Record actual image reviews and publish bounded diagnostic conclusions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import re
import numpy as np
from PIL import Image
import audit_gaussianavatars_params123 as a

ROOT=a.REC/'cross-subject-sh0-124'
OCC=ROOT/'indoor-visibility124'
DEPTH=ROOT/'quality-first-eight124/grass-multiview-joint-check'
stamp=datetime.now(timezone.utc).isoformat()


def local(p):
    p=str(p)
    if p.startswith('/mnt/'):p=p[5].upper()+':/'+p[7:]
    if p.startswith('/opt/'):return Path('//wsl.localhost/Ubuntu-20.04'+p)
    return a.lp(p)


def digest(p):return hashlib.sha256(local(p).read_bytes()).hexdigest()


def main():
    assert not a.lp(OCC/'visual-review.json').exists()
    occ=a.read(OCC/'result.json');depth=a.read(DEPTH/'result.json');plan=a.read(DEPTH/'plan.json')
    notes={
        'frame_000001':'The couch behind the torso is masked as unknown, while the foreground round tabletop at the lower image edge is not adequately covered. Forearm-person boundary also remains uncertain.',
        'frame_000028':'The rear couch is extensively excluded; some foreground table pixels are included, but the table boundary and nearby leg support are incomplete. Object membership is not depth ordering.',
        'frame_000056':'Rear-view lower legs are hidden by foreground furniture. The detected couch fragments do not cover the central occluding boundary reliably.',
        'frame_000083':'Rear couch and one foreground tabletop are both selected. The front upper tabletop and object/leg intersections are not consistently covered; the person mask has small front-torso holes.',
        'frame_000111':'Large rear-couch regions are excluded despite being behind the visible torso; cropped lower legs and uncertain foreground occlusion are not resolved.'}
    reviewed=[]
    for r in occ['records']:
        assert digest(r['evidence_image'])==r['evidence_sha256']
        assert digest(r['person_mask'])==r['person_mask_sha256']
        assert digest(r['source_image'])==r['source_sha256']
        assert digest(r['unknown_mask'])==r['unknown_sha256']
        person=np.asarray(Image.open(local(r['person_mask'])).convert('L'))
        unknown=np.asarray(Image.open(local(r['unknown_mask'])).convert('L'))
        assert set(np.unique(person))<={0,255} and set(np.unique(unknown))<={0,255}
        assert not np.any((person>0)&(unknown>0))
        reviewed.append(dict(frame=r['frame'],status='rejected_as_training_validity',
            actual_image_review=True,evidence=r['evidence_image'],evidence_sha256=r['evidence_sha256'],
            reason=notes[r['frame']],person_foreground_preserved=True))
    occ_review=dict(updated=stamp,state='completed_not_admitted',records=reviewed,
        conclusion='Furniture detection and segmentation alone cannot establish front/behind ordering. This fixed policy loses known-background supervision and misses some actual occluder boundaries.',
        policy_not_retried=True,original_inputs_unchanged=True,new_training_steps=0,
        loss_validity_module_integration=False,training_admitted=False,quality_accepted=False)
    a.save(OCC/'visual-review.json',occ_review)
    a.save(OCC/'conclusion.json',occ_review)
    a.save(OCC/'progress.json',dict(updated=stamp,state='completed_not_admitted',pid=None,done=5,total=5))

    assert len(depth['images'])==12 and len(plan['verification_frames'])==4
    assert not(set(plan['verification_frames'])&set(plan['fit_frames']))
    for r in depth['images']:assert a.sha(r['path'])==r['sha256']
    stats={}
    for name,r in depth['joints'].items():
        stats[name]=dict(selected_mean_px=float(np.mean([x['error_px'] for x in r['per_selected']])),
            verification_mean_px=float(np.mean([x['error_px'] for x in r['per_verification']])),
            verification_max_px=max(x['error_px'] for x in r['per_verification']),
            leave_one_out_max_over_shoulder_width=max(r['leave_one_out_displacement_over_shoulder_width']))
    review=dict(updated=stamp,state='completed_rejected_as_3d_supervision',
        actually_viewed_frames=[r['frame'] for r in depth['images']],actual_images_reviewed=12,
        evidence=depth['images'],statistics=stats,
        visual_finding='The recorded raised hand remains near the chest/hair in the front and both observed side directions. The conditional triangulated skeleton projects away from shoulders, hips and wrists, especially at the sides.',
        interpretation='The frozen conditional cameras, detected joints and a static-joint assumption are not jointly consistent. This does not isolate camera error from observation errors or body motion.',
        not_an_independent_3d_measurement=True,old_poses_used=False,images_warped=False,
        no_bad_points_removed=True,verification_frames_not_used_for_estimation=True,
        new_SMPLify_fits=0,new_Gaussian_steps=0,training_admitted=False,
        decision='Keep the original user-rejected26-29 poses rejected. Do not feed these triangulated points to SMPLify-X or increase temporal weights as a substitute for depth evidence.')
    a.save(DEPTH/'visual-review.json',review);a.save(DEPTH/'conclusion.json',review)

    summary=('胸前手臂仍未修复：现有SMPLify-X的二维投影、肢体先验和时间平滑不保证手腕深度。'
        '新增8帧固定相机检查中，右手腕在4个未参与计算的检查帧最大偏差144.84px；该三维关节结果已拒绝作为监督。'
        '室内5帧家具候选未知区也未通过实际审阅，未接入训练。')
    next_action=('Keep the about8 good-frame scope and rejected chest-depth poses excluded; do not require all frames or relax the pose filter. '
        'The fixed-camera triangulated targets and five furniture unknown masks are closed rejected diagnostics; do not repeat them or use them for training. '
        'The next actionable design is a bounded selected-view joint-constraint adapter to the pinned SMPLify-X loss and original optimizer, '
        'using actual native2D observations jointly instead of claiming these pseudo3D targets are truth. '
        'Before any fitting, register the one candidate, state which body pose is shared and which per-view placement is free, '
        'check the static-gesture assumption against the already reviewed source images, and verify single-view upstream parity. '
        'Do not rewrite a full sequence solver, hardcode wrists onto the chest, change K, claim background camera calibration, '
        'restart a closed fit, or train GS on rejected poses. Keep a finite eight-view candidate and native face/body/feet and side review; '
        'if shared-pose assumptions are unsupported, retain the evidence instead of forcing it. '
        'Current demo120k SH0 remains; goal active and15min target unverified.')
    note=dict(updated=stamp,summary=summary,grass_result=str(DEPTH/'conclusion.json'),
        indoor_result=str(OCC/'conclusion.json'),grass_images_reviewed=12,indoor_images_reviewed=5,
        new_fit_calls=0,new_gaussian_steps=0,all_frames_required=False,quality_accepted=False)
    for p in [a.R/'protocol.json',a.R/'queue-state.json',a.R/'evidence-ledger.json',a.C/'local-research-status.json']:
        d=a.read(p);d.update(updated=stamp,current_stage='sparse_selection124_depth_support_rejected',
            summary=summary,next_action=next_action,next=next_action,visibility_and_depth124=note,
            current_active_experiment=None,active_experiment=None,running=[],current_worker_pid_wsl=None,
            worker_pid=None,active_pid=None,worker_command=None,fitting_running=False,fitter_running=False,
            own_gpu_task_running=False,local_GPU_task_active=False,GPU_task_running=False,
            training_running=False,gaussian_training_running=False,training_started=False,
            training_allowed=False,quality_accepted=False,goal_complete=False,
            last_goal_turn=dict(classification='progress_with_rejected_depth_evidence',evidence=summary,goal_complete=False))
        if p in [a.R/'queue-state.json',a.C/'local-research-status.json']:d.update(state=d['current_stage'],updated_at=stamp)
        a.save(p,d)
    url='diagnostics/video10-global-initializer/index.html#quality-chest-depth124'
    block='''<div id="quality-chest-depth124"><h3>胸前手臂：深度问题仍未修复</h3>
<p><strong>手应该靠近胸前。这组向前伸手的SMPL-X姿态未通过，26—29帧继续排除；正面重合不能作为通过依据。</strong></p>
<p>原版人体先验限制不合理关节姿态，时间约束减少相邻帧跳变；它们都没有直接测量“手离胸口多远”。错误已出现在高斯训练之前的网格中。</p>
<p>本次用原先登记的8帧二维肩／肘／腕／髋和固定条件相机进行一次三角化，另外4帧仅作检查。未改相机，未删高误差点，未重新拟合或训练高斯。右手腕检查误差最大144.84px，肩与髋也偏离，故拒绝把这批三维点作为监督。相机、检测误差和人物微动仍不能唯一分离。</p>
<figure id="quality-chest-side75"><img loading="lazy" src="smplify-temporal-124/quality-chest-side75.png" alt="草地75帧原图与二维观测和条件三维投影"><figcaption>第75帧仅用于检查：左为原图，右为二维检测（绿）与条件三角化投影（紫）。真实侧视图中手仍在胸前；紫点并非真值，也不是修复后的结果。</figcaption></figure>
<p>仍按“约8个好帧，尽量覆盖视角”推进。筛选可以拒绝错误姿态，不能修好深度。下一项应在原SMPLify-X接口内验证选定视角的共同约束；先登记共享姿态假设与单视图等价检查，不增加全序列求解器，不把所有人物的手强制放胸前。</p>
<details><summary>室内遮挡输入：5帧检查未通过</summary><p>家具检测把后方沙发也划为未知区，部分前方遮挡边界却漏掉。检测家具不等于确定遮挡先后。此候选没有接入高斯损失，也没有修改原人物mask。</p>
<figure><img loading="lazy" src="smplify-temporal-124/quality-indoor-unknown28.png" alt="室内28帧原图人物mask与失败的候选未知区"><figcaption>原图／人物mask／候选未知区／橙色叠加。该候选未通过，不能用它证明腿部完整或正确。</figcaption></figure></details>
<p>新证据图片位于本机 D:/00/记录/005 和008。查看器保留原demo120k SH0模型；本轮没有新模型导出。</p></div>'''
    source=a.R/'video10-global-initializer/index.html';content=a.lp(source).read_text(encoding='utf-8')
    begin='<!-- QUALITY_CHEST_DEPTH124_BEGIN -->';end='<!-- QUALITY_CHEST_DEPTH124_END -->'
    assert begin not in content
    marker='<!-- QUALITY_CROSS_SUBJECT124_END -->';assert content.count(marker)==1
    content=content.replace(marker,marker+begin+block+end)
    grass_image=next(r['path'] for r in depth['images'] if r['frame']=='frame_000075')
    indoor_image=next(r['evidence_image'] for r in occ['records'] if r['frame']=='frame_000028')
    for base in [a.R/'video10-global-initializer',a.V/'diagnostics/video10-global-initializer',a.S/'diagnostics/video10-global-initializer']:
        a.lp(base/'index.html').write_text(content,encoding='utf-8')
        a.lp(base/'smplify-temporal-124/quality-chest-side75.png').write_bytes(local(grass_image).read_bytes())
        a.lp(base/'smplify-temporal-124/quality-indoor-unknown28.png').write_bytes(local(indoor_image).read_bytes())
    models=[]
    for base in [a.V,a.S]:
        d=a.read(base/'latest-result.json');model=d.get('modelId');models.append(model)
        d['latestDiagnostic']=dict(updated=stamp,round='124',title='胸前手臂：深度证据检查',summary=summary,
            status='depth_evidence_not_admitted',reportUrl=url,qualityAccepted=False,goalComplete=False)
        a.save(base/'latest-result.json',d);assert a.read(base/'latest-result.json')['modelId']==model
    p=a.R/'long-term-plan.md';text=a.lp(p).read_text(encoding='utf-8')
    text+='\n## 2026-11-03 Selected-view depth and occlusion input review\n\n'+summary+'\n\n'+next_action+'\n'
    a.lp(p).write_text(text,encoding='utf-8')
    a.save(DEPTH/'publication.json',dict(updated=stamp,reportUrl=url,summary=summary,
        html_sha256=a.sha(source),models=models,actual_UI_verified=False,model_unchanged=True,
        cloud_sync_verified=False,company00_verified=False,goal_complete=False))
    print(summary)


if __name__=='__main__':main()
