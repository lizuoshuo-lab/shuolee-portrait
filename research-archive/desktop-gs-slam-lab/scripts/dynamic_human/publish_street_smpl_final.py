"""Record inspected final street results without claiming clear reconstruction."""
from pathlib import Path
import datetime
import json

LAB = Path(__file__).resolve().parents[2]
BODY = LAB / 'outputs/dynamic-human-street-smpl-joint-011'
TIME = LAB / 'outputs/dynamic-human-street-temporal-012'
ALIGN = LAB / 'outputs/dynamic-human-training-alignment-013'


def write_json(path, value):
    assert not path.exists(), f'Preserve existing record: {path}'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


def main():
    pair = json.loads((BODY / 'paired-final-audit/independent-audit.json').read_text())
    temporal = json.loads((TIME / 'independent-audit-v2/independent-audit.json').read_text())
    assert pair['views_per_branch'] == 105 and len(pair['figures']) == 13
    assert len(temporal['images']) == 160 and len(temporal['transitions']) == 152
    write_json(BODY / 'paired-final-audit/visual-review.json', dict(
        status='all13_main_person_pairs_visually_reviewed_clear_reconstruction_not_accepted',
        viewed_files=[r['file'] for r in pair['figures']],
        additional_full_view=dict(frame=130, camera=6, selection='lowest articulated full-image PSNR',
            observation='Ego hood absent; pavement, vegetation and people also inaccurate. Whole-image error is not isolated human error.'),
        observations=[
            'Early/front views60-100 show recognizable people but clothing and extremities remain soft; representation benefit varies by view.',
            'Near views110/120/130 still lose original sleeve folds, hands, bag silhouette and separated legs; articulated geometry has broken protrusions.',
            'Side view130 worsens from20.24 to19.16dB despite mean main-person improvement.',
            'Camera6 frame140 gains strongly in box PSNR but includes large background region and still lacks sharp human detail.',
            'Back view150 remains soft in jacket, hands and ankles for both representations.'],
        scores=pair['branch_scores'], quality_accepted=False,
        decision='Proceed to training-only pose alignment and controlled geometric-offset refinement, not additional capacity by default.',
        boundaries=pair['limitations']))
    write_json(ALIGN / 'geometry-decomposition-15000/visual-review.json', dict(
        status='two_training_views_reviewed_pose_error_persists_without_free_offsets',
        viewed_files=['frame129-camera4-decomposition.png','frame151-camera6-decomposition.png'],
        other_computed_views_not_visually_reviewed=[131,149], optimized_steps=0,
        observations=['Fixed-template posed body still puts head/legs away from observed gray-suited person.',
            'Connecting learned Gaussian centers produces extra jagged boundaries; this is not an extracted clothing mesh.',
            'Free offsets and voxel corrections do not eliminate the underlying pose misalignment.'],
        decision='Use a staged pose-fitting diagnostic before granting more shape freedom.',
        quality_accepted=False))
    write_json(TIME / 'independent-audit-v2/visual-review.json', dict(
        status='four_window_contact_sheets_reviewed_temporal_quality_not_accepted',
        contact_sheets=[r for r in temporal['contact_sheets'] if r['view']=='fixed-union'],
        viewing_scope='Four sampled frames per window inspected; all video frames decoded computationally, not a claim of exhaustive frame-by-frame visual review.',
        observations=['Far pose-onset window cannot establish near clothing clarity or certify handoff quality.',
            'Approach and side windows retain broken or diffuse person silhouettes even at sampled training times.',
            'Side fixed-union mean PSNR and temporal-difference residual both worsen for articulated model; other windows vary.',
            'Rear window preserves moving-person presence but jacket/limbs remain soft; background and ego hood affect region metrics.'],
        quality_accepted=False, boundaries=temporal['boundaries'],
        presentation_revision='v2 shortens headings to prevent overlap in small far-person crops; v1 retained locally.'))

    knowledge_path = LAB / 'knowledge/32_dynamic_human_scene_reconstruction.md'
    text = knowledge_path.read_text(encoding='utf-8')
    text = replace_once(text, '室内人物运动基准和街景无人体模板基线均已完成训练及留出图像评测；街景人体关节模型仍在验证中。',
        '室内人物运动基准、街景可变形模型与人体关节模型均已完成训练及留出图像评测，但街景近处衣物和肢体仍未达到清晰重建要求。')
    text = replace_once(text, '背景、刚体与可变形人物联合模型已完成 30,000 次更新和全部 105 张留出图评测，近处人物仍未达到清晰重建要求。人体模板已就绪，完整人体关节分支与姿态对齐继续单独验证。',
        '可变形模型与人体关节模型均已完成 30,000 次更新和全部 105 张留出图评测，近处人物仍未达到清晰重建要求；后续重点是姿态对齐和衣物表面。')
    text = replace_once(text, '完整 156 帧、7 路相机的 SMPL 联合模型已通过四次被丢弃的前后向检查及留出渲染，正式 30,000 次对照已启动。',
        '完整 156 帧、7 路相机的 SMPL 联合模型已完成正式 30,000 次更新；两支最终 210 张完整图和 870 个人物框均通过独立误差复算。')
    text = replace_once(text,
        '![真实街景人物的原始观测与三万次更新后的重建，衣物与肢体细节仍未恢复](../outputs/dynamic-human-street-joint-009/final-image-audit/frame130-camera4-person79-observation-reconstruction.png)',
        '![真实街景的原始观测、可变形模型和人体关节模型，同为三万次更新](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame130-camera4-person79-comparison.png)')
    text = replace_once(text,
        '这组街景对照说明需要区分“原图没有记录细节”和“模型没有拟合出已有细节”。左图在当前训练分辨率下仍能看见衣物褶皱、手提袋边缘和分开的双腿；右图的躯干已经出现，但这些细节仍模糊。右图是无人体关节模板的可变形模型完成 30,000 次更新后的结果，尚非完整 OmniRe 人体模型。继续优化降低了图像误差，却没有自动带来清晰的衣物表面；关节对齐、衣物表面与可见性还需分别验证，也不能预先断言加入人体模板就一定解决问题。',
        '这组街景对照说明需要区分“原图没有记录细节”和“模型没有拟合出已有细节”。左图在当前训练分辨率下仍能看见衣物褶皱、手提袋边缘和分开的双腿；中、右两种表示均完成 30,000 次更新，人体关节模型仍有肢体错位和破碎外观。主目标全部 13 个留出区域的平均 PSNR 从 19.07 提高到 19.70 dB，但本图的近处侧面反而下降，衣物清晰度也未通过。人体先验为运动提供了结构，姿态、表面和可见性仍需与观测一致，不能用平均误差改善替代这三项检查。')
    theory_anchor = '该式解释中心位置的变化，高斯方向也随人体变形更新，不能只移动中心而保持整个椭球朝向不变。'
    text = replace_once(text, theory_anchor, theory_anchor + r'''

人体表面的自由度也需要受到约束。参考姿态下的高斯中心可写成

$$
\overline{\boldsymbol\mu}_p=\mathbf v_p(\boldsymbol\beta)+\Delta\boldsymbol\mu_p.
$$

$\mathbf v_p(\boldsymbol\beta)$ 是体形参数 $\boldsymbol\beta$ 决定的参考人体表面点，$\Delta\boldsymbol\mu_p$ 是高斯相对该表面的位移，单位为米。关节姿态和这个位移都能改变投影位置，因此相同的图像残差可能通过不同参数得到补偿。若姿态仍错位，允许高斯自由移动并不保证形成正确衣物表面。检查时应分别比较只随关节运动的参考表面和完整高斯表示；随后再验证姿态校正、表面自由度与外观细节的作用，不能仅看损失是否下降。
''')
    final_appendix = '''
街景人体对照的主目标全部 13 组最终裁图均已查看，最差全图也单独检查；两支均未通过清晰度验收。[最终配对核验](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/independent-audit.json)、[视觉结论](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/visual-review.json)。四段时间窗口另保存两支共 160 张渲染、152 组相邻变化及 8 段对照视频。视频按原拍摄约 10 Hz 播放，明确区分训练与留出时刻；矩形区域含背景和遮挡者，相邻差分残差不是纯闪烁或光流误差。[侧面连续对照](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-fixed-union.mp4)、[时间窗口独立核验](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/independent-audit.json)。

训练帧的进一步几何分解显示：移除高斯位置偏移及学习到的蒙皮修正后，保留已优化关节的参考人体仍有头部和腿部偏移；完整高斯中心则出现额外锯齿状边界。该投影仅是诊断，不是衣物网格，也没有场景遮挡合成。下一步应优先检验姿态拟合，再评估表面位移的约束。[训练帧几何分解](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/frame129-camera4-decomposition.png)、[检查范围](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/visual-review.json)。

'''
    text = replace_once(text, '## 参考资料', final_appendix + '## 参考资料')
    knowledge_path.write_text(text, encoding='utf-8')

    evidence_path = LAB / 'experiments/37_dynamic_human_reconstruction.md'
    text = evidence_path.read_text(encoding='utf-8')
    text = replace_once(text, '仍采用 30,000 次正式更新，完成后使用原先固定的全部 105 张留出图和 435 个人物框评测。',
        '两支现均完成 30,000 次正式更新，并使用原先固定的全部 105 张留出图和 435 个人物框完成评测。')
    text = replace_once(text, '初始网格仍有偏移，因此必须等待真实渲染对照，不能由梯度非零推断人物清晰度改善。',
        '初始网格存在偏移，最终渲染对照也未通过清晰度验收，不能由梯度非零推断人物清晰度改善。')
    text += '''

## 三万次人体表示对照：平均改善与近处退步并存

两种表示均已完成 30,000 次更新，各自 105 张完整留出渲染、435 个人物区域全部独立核验。主目标的全部 13 组原始观测／可变形／人体关节裁图逐一查看。目标 PNG 与原始 JPEG 经规定缩放的像素完全一致，区域边界相同，保存图片的 PSNR 均处于浮点评测量化误差界内。此比较同时改变表示、拓扑及正则，不是单因素消融。

![最终近处侧面原始观测与两种人体表示](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame130-camera4-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 帧 130、相机 4 · 同预算、同区域、未锐化 · 原始数据 CC BY-NC-SA 4.0。

主目标区域 PSNR 均值从 19.07 提高到 19.70 dB，SSIM 从 0.566 提高到 0.580；全图平均 PSNR 则为 22.41 与 22.39 dB。帧 130／相机 4 的人物区域从 20.24 下降到 19.16 dB，原图清楚的衣褶、手提袋边界和分开的双腿仍未恢复。帧 140／相机 6 的区域分数提高较多，但含大量背景，人物本身仍模糊。不能把总体均值改善解释为各视角均清晰。

![最终侧后方人物对照](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/frame150-camera6-person79-comparison.png)
> 图源：Argoverse 2 原始观测与本站实测 · 帧 150、相机 6 · 同区域显示，未进行图像修复。

人体分支的全图 PSNR 最低视图也是帧 130／相机 6。核对发现自车引擎盖未重建，道路、植被与行人也存在误差，所以最低全图分数并非独立的人体质量。当前结果是两支训练和数值评测完成，清晰人物重建未通过。

证据：[最终配对核验与全部指标](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/independent-audit.json)、[十三组实际看图范围](../outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/visual-review.json)。

## 连续运动、遮挡与人体表示切换

四个窗口在渲染前固定，共 80 个不同的帧／相机组合。两支均从最终模型恢复，每个保存张量精确一致；与已保存留出图重叠的视图重新渲染，每通道差异不超过一个 8 位灰度级。共 160 张新渲染和 152 组相邻差分完成独立复算。视频保持约 10 Hz 的原拍摄节奏，包含训练时刻和留出时刻，不能称为 80 个全新视图。

四段人物固定区域对照：[姿态起始与表示切换](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/pose-onset-fixed-union.mp4)、[人物接近](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/approach-fixed-union.mp4)、[侧面遮挡](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-fixed-union.mp4)、[后方遮挡](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/rear-occlusion-fixed-union.mp4)。完整画面对应为：[起始](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/pose-onset-full-image.mp4)、[接近](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/approach-full-image.mp4)、[侧面](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/side-occlusion-full-image.mp4)、[后方](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/rear-occlusion-full-image.mp4)。

每个窗口抽看四张对照，共 16 组；视频全部帧通过解码检查，但尚不等于每一帧均逐帧视觉验收。近处侧面在训练帧中也可见破碎轮廓和不清晰肢体。该窗口固定区域平均 PSNR 从 24.19 降到 22.94 dB，相邻图像变化残差从 0.04394 升到 0.04840；其他窗口表现不一。固定区域会包含背景与其他人物，差分也含自车运动、人物运动及光照变化，不能直接称作闪烁指标。远处第 56 帧的表示切换只能检查该距离下的连续性，不能证明近处衣物清晰。

证据：[预先固定的窗口](../outputs/dynamic-human-street-temporal-012/protocol.json)、[全部图像与差分独立核验](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/independent-audit.json)、[视觉检查边界](../outputs/dynamic-human-street-temporal-012/independent-audit-v2/visual-review.json)。MP4 是有损观看副本，所有数值来自原 PNG，未锐化或生成式增强。

## 区分姿态误差与高斯表面偏移

![同一训练帧的原始观测、仅保留优化姿态的参考人体、实际训练高斯中心](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/frame129-camera4-decomposition.png)
> 图源：Argoverse 2 原图与实际 15,000 次模型的 CPU 投影 · SMPL：Loper 等（2015），MPI · 几何诊断，没有场景遮挡合成。

左列是原始训练照片，中列保留实际优化的根平移、根旋转和关节旋转，恢复固定参考顶点和固定蒙皮权重，右列使用实际训练高斯中心及学习到的蒙皮。帧 129／相机 4、帧 151／相机 6 的中列仍有头部、腿部偏移，右列出现额外的锯齿状边缘。右列用原拓扑连接高斯中心仅作诊断，不能称作衣物表面网格。该分解同时去掉位置偏移和体素蒙皮修正，不能据此单独量化某一项的因果作用。

因此下一步优先进行训练图像上的姿态拟合，再检查高斯偏移约束和衣物细节。已发现二维关节预测混入旁人肢体，不能直接当作真值加入监督，也不继续以增加背景高斯或延长总训练替代对齐。

证据：[几何分解计算](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/receipt.json)、[实际看图范围](../outputs/dynamic-human-training-alignment-013/geometry-decomposition-15000/visual-review.json)。
'''
    evidence_path.write_text(text, encoding='utf-8')
    manifest_path = LAB / 'manifests/dynamic-human-work.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['status'] = 'both_street_representations_30000_complete_heldout_and_temporal_audited_person_clarity_not_accepted'
    manifest['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest['completed'] += [
        'Both30000update street models: all210heldout full images and870person boxes independently verified; all13main-person pairs visually reviewed; mean improvement does not resolve near clothing/limb failure',
        'Both final models restored exactly,80temporal renders each,152consecutive difference checks,8videos decoded; sampled visual inspection rejects clarity acceptance',
        'Actual15000update training-view geometry decomposition separates fixed-template posed geometry from learned Gaussian centers; pose error persists without free offsets']
    manifest['pending'] = ['training-only pose alignment with controlled appearance and geometric freedom, followed by heldout verification',
        'restore near-person clothing and limb clarity; both completed street models still fail',
        'complete temporal/handoff visual acceptance after alignment repair; current videos expose unresolved issues']
    manifest['current_runs'][0]['status'] = 'completed30000updates_and105heldout_views_independently_audited_quality_not_accepted'
    manifest['street_current_quality_issue'] = 'Articulated mean target-box PSNR19.70 versus19.07dB, but side view130 worsens and both fail clothing/limb clarity; training-view pose misalignment persists'
    manifest['smpl_download']['next_action'] = 'Training complete; proceed to training-only pose alignment and constrained Gaussian offsets'
    manifest['street_articulated_final_audit'] = 'outputs/dynamic-human-street-smpl-joint-011/paired-final-audit/independent-audit.json'
    manifest['street_temporal_audit'] = 'outputs/dynamic-human-street-temporal-012/independent-audit-v2/independent-audit.json'
    manifest['street_temporal_visual_review'] = 'outputs/dynamic-human-street-temporal-012/independent-audit-v2/visual-review.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Recorded final visual decisions and updated knowledge, evidence and manifest.')


if __name__ == '__main__':
    main()
