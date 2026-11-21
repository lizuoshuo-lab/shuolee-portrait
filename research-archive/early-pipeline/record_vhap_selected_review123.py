"""Save the actual all-frame visual review, outside the images-only records."""
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now

def main():
    root=a.REC/'demo-vhap-initialization/vhap-selected-reference'
    stats=a.read(root/'review/summary.json')
    review=dict(updated=now(),state='reviewed_failed',actual_visual_review=True,frames_reviewed=59,
        body_pages=[f'review/body-page-{i:02d}.jpg' for i in range(6)],face_page='review/faces-all59.jpg',
        detailed_checks=['surface-audit/frame_000001_raw_cuda_mask.png','surface-audit/frame_000033_raw_cuda_mask.png',
            'surface-audit/frame_000001_face_observed_projected.png','surface-audit/frame_000033_face_observed_projected.png',
            'surface-audit/frame_000059_face_observed_projected.png'],
        findings_zh=[
            '已逐张查看59帧整身和面部。优选参考帧附近的正面脸部较贴合，但两侧视角的头部、肩膀和手臂朝向明显不匹配。',
            '侧面轮廓大幅偏出真人；腿脚和抬臂仍有错位，不能仅归入用户同意后期处理的手指遮挡问题。',
            '躯干衣物轮廓未被学到；非零vertex offset主要集中在手部，不能用最大位移证明衣服贴合。',
            '全部59帧已参与联合拟合，共享体型和顶点偏移确有更新；完成迭代不代表质量通过。',
            '检测器残差较小的面部仍不保证可见表面和三维朝向正确；不以均值或少量好帧代替整段审阅。',
            '按同一体型未偏移基准，发现52个三角形法向反转，需进一步定位；这不是完整自交检测。'],
        mesh_pose_accepted=False,clothing_surface_accepted=False,face_surface_accepted=False,
        hand_repair_deferred=True,Gaussian_training_admitted=False,quality_accepted=False,
        statistics=stats)
    a.save(root/'visual-review.json',review)
    p=a.R/'long-term-plan.md'
    text=a.lp(p).read_text(encoding='utf-8-sig');heading='## 2026-10-30 demo优选帧联合拟合'
    if heading in text:text=text[:text.index(heading)]
    text+='\n\n'+heading+'\n'
    text+='当前用户路线：3fps预处理，选清晰近正面帧初始化，VHAP_smplx共享体型/偏移/纹理联合59帧，随后接绑定GaussianAvatars_SMPLX。手部遮挡后期处理；不回到首帧、不只修复一帧。\n'
    text+='本轮参考为采样第33帧（源256）。59帧7220次更新完成，耗时692.44秒；实际逐帧审阅未通过侧面头身姿态、衣物表面和腿脚。暂不将该结果作为合格高斯初始化。\n'
    text+='只读CUDA诊断确认未覆盖衣物在target背景中RGB误差为零，silhouette项被注释；躯干P90位移0.428毫米，而手部最大50.19毫米。下一步准备原库支持的white背景单变量对照，并独立核对姿态/表面可见性。仍保留原优化器、共享参数和条件K。任何存在负面质量影响的新训练先按用户既有要求说明具体取舍，不另写序列求解器。\n'
    text+='目标仍是视频输入到合格模型导出15分钟；本轮只有拟合完成，尚未完成该端到端目标。\n'
    a.lp(p).write_text(text,encoding='utf-8')
    print('all59 mesh review saved: failed; goal remains active')

if __name__=='__main__':main()
