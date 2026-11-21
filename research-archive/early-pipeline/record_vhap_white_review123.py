"""Record actual white-control review and the verified VHAP input boundary."""
import numpy as np
import audit_gaussianavatars_params123 as a
from run_official_gaussians123 import now


def main():
    root=a.REC/'demo-vhap-initialization';run=root/'vhap-selected-reference-white'
    baseline=a.read(root/'vhap-selected-reference/surface-audit/result.json')
    current=a.read(run/'surface-audit/result.json');stats=a.read(run/'review/summary.json')
    contract=a.read(root/'vhap-input-contract.json')
    metrics={}
    for label,data in [('target',baseline),('white',current)]:
        rows=data['records']
        metrics[label]={k:float(np.median([r[k] for r in rows])) for k in
            ['mask_IoU','target_uncovered_fraction','mesh_outside_target_fraction']}
        metrics[label]['worst_mask_IoU']=min(r['mask_IoU'] for r in rows)
        metrics[label]['worst_mesh_outside_target_fraction']=max(r['mesh_outside_target_fraction'] for r in rows)
        metrics[label]['trunk_offset_mm']=data['offset_regions_mm']['trunk']
    summary='demo白背景对照已完成：轮廓覆盖改善，侧面朝向与区域外泄漏仍不合格；先核对VHAP单目输入条件。'
    findings=[
        '已实际查看59帧整身和面部。正面附近更贴合，但两端侧面头部、躯干、抬臂和腿脚仍明显偏离，不能视作可用的高斯初始化。',
        '仅改target→white，mask交并比中位数0.666→0.747，人物漏覆盖中位数24.35%→14.64%；改善支持原背景设置削弱轮廓监督。',
        '网格可见投影落在人物mask外的比例中位数15.11%→14.57%，最差仍39.24%；第59帧还从16.09%恶化至21.34%。区域外泄漏未解决，不以平均改善替代逐帧保护。',
        '躯干90%顶点偏移仍不足0.343毫米；最大51.94毫米来自手部。覆盖改善不能直接解释为衣服三维表面已经学到。',
        'canonical三角法向反转52→36，仍有局部风险；此统计不是完整自交检测。手部遮挡按用户指示后期处理，但侧面头身不合格不能归入手部问题。',
        '本轮7220更新、59帧联合拟合耗时700.42秒；未开展新的高斯训练，15分钟视频到合格模型目标未达成。',
        '输入审计确认：只有2D检测点，没有视频恢复的3D特征点；各帧单位外参仅定义相机相对坐标，未提供场景外参或同步多视角约束。',
        '原版VHAP单目入口处理FLAME头部；当前全身分支示例是已标定多相机SMPL-X。缺3D点不是接口必需项，输入场景与分支假设不一致是必须先处理的缺口。',
        '原库已有相邻帧平滑，但没有显式按真实帧间隔加权。连续动作修正已登记后续，不混入本轮背景对照；平滑不能修复系统性的错误朝向。']
    review=dict(updated=now(),state='reviewed_failed',summary_zh=summary,findings_zh=findings,
        actual_visual_review=True,frames_reviewed=59,
        body_pages=[f'review/body-page-{i:02d}.jpg' for i in range(6)],face_page='review/faces-all59.jpg',
        direct_comparison_reviewed_stage='global-epoch-20',direct_comparison_slots=[0,32,58],
        final_original_mesh_all59_reviewed=True,comparison=metrics,statistics=stats,
        source_input_audit=str(root/'vhap-input-contract.json'),
        mesh_pose_accepted=False,clothing_surface_accepted=False,face_surface_accepted=False,
        outside_mask_accepted=False,hand_repair_deferred=True,Gaussian_training_admitted=False,
        quality_accepted=False,control='Only supported background_train target→white; no fit/camera/loss changes during execution.')
    a.save(run/'visual-review.json',review)
    p=a.R/'long-term-plan.md';text=a.lp(p).read_text(encoding='utf-8-sig')
    heading='## 2026-10-30 demo白背景对照与VHAP输入核验'
    if heading not in text:
        text+='\n\n'+heading+'\n'
        text+='用户授权只改white背景后从优选第33帧重新拟合59帧；7220更新已完成、实际全59帧审阅未通过。轮廓IoU中位数0.666→0.747，但区域外可见网格比例中位仍14.57%、最差39.24%，侧面头身朝向错误，躯干P90偏移0.343毫米；不自动进入高斯训练。\n'
        text+='当前优先事项改为用户要求的VHAP输入合同：已核对运行提交、入口、数据加载器与原版官方单目文档。当前是单目59时间步×1视角，不是同步多目；只有条件K和相机相对单位外参，没有从图像恢复的三维点或场景外参。原版不强制3D点文件，原全身分支依赖已标定多相机提供空间约束。当前适配不能称完整多目复现或网络直接预测3D初始化。\n'
        text+='下一步先对齐单目数据模式、姿态初始化和已有模型参数导入接口，明确可复用原库能力及最小差异；不新增全序列求解器，不原样重跑白背景试验，不把本次SMPL-X预测点当作独立3D观测。后续按用户建议核验真实时间戳下连续动作，不能将不同时间的人体点直接当静态点三角化。区域外不能靠显示裁剪掩盖。\n'
        a.lp(p).write_text(text,encoding='utf-8')
    print(summary,metrics)


if __name__=='__main__':main()
