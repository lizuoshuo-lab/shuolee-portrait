from pathlib import Path
import json,hashlib,datetime
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
P=B/'current/static-ncc-paired-fresh-093'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    r=read(P/'review/paired-result.json');q=read(P/'review/common-query-comparison.json')
    a,b=r['summary']['control'],r['summary']['no_ncc']
    reduction=100*(1-b['ghost_alpha_mean']/a['ghost_alpha_mean'])
    lower=q['surface_transfer']['face_lower'];upper=q['surface_transfer']['face_upper']
    lm,um=lower['metrics']['rgb_center'],upper['metrics']['rgb_center']
    boundary=read(P/'no_ncc/review/same-camera-result.json')
    assert r['cameras']==175 and r['identical_initial_positions'] and r['identical_view_and_neighbor_schedule']
    assert lower['common_pairs']==717 and upper['common_pairs']==588
    # These exact outputs have been visually inspected in the current turn.
    paths=[P/'review'/f'{kind}-{f:06d}.jpg' for kind in ['head','body'] for f in [1,34,70,100,136,175]]
    paths += [P/'no_ncc/review'/f'novel-{n}-bg0.25.jpg' for n in ['front','left45','right45','left90','right90','above25','below25']]
    paths += [P/'no_ncc/review'/f'continuous-{i:03d}.jpg' for i in [0,30,60,90,120,149]]
    paths += [P/f'{arm}-standalone-{mode}.png' for arm in ['control','no_ncc'] for mode in ['gaussian','points']]
    assert all(p.exists() for p in paths)
    d=dict(created=datetime.datetime.now().astimezone().isoformat(),round='093',training_completed=True,
        actual_visual_review_complete=True,quality_accepted=False,goal_complete=False,
        adopt_ncc_off_as_mainline=False,numeric_screening_passed=r['numeric_screening_passed'],
        summary=f'093：关闭NCC使固定鼻口重影区域的平均不透明度降低{reduction:.1f}%，但跨帧几何对应变差，鼻口和俯仰视角仍错位，质量未通过。',
        early_result='两组在第1,800步已经出现鼻口重复和眼镜外散，此时NCC及多视图几何损失尚未启用。固定重影区域的平均不透明度分别为0.4787与0.4804。第6,999步仍接近，分别为0.3981与0.4019。NCC可以影响后期结果，但不是重影开始形成的必要条件。',
        paired_numeric_result=f'在训练前已冻结的第100帧2,420个像素上，平均不透明度从{a["ghost_alpha_mean"]:.4f}降到{b["ghost_alpha_mean"]:.4f}，减少{reduction:.1f}%。这个数只表示该区域变淡，不等于三维精度提高{reduction:.1f}%。',
        paired_visual_result='已逐一查看6个固定实拍视角。关闭NCC后第100帧鼻口前方的半透明副本减弱，第70帧肩部外溢减少，但第100帧仍有重复鼻口，正面和侧面的眼镜外散也仍在。人体主要特征保留，不能据此认定对齐完成。',
        protection_result=f'175帧前景颜色平均绝对误差为{a["foreground_rgb_mae"]:.5f}与{b["foreground_rgb_mae"]:.5f}，头部高度带误差为{a["head_band_rgb_mae"]:.5f}与{b["head_band_rgb_mae"]:.5f}，轮廓IoU为{a["silhouette_IoU"]:.4f}与{b["silhouette_IoU"]:.4f}，均通过预设的图像保护条件。但对完全相同的717个下半脸跨帧对应，表面深度转移的中位误差从{lm["control_median"]:.2f}增至{lm["no_ncc_median"]:.2f}像素；588个上半脸对应从{um["control_median"]:.2f}增至{um["no_ncc_median"]:.2f}像素。两组查询覆盖相同，因此几何退步不能解释为只保留了更难的样本。',
        novel_result='已查看7个固定检查视角和连续路径的6个抽样位置。部分侧面衣服边界更整齐，但正面仍有眼镜和脸部散射，上方25度仍有脸部重叠，下方25度的衣服和颈部出现明显半透明拖影。连续路径90、120两个抽样位置也保留该问题。没有这些视角的实拍真值，不能报告新视角重建精度，也没有宣称逐帧完整观看了150帧视频。',
        interpretation='结论是局部不透明度减少与三维一致性改善并不等价。本轮不把关闭NCC作为已通过的优化方案。训练器、标准PLY及查看器坐标的数值核验通过，说明导出没有新增局部变形，但仍不能证明真实相机标定完全准确。当前证据尚未区分相机或观测残差、初始化表面误差以及高斯自由位置和协方差对矛盾图像约束的吸收，不能将错位只归因于某一个因素。',
        limitations='本轮是一个配对随机种子，两组均独立重新拟合人体和生成20万初始点。初始位置及18,000步相机和邻居采样顺序相同，但GPU归约和自适应增密有非确定性：NCC启用前，点数最多相差2,407，固定区域平均不透明度最多相差0.0108。最终差异远大于已记录的早期差异，仍不能替代重复实验。颜色和轮廓来自训练观测，不是独立三维真值；头部高度带是按人体高度定义的区域，不是精确解剖掩码。对应误差采用一致像素中心，单位为完整去畸变图像像素，包含特征和相机不确定性。140个深度查询显式报告共同可见覆盖，薄的深度分布不自动等于准确表面。',
        model_status={
            'control':'093 保留NCC对照：175帧、20万点从头训练18,000步，299,824个最终高斯。鼻口重影仍在，质量未通过。',
            'no_ncc':f'093 关闭NCC诊断对照：175帧、20万点从头训练18,000步，257,117个最终高斯。固定重影区域不透明度降低{reduction:.1f}%，跨帧几何对应变差，未采用为通过方案。'},
        next_bounded_work={
            'scope':'保持175帧整图相机与静态人体路线，进行一次0至1,800步的从头来源诊断。记录重复结构对应高斯的初始位置、增密父点、每步实际位置及协方差和不透明度变化，并关联当步原始图像。结合固定跨帧特征的重投影与深度范围，区分观测冲突和自由高斯吸收误差的过程。',
            'gate':'先验证新增记录不改变训练采样和优化，再定位重影首次形成的更新及贡献点。分析完成后只选一个有证据的修改，重新从头训练。暂不扩展NCC权重搜索、不再增加训练步数，也不恢复动态或眼镜专项分支。相机真实性与几何约束继续分别检验。'},
        slide_conclusion='关闭NCC减弱局部重影，但跨帧几何对应变差，鼻口与俯仰视角问题仍未解决。',
        presentation_notes=[],
        actual_review_images=[dict(path=str(p),sha256=sha(p)) for p in paths],
        continuous_video_rendered=True,continuous_video_complete_visual_playback=False,
        local_standalone_ply_verified=True,pc00_opened_this_round=False,
        measured_facts=dict(ghost_relative_reduction_percent=reduction,all175=r['summary'],
            numeric_gates=r['gates'],silhouette_guard_gates=boundary['gates'],
            shared_lower_correspondences=717,shared_upper_correspondences=588,
            lower_rgb_center=lm,upper_rgb_center=um))
    d['presentation_notes']=[d[k] for k in ['early_result','paired_numeric_result','paired_visual_result','protection_result','novel_result','interpretation','limitations']]
    d['presentation_notes'] += ['方法：两组独立从原始观测重新拟合人体、三角化衣物特征和生成20万表面初始点。175帧相机固定，均训练18,000步。NCC权重是唯一配置差异，0.15与0。两组使用一致像素中心，前景颜色、轮廓、法向和几何约束保持相同。',
        'NCC通过当前高斯深度与法向构造局部平面诱导的跨视角图像块匹配。本次关闭该项仍计算它以保留采样行为。局部亮度相关性不等价于唯一三维表面，这个对照也未证明NCC是重影起因。',
        '下一项：'+d['next_bounded_work']['scope'],
        '证据：current/static-ncc-paired-fresh-093/protocol.json、review/paired-result.json、review/common-query-comparison.json及decision.json。原生模型和完整标准PLY均已核验，本地通过查看器打开文件检查高斯与点云两种模式。']
    (P/'decision.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=d['summary'],adopt_ncc_off_as_mainline=False,quality_accepted=False),ensure_ascii=False))

if __name__=='__main__':main()
