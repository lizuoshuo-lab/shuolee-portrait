from pathlib import Path
import datetime,json

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-fullscene-camera-fresh-075'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

result=read(R/'gaussian-run/result.json')
metrics=read(R/'review/same-camera-result.json')
receipt=read(R/'gaussian-run/viewer-export/delivery.json')
assert result['steps']==18000 and result['points']==290458
assert receipt['compatibility_passed'] and not metrics['numeric_screening_passed']
a,b=metrics['summary']['072']['all'],metrics['summary']['075']['all']
rgb_gain=100*(1-b['inside_rgb_mae']/a['inside_rgb_mae'])
contour_change=100*(b['contour_mean_px']/a['contour_mean_px']-1)
files=[f'head-{i:03d}.jpg' for i in [0,33,69,99,135,174]]
files+=['body-099.jpg','novel-below25-bg0.25.jpg']
files += [f'continuous-{i:03d}.jpg' for i in [0,30,60,90,120,149]]
assert all((R/'review'/p).is_file() for p in files)
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
decision=dict(updated=now,actually_inspected=True,final_evaluation_inspected=True,
    inspected_files=files,observed_metrics_frames=175,observed_visual_frames=[1,34,70,100,136,175],
    continuous_review='Six fixed sampled frames from the continuous path; not a full frame-by-frame video review',
    unknown_view_GT=False,quality_accepted=False,geometry_accepted=False,
    root_cause_established=False,camera_accuracy_accepted=False,
    workflow_completed=True,from_scratch=True,old_gaussian_loaded=False,
    representation='shared static clothed-person Gaussian',
    viewer_status='075整图相机与人体初始化已完成18,000步；鼻口、眼镜及肩部仍有重影，质量未通过。',
    report='body-smplx.html#full-scene-person',
    observed_view_conclusion=f'人物内部RGB误差降低{rgb_gain:.1f}%，内部欠覆盖比例由{100*a["inner_undercovered"]:.2f}%降至{100*b["inner_undercovered"]:.2f}%。但第70帧眼镜横向散开、第100帧鼻口与眼镜重影仍清楚可见，肩部边界的漂浮区域也没有消除。不能把内部颜色拟合改善等同于几何对齐成功。',
    novel_view_conclusion='统一的下方25度相机和连续路径抽查中，脸部重复结构、肩部散点、手部和鞋底的破碎仍在。未看到足以通过验收的整体改善。这里没有新视角实拍真值，只作结构与外观诊断。',
    next_step='保留本轮完整流程作对照，先追踪异常鼻口和眼镜高斯由哪些原始帧、哪些邻域图像块提供约束，再对照纹理三角化深度与高斯深度，定位错误结构从哪里产生。相机误差仍是候选原因；本轮只说明整图相机替换尚不足以消除重影，不能据此排除外参问题。确认冲突来源后再定义下一轮从头训练。',
    numeric=dict(interior_rgb_reduction_percent=rgb_gain,contour_increase_percent=contour_change,
        before=a,after=b,gates=metrics['gates']),
    ppt_effect=f'人物内部颜色误差降低{rgb_gain:.1f}%，欠覆盖比例由{100*a["inner_undercovered"]:.2f}%降至{100*b["inner_undercovered"]:.2f}%。',
    ppt_summary='整图定位与人体初始化已贯通，鼻口和眼镜重影仍在。',
    ppt_limit=f'轮廓距离由{a["contour_mean_px"]:.3f}增至{b["contour_mean_px"]:.3f}像素，未知视角仍有破碎。\n下一步追踪异常高斯的观测来源与深度冲突。',
    pc00_opened_this_round=False,remote_00_deployed=False)
save(R/'review/visual-decision.json',decision)
save(R/'review/viewer-interaction-review.json',dict(updated=now,
    preset_opened=True,manual_open_file=True,gaussian_mode_seen=True,point_mode_seen=True,
    points_displayed=290458,default_upright_verified=True,template_loaded=False,
    remote_00_opened=False,shape_quality_accepted=False,
    note='Actual local shuolee viewer via file chooser, both Gaussian and point modes. No extra skeleton or local mesh deformation applied.'))
path=read(R/'review/continuous-path.json')
path.update(sampled_visual_frames_reviewed=[0,30,60,90,120,149],
    actual_continuous_visual_review=False,full_frame_by_frame_review=False,quality_accepted=False)
save(R/'review/continuous-path.json',path)
print(json.dumps(dict(rgb_gain_percent=rgb_gain,contour_increase_percent=contour_change,
    workflow_completed=True,quality_accepted=False),ensure_ascii=False))
