"""Record the completed visual review; does not train or alter model data."""
from datetime import datetime, timezone
import json
import audit_gaussianavatars_params123 as a
from publish_import_teacher123 import win, live_state

out=a.REC/'demo-vhap-initialization/body-gaussian-smplify-eight-dc-reviewedfit'
snap=out/'gaussian-run/step-018000'
stamp=datetime.now(timezone.utc).isoformat()
review=a.read(snap/'review/result.json')
orbit=a.read(snap/'orbit-comparison/result.json')
result=a.read(snap/'result.json')
assert review['frames']==59 and len(review['pages'])==30 and result['active_sh_degree']==0
for page in review['pages']+orbit['pages']:
    assert a.sha(win(page['path']))==page['sha256']
findings={
 '身体':'59帧原图/mesh/高斯及30张区域总览均已实际审阅。18～23帧的抬臂和袖子连接较连续，但衣服仍有斑驳、亮点、袖口和衣摆锯齿。不能据此认定全身几何合格。',
 '面部':'59帧均已审阅，眉眼、鼻嘴和脸颊仍有白亮点、颗粒与破碎边缘。面部保护区未接受衣物几何引导，但这不证明原拟合准确，也没有独立3D面部真值。',
 '持物手':'指缝、指形、局部空洞和亮斑仍不合格。身体姿态改变12帧中持物手RGB指标12/12改善，只能证明该指标改善。',
 '口袋手':'部分裁片只有衣摆和裤兜，遮挡的手指没有真实外观可供验收，不声明恢复正确。',
 '脚':'鞋整体可识别，侧面鞋底/鞋口仍有条纹、突刺、拉丝和暗块；42/59帧脚部固定ROI误差上升。保留全部帧，不筛除退化帧。',
 '未知视角':'同一36个连续合成视角已实际审阅，左右均为SH0。侧背面仍有亮斑、缺面和口袋侧手臂异常，没有整体修复；未拍背面没有真实图像真值。',
 '颜色':'高阶SH已关闭，仍有37.97%的高斯基础RGB至少一个分量超出0～1，最大3.547。该读数不是显示像素比例；与亮斑的因果关系仍需固定几何单变量验证。',
 'mesh':'新增顶点位移最大0.376毫米，面部梯度与位移为0，没有新增法线翻转；几何学习幅度仍小，不能称已到衣服表面。'
}
a.save(snap/'visual-review.json',dict(updated=stamp,actual_visual_review=True,step=18000,reviewed_pages=review['pages'],all_frames=59,native_groups=295,findings=findings,decision='bounded_candidate_completed_quality_not_accepted',quality_accepted=False,model_promoted=False,frame_filtering=False))
a.save(snap/'orbit-comparison/visual-review.json',dict(updated=stamp,actual_visual_review=True,reviewed_pages=orbit['pages'],camera_count=36,left='early-fit SH0 step18000',right='reviewed-fit SH0 step18000',findings=findings['未知视角'],ground_truth_available=False,quality_accepted=False,model_promoted=False))
live=live_state()
assert not live['foreign'] and live['gpu_pids']==[331] and live['baseline']['start_identity']=='9229'
a.save(out/'closeout.json',dict(updated=stamp,state='bounded_candidate_completed_reviewed',step=18000,points=result['points'],active_sh_degree=0,world_ply_sha256=result['world_ply_sha256'],visual_review_sha256=a.sha(snap/'visual-review.json'),orbit_visual_review_sha256=a.sha(snap/'orbit-comparison/visual-review.json'),colour_scale_audit_sha256=a.sha(snap/'colour-scale-audit.json'),quality_accepted=False,model_promoted=False,goal_complete=False,live_processes=live,scope='Fixed8 SH0, only reviewed SMPLify-X input handoff changed; no further candidate launched.',next='Prepare one fixed-geometry RGB-range render diagnostic and inspect mesh-vs-Gaussian gradient allocation before choosing a training change. Not yet executed.'))
print(json.dumps(dict(reviewed_native_groups=295,reviewed_orbit_cameras=36,points=result['points'],quality_accepted=False,live=live),ensure_ascii=False))
