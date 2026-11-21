import json,datetime
from pathlib import Path
base=Path(r'\\?\C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
photo=base/'current/novel-view-face/head-structure/flame-photometric-registration'
pilot=photo/'whole-body-gaussianavatar/all-views-background/surface-gaussian-pilot'
stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def read(p):return json.loads(p.read_text('utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def review(folder,state,conclusion,images,next_step,extra=None):
    result=read(folder/'result.json');result['state']=state;result['reviewed_at']=stamp;result['quality_accepted']=False
    save(folder/'result.json',result)
    d=dict(updated=stamp,state=state,quality_accepted=False,visual_conclusion=conclusion,visual_files_reviewed=images,
        next=next_step,parent_unchanged=True,viewer_replaced=False,deployed_to_00=False,eyewear_optimization=False)
    if extra:d.update(extra)
    save(folder/'root-review.json',d)
review(pilot/'parameter-isolation','reviewed_no_single_family_fix',
    '把强几何候选的切向尺度、局部旋转或位置单独换回图像对照值，均未明显修复拉丝，全部59张训练观测的误差也变大。参数已经相互适应，不能据此把问题归因于某一个参数族。',
    ['pitch-15-group0.png','pitch-15-group1.png','000058-group0.png','000058-group1.png'],
    '单参数替换诊断结束，不修改主模型。',dict(optimizer_updates=0,render_forwards=325))
review(pilot/'staged-controls','reviewed_not_adopted',
    '从1180步图像对照继续885步，并分别加入较弱法向或深度畸变。分离与减弱约束降低了实拍回退，但连续俯仰和侧身衣物仍无明显改善；两个几何分支不采用，停止这一轮权重调整。',
    ['pitch-15.png','pitch-45.png','000000-body-comparison.png','000058-body-comparison.png'],
    '转向原衣物观测支持与动态绑定检查，保留必要图像对照和精简失败证据。',
    dict(steps_per_arm=885,actual_optimizer_updates=2655,independent_test=False))
review(pilot/'clothing-support','reviewed_coverage_and_appearance_both_present',
    '身体区域存在覆盖不足，也有高不透明度下的较大RGB误差。不能将此前头部内部覆盖充分的结论扩展到整个身体；裸身网格偏移或达到边界比例不是衣物几何真值。',
    ['000058-support.png'],'继续使用剔除轮廓带的同相机诊断。',dict(optimizer_updates=0))
review(pilot/'clothing-interior-reference','reviewed_old_clothing_reference_useful_not_gt',
    '剔除15像素轮廓带后，当前人体身体内部低于0.5的透明度占比降为0.745%，高RGB误差中仍有27.8%低覆盖、59.0%高覆盖。旧场景衣服轮廓和内部覆盖更好，但其头部重影与人物分离残留地面清楚可见，不能整体替换动态人体。',
    ['000058-reference.png','pitch-45.png'],
    '仅筛选真实观测支持的衣物高斯，保持头像与原动态人体；比较衣物静态和独立时间形变。',dict(optimizer_updates=0,render_forwards=242))
initial=photo/'whole-body-gaussianavatar/all-views-background/observed-clothing-pilot/initial'
review(initial,'reviewed_clothing_support_improves_motion_transfer_fails',
    '叠加旧场景衣物后覆盖增加，直接沿用已拟合的人体形变则使衣服侧面拉偏。静态参考中还出现手部重叠。后续版本加入SMPL-X手部影响排除，并为衣物单独学习已有形式的时间形变。',
    ['000058-clothing.png','000000-clothing.png'],
    '两组衣物参数相同初始化的885步短程对照；原人体和背景不动。',
    dict(selection_version=1,optimizer_updates=0,render_forwards=362))
r=read(pilot/'root-review.json');r.update(updated=stamp,state='reviewed_followups_no_adoption',
    followup_actual_updates=2655,parameter_isolation_forwards=325,
    followup_conclusion='单参数替换和两种较弱分离几何约束均没有带来明显未知视角改善；停止这轮表面几何权重调整。',
    next='现有动态人体上，单独验证真实观测支持的衣物高斯与独立时间形变；不变更主3DGS查看器。')
save(pilot/'root-review.json',r)
print('RECORDED',stamp)

