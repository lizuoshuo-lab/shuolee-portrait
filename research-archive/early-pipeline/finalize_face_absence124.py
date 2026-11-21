"""Record actual review of the once-only temporal input control; no fitting."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
OUT=ROOT/'local-clothed-surface-123/cross-subject-sh0-124/face-absence-input124/indoor-transition-control'
R=ROOT/'robust-portrait-122'
def read(p):return json.loads(p.read_text('utf-8-sig'))
def save(p,d):
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8');tmp.replace(p)
def local(s):
    return Path(s[5].upper()+':/'+s[7:]) if s.startswith('/mnt/') else Path(s)

assert not (OUT/'conclusion.json').exists()
r=read(OUT/'result.json');v=read(OUT/'visual-review.json')
assert r['fitting_calls']==6 and read(OUT/'progress.json')['pid'] is None
assert r['comparisons']['frame_000040']['camera_translation_max_delta_m']==0
stamp=datetime.now(timezone.utc).isoformat()
images={k:x['images'] for k,x in v['frames'].items()}
for items in images.values():
    for item in items.values():assert hashlib.sha256(local(item['path']).read_bytes()).hexdigest()==item['sha256']
findings={
    'shared_prefix38_39':'原生叠加显示身体大致转向后方，但头部朝向偏向侧前方；头部轮廓明显超出原图，手臂有投影偏差。前段本身未通过质量，不能当新序列初始化。',
    'matched40':'同39初值与相机的直接对照：停用face51后，头部从错误侧前方向原图侧后方转动；但头部位置偏高、外轮廓仍错，左肩检测点误差10.26→19.01px，左腕18.07→21.39px。',
    'branch41':'沿各自40传播后，后脑方向变化保留，平均身体检测误差降低；左肩10.42→21.29px，右肘19.13→22.39px，未通过逐点保护。',
    'native_arms_and_side_views':'四帧全部原生身体、头部、手臂及共相机四方向图已实看。手在腰部的深度关系仍不可靠；两组均保留倾斜的相机坐标几何，未恢复重力/世界外参，不能仅凭倾斜宣称解剖错误。出画腿脚没有真实监督，不能验收。',
    'measurement_boundary':'相对检测置信度加权的投影距离不是独立真值。不可见51脸点误差变大不作为退化依据，也不通过删点将其变为精度通过。'}
summary='SMPLify-X连续帧约束已接入原版求解器。室内38—41帧一次6拟合对照完成（173.2秒）：同初值下停用无支持的51个脸点，头部朝向更接近原图侧后方；40/41帧身体检测加权误差30.03/30.44→28.32/28.33px，但左肩10.26/10.42→19.01/21.29px，逐关节保护失败。全部4帧原生叠加及侧视已检查；此候选不设为默认，不启动整段或高斯。'
next_action='The one face51-absence control is complete and rejected for whole-body protection; do not rerun its6fits or promote its local38/39 prefix. Preserve the original temporal interface, solver and all old models. Next inspect saved outputs and original observation semantics only: distinguish the still-active COCO nose/eye/ear5 from inner-face51, examine reference105 shared-beta and camera-relative scale support, and separate prior/initialization effects from invalid targets. No new detector, weight/threshold sweep, old-sequence restart or Gaussian before a distinct bounded plan with visible-body protections. A coherent absence policy is not yet evidence of correct 3D; current knees/feet out of raster remain unobserved. Continue the active multi-person goal, not a timer.'
metrics={}
for k,d in r['comparisons'].items():
    metrics[k]=dict(original_body=d['original']['body'],candidate_body=d['face_absence']['body'],
        body_protection_passed=all(d['gates'].values()),original_gates=d['gates'],
        left_shoulder_old=d['original']['distance_px'][5],left_shoulder_new=d['face_absence']['distance_px'][5],
        camera_translation_max_delta_m=d['camera_translation_max_delta_m'])
v.update(updated=stamp,state='actually_reviewed_quality_failed',actual_visual_review=True,
    reviewed_images=16,reviewed_distinct_frames=4,reviewed_fit_endpoints=6,findings=findings,
    accepted=False,quality_accepted=False,default_applied=False)
save(OUT/'visual-review.json',v)
verdict=dict(updated=stamp,state='completed_rejected',summary=summary,next_action=next_action,
    accepted=False,quality_accepted=False,default_applied=False,actual_visual_review=True,
    fitting_calls=6,seconds=r['seconds'],images=images,metrics=metrics,findings=findings,
    full_sequence_admitted=False,gaussian_admitted=False,physical_camera_calibration_accepted=False,
    unchanged_originals=True,causal_scope='Frame40 is exact shared-seed input-only contrast. Frame41 additionally includes downstream propagation; both recorded camera translations match exactly. No complete-sequence quality claim.')
save(OUT/'conclusion.json',verdict)
for path in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/local-research-status.json')]:
    d=read(path);d.update(updated=stamp,current_stage='cross_subject_temporal124_face_absence_control_reviewed',
        next_action=next_action,summary=summary,own_gpu_task_running=False,current_worker_pid_wsl=None,
        gaussian_training_running=False,training_allowed=False)
    d['face_absence_control124']=dict(plan=str(OUT/'plan.json'),result=str(OUT/'result.json'),conclusion=str(OUT/'conclusion.json'),
        actual_visual_review=True,quality_accepted=False,default_applied=False,gaussian_admitted=False)
    d['last_goal_turn']=dict(classification='progress',evidence=summary,goal_complete=False)
    save(path,d)
print(summary)
