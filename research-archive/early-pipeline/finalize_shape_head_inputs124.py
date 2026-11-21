"""Finalize the actually viewed frozen-state audit; no optimization or fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

CURRENT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
OUT=CURRENT/'local-clothed-surface-123/cross-subject-sh0-124/shared-shape-head5-audit124'
R=CURRENT/'robust-portrait-122'

def read(p):return json.loads(p.read_text('utf-8-sig'))
def save(p,d):
    tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
    tmp.replace(p)
def local(p):return Path(p[5].upper()+':/'+p[7:]) if p.startswith('/mnt/') else Path(p)

assert not (OUT/'conclusion.json').exists()
r=read(OUT/'result.json'); v=read(OUT/'visual-review.json'); progress=read(OUT/'progress.json')
assert r['new_fit_calls']==0 and r['optimizer_steps']==0 and progress.get('pid') is None
assert len(r['replay_checks'])==9 and all(x['passed'] for x in r['replay_checks'].values())
images={'rest':v['rest']}
images.update({'native-'+k:x for k,x in v['native'].items()})
images.update({'points-'+k:x for k,x in v['points'].items()})
assert len(images)==11
for item in images.values():
    assert hashlib.sha256(local(item['path']).read_bytes()).hexdigest()==item['sha256']

stamp=datetime.now(timezone.utc).isoformat()
findings={
    'numerical_reconstruction':'原SMPL-X模型按保存的参数重建9份网格，最大顶点差5.25e-7模型米、投影差0.000423像素以内；本次图片分解不是导出或显示坐标错误。这仅验证计算一致性，不是实拍标定精度。',
    'shared_shape':'同一静态姿态、同一相机与尺度下，室内105共享体型的身材比例明显不同于中性、冬帽与沙发；该体型在后续6份诊断结果中保持相同。模型竖向范围1.402米是模型坐标值，不是测得人物身高，不能以偏离中性模板单独判定真实体型。',
    'expression_counterfactual':'同姿态同相机下只将表情置零，室内105及40帧的主要头部轮廓异常仍在；停用内脸51点的40帧，表情置零仅改变头部中位0.012毫米。因此剩余头形不能只归因于表情。',
    'shape_counterfactual':'只将体型置零会明显改变头形，同时让整个头部和身体投影严重偏移；它包含骨架比例与蒙皮效应，未重新拟合，不能作为修复结果、真实体型或新初始化直接采用。',
    'head_observations':'已查看38、39、40、41、53、61、105的原生像素对照。侧后方的远侧耳点落在后脑勺；53帧背面对着相机，仍在后脑检测出眼鼻及内脸点；105正面相对合理。身体鼻眼耳5点与内脸51点是不同输入项，后者停用并未停用前者。检测高分不等于实际可见。',
    'causal_limit':'已证实共享体型传递与不可信头部观测同时存在，尚不能把全部肢体/头部错误唯一归因于其中之一。参考帧的相机尺度、未观测下半身和原版分阶段目标仍耦合；此次不改权重、不优化体型、不重启整段。'}
summary='连续帧约束已接入原SMPLify-X并记录到GPU梯度；新完成的只读分解又确认两个上游问题：参考105的共享体型被原样传递，转到背面的帧仍有鼻眼耳身体点落在后脑。11张实际对照图已审阅；仅清零表情没有消除主要头形异常，直接清零体型则破坏投影，不能当作修复。9份保存结果重建一致，未运行新拟合或高斯。'
next_action=('Read shared-shape-head5-audit124/conclusion.json and its11 actually reviewed images; the one CPU audit is complete, do not rerun it. '
             'Temporal root/body rotation constraints and same-run chronological seed are implemented; quality remains rejected. '
             'Do not strengthen temporal weights to hide invalid targets, adopt neutral beta, reuse rejected local38/39 prefix as sequence initialization, or restart historical sequences. '
             'Next prepare one coherent head-observation absence input contract that distinguishes body nose/eyes/ears5 from innerface51, retaining original observations and using only already frozen face-support evidence; no new detectors or threshold scan. '
             'Any matched diagnostic must change only head5 relative to the preserved face51-absence control, keep the exact same40 seed/camera/shape/stages and original visible shoulder/elbow/wrist protection, and register a finite budget before fitting. '
             'Do not promote the policy based on lower aggregate error: side-face SFD positives can still be false, visible ears can exist without a face box, and shared beta remains unaccepted. '
             'Shared shape must later be supported across visible frames rather than frozen forever from an unaccepted reference; first read the existing check_official_shared_shape123.py results to avoid duplicating its control. '
             'No Gaussian until pose and occlusion admission, current demo120k SH0 and all original data remain unchanged. Continue the active goal, not a scheduled task.')
v.update(updated=stamp,state='actually_reviewed_diagnostic_complete',actual_visual_review=True,
         reviewed_images=11,findings=findings,quality_accepted=False,default_applied=False)
save(OUT/'visual-review.json',v)
conclusion=dict(updated=stamp,state='completed_readonly_diagnosis',summary=summary,next_action=next_action,
    actual_visual_review=True,images=images,findings=findings,
    computational_parity_passed=True,maximum_vertex_replay_delta_m=max(x['vertex_delta_m'] for x in r['replay_checks'].values()),
    maximum_projection_replay_delta_px=max(x['projection_delta_px'] for x in r['replay_checks'].values()),
    seconds=r['seconds'],new_fit_calls=0,optimizer_steps=0,new_detector_calls=0,
    quality_accepted=False,default_applied=False,gaussian_admitted=False,
    neutral_beta_is_truth=False,unique_physical_cause_identified=False,original_files_unchanged=r['original_files_unchanged'])
save(OUT/'conclusion.json',conclusion)
for path in [R/'protocol.json',R/'queue-state.json',R/'evidence-ledger.json',
             Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/local-research-status.json')]:
    d=read(path)
    d.update(updated=stamp,current_stage='cross_subject_temporal124_shape_head5_reviewed',summary=summary,
             next_action=next_action,own_gpu_task_running=False,current_worker_pid_wsl=None,
             current_active_experiment=None,gaussian_training_running=False,training_allowed=False)
    d['shape_head5_audit124']=dict(plan=str(OUT/'plan.json'),result=str(OUT/'result.json'),conclusion=str(OUT/'conclusion.json'),
        actual_visual_review=True,images=11,fit_calls=0,quality_accepted=False,default_applied=False)
    d['last_goal_turn']=dict(classification='progress',evidence=summary,goal_complete=False)
    save(path,d)
print(summary)
