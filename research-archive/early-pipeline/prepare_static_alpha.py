from pathlib import Path
import shutil,json,datetime,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-alpha-fresh-069';assert not R.exists()
S=B/'current/person-clothing-pipeline-fresh-060';R.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
inputs=[]
for rel in ['data/inputs.json','data/training-inputs.json','data/observations.npz','inference/prediction-postprocessed.npz','inference/prediction-raw.npz','inference/vitpose-keypoints.pt','inference/time-camera-inputs.npz','preparation.json','mask-audit.json','feature-contract.json']:
    src=S/rel;dst=R/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);inputs.append(dict(file=rel,sha256=sha(src)))
stamp=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
save(R/'source-inputs.json',dict(created=stamp,inputs=inputs,from_scratch=True,old_subject_fit_loaded=False,
    old_gaussian_loaded=False,old_optimizer_loaded=False,real_frames=175,initial_points=200000,steps=18000,
    representation='shared_static_world',dynamic_route_abandoned=True,reference_round='064',
    controlled_change='paired black/white associated RGB and ROI/contour alpha; other PGSR geometry terms unchanged',quality_accepted=False))
frozen=R/'source-final';frozen.mkdir();src=B.parents[2]/'repos/PGSR-shuolee/shuolee_static.py';shutil.copy2(src,frozen/src.name)
save(frozen/'manifest.json',dict(created=stamp,files={src.name:sha(src)}))
st=json.loads((B/'work-status.json').read_text(encoding='utf-8-sig'))
st.update(updated=stamp,current_phase='static_alpha_edge_correction',current_decision='动态路线废弃。当前首要修复白边：原生黑背景复现，改为颜色与透明度联合监督，从头训练。',
    local_training_running=False,dynamic_route_abandoned=True,current_experiment='069 static alpha edge correction')
st['latest_local_execution']=dict(kind='static_alpha_069',directory=str(R),state='fresh_preparation',running=False,from_scratch=True,quality_accepted=False)
save(B/'work-status.json',st)
p=B/'long-term-plan.md';t=p.read_text(encoding='utf-8-sig');start=t.index('<!-- GVHMR_059_ACTIVE -->');end=t.index('<!-- END_GVHMR_059_ACTIVE -->')+len('<!-- END_GVHMR_059_ACTIVE -->')
active='''<!-- GVHMR_059_ACTIVE -->
## 当前执行：共享静态人物与轮廓白边修复

用户已明确废弃动态路线，不再作为后续扩展。175个有效实拍帧只监督一个穿衣人物的静态三维高斯场；不使用逐帧人体、衣物或时间网络。眼镜专项停止，不同步E盘，本地验证通过后才考虑00部署。

每轮从原始观测重新拟合初始化范围，重新生成20万高斯和优化器，训练18000步。旧模型只读对照，不续训、不按颜色阈值剪白点。

062普通静态与064表面约束已完成；065取消尺度压缩完成172643点，066取消多视图约束完成139941点。整体几何未通过，不按训练结束宣布最佳。065的尺度分布改变，但鞋部缺口与仰视重影未见明确总体改善。

最新优先修复白边：064原生渲染在黑灰背景下也出现后脑白壳；纯白背景训练使颜色与覆盖度可相互补偿。069从头验证黑白双背景、预乘前景重采样、人物区域与轮廓透明度监督，保持静态几何项、175帧与18k预算。原分割掩码保留，不强制侵蚀人物。

验收包括同相机黑灰白背景、头发/衣袖/鞋边放大、新视角诊断、原生与独立PLY重载及查看器实测。没有未知视角真值，不把清晰度当几何正确。每轮交付可独立导入00查看器的+Y标准PLY、路径与核验；质量与格式分别记录。

过去的动态和眼镜计划仅为历史记录，均不是当前或后续默认执行项。
<!-- END_GVHMR_059_ACTIVE -->'''
p.write_text(t[:start]+active+t[end:],encoding='utf8')
agents=B.parents[2]/'AGENTS.md'
with agents.open('a',encoding='utf8') as f:f.write('\n- 2026-10-23 最新用户明确废弃动态路线，不再写作后续扩展；当前仅共享静态人物。首要修复黑灰背景可见的白边，所有新训练仍从头开始，禁止用删除浅色衣物或切换白背景冒充修复。\n')
print(R)
