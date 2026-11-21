from pathlib import Path
import json,datetime,hashlib,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
O=B/'current/shared-point-position-preflight-100'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
a=read(O/'result.json');assert a['failed']==['slanted-footprint-approximation']
decision=dict(created=datetime.datetime.now().astimezone().isoformat(),round='100',state='candidate_rejected_before_training',
  formulation='同名特征共享唯一世界点。此候选用每个贡献高斯的中心深度构成射线上位置，再计算到该共享点的三维距离。',
  passed='原生RGB/几何输出及原有梯度一致，新损失有限且位置梯度非零。两层不能通过平均深度抵消。144个原生贡献查询的解析矩与直接加权计算最大差异5.43e-8。',
  rejected='中心深度不能代替倾斜表面的实际交点。144个固定查询中，非掠射正深度贡献的近似偏差相对既定共享点不确定度，中位数1.086，90分位1.984，超过预先规定的1.0门槛。',
  limits='该比较说明此近似可能产生额外几何偏差，不是对真实人体几何误差的测量。冻结模型的已有平面也可能错误；不把它当三维真值。',
  action='没有开始新的高斯训练、没有更改相机或替换098交付点云。保留一个简明诊断与源码，不产生失败模型包。',
  next='检查联合相机/共享点求解器的默认终止设置及实际收敛证据。先解决唯一空间点与多视图观测之间的误差，再决定训练；不提高本轮门槛或扫权重。',
  training_started=False,optimizer_updates=0,quality_accepted=False,goal_complete=False)
save(O/'decision.json',decision)
snapshot=O/'source';snapshot.mkdir(exist_ok=True)
manifest=[]
for p,d in a['consumer_sources'].items():
    path=Path('C:/'+p.split('/mnt/c/',1)[1]);assert hashlib.sha256(path.read_bytes()).hexdigest()==d
    dest=snapshot/path.name;assert not dest.exists();shutil.copy2(path,dest)
    manifest.append(dict(path=str(path),sha256=d,copy=dest.name))
save(snapshot/'manifest.json',manifest)
state=read(B/'work-status.json');state.update(updated=decision['created'],local_training_running=False)
state['latest_diagnostic']=dict(round='100',state=decision['state'],path=str(O),quality_accepted=False,new_gaussian=False)
state['next_bounded_work']=dict(scope=decision['next'],training_running=False,goal_complete=False)
save(B/'work-status.json',state)
print(json.dumps(dict(recorded=str(O/'decision.json'),training_started=False)))
