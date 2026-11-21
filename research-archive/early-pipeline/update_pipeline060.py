import json
from pathlib import Path
from datetime import datetime

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
p=B/'work-status.json'
d=json.loads(p.read_text(encoding='utf-8-sig'))
d.update(updated=datetime.now().isoformat(timespec='seconds'),current_phase='fresh_all_frames_pose_and_clothing_pipeline_060',
    local_training_running=False,current_decision='按用户最新两项要求：175个独立原视频帧重新提取观测并共同拟合姿态；SMPL-X仅作运动先验，衣物高斯需要独立表面支持及自由位置。059位姿与穿衣分布未通过；暂停表面绑定局部实验。全部新实验从头，旧个体结果只作对照。')
d['latest_local_execution']=dict(kind='person_clothing_pipeline_fresh_060',state='fresh_feature_inference',running=True,
    directory=str(B/'current/person-clothing-pipeline-fresh-060'),real_frames=175,calibrated_cameras=59,interpolated_cameras=116,
    old_person_fit_loaded=False,old_gaussian_loaded=False,old_optimizer_loaded=False,geometry_accepted=False,quality_accepted=False)
tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
print('Current decision and execution updated; prior evidence retained.')
