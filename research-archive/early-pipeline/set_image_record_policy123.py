from pathlib import Path
from datetime import datetime, timezone
import hashlib,json

H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
R=T/'shuolee/current/robust-portrait-122'
COORD=H.parent/'shuolee-coordination'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
rule=Path('D:/00/记录/规则.md')
policy={'updated':stamp,'authorization':'User: 将图片按照记录里面的规则放到00的记录里，不要放日志了',
        'root':'D:/00/记录','rules':str(rule),
        'rules_sha256':hashlib.sha256(rule.read_bytes()).hexdigest(),
        'dataset_naming':'videoN.mp4 -> zero-padded three-digit dataset number, e.g. video5 -> 005',
        'allowed_files':'images only inside each numbered dataset; no text, JSON, HTML, scripts, or weights',
        'image_annotations':'No words on images. Describe stage, frame, region and comparison order in file names.',
        'stages':{'mask':['原图','mask'],'特征点':['原图','2d特征点','空间特征点'],
                  'smpl-x':['原图','smpl-x'],'3DGS':['原图','高斯'],'分析':['actual pose/distance/view comparisons']},
        'preservation':'Use original native image/mask bytes where possible; overlay points and meshes without geometric image warps.',
        'no_fake_stage':'Do not fill an unrun Gaussian stage with mesh renders or a different experiment.',
        'new_work_log_image_writes':False,
        'receipt_root':str(REC),'prior_artifacts_preserved':True,'WPS_cloud_verified':False}
save(COORD/'image-record-policy.json',policy)
for root in [REC,R]:
    p=root/'queue-state.json';state=read(p)
    state.update(updated=stamp,image_delivery_root=policy['root'],image_delivery_rules=str(rule),
                 write_new_images_to_work_log=False,
                 previous_goal_turn_classification='progress: numerical repair completed on three views; duplicate target contradiction verified; published actual review and slide52')
    save(p,state)
p=R/'protocol.json';state=read(p)
state['image_delivery_override']=policy
save(p,state)
p=REC/'smplifyx-reference-control/observation-contract-plan.json';state=read(p)
state['image_delivery']=policy['root'];state['no_new_images_in_work_log']=True
save(p,state)
p=COORD/'local-research-status.json';state=read(p)
state.update(updated_at=stamp,image_delivery_policy=str(COORD/'image-record-policy.json'),
             image_delivery_root=policy['root'],new_images_in_work_log=False)
save(p,state)
# The latest report exporter must also stop automatically duplicating its images
# into the old work-log mirror. Historical report/image links remain intact.
p=H/'publish_smplify_conditioned123.py';source=p.read_text(encoding='utf-8')
old="destinations=[G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer',LOG]"
new="destinations=[G,T/'viewer/diagnostics/video10-global-initializer',SYNC/'diagnostics/video10-global-initializer'] # User image delivery now uses D:/00/记录; export_smplify_record_images123.py"
assert source.count(old)==1
p.write_text(source.replace(old,new),encoding='utf-8',newline='\n')
print(json.dumps({'image_delivery_root':policy['root'],'new_work_log_image_writes':False,'training_code_changed':False},ensure_ascii=False))
