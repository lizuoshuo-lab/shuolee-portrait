from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';OUT=REC/'remaining-native-input-audit';H=Path(__file__).parent
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat();r=read(OUT/'result.json')
assert not (OUT/'visual-review.json').exists()
notes={
 'grass':[
  '0000: 全身与鞋可见，侧脸部分被头发遮挡；背景有灯杆、铺装路和树干，但大量草叶树叶不能自动算静止材料。',
  '0122: 正侧面，双腿鞋可见；背景路上有人，头发遮挡一侧脸。此幅没有材料轨迹或相机真值。',
  '0245: 近正面，眼鼻嘴有图像信息；草坪纹理重复，树干与灯杆仅为候选固定结构。',
  '0367: 另一侧面，远侧眼鼻嘴存在自遮挡；全身鞋可见。',
  '0469: 侧后方，头发遮住面部；仅凭这5幅不能量化运动或连续相机轨迹。'],
 'indoor':[
  '0000: 正面双眼鼻嘴可见，白衣和手臂可见，画面下边截断腿脚；置物架、墙面温控器和插座提供固定结构候选。灯饰高光、窗帘和软沙发不是自动有效材料点。',
  '0230: 右向侧面，远侧面部自遮挡；桌沿与下边界遮挡/截断腿脚。柜体、桌腿和墙地交界是物体候选，边缘本身未必是固定表面点。',
  '0461: 后脑勺，没有可见眼鼻嘴；有楼梯与门口结构、相机三脚架、柜体。重复栏杆、发光环和绿植需排除身份/高光歧义。腿脚被家具及下边界挡住。',
  '0691: 左向侧面，近侧面部可见；门框、楼梯连接和墙面开关可辨。地上有猫，不作静止场景真值；部分脚与桌子重叠，另有截边。',
  '0883: 回到近正面且画面尺度变大，眼鼻嘴可见；腿脚出画。物体布局相似不能证明人物或相机静止。']}
review=dict(updated=stamp,actual_original_images_reviewed=10,full_native_resolution=True,source_samples_reused_from121=True,
    subjects=[dict(subject=s['subject'],sample_files=[x['file'] for x in s['samples']],notes=notes[s['subject']],
                   physical_static_material_certified=False,face_visibility_only_qualitative=True,subpixel_accuracy_verified=False) for s in r['subjects']],
    decision='Prioritize existing indoor video8 full177 intake because visible multi-depth rigid-object candidates justify checking the denser predeclared observations. Grass remains in queue. No feature/geometry run admitted by this review.',
    no_unique_ghosting_cause_claim=True,training_started=False,quality_accepted=False)
save(OUT/'visual-review.json',review)
indoor=next(s for s in r['subjects'] if s['subject']=='indoor')
assert len(indoor['previously_unexamined123_indices'])==173
plan=dict(registered=stamp,subject='indoor',video='D:/00/工作/数据集/video8.mp4',video_sha256=indoor['video_sha256'],source_frames=884,
    selected_indices=list(range(0,884,5)),selected_frames=177,native_wh=[2704,1520],review_slots=[0,44,88,132,176],
    stage='native_intake_only',hypothesis='Existing24frame diagnostic did not examine173 of the predeclared177 source indices. More densely sampled original imagery may retain rigid-object views across large historical gaps; this intake only verifies pixel domain, timestamps and visible coverage.',
    changed_information='Temporal sampling gap5 instead of historical max39; same source bytes and raster. No claim that sampling alone resolves the failed camera graph.',
    historical_review=str(OUT/'visual-review.json'),historical_review_sha256=sha(OUT/'visual-review.json'),
    operation='Once-only extraction of177 frozen frames with existing FFmpeg extractor, no resize at2704max; compare4 shared historical frames by decoded pixel values and PTS, never mix historical feature observations across changed decoder output.',
    budget=dict(seconds=240,decode_passes=1,CPU_threads=4,GPU_jobs=0,feature_extractions=0,camera_candidates=0,SMPLX=0,training=0),
    stop='Hash, frame count, raster, rotation or PTS failure stops intake. If historical RGB differs, record exact difference and prohibit blindly importing121 coordinates. Inspect fixed5 fullframes before further registration.',
    no_second_undistortion=True,no_K_change=True,no_interpolated_camera=True,old_individual_geometry_reused=False,physical_accuracy_accepted=False)
save(OUT/'indoor-intake-plan.json',plan)
print(json.dumps(dict(reviewed_images=10,selected='indoor',registered_frames=177,training=False),ensure_ascii=False))
