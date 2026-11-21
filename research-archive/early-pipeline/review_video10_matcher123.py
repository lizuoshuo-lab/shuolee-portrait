"""Record the actual72row and16context visual review, without new inference."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,itertools
import numpy as np
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122';Q=R/'video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (Q/'visual-review.json').exists()
meta=read(Q/'review/index.json');rows=meta['rows'];p=read(Q/'plan.json');result=read(Q/'run/result.json')
assert len(rows)==72 and len(meta['pages'])==12
fixed={3,6,26,31,32,36,39,46,48,49,50,52,56,*range(60,72)}
person={8,18,28,29,58}
unusable={1,4,7,9,11,13,16,17,23,30,33,34,37,40,41,42,45,47,51,53}
notes=[
'边界附近的顶棚木纹缺少唯一局部身份，目标接近视场外；保留未知。',
'原查询在墙顶支架一带，目标被后脑与衣领覆盖，不能作为同一背景材料点。',
'可辨认柜顶同一黑色玻璃/金属器物，但中心附近有反射、亮边和镜面背景，不认证精确材料点。',
'扩大上下文可辨认同一木质蘑菇摆件及局部形状；仅粗材料支持。',
'墙顶支架/照明附近在目标被人物覆盖，未建立可见同名物理点。',
'黑色玻璃器物局部相似，反射与亮边身份不确定。',
'扩大上下文可辨认同一小白花盆与柜顶局部；叶片及边缘不能作为亚像素真值。',
'原查询靠桌边/手机及画面边缘，预测贴近或超出有效图幅，不认证材料对应。',
'后颈/花纹衣领的同一粗局部可辨认；属于人物，不能作静止场景或面部眼鼻嘴真值。',
'头部矩形内实际接近被人物遮挡的背景，预测转到头面附近；ROI编号不等于人脸材料点。',
'吊篮/植物区域可辨认，但叶片和阴影不保证刚体固定。',
'墙顶支架区域在目标被头面遮挡，原材料未证实可见。',
'可辨认黑色器物及镜框背景，查询中心的反射亮边仍不确定。',
'木质摆件附近的预测转到另一处门边/地面背景，与原物体不相同。',
'墙面光斑与支架附近被人体部分遮挡；亮度边缘不是已确认固定材料。',
'黑色器物相似，不能分辨中心是否同一反射或实体纹理。',
'柜顶灯笼附近的预测转到门边/大花盆区域，不接纳为原物体对应。',
'桌角/手机附近预测转到地面，材料不一致。',
'后颈/花纹衣领粗局部可辨认，不认证人体静止或面部亚像素。',
'头部ROI实际主要是重复顶棚木纹；扩大上下文仍不能认证同一精确纹理位置。',
'吊灯框架附近有强过曝，实体线与光晕混合，不认证角点材料。',
'顶棚重复木纹且邻近头部遮挡，保留身份歧义。',
'盆栽粗区域相似，但中心含叶片与反光盆壁，不能当作固定精确材料。',
'蘑菇摆件区域的预测转到大花盆/植株，材料不一致。',
'同一盆栽附近仍存在叶片/阴影/盆壁混合，保留未知。',
'扩大上下文显示灯笼边与干草交叠，中心相对边框位置漂移，精确物理身份不确定；不把小往返误差当通过。',
'同一门板的门边五金/接缝粗局部可辨认；未测量真实亚像素位置。',
'黑色玻璃器物可辨认，中心反射/亮边仍未知。',
'后颈/衣领可辨认粗对应，不是眼鼻嘴留出真值。',
'后颈/衣领可辨认粗对应，不认证人物静止。',
'墙面画框的预测离开原画框，接近另一处边缘或视场外。',
'冰箱磁贴及篮子边的同一粗区域可辨认；不认证精确角点。',
'架上小房屋/包装的排列可辨认，粗材料对应。',
'墙面摆架附近的预测转到另一片物体，未找到同一材料。',
'墙画框预测离开原画框并靠图幅边界，不接纳。',
'黑色玻璃器物/镜框可辨认，但反射与实体位置不分明。',
'架上小房屋与包装可辨认粗对应。',
'画框/墙面查询的预测转到另一片支架或画面外区域，不接纳。',
'扩大上下文显示查询紧邻侧脸轮廓及背景；不能据头部框将它认成人脸特征。精确前景/背景身份保留未知。',
'扩大上下文可辨认查询实际上靠柜门小照片/边框，在目标仍对应此背景；不是面部点。',
'重复顶棚木纹的预测贴近图幅外，未证明同一位置。',
'原冰箱背景在目标被人物遮住，预测落在头颈，不接纳为同一背景材料。',
'原冰箱篮子/磁贴附近的固定背景在目标被人物遮住；预测落在脸颊或衣领。往返约0.08px仍不能证明物理同名，本轮0.9置信度门槛拒绝。',
'电脑屏幕区域与反光混合，显示内容不是已确认静止材料。',
'木纹预测靠边界/画面外，重复纹理身份不确定。',
'原冰箱背景被人物覆盖，预测落到人物，不接纳。',
'架上小房屋/包装排列可辨认粗对应。',
'画框原物体与目标另一处墙架不一致。',
'扩大上下文显示查询在头部框内的柜门小照片附近，目标为该背景，不能计入人脸约束。',
'扩大上下文显示柜门小照片/木边的背景局部，不是面部留出点。',
'镜框/墙面上方的实体边框粗局部可辨认，精确点仍未测。',
'冰箱背景与目标人物遮挡混合；原材料身份不能由小往返误差认证。',
'架上小房屋/包装局部可辨认粗对应。',
'原冰箱日历/磁贴被肩袖遮住，预测转到衣服，不接纳。',
'顶棚木纹重复且靠图幅外，保留未知。',
'黑色玻璃器物中心有反射/亮边，物理点身份不确定。',
'架上小房屋/包装局部可辨认粗对应。',
'飞镖盘在人体旁部分遮挡，重复扇区与边界存在歧义；高置信度不能认证具体材料点。',
'唯一可辨认眼镜/眼周粗局部的所选正面查询；FIT，未认证精确物理点、反射稳定性或亚像素。',
'头部ROI实际为人物旁的部分飞镖盘/轮廓区域；重复扇区及遮挡未知，不计人脸HELD。',
'冰箱磁贴排列的原控制粗局部可辨认。',
'扩大上下文可辨认冰箱顶部/篮子边的原粗控制；直接RootSIFT未给点，RoMa有预测，但未通过固定置信度门槛。',
'冰箱日历/磁贴的原控制粗局部可辨认。',
'冰箱磁贴/篮子边原控制粗局部可辨认。',
'冰箱侧面白字与相邻贴纸原粗控制可辨认。',
'冰箱侧面白字原粗控制可辨认，直接RootSIFT缺失但RoMa未通过固定置信度门槛。',
'柜体底部与地砖接缝/鞋旁原粗控制可辨认，未证亚像素。',
'冰箱日历/磁贴原粗控制可辨认。',
'冰箱磁贴与白纸边原粗控制可辨认。',
'柜顶设备与白色器件原粗控制可辨认；设备是否曾移动不能仅由两帧证明。',
'柜顶设备的绿色外框/外壳原粗控制可辨认。',
'柜体下缘与地砖接缝原粗控制可辨认；不将鞋视为场景固定先验。'
]
assert len(notes)==72
review=[]
for r,note in zip(rows,notes):
    i=r['row'];category='coarse_fixed_patch' if i in fixed else 'coarse_person_patch' if i in person else 'not_same_visible_material' if i in unusable else 'ambiguous'
    cycle_results=[]
    for tri in p['triplets']:
        if r['source'] not in tri or r['target'] not in tri:continue
        for b,c in itertools.permutations([f for f in tri if f!=r['source']]):
            file=Q/'run'/f'cycle_{r["source"]:06d}_{b:06d}_{c:06d}.npz'
            with np.load(file) as d:cycle_results.append(dict(cycle=[r['source'],b,c,r['source']],gate_pass=bool(d['eligible'][r['query_feature']]),closure_native_px=float(d['closure'][r['query_feature']])))
    review.append(dict(row=i,family=r['family'],split=r['split'],source=r['source'],target=r['target'],category=category,note=note,
        gate_pass=r['gate_pass'],source_review_group=r['review_group'],cycle_checks=cycle_results,
        physical_point_certified=False,subpixel_certified=False))
stamp=datetime.now(timezone.utc).isoformat()
stats=dict(category_counts=dict(collections.Counter(r['category'] for r in review)),
    passed_category_counts=dict(collections.Counter(r['category'] for r in review if r['gate_pass'])),
    coarse_control_passed=sum(r['gate_pass'] for r in review[60:]),coarse_control_total=12,
    head_roi_rows=sum(r['source_review_group']=='headROI_unknown_material' for r in review),
    eye_nose_mouth_candidate_rows=[58],candidate58_cycle_checks=review[58]['cycle_checks'],
    certified_face_heldout=0,certified_face_subpixel=0)
review_record=dict(updated=stamp,reviewed_all72=True,reviewed_pages=12,reviewed_context_rows=read(Q/'review/contexts/allocation.json')['rows'],
    pages_sha256={x['file']:sha(Q/'review'/x['file']) for x in meta['pages']},rows=review,statistics=stats,
    interpretation='Visual labels are coarse recognition or visible rejection, not subpixelGT. All frozen splits and old raw tracks remain unchanged. Labels from this selected diagnostic are not a globalaccuracy estimate.')
save(Q/'visual-review.json',review_record)
decision=dict(updated=stamp,state='bounded_matcher_contrast_complete_not_admitted',
    summary='同一查询位置的七组RoMa对照完成，72个预登记位置和16个扩大上下文全部审阅；15个通过固定置信度/往返筛查，其中8个粗背景、1个粗人物、6个身份仍有歧义。12个旧粗背景控制只有4个通过该筛查。可见遮挡误配仍可能有0.08px往返误差，但本轮置信度门槛拦下该例。唯一眼镜/眼周粗候选为FIT，三视角循环通过；循环误差0.075至0.174px只是自洽误差，独立面部HELD仍为0。没有证据准入新相机或SMPL-X重拟合。',
    stats=stats,cycles_are_self_consistency_only=True,selected_probe_not_all_video=True,
    old_SIFT_not_rerun=True,RoMa_attempt_consumed=True,thresholds_changed=False,new_correspondence_inference_allowed=False,
    new_camera=False,new_fit=False,new_training=False,quality_accepted=False,
    next='Audit remaining already-extracted, not-yet-reviewed background families for missing temporal support before deciding whether any fixedKcamera graph is observable. No newfeature/matcher pass or threshold sweep; inventory first, freeze once-only review scope and physical-point deduplication.')
assert stats['coarse_control_passed']==4
assert len(stats['candidate58_cycle_checks'])==2
save(Q/'decision.json',decision)
prog=read(Q/'progress.json');prog.update(updated=stamp,state='complete_reviewed_not_admitted',pid=None,quality_accepted=False);save(Q/'progress.json',prog)
print(json.dumps(dict(summary=decision['summary'],statistics=stats),ensure_ascii=False))
