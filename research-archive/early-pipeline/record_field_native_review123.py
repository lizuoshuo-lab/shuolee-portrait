from pathlib import Path
from datetime import datetime, timezone
import json, hashlib

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/local-clothed-surface-123'
F=R/'field-preflight'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (F/'visual-review.json').exists()
result=read(F/'result.json')
for item in result['images']:
    assert sha(F/item['file'])==item['sha256']
stamp=datetime.now(timezone.utc).isoformat()
views=[
    dict(source_frame=0,view='正面',observed='头部、躯干、双手、两只鞋均在画面内。上衣为深色低纹理面料，斜挎带横过胸部。'),
    dict(source_frame=220,view='一侧',observed='鼻部侧面、近侧耳朵及全身可见，两脚有前后遮挡。背后小包可见。'),
    dict(source_frame=440,view='背面',observed='后脑、背部与包可见，脸部不可见。裤脚遮挡部分鞋后跟，不能据此评价足部内部几何。'),
    dict(source_frame=660,view='另一侧',observed='另一侧脸、肩臂及全身可见，远侧鞋有部分遮挡。'),
    dict(source_frame=880,view='再次正面',observed='头部到两只鞋均在画面内。面部外观与首帧相近，但没有测量刚体位姿或表情变化。')]
review=dict(updated=stamp,subject='field',source_sha256=result['source_sha256'],
    selected_frames=177,actual_full_frames_reviewed=5,actual_native_head_crops_reviewed=5,
    actual_native_feet_crops_reviewed=5,source_indices=[0,220,440,660,880],views=views,
    review_method='view_image high for each complete original frame, followed by original-detail unresized head and foot strips. Full frames were display-downscaled, crop pixels were native.',
    image_text_or_warp_added=False,original_pixel_hashes_unchanged=True,
    observations=[
        '五个抽检时刻显示正面、两侧与背面，首尾接近同一朝向。不是全部177帧可见性验收。',
        '面部在整幅图中较小。原生裁片能分辨眼鼻嘴，但没有测量单像素定位误差或不确定性。',
        '建筑、桥梁、凉亭、土地及部分灌溉接头可作为固定场景候选，尚未建立同名物理对应。',
        '成行植株、重复立杆、平行水管及树叶具有歧义，不能仅以高特征数或闭环认定为可靠静态材料。',
        '树叶和停放车辆未进行时间静止性验证。玻璃反射不能自动当作固定三维点。',
        '上衣、裤子和包的外轮廓不属于裸SMPL-X皮肤表面。人体骨架拟合与最终自由衣物表面应分别检查。',
        '五帧没有量化人物微动。首尾相似不能证明全视频人物完全静止。'],
    physical_material_identity_verified=False,camera_registered=False,SMPLX_fitted=False,
    independent_face_subpixel_pass=False,training_started=False,quality_accepted=False,
    next='先修复并测试读取已冻结观测的工程入口，避免旧相机函数重新提取未审阅轨迹。之后在单独登记的田地材料预算内核验固定背景及人体留出，不从这五幅图宣称标定或对齐通过。')
save(F/'visual-review.json',review)
progress=read(F/'progress.json')
progress.update(updated=stamp,state='native_input_review_complete_no_camera',running=False,visual_review=str(F/'visual-review.json'))
save(F/'progress.json',progress)
print(json.dumps(dict(review=str(F/'visual-review.json'),actual_frames=5,native_crops=10,quality_accepted=False),ensure_ascii=False))
