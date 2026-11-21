"""Record the completed all-node visual review before reading saved held errors."""
from datetime import datetime, timezone
from collections import Counter
from audit_grayT123_camera_support import REC, read, save, sha

OUT = REC / 'grayT-material-identity-review'
# Written after actual review of all 28 native-pixel sheets, not error ranking.
NOTES = [
 ('outline', '鼻部/鼻孔侧面边缘；同一区域，但轮廓切点可能随视角改变。'),
 ('mouth', '牙齿与唇边界；笑容和口型变化，不能当作固定皮肤材料点。'),
 ('eye', '眼裂/瞳孔或眼睑暗部；跨视角仍在眼周，视线与眼睑静止性未证实。'),
 ('mouth', '下唇/下巴轮廓；口型改变，轮廓材料身份未证实。'),
 ('outline', '鼻梁/鼻部侧面轮廓；没有独立固定纹理标记。'),
 ('eye', '上眼睑暗部；同一眼周区域不等于同一固定材料点。'),
 ('mouth', '下唇/下巴轮廓；笑容变化。'),
 ('eye', '眼裂/眼角附近；没有可认证的静止皮肤纹理。'),
 ('eye', '眉睫/上眼睑附近；只有三帧，具体材料定位不确定。'),
 ('mouth', '上唇与鼻部交界侧面轮廓；口型及轮廓效应未分离。'),
 ('eye', '眼裂附近；跨帧都在眼周，未证实物理静止性。'),
 ('eye', '眼角/眼裂暗部；未校准亚像素定位。'),
 ('outline', '远侧眉额/鼻梁与背景的轮廓；不能作为固定表面纹理。'),
 ('eye', '眉毛或上睑纹理/褶皱；有局部纹理，但形变与定位不确定。'),
 ('outline', '远侧眼周与鼻梁边缘；接近轮廓，材料切点不确定。'),
 ('outline', '鼻梁侧面轮廓；眼睑开闭可见，未建立固定材料身份。'),
 ('eye', '上眼睑/外侧眼裂附近；29帧都在相近眼周区域，不能据此认证静止。'),
 ('eye', '眼裂/虹膜暗部；37帧中眼裂形状可见变化。'),
 ('mouth', '嘴角/牙齿边缘；笑容变化。'),
 ('eye', '上眼睑/睫毛边缘；视角变化，固定皮肤材料未证实。'),
 ('mouth', '侧面嘴唇/牙齿边界；具体表面点受口型和遮挡影响。'),
 ('outline', '眉额与背景轮廓；只有三帧，不是独立纹理标记。'),
 ('mouth', '侧面唇缘；口型和轮廓材料身份不确定。'),
 ('eye', '侧面眉眼/上睑近轮廓暗部；姿态与眼睑变化未分离。'),
 ('eye', '上睑/眼裂区域；只有三帧。'),
 ('mouth', '嘴角/唇齿交界；18帧中笑容和张口程度变化，非静止性未排除。'),
 ('mouth', '侧面唇齿轮廓；四帧不具备固定皮肤材料认证。'),
 ('outline', '鼻梁侧面轮廓；三帧都在相近轮廓区域，不能认证同一表面切点。'),
]

def main():
    target = OUT / 'actual-review.json'
    if target.exists():
        raise RuntimeError('Actual review already recorded; preserve it.')
    index = read(OUT / 'index.json')
    assert len(index['images']) == len(NOTES) == 28
    records = []
    for i, (im, (category, note)) in enumerate(zip(index['images'], NOTES), 1):
        assert im['locator'].split('_')[1] == f'{i:02d}'
        assert sha(OUT/im['locator']) == im['locator_sha256']
        records.append(dict(track_id=im['frame'], split=im['region'].split('_')[0],
            locator=im['locator'], observations=im['all_original_nodes'], category=category,
            observations_actually_reviewed='all', note=note,
            gross_cross_object_jump_observed=False,
            same_anatomical_region_observed=True,
            fixed_physical_material_identity_certified=False,
            subpixel_location_uncertainty_measured=False,
            static_world_point_certified=False,
            use_for_training=False))
    result = dict(created=datetime.now(timezone.utc).isoformat(), actual_visual_review=True,
        source_index_sha256=sha(OUT/'index.json'), selected_components=28,
        fit_components=25, held_components=3, all_original_observations=279,
        all_nodes_actually_reviewed=True, categories=dict(Counter(r['category'] for r in records)),
        labels_fixed_before_stage2_saved_error_read=True,
        prior_knowledge='The previous review already reported the mouth track344fc frame76 residual; all28components/all279nodes were nevertheless predetermined, without residual selection.',
        observations_removed=0, split_changes=0, cameras_changed=False,
        independent_static_face_truth_count=0, records=records,
        conclusion='28条候选均仍在相近的脸部区域，完整序列未见新增的跨物体跳接。但它们是眼睑/眼裂、唇齿嘴角或轮廓点；未取得可认证的静止皮肤材料轨迹。不能把25条FIT候选直接补入相机优化，也不能把3条HELD当作物理亚像素真值。此前另一个眼部跳到栏杆的错误轨迹仍保留，未混入这28条全脸区域组件。',
        quality_accepted=False)
    assert sum(r['observations'] for r in records) == 279
    save(target, result)
    print(target)

if __name__ == '__main__':
    main()
