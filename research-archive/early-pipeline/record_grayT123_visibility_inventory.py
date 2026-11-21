"""Actual native review labels; no changed weights or geometry."""
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import json
from audit_grayT123_camera_support import REC,ROOT,read,save,sha
OUT=REC/'grayT-visibility-material-review';H=Path(__file__).parent
if (OUT/'actual-review.json').exists():raise RuntimeError('Actual review already recorded.')
r=read(OUT/'result.json');stamp=datetime.now(timezone.utc).isoformat();labels=[]
for o in r['observations']:
    f,j=o['frame_number'],o['IBUG']
    if f<=57:label='visible_nose_region_not_material_identity'
    elif f<=65:label='uncertain_far_side_semantic_point' if j in [34,35] else 'visible_nose_region_not_material_identity'
    elif f<=69:label='visible_nose_region_not_material_identity' if j in [31,32,33] else 'boundary_or_inferred_semantic_point'
    else:label='visible_nose_region_not_material_identity' if j in [31,32] else ('uncertain_far_side_semantic_point' if j==33 else 'boundary_or_inferred_semantic_point')
    if not o['person_mask_nearest_foreground']:label='boundary_or_inferred_semantic_point'
    labels.append(dict(frame_number=f,IBUG=j,label=label,physical_material_identity_certified=False,
        location_accuracy_certified=False,uses_projection_error=False))
assert len(labels)==369 and len({(o['frame_number'],o['IBUG']) for o in labels})==369
summary=('已逐张看完41帧鼻部原图/原点对照及41帧头部原生裁片。前段代理大体位于可见鼻区，后段远侧鼻翼与鼻梁逐渐贴近轮廓或成为遮挡推断；'
    '这些只是图像可见区域标签，没有把平滑鼻部上的语义编号认证成同一皮肤材料。369个原鼻部观测全部保持正权重，其中12个位于原人物mask外，'
    '距离0.786–3.660像素，仍被原4像素边界余量保留，权重0.748–0.979；mask本身也不是精细鼻部真值。'
    '另从实际RTMW解码及全部坐标核实：人体先压到288×384，脸部ROI在网络输入中位只有29.75像素高；'
    '单次argmax解码原生步长约2.82像素，翻转平均后落在约1.41像素的格点上，369点距解析格点最大仅3.23e-13像素。'
    '这不是亚像素物理观测，能够引入跳动；它不能单独解释先前18.83像素静态残差。'
    '原生图中另辨认到眼外上方颞部独立色素斑，与原眼外侧标记分离，41帧均未见明显遮挡。下一步可做一次独立测量诊断，不能用鼻部低残差或当前mask直接删除观测。')
review=dict(created=stamp,actual_visual_review=True,all41_nose_pairs_and41_heads_actually_reviewed=True,
    all369_points_labeled=True,labels=labels,counts=dict(Counter(o['label'] for o in labels)),summary=summary,
    labels_are_conservative_visual_regions_not_anatomical_ground_truth=True,
    no_observations_removed_or_reweighted=True,no_new_features_or_fits=True,quality_accepted=False,
    source_sha256=sha(OUT/'result.json'),
    new_material_inventory=[dict(id='temple_pigment_M1',seed_frame=54,seed_native_xy=[792.5,965.5],
        description='原眼外侧色素标记上方约30px的独立小深色颞部斑。不是眉毛边缘，非鼻/眼语义编号。',
        fixed_view_inventory=list(range(40,81)),identity_visually_plausible=True,subpixel_center_not_measured=True,
        support_limit='只认证可继续测量的候选，未证明皮肤刚性或亚像素真值；与旧点近邻，测量误差可能相关。'),
        dict(id='neck_pigment_candidates',geometry_group='neck_not_rigid_head',
            description='颈部若干斑点前段可见，侧面和衣领遮挡导致完整41帧支持不足，不用作头部刚体真值。',usable_for_registered_head_test=False)])
save(OUT/'actual-review.json',review)
idx=read(OUT/'index.json')
for s in r['sheets']:idx['images'].append(dict(frame=str(s['frames']),region='候选皮肤材料支持的原生头部清点',file=s['head'],columns=['native384px, no resampling']))
save(OUT/'index.json',idx)
save(REC/'grayT-proxy-visibility-material-completion.json',dict(completed=stamp,plan=str(REC/'grayT-proxy-visibility-material-plan.json'),
    result=str(OUT/'result.json'),actual_review=str(OUT/'actual-review.json'),observations=369,new_features=0,new_pose_fits=0,quality_accepted=False))
print(json.dumps(dict(summary=summary,labels=review['counts']),ensure_ascii=False))
