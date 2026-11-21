"""Record actual full-strip review and freeze one real consumer replay."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';F=REC/'field-preflight'
B=F/'native-material/topology-review';OUT=B/'background-full-review'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'review.json').exists()
index=read(OUT/'index.json');assert sum(len(r['views']) for r in index['rows'])==102
notes={
4:('ground_straw','相同地面枯草与石块的局部构型在3帧中可辨。精确草茎交点的定位与微动未独立测量。'),
5:('building_foliage_boundary','窗框与建筑区域可以对应，但中心查询靠近树叶和白色建筑边界，遮挡及混合深度使具体材料身份仍不确定。'),
9:('small_plant','灌溉管旁的植株区域可持续辨认，中心落在细小叶片附近，不能据此证明叶片静止或亚像素位置不变。'),
10:('irrigation_ground','3帧中可辨相同灌溉设施与地面构型；具体中心点的独立定位和静止误差尚未测量。'),
13:('ground_texture','13帧的土壤及邻近植株构型相似，中心位于弱纹理土面；尚不能逐像素确认同一个表面纹理点。'),
14:('tube_soil_boundary','18帧可见同一管线与周围支架，但中心在管线边界附近。轮廓和高光不能直接当作同一固定表面点。'),
19:('foliage_sky_boundary','3帧中心位于树冠与天空边界，局部叶片重叠且模糊。不能将天空空隙或叶片轮廓当作固定背景真值。'),
20:('small_plant','19帧中灌溉接头和附近植株构型可辨，中心落在小叶片簇附近。材料点身份和植物微动仍未独立确认。'),
25:('ground_stone','11帧中土块、暗缝与邻近枯枝的局部构型可持续辨认，是静止地面候选；精确点坐标和亚像素不确定性仍未测量。'),
26:('ground_plant_texture','7帧可辨管线、支架和地面植株布局，中心落在重复的土面与小叶片纹理附近，未取得独立点级身份。'),
}
rows=[]
for r in index['rows']:
    assert sha(OUT/r['file'])==r['sha256']
    obj,note=notes[r['review_row']]
    rows.append(dict(row=r['review_row'],category='background_query',object_context=obj,
        segment_id=r['segment_id'],family_id=r['family_id'],split=r['split'],development_only=r['development_only'],
        classification='unresolved_physical_identity',note=note,all_nodes_actually_reviewed=True,
        exact_material_accepted=False,staticity_verified=False,localization_uncertainty_px=None,
        actual_saved_views_reviewed=[dict(slot=v['slot'],source_frame=v['source_frame'],keypoint=v['keypoint'],uv=v['uv_native']) for v in r['views']],
        reviewed_nodes_count=len(r['views']),total_nodes=len(r['views']),
        reviewed_image=r['file'],reviewed_image_sha256=r['sha256'],
        supporting_evidence=[dict(path='index.json',sha256=sha(OUT/'index.json'))]))
stamp=datetime.now(timezone.utc).isoformat()
save(OUT/'review.json',dict(updated=stamp,actual_saved_pixels_reviewed=True,all10_native_sheets_reviewed=True,
    reviewed_native_views=102,rows=rows,confirmed_identity_contradictions=0,
    conclusion='The same local object/region can be recognizable without confirming a unique stationary material point. These ten previously opened segments do not establish177-frame calibration or subpixel truth.',
    limitations=['This is the fixed10-family development sample, not a representative estimate of all46299 eligible segments.',
        'Uncertainty is not a negative identity result. No arbitrary coordinate corrections or residual-based deletions were performed.',
        'Tree/plant motion is not numerically measured here. It is a staticity concern, not a proven unique cause of portrait ghosting.'],
    all_original_coordinates_preserved=True,camera_estimated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False))
save(OUT/'review-result.json',dict(updated=stamp,state='bounded_full_view_review_complete',families=10,observations=102,
    unique_source_frames=76,all_selected_views_reviewed=True,confirmed_static_material_families=0,
    confirmed_identity_conflicts=0,independent_subpixel_measurements=0,
    background_fit_admitted=False,camera_solver_calls=0,SMPLX_fit_calls=0,training_calls=0,
    interpretation='More views resolve the scope of support but do not make coarse object recognition an exact stationary material measurement.',
    next='Reuse the existing reviewed observation entry, not the legacy extractor; one bounded real-field replay must preserve every native coordinate and original split and deny unsupported camera solve.'))
save(OUT/'consumer-plan.json',dict(registered=stamp,purpose='One real-field input replay through the already implemented reviewed camera entry. No new entry, no repeated synthetic controls.',
    sources={name:sha(H/name) for name in ['local_surface123.py','local_surface123_camera_reviewed.py','local_surface123_frozen_observations.py']},
    review_sha256=sha(OUT/'review.json'),index_sha256=sha(OUT/'index.json'),
    source_video=read(F/'plan.json')['verified_source_path'],subject=read(F/'plan.json')['source'],
    budget=dict(passes=1,CPU_seconds=120,threads=2,GPU=0,detector=0,matcher=0,geometry=0,training=0),
    assertions=['Consume actual field177-frame immutable tracks and this102-view review.',
        'Exactly preserve observed native pixels, source identities, original family and FIT/HELD.',
        'No unknown claim becomes static background. No development family becomes a new independent holdout.',
        'Unsupported solver entry must fail before creatingfield runroot or importing geometric solver.'],
    stop='If data binding fails, save error. Do not rebuild extraction, coerce labels, lower thresholds or run a camera.'))
print(json.dumps(dict(review=str(OUT/'review.json'),reviewed=102,accepted_static_families=0,consumer_plan_registered=True),ensure_ascii=False))
