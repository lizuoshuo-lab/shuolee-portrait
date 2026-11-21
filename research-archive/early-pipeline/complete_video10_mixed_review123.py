from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
import json,hashlib
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'mixed-region-cases/visual-review.json').exists()
notes=['水烟器支架的固定物体局部，被头部框和身体框覆盖；反光影响点位精度未定。','拱门上部角落处于头部矩形内，但实际位于可见背景。','水烟器竖杆局部属于背景，人体矩形含有这片背景。','水烟器弯管区域是背景物体；管的轮廓不保证是固定材料点。','地砖/设备底部局部落入身体矩形；阴影及重复地纹仍需物理身份核验。','地砖局部被身体矩形误包；不能据此认证亚像素材料对应。','水烟器顶部局部被头部矩形误包；金属表面精度未定。','水烟器托盘局部被头部矩形误包；光泽与遮挡仍保留不确定性。']
rows=[]
for x,n in zip(read(O/'mixed-region-cases/index.json')['rows'],notes):
 rows.append(dict(case=x['case'],family=x['family'],track=x['track'],frames=x['cell_frames'],roles=x['cell_roles'],note=n,nonperson_location_visually_confirmed=True,bbox_role_not_semantic_identity=True,physical_point_accuracy_verified=False,camera_promoted=False))
summary='第349帧的38条带背景观测轨迹中，36条所在族跨过背景/人体ROI标签而被筛除。按预登记规则追加审阅8族，8族的所示位置均是水烟器、拱门或地砖等背景区域，曾被外接矩形误标为人体/头部ROI。矩形区域不能充当语义分割；这是当前诊断中的约束丢失来源，不是旧版SMPL-X全部错位的唯一归因。'
save(O/'mixed-region-cases/visual-review.json',dict(updated=datetime.now(timezone.utc).isoformat(),summary=summary,actual_saved_visual_review=True,reviewed_rows=8,rows=rows,total_native_context_budget_used=16,raw_features_or_splits_changed=False,camera_promotions=0,scope='Diagnostic case set selected by frame349 and longest track, not random prevalence estimate; metallic highlights, shadow, contour and precise physical identity remain unverified.'))
with Image.open(O/'mixed-region-cases/case_01_family_565.png') as im:im.crop((0,0,1152,384)).save(O/'bbox-background-evidence.png')
r=read(O/'result-reviewed.json');r.update(mixed_role_review=str(O/'mixed-region-cases/visual-review.json'),additional_finding=summary,reviewed_rows_total=104,contexts_total=16,next='Register one mask-based role diagnostic with frozen cached observations, graph and FIT/HELD splits; no geometry or retraining yet.');save(O/'result-reviewed.json',r)
print(summary)
