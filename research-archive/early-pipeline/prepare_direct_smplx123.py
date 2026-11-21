"""Register the direct semantic-fitting implementation and close finished CPU inventory."""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
R=REC.parent/'robust-portrait-122'
OUT=REC/'direct-smplx-bindings'
B=REC/'field-preflight/native-material/topology-review'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
stamp=datetime.now(timezone.utc).isoformat()
assert not OUT.exists()
OUT.mkdir()
save(OUT/'plan.json',dict(created=stamp,authorization='要做的是2d的特征点要和3d的smpl-x匹配上',
    objective='Explicit anatomical pixel-to-mesh bindings consumed by the actual fresh SMPL-X fitting loss.',
    baseline_fitter_sha256='fd6f3bb11730a97ea98f48be9dda99481f93b6952f9b9f9da5ede71618fe6db68'.replace('b9f9','b9f'),
    implementation=dict(face='RTMW133 40..90 / IBUG68 17..67 -> fixed SMPL-X triangle+barycentric anchors',
        nonfixed_contour='Excluded, never replace by nearest fixed vertex',
        visibility='Independent reviewed observed/occluded/outside/unknown; missing evidence cannot mean visible',
        loss='Native pixels, recorded uncertainty, group-normalized eyes/nose/mouth/brows, fixed FIT/HELD',
        variables='Existing static body/root/shape/hand plus shared jaw and expression; no camera or image warping',
        release='Separate candidate, frozen historical runtime preserved'),
    validation=dict(kind='known synthetic SMPL-X multiview controls, not a new real calibration or fit',
        independent_reference='Official model output landmarks plus OpenCV projection; new binding uses explicit mesh barycentrics',
        cameras='3 fixed known cameras, deliberately off-centre principal points, 1520x2704 integer centres',
        optimizer='One fixed schedule: Adam 250 updates then LBFGS <=150, no parameter sweep',
        variables='shared global/translation/neck/head/jaw/expression; fixed template shape for the controlled recovery',
        checks=['anchor equivalence to official output','projection equivalence to OpenCV',
                'gradients reach eyes, neck/head, jaw, expression',
                'known target fitting and unseen-camera reprojection',
                'unknown/occluded/HELD targets do not influence fitted objective',
                'reject frame/semantic/physical-family split mismatch'],
        budget_seconds=600,max_gpu_tasks=1,stop_on_failure=True),
    real_data='No real camera/SMPL-X/Gaussian run until verified observations and camera admission. No historical fit restart.',
    no_training=True,quality_accepted=False))
if not (B/'actual-review.json').exists():
    idx=read(B/'native-review/index.json');rows=idx['rows']
    notes={2:'person ROI中心是人物旁的地面',3:'person ROI中心是腿旁地面',
           11:'人物旁地面，不是皮肤点',17:'人物旁地面，ROI变化不能直接认定错误对应',
           18:'鞋旁地面，不能充当足部解剖点',19:'树冠和天空边缘，静止性未知',
           24:'头旁地面，不是人体解剖点'}
    save(B/'actual-review.json',dict(updated=stamp,actual_images_reviewed=True,
        reviewed_sheets=[f'native-review/sheet_{i:02d}.png' for i in range(6)],
        native_crops_reviewed=81,queries=27,images_resized=False,image_text_added=False,
        review=[dict(row=x['row'],family_id=x['family_id'],segment_id=x['segment_id'],
                     actual_three_crops_reviewed=True,all_segment_views_reviewed=False,
                     physical_identity_accepted=False,subpixel_precision_measured=False,
                     note=notes.get(x['row'],'局部纹理/结构有可比较内容；未验证整轨物理身份、静止性及亚像素定位')) for x in rows],
        conclusion='ROI内的纹理点不等于人体解剖点。已看的留出家族只作开发审阅，不能加入FIT。',
        camera_admitted=False,SMPLX_fitted=False,training_started=False,quality_accepted=False))
nexttext='按最新指示实现2D语义点到SMPL-X固定网格点的直接拟合：真实可见性+逐点绑定+实际损失，包含眼周、下颌和表情。先做独立已知几何工程验证，不重复旧拟合或启动未准入人物训练。'
for p in [REC/'queue-state.json',R/'queue-state.json']:
    q=read(p);q.update(updated=stamp,current_stage='direct_2d_3d_smplx_implementation',
        worker_pid=None,worker_command=None,running=None,training_running=False,
        local_GPU_task_active=False,next_plan=str(OUT/'plan.json'),next=nexttext,next_action=nexttext,
        latest_field_material_review=str(B/'actual-review.json'),latest_evidence=str(B/'result.json'))
    save(p,q)
save(OUT/'progress.json',dict(updated=stamp,state='implementation',training_started=False,quality_accepted=False))
print(json.dumps(dict(plan=str(OUT/'plan.json'),field_inventory_complete=True),ensure_ascii=False))
