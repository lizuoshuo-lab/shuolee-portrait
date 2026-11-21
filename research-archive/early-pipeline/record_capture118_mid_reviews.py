"""Persist human-visible image review already performed in this turn."""
from pathlib import Path
import json,hashlib,datetime
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118/all-persons-20261027')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
d=R/'sofa-gaussian-012000-review';idx=read(d/'index.json')
save(d/'actual-review.json',dict(updated=now,step=12000,snapshot_sha256=idx['snapshot_sha256'],actual_same_camera_review=True,
    reviewed_images=[x['file'] for x in idx['images']],quality_accepted=False,
    summary='已实际审阅六个固定相机的全身与面部三列图。坐姿与整体位置基本一致；正面五官更集中，但左右侧鼻唇、眼镜边缘仍有散边，头发外圈、袖口、衬衣背部、裤腿与脚趾仍有拖影或破碎。允许已授权同一18k预算结束，不能称重影修复。',
    continue_existing_budget=True,new_training_authorized=False,physical_calibration_accepted=False))
d=R/'indoor118-external-occluders';result=read(d/'result.json')
review=dict(updated=now,actual_pixel_review=True,images=[f'review/frame_{i:06d}_candidate_ignore_overlay.png' for i in [1,18,36,53,71,89]],
    candidate_masks_usable_for_occlusion_training=False,allow_new_training=False,
    summary='六张原图叠加已逐张审阅：候选大多为人物后方沙发，18和71帧遮挡小腿的小圆桌几乎未被包含；36帧仅含一处桌面。当前2D家具标签没有前后深度依据，直接忽略会放松大量真实背景而没有覆盖主要腿部遮挡。未应用到训练。',
    positive_person_pixels_removed=sum(x['person_positive_removed'] for x in result['records']),
    frame_counts_with_class60=sum(any(o['class_id']==60 and o.get('included') for o in x['objects']) for x in result['records']),
    material_limitation='Mask zero may be true background, an external occluder or camera/fit error. Classifying furniture is insufficient to distinguish them.',
    remaining_requirements=['可靠的小圆桌前景遮挡及深度关系','腿脚实拍覆盖','相机与人体微动独立验证'],
    training_input_changed=False,camera_changed=False,old_18k_result_preserved=True)
save(d/'actual-review.json',review)
st=read(R/'indoor118-state.json');st.update(updated=now,state='occlusion_candidate_rejected',training_running=False,
    summary=review['summary'],next_action='Preserve old18k; no blind new run using rejected masks.',evidence=str(d/'actual-review.json'))
save(R/'indoor118-state.json',st)
ledger=read(R/'evidence-ledger.json')
if isinstance(ledger,dict):
    ledger['indoor118_external_occluder_review']=review
    ledger['sofa_12000_review']=str(R/'sofa-gaussian-012000-review/actual-review.json')
    save(R/'evidence-ledger.json',ledger)
print('Actual mid-stage and rejected occluder reviews recorded.')
