"""Record the actual T5 image review and its concrete segmentation defects."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/robust-portrait-122/native-supervision';B=O/'review-inputs'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def windows(p):return Path(p.replace('/mnt/d/','D:/').replace('/mnt/c/','C:/'))
assert not (O/'review.json').exists()
rows=read(O/'mask-rows.json');r=read(O/'result.json');summary=read(B/'summary.json')
assert len(rows)==464 and summary['pages']==15 and summary['native_detail_frames']==5
# Coordinates are selected from actually viewed native crops, not residuals.
samples=[]
for f,xy in [(0,(394,387)),(348,(382,371))]:
 row=rows[f];x0,y0,x1,y1=row['mask_bbox_xyxy'];cx=(x0+x1)//2;waist=int(y0+.52*(y1-y0))
 left=max(0,cx-384);top=max(0,waist-384);x,y=xy[0]+left,xy[1]+top
 mask=np.array(Image.open(windows(row['alpha_proposal']['path'])))
 rgb=np.array(Image.open(windows(row['rgb']['path'])).convert('RGB'))
 samples.append(dict(frame_id=f,native_xy=[x,y],crop_xy=list(xy),rgb=rgb[y,x].tolist(),alpha=int(mask[y,x]),
  semantic_review='Visible opaque whiteT fabric below/beside hand in unchanged source; alpha incorrectly zero.',image=f'hands_clothing_{f:06d}.png'))
 assert mask[y,x]==0,('Selected point is not the visually identified hole',f,x,y)
im=Image.new('RGB',(1824,541))
for j,f in enumerate([0,232,463]):im.paste(Image.open(B/f'full_{f:06d}.png'),(j*608,0))
im.save(B/'report-comparison.png')
findings=[
 '464个轮廓缩略图中未见整个人物丢失或切换为背景行人的粗错误。全部mask未触碰画框，5帧原生细节中脸、手、鞋的主体保留。此结论限于粗轮廓与已审阅裁片。',
 '第0帧和第348帧，手旁白T恤内部出现误删小洞。原图是连续布料，二值alpha却给出0。具体原生像素位置和颜色已保存，未自动填洞或改原图。',
 '第348帧发束与颈部之间仍残留绿色背景。头发细丝、衣袖/鞋边的混色也未完成逐像素验收；高SAM分数不能代表边界小于1像素。',
 '48帧检测到多个候选人物。连续框身份检查没有中断，5张完整原图中背景人物未并入主人物，但这不等于全段外部遮挡已经验证。',
 '相邻mask最低IoU为0.9264，发生在第429到430帧。这个数混合了相机运动、人物变化和分割变化，不能据此宣称真实人物静止或相机正确。',
 '整段视角从一侧经过正面到另一侧，没有完整背面观测。5张原图中的面部/肩部关系需要结合稳定场景与固定材料轨迹进一步检查；仅凭这些不同视角不能量化人物微动。']
review=dict(stage='T5',updated=datetime.now(timezone.utc).isoformat(),actual_review_complete=True,
 state='native_preparation_reviewed_segmentation_defects_retained',native_frames=464,alpha_proposals=464,
 all_silhouettes_coarsely_reviewed=464,coarse_pages=15,native_detail_frames=[0,116,232,348,463],
 native_detail_regions=['head','hands_clothing','feet'],complete_original_and_cutout_pairs_reviewed=5,
 coarse_no_whole_subject_loss_observed=True,physical_supervision_accepted=False,source_pixel_identity_passed=True,
 observations=samples,findings_zh=findings,
 hair_background_leak=dict(frame_id=348,image='head_000348.png',location='gap between ponytail and neck',
  finding='Green background pixels retained in the foreground proposal; no manual correction applied.'),
 minimum_adjacent_mask_iou=summary['minimum_adjacent_mask_iou'],minimum_iou_frame=summary['minimum_iou_frame'],
 multi_detection_frames=len(r['multi_detection_frames']),frozen7_byte_identical=True,
 K_changed=False,poses_changed=False,physical_tracks_changed=False,person_fit=False,GS_training=False,
 registered_training_frames=464,full_camera_registration=False,old_diagnostic_cameras=7,
 heldout_face_max_native_px_unchanged=13.04875764,physical_admission=False,
 limitations=['No all464frame native-edge groundtruth; thumbnail inspection only screens gross errors.',
 'No allvalid observation masks generated; unknown/occlusion/colorsupervision unresolved.',
 'Masks cannot establish camera calibration or static material identity; no1px orghosting improvement claim.',
 'Same5source views contain no fullback; no recovery claim for unobserved regions.'],
 evidence={str(p.relative_to(O)):sha(p) for p in [O/'plan.json',O/'frame-plan.json',O/'frame-clock.json',O/'result.json',O/'decode-receipt.json',B/'index.json',B/'summary.json',*sorted(B.glob('*.png'))]})
save(O/'review.json',review)
print(json.dumps(dict(state=review['state'],native_defect_samples=samples,multi_detection_frames=review['multi_detection_frames']),ensure_ascii=False))
