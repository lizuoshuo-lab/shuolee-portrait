from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
H=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'run').exists() and not (O/'execution-admission.json').exists()
p=json.loads((O/'regions.json').read_text(encoding='utf-8'))
save(O/'region-review.json',dict(updated=datetime.now(timezone.utc).isoformat(),actual_saved_visual_review=True,
 raw_overview_pages=list(range(8)),final_annotation_pages=list(range(8)),corrected_page01_rechecked=True,
 regions_sha256=sha(O/'regions.json'),safe_for_candidate_extraction=True,
 scope='24 images reviewed. Head boxes cover visible heads, body boxes contain visible person. Page01 TV exclusion was corrected and checked before extraction. Masks are conservative regional budgets, not physical identities or segmentation truth.',
 limitations=['Plants occlude torso/legs and sometimes lower face.','Glass reflections/screens excluded where observed; remaining non-person pixels not certified static.','Body/head boxes can contain foreground occluders and background. Every proposed camera point needs physical review.','Many feet lie outside image or behind plants; no negative empty-pixel supervision inferred.','33 diagnostic frames do not establish all860frame alignment.'],
 causal_boundary=p['causal_boundary'],images={str(r['frame']):r['source_sha256'] for r in p['rows']},pages={f'regions_{j:02d}.jpg':sha(O/f'regions_{j:02d}.jpg') for j in range(8)}))
save(O/'execution-admission.json',dict(registered=datetime.now(timezone.utc).isoformat(),matcher_sha256=sha(H/'match_video10_temporal_bridge123.py'),baseline_module_sha256=sha(H/'match_video10_material123.py'),regions_sha256=sha(O/'regions.json'),plan_sha256=sha(O/'plan.json'),region_review_sha256=sha(O/'region-review.json'),
 new_feature_passes=1,new_frames=24,cached_frames=9,new_pairs=63,cached_pairs=36,camera_operations=0,GPU_tasks=0,CPU_deadline_seconds=600,training=False,
 observation_review='One representative per selected family; every observation shown as unresized128px crop, wrapped to9columns. Up to113rows. All conflicts and inherited negative reviews retained. Whole-family FIT/HELD collisions quarantined.',
 inference_boundary='Candidate extraction only. No K, pose, SMPL or Gaussian change. Temporal and pre-extraction allocation changes jointly observed; no unique causal attribution.',state='admitted_once'))
print('24 new frames admitted once; old9frames and36pairs immutable; no geometry or training.')
