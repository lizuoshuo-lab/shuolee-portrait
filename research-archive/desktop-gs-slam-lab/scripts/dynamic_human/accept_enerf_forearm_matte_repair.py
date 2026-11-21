"""Independently reconstruct all proposed matte changes before accepting three."""
from pathlib import Path
from datetime import datetime, timezone
import json,hashlib,shutil,numpy as np
from scipy.ndimage import binary_erosion
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-forearm-matte-repair-072';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (R/'residual_graph/completion.json').exists()
protocol=read(R/'protocol.json');prep=read(R/'preparation.json');records=[]
for row in prep['records']:
 t=row['frame'];source=I/'native-input'/f'camera01-frame{t:06d}.npz';candidate=R/'candidates'/f'camera01-frame{t:06d}.npz'
 assert sha(source)==row['source_sha256'] and sha(candidate)==row['candidate_sha256']
 z=np.load(source);c=np.load(candidate);assert z['rgb'].shape==(1080,1920,3)
 core=binary_erosion(c['raw_mask'],structure=np.ones((3,3)),border_value=1)
 x0,y0,x1,y1=protocol['region_xyxy'];region=np.zeros_like(core);region[y0:y1,x0:x1]=True
 expected=z['matte'].copy();expected[core&region&z['valid']]=1
 assert np.array_equal(expected,c['matte']) and np.array_equal(c['changed'],expected>z['matte'])
 assert np.array_equal(c['matte'][~region],z['matte'][~region]) and np.isfinite(c['matte']).all()
 assert int(c['changed'].sum())==row['changed_pixels']
 records.append(dict(**row,accepted=t in [750,752,754],source_unchanged_sha256=sha(source)))
v=dict(status='all7candidate_panels_actually_viewed_three_accepted_before_training',reviewed_utc=datetime.now(timezone.utc).isoformat(),reviewed_files=[f'candidates/camera01-frame{r["frame"]:06d}-review.png' for r in records],accepted_frames=[750,752,754],narrative_zh='第750、752、754帧原照片中可见的前臂和手被原matte明显删去，候选恢复了这些可见区域；未看到明显将墙面作为前景补入。其余四帧原覆盖较完整，保留原matte，避免无必要的边缘改动。固定区域也包含少量袖口和衣服边缘，并非纯皮肤标注。',qualification='Visual acceptance of predicted labels, not ground-truth segmentation. No reconstruction outcome yet.',goal_complete=False)
save(R/'mask-visual-review.json',v)
audit=dict(status='all7candidate_files_hashes_and_SciPy_recomputed_local_mattes_verified',records=records,accepted_frames=v['accepted_frames'],accepted_changed_pixels=sum(r['changed_pixels'] for r in records if r['accepted']),accepted_raised_from_below_half=sum(r['raised_from_below_half'] for r in records if r['accepted']),rgb_and_original_sources_unmodified=True,rule='Original maximum with3x3eroded SAM mask inside fixed native ROI and valid mask; SciPy independently reproduced OpenCV morphology.',goal_complete=False)
save(R/'mask-independent-audit.json',audit);shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps({k:v for k,v in audit.items() if k!='records'}))
