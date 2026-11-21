"""Save the actual five-view image review and explicit 2D/SMPL-X bindings.

This is an input review, not a fitted human or subpixel ground truth. No camera
or FIT/HELD designation is invented for these new semantic observations.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, hashlib,sys,shutil
import numpy as np

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
OUT=REC/'field-preflight/semantic-face-binding'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

if '--correct-profile-side' in sys.argv:
    assert (OUT/'visual-review.json').exists() and not (OUT/'profile-side-correction.json').exists()
    for name in ['visual-review.json','real-2d-smplx-bindings.json','review-result.json']:
        shutil.copy2(OUT/name,OUT/(Path(name).stem+'-before-profile-correction.json'))
    save(OUT/'profile-side-correction.json',dict(updated=datetime.now(timezone.utc).isoformat(),
        cause='Reviewer confused image-facing direction with the visible anatomical side.',
        correction='Facing image right shows subject right eye/mouth (IBUG36/39/48). Facing image left shows subject left (42/45/54).',
        evidence='Original native head and point panels, standard fixed IBUG/SMPL-X labels. Near eye pair spans the visible eye; far pair collapses at the nose silhouette.',
        scope='Review annotations only. No changed detector pixels, topology indices, camera, model weights, or inference.',
        original_review_sha256=sha(OUT/'visual-review.json'),source_sha256=sha(Path(__file__))))
else:
    assert not (OUT/'visual-review.json').exists()
r=read(OUT/'result.json'); index=read(OUT/'point-review-index.json')
obs=dict(np.load(OUT/'observations.npz'))
anchors=read(REC/'direct-smplx-bindings/bindings.json')
by_id={b['ibug68_index']:b for b in anchors['anchors']}
assert r['network_forwards']==10 and r['original_inputs_unchanged']
stamp=datetime.now(timezone.utc).isoformat()
names={30:'鼻尖',33:'鼻底中央',36:'人物右眼外眼角',39:'人物右眼内眼角',
       42:'人物左眼内眼角',45:'人物左眼外眼角',48:'人物右嘴角',54:'人物左嘴角'}
# These labels record the images actually inspected by the assistant. They are
# not inferred from confidence, residual, the image box, or model visibility.
visible={0:set(names),44:{30,36,39,48},88:set(),132:{30,42,45,54},176:set(names)}
occluded={0:set(),44:{42,45,54},88:set(names),132:{36,39,48},176:set()}
view_notes={0:'正面，眼角、鼻尖鼻底和嘴角可见。精确像素位置仍不是真值。',
    44:'朝画面右侧的侧脸。人物右眼和右嘴角可见，远侧左眼和左嘴角被遮挡。鼻底中央保留未知。',
    88:'后脑勺，八个目标面部部位全部不可见。两个检测器仍给出面部坐标，不能进入拟合。',
    132:'朝画面左侧的侧脸。人物左眼和左嘴角可见，远侧右眼和右嘴角被遮挡。鼻底中央保留未知。',
    176:'正面，眼角、鼻尖鼻底和嘴角可见。精确像素位置仍不是真值。'}
rows=[]
for panel in index['panels']:
    q=dict(panel);slot=q['slot'];point=q['ibug']
    state='observed' if point in visible[slot] else ('occluded' if point in occluded[slot] else 'unknown')
    assert q['anchor']==by_id[point]
    q.update(semantic_label=names[point],anatomical_visibility=state,
        anatomical_identity_review_scope='semantic region and side only; not independently localized physical truth',
        semantic_region_supported=state=='observed',review_evidence=f'points_slot_{slot:03d}.png',
        review_image_sha256=sha(OUT/f'points_slot_{slot:03d}.png'),review_method='actual_native_image_inspection',
        coordinate_localization_uncertainty_px=None,subpixel_accuracy_verified=False,
        eligible_for_training=False,reason='Camera and observation admission incomplete',
        integer_rounding_shift_px=float(np.linalg.norm(np.asarray(q['fan_uv_continuous'])-q['fan_uv_official'])))
    rows.append(q)
counts=Counter(q['anatomical_visibility'] for q in rows)
assert dict(counts)=={'observed':24,'occluded':14,'unknown':2},counts
all_records=[]
for i,slot in enumerate(obs['slots'].tolist()):
    selected={q['ibug']:q for q in rows if q['slot']==slot}
    for point in range(17,68):
        q=selected.get(point)
        state=q['anatomical_visibility'] if q else 'unknown'
        all_records.append(dict(slot=int(slot),source_frame=int(obs['source_indices'][i]),
            anchor=by_id[point],uv_native=obs['fan_native_continuous'][i,point].tolist(),
            coordinate_domain='native_integer_pixel_centres',coordinate_dtype='float64',
            detector='local 3DFAN4 with official crop and quarter-pixel heatmap decoder',
            inverse_transform='continuous; crop sampling unchanged',visibility=state,
            semantic_region_reviewed=q is not None,
            semantic_region_supported=bool(q and q['semantic_region_supported']),
            visibility_evidence=q['review_evidence'] if q else None,
            coordinate_localization_uncertainty_px=None,
            physical_family_id=f'field:anatomical:{point}',split='UNASSIGNED',
            independent_physical_truth=False,training_eligible=False))
assert len(all_records)==255
selected=np.array([q['integer_rounding_shift_px'] for q in rows])
observed=np.array([q['integer_rounding_shift_px'] for q in rows if q['anatomical_visibility']=='observed'])
delta=np.linalg.norm(obs['fan_native_continuous']-obs['fan_native_official'],axis=-1)
review=dict(updated=stamp,actual_images_reviewed=True,
    reviewed_images=[dict(file=n,sha256=sha(OUT/n)) for n in ['five-native-heads.png']+[f'points_slot_{s:03d}.png' for s in obs['slots']]],
    source_views=5,selected_point_triplets=40,review_counts=dict(counts),views=view_notes,
    limits=['These 24 observed semantic locations are not 24 independently accurate coordinates.',
            'The 215 unreviewed fixed-face observations remain unknown.',
            'Side-view nose-base positions remain unknown; no residual-based filtering was used.',
            'FAN versus RTMW changes the detector and crop; differences cannot isolate a camera error.',
            'Continuous coordinates remove rounding only, not detector or camera error.'],
    points=rows,training_started=False,physical_truth=False,quality_accepted=False)
save(OUT/'visual-review.json',review)
save(OUT/'real-2d-smplx-bindings.json',dict(updated=stamp,template_sha256=anchors['template_sha256'],
    bindings_sha256=sha(REC/'direct-smplx-bindings/bindings.json'),observations_sha256=sha(OUT/'observations.npz'),
    reviewed_points=40,fixed_face_observations=255,records=all_records,
    geometry='Official topology anchors only. No fitted real-person geometry or accepted camera in this file.',
    training_admitted=False))
summary=dict(updated=stamp,review_counts=dict(counts),full_record_visibility_counts=dict(Counter(q['visibility'] for q in all_records)),
    all_68_prediction_rounding=dict(count=int(delta.size),max_px=float(delta.max()),median_px=float(np.median(delta)),count_ge1=int((delta>=1).sum())),
    selected_40_rounding=dict(max_px=float(selected.max()),median_px=float(np.median(selected)),count_ge1=int((selected>=1).sum())),
    observed_24_rounding=dict(max_px=float(observed.max()),median_px=float(np.median(observed)),count_ge1=int((observed>=1).sum())),
    causal_finding='int32 inverse conversion alone discards up to 1.394719 native pixels in these predictions. This is coordinate loss, not measured ground-truth error.',
    false_visibility_finding='Back-of-head images still produce face landmarks; confidence or ROI alone cannot enable the face loss.',
    direct_bindings_saved=True,camera_updated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False)
save(OUT/'review-result.json',summary)
print(json.dumps(summary,ensure_ascii=False))
