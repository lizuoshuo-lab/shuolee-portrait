"""Record the completed native-image review, without inventing camera admission."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import json, hashlib
import numpy as np
from PIL import Image

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
OUT=REC/'field-preflight/semantic-face-all'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

assert not (OUT/'visual-review.json').exists(), 'Review already recorded; do not overwrite'
r=read(OUT/'result.json'); sheets=read(OUT/'review-sheets.json')
meta=read(OUT/'observations.json')
z=dict(np.load(OUT/'observations.npz',allow_pickle=False))
anchors=read(REC/'direct-smplx-bindings/bindings.json')
byid={q['ibug68_index']:q for q in anchors['anchors']}
assert r['frames']==177 and r['new_network_forwards']==344 and r['reused_predictions_byte_equal']
assert r['observations_sha256']==sha(OUT/'observations.npz')
assert len(sheets['sheets'])==15 and np.array_equal(z['slots'],np.arange(177))
stamp=datetime.now(timezone.utc).isoformat()
names={30:'鼻尖',33:'鼻底中央',36:'人物右眼外眼角',39:'人物右眼内眼角',
       42:'人物左眼内眼角',45:'人物左眼外眼角',48:'人物右嘴角',54:'人物左嘴角'}
all8=set(names); right={30,36,39,48}; left={30,42,45,54}
farleft={42,45,54}; farright={36,39,48}
# These are conservative annotations from the 15 native sheets actually read.
# They are NOT automatic inferences from residual, confidence, or head angle.
ranges=[
 (0,17,all8,set(),'正面到轻侧面，八个所选部位可辨；坐标精度未测量。'),
 (18,35,right,set(),'向画面右侧转，近侧为人物右眼与右嘴角；其余保留未知。'),
 (36,49,right,farleft,'右侧面，远侧左眼与左嘴角遮挡；鼻底中央未知。'),
 (50,69,set(),farleft,'右侧后方过渡；预测开始落在耳、鬓角附近，近侧身份也保留未知。'),
 (70,107,set(),all8,'后脑勺，所选八个面部部位全部遮挡。'),
 (108,123,set(),farright,'左侧后方过渡；近侧材料位置尚不能可靠认定。'),
 (124,137,left,farright,'左侧面，近侧为人物左眼与左嘴角；远侧右眼与右嘴角遮挡。'),
 (138,155,left,set(),'返回前方，近侧左眼与左嘴角可辨，其余保留未知。'),
 (156,176,all8,set(),'正面，八个所选部位可辨；眼睑开合变化不当作静止表面证据。'),
]
labels={}; frame_reviews=[]; points=[]; all_records=[]
for sheet in sheets['sheets']:
    assert sha(OUT/sheet['file'])==sheet['sha256']
    for f in sheet['frames']:
        s=f['slot']; rr=next(q for q in ranges if q[0]<=s<=q[1])
        vis,occ=rr[2:4]; assert not vis&occ
        states={p:('observed' if p in vis else 'occluded' if p in occ else 'unknown') for p in names}
        labels[s]=states
        frame_reviews.append(dict(**f,evidence=sheet['file'],evidence_sha256=sheet['sha256'],
            actual_native_panels_reviewed=True,selected_point_visibility=states,note=rr[4],
            physical_identity_precision_verified=False))
        for p in range(17,68):
            state=states.get(p,'unknown')
            record=dict(slot=s,source_frame=int(z['source_indices'][s]),frame=f['frame'],
                anchor=byid[p],uv_native=z['fan_native_continuous'][s,p].tolist(),
                coordinate_domain='native_integer_pixel_centres',coordinate_dtype='float64',
                detector='local 3DFAN4 with official crop and quarter-pixel heatmap decoder',
                inverse_transform='continuous; original crop sampling unchanged',
                visibility=state,semantic_region_reviewed=p in names,
                semantic_region_supported=state=='observed',
                visibility_evidence=sheet['file'] if p in names else None,
                coordinate_localization_uncertainty_px=None,
                physical_family_id=f'field:anatomical:{p}',split='UNASSIGNED',
                independent_physical_truth=False,training_eligible=False)
            all_records.append(record)
            if p in names:
                points.append(dict(slot=s,source_frame=int(z['source_indices'][s]),frame=f['frame'],
                    ibug=p,semantic_label=names[p],anchor=byid[p],
                    fan_uv_continuous=record['uv_native'],
                    fan_uv_official=z['fan_native_official'][s,p].tolist(),
                    anatomical_visibility=state,review_evidence=sheet['file'],
                    coordinate_localization_uncertainty_px=None,subpixel_accuracy_verified=False))
assert len(labels)==177 and len(points)==1416 and len(all_records)==9027
counts=Counter(q['anatomical_visibility'] for q in points)
assert counts==dict(observed=568,occluded=496,unknown=352),counts
fullcounts=Counter(q['visibility'] for q in all_records)
assert fullcounts==dict(observed=568,occluded=496,unknown=7963)
old=read(REC/'field-preflight/semantic-face-binding/visual-review.json')
assert all(labels[q['slot']][q['ibug']]==q['anatomical_visibility'] for q in old['points'])
review=dict(updated=stamp,actual_images_reviewed=True,reviewed_images=sheets['sheets'],
    source_views=177,selected_points_per_view=8,selected_observations_reviewed=1416,
    unique_anatomical_anchors_reviewed=8,review_counts=dict(counts),
    views=frame_reviews,points=points,
    eyelid_state_evidence=dict(slots=[[3,4,5],[7,8,9],[165,166,167],[169,170,171]],
        finding='连续原生裁片可见眼睑开合变化；眼角语义与眼睑点需要分别处理。',
        scope='Image observation only; not a quantified motion model or unique ghosting cause.'),
    limits=['Reviewed semantic regions are not independently accurate pixel coordinates.',
        'The other 43 fixed landmarks per frame remain unknown, not automatically visible.',
        'Visibility decisions preserve all original predicted coordinates; no residual pruning.',
        '177 source frames are not 177 accepted camera registrations.',
        'No direct real SMPL-X fitting or Gaussian training has run.'],
    training_started=False,physical_truth=False,quality_accepted=False)
save(OUT/'visual-review.json',review)
save(OUT/'real-2d-smplx-bindings.json',dict(updated=stamp,
    template_sha256=anchors['template_sha256'],
    bindings_sha256=sha(REC/'direct-smplx-bindings/bindings.json'),
    observations_sha256=sha(OUT/'observations.npz'),reviewed_points=1416,
    fixed_face_observations=9027,records=all_records,
    geometry='Official topology anchors only; no fitted real person or accepted cameras.',
    training_admitted=False))
summary=dict(updated=stamp,state='complete_native_visibility_review',frames=177,
    reused_original_views=5,new_network_forwards=344,review_sheets=15,
    selected_review_counts=dict(counts),full_record_visibility_counts=dict(fullcounts),
    frames_with_visible_selected_regions=sum(any(x=='observed' for x in v.values()) for v in labels.values()),
    selected_anatomical_anchors=8,fixed_face_observations=9027,
    independently_localized_physical_ground_truth_observations=0,
    false_visibility_finding='Rear transitions still return eyes/nose/mouth on ear/hair/background; exclude absent observations.',
    eye_motion_finding='Native sequential panels show eye open/closed changes; not isolated as the sole reconstruction error.',
    direct_bindings_saved=True,camera_updated=False,SMPLX_fitted=False,
    training_started=False,quality_accepted=False)
save(OUT/'review-result.json',summary)
# Image evidence for the existing slide: raw row and prediction row; no text,
# stretching, interpolation or new inference. Same 96px crop in each panel.
fig=Image.new('RGB',(288,192),'white')
for col,slot in enumerate([0,52,88]):
    sh=next(q for q in sheets['sheets'] if q['first_slot']<=slot<=q['last_slot'])
    cell=slot-sh['first_slot']; x=(cell%3)*384; y=(cell//3)*192
    with Image.open(OUT/sh['file']) as im:
        for row in [0,1]: fig.paste(im.crop((x+row*192+48,y+48,x+row*192+144,y+144)),(col*96,row*96))
fig.save(OUT/'front-transition-back-native.png')
print(json.dumps(summary,ensure_ascii=False))
