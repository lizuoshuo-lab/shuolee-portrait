"""Persist actual inspection of all177 native head panels and fixed bindings."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'indoor-preflight';OUT=PRE/'semantic-face-all'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (OUT/'visual-review.json').exists()
r=read(OUT/'result.json');sheets=read(OUT/'review-sheets.json');meta=read(OUT/'observations.json')
z=dict(np.load(OUT/'observations.npz',allow_pickle=False));anchor=read(REC/'direct-smplx-bindings/bindings.json')
byid={b['ibug68_index']:b for b in anchor['anchors']}
assert r['frames']==177 and r['new_network_forwards']==344 and r['reused_predictions_byte_equal']
assert r['observations_sha256']==sha(OUT/'observations.npz') and len(sheets['sheets'])==15
assert np.array_equal(z['slots'],np.arange(177))
names={30:'鼻尖',33:'鼻底中央',36:'人物右眼外眼角',39:'人物右眼内眼角',42:'人物左眼内眼角',45:'人物左眼外眼角',48:'人物右嘴角',54:'人物左嘴角'}
all8=set(names);right={30,36,39,48};left={30,42,45,54};farleft={42,45,54};farright={36,39,48}
# Conservative human image observations from the15sheets actually opened.
# These ranges compress per-frame annotations; they are not computed from pose,
# confidence, residuals, model projections or the previous field subject.
ranges=[
 (0,11,all8,set(),'近正面，所选八个解剖部位可见；像素精度未独立测量。'),
 (12,35,right,set(),'向画面右侧转，近侧为人物右眼和右嘴角。远侧及鼻底中央保留未知。'),
 (36,47,right,farleft,'右侧面，远侧左眼和左嘴角遮挡；鼻底中央未知。'),
 (48,50,{30},farleft,'向右后方过渡，鼻尖轮廓仍可辨；近侧眼角和嘴角的完整身份保留未知。'),
 (51,67,set(),farleft,'右后方过渡，部分预测转移到耳和发际线。近侧点无法确认，保留未知。'),
 (68,99,set(),all8,'后脑勺，所选八个面部部位均不可见。'),
 (100,113,set(),farright,'左后方过渡，近侧点在耳、脸侧和背景之间预测，身份保留未知。'),
 (114,119,{30},farright,'鼻尖轮廓重新可辨，眼角和嘴角仍处于遮挡过渡，近侧保留未知。'),
 (120,137,left,farright,'左侧面，近侧人物左眼和左嘴角可见，远侧右眼和右嘴角遮挡。鼻底中央未知。'),
 (138,161,left,set(),'回到左前方，近侧可辨，远侧与鼻底中央暂不作已验证观测。'),
 (162,176,all8,set(),'近正面，八个重点部位可见；预测位置不是独立像素真值。')]
labels={};views=[];points=[];records=[]
for sh in sheets['sheets']:
    assert sha(OUT/sh['file'])==sh['sha256']
    for f in sh['frames']:
        s=f['slot'];matching=[a for a in ranges if a[0]<=s<=a[1]];assert len(matching)==1
        rr=matching[0];vis,occ=rr[2:4];assert not vis&occ
        states={i:('observed' if i in vis else 'occluded' if i in occ else 'unknown') for i in names};labels[s]=states
        views.append(dict(**f,evidence=sh['file'],evidence_sha256=sh['sha256'],actual_native_panels_reviewed=True,
            selected_point_visibility=states,note=rr[4],physical_identity_precision_verified=False))
        assert meta['rows'][s]['frame']==f['frame'] and int(z['source_indices'][s])==f['source_frame']
        for i in range(17,68):
            state=states.get(i,'unknown')
            row=dict(slot=s,source_frame=int(z['source_indices'][s]),frame=f['frame'],anchor=byid[i],
                uv_native=z['fan_native_continuous'][s,i].tolist(),coordinate_domain='native_integer_pixel_centres',coordinate_dtype='float64',
                detector='Existing3DFAN4, unchanged official crop and heatmap decoder; continuous inverse',visibility=state,
                semantic_region_reviewed=i in names,semantic_region_supported=state=='observed',
                visibility_evidence=sh['file'] if i in names else None,coordinate_localization_uncertainty_px=None,
                physical_family_id=f'indoor:anatomical:{i}',split='UNASSIGNED',independent_physical_truth=False,training_eligible=False)
            records.append(row)
            if i in names:points.append(dict(slot=s,source_frame=f['source_frame'],frame=f['frame'],ibug=i,semantic_label=names[i],anchor=byid[i],
                fan_uv_continuous=row['uv_native'],fan_uv_official=z['fan_native_official'][s,i].tolist(),anatomical_visibility=state,
                review_evidence=sh['file'],coordinate_localization_uncertainty_px=None,subpixel_accuracy_verified=False))
assert len(labels)==177 and len(points)==1416 and len(records)==9027
old=read(PRE/'semantic-face-binding/visual-review.json')
assert all(labels[q['slot']][q['ibug']]==q['anatomical_visibility'] for q in old['points'])
counts=dict(Counter(p['anatomical_visibility'] for p in points));fullcounts=dict(Counter(p['visibility'] for p in records))
stamp=datetime.now(timezone.utc).isoformat()
review=dict(updated=stamp,actual_images_reviewed=True,reviewed_images=sheets['sheets'],source_views=177,
    selected_points_per_view=8,selected_observations_reviewed=1416,unique_anatomical_anchors_reviewed=8,
    review_counts=counts,views=views,points=points,
    limits=['All177head panels actually inspected; the other43 fixedface landmarks remain unknown.',
            'Observed means the anatomical region is visible, not that the detector coordinate is independently accurate.',
            'No error-based filtering, temporal smoothing, camera update or image warp.',
            'No staticity or unique ghosting-cause claim follows from semantic visibility.',
            '177 decoded frames are not177 registered cameras.'],
    training_started=False,physical_truth=False,quality_accepted=False)
save(OUT/'visual-review.json',review)
save(OUT/'real-2d-smplx-bindings.json',dict(updated=stamp,template_sha256=anchor['template_sha256'],
    bindings_sha256=sha(REC/'direct-smplx-bindings/bindings.json'),observations_sha256=sha(OUT/'observations.npz'),
    reviewed_points=1416,fixed_face_observations=9027,records=records,
    geometry='Fixed official anatomical surface anchors only; no real fitted individual or accepted cameras.',training_admitted=False))
summary=dict(updated=stamp,state='complete_native_visibility_review',frames=177,reused_original_views=5,new_network_forwards=344,
    review_sheets=15,selected_review_counts=counts,full_record_visibility_counts=fullcounts,
    frames_with_visible_selected_regions=sum(any(v=='observed' for v in states.values()) for states in labels.values()),
    selected_anatomical_anchors=8,fixed_face_observations=9027,independently_localized_physical_ground_truth_observations=0,
    confirmed_input_risk='Occluded facial regions still receive semantic predictions on ears, hair and background. Such predictions cannot constrain the corresponding SMPL-X facial surface.',
    direct_bindings_saved=True,camera_updated=False,SMPLX_fitted=False,training_started=False,quality_accepted=False)
save(OUT/'review-result.json',summary)
fig=Image.new('RGB',(576,384),'white')
for col,slot in enumerate([0,56,88]):
    sh=next(x for x in sheets['sheets'] if x['first_slot']<=slot<=x['last_slot']);cell=slot-sh['first_slot'];x=(cell%3)*384;y=(cell//3)*192
    with Image.open(OUT/sh['file']) as im:
        for row in [0,1]:fig.paste(im.crop((x+192*row,y,x+192*(row+1),y+192)),(col*192,row*192))
fig.save(OUT/'front-transition-back-native.png')
print(json.dumps(summary,ensure_ascii=False))
