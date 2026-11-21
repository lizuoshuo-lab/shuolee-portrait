"""Save actual original-image review and selected explanatory evidence after the fixed run."""
from pathlib import Path
import datetime as dt
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
B=TOP/'shuolee';O=B/'current/portrait-alignment-open-source-116';R=O/'roma';D=R/'smoke';S=B/'current/full-scene-localization-074'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
now=dt.datetime.now(dt.timezone.utc).isoformat();result=read(D/'result.json');fixed=read(D/'visual-sample-manifest.json')
assert result['state']=='completed_waiting_original_image_review' and fixed['tracks']==90
notes=[
    ['0: plausible blurred desk feature;12: door frame becomes occluding yellow clothing;23: ceiling point lands on subject head inB.',
     '24 and28: low-texture chin/jaw positions remain uncertain;32: visible lip feature is compatible, not exact physical GT.',
     '33,41,48: glasses-arm or rim locations are visually compatible, but reflections and boundaries do not identify exact surface material.',
     '49: cloth fold is ambiguous;57: cuff detail compatible;64: trouser detail near hand compatible.',
     '65: shoe detail compatible;73: smooth trouser surface ambiguous;80: shoe silhouette changes with view.'],
    ['81: exit-sign region changes to ceiling/speaker;93 and104: background desk features become other people or furniture.',
     '105 and119: chin/lip region compatible but held residuals4.98/4.70px;112: skin-neck boundary remains ambiguous.',
     '120: lens reflection cannot establish facial material;128: rim/temple connection compatible;135: nose feature has6.40px held residual.',
     '136 and151: printed textile details compatible;144: visible hand detail with posture/motion uncertainty.',
     '152: shoe detail compatible;160: trouser edge has uncertain correspondence;167: shoe boundary is view dependent.'],
    ['168: window feature is lost inB;180: shoe belongs to another person, so outside-mask is not static GT;191: person becomes wall.',
     '192: cheek point falls on background inC;199 and206: chin/lip matches compatible with residuals2.58/3.18px.',
     '207: lens/reflection ambiguity;213: glasses bridge compatible, cycle0.14px but held3.28px;219: nose region compatible, cycle0.40px but held4.84px.',
     '220: hand detail uncertain;228 and235: printed shirt/jacket features compatible and small held residuals.',
     '236 and244: folds/occlusion make identity uncertain;251: trouser fold compatible but lacks a unique texture point.'],
    ['252: chair feature changes/occludes;264: repeated window pattern has ambiguous physical identity despite small cycle;275: desk point lands on jacket inB.',
     '276: lip correspondence compatible forward but large backward error;279: changing philtrum/nose contour is ambiguous;282: cheek becomes background inC.',
     '283: glasses/highlight boundary ambiguity;291: upper ear compatible;298: nose falls on background inC.',
     '299 and307: printed letters match forward approximately but backward support fails;314: inner-shirt glyph compatible.',
     '315: fold material uncertain;323: trouser seam compatible;330: hem fold/edge lacks exact material identity.'],
    ['331 and354: background becomes jacket inC;343: repeated ceiling beams remain identity-ambiguous even with low residual.',
     '355,357,358: side-view nose/lip boundaries are visible but their exact material positions are uncertain and backward checks fail.',
     '359: glasses-arm highlight uncertain;363: upper ear compatible with2.55px held residual;367: head point becomes ceiling.',
     '368: self-occluded inner shirt detail;376: pocket-flap corner compatible;383: plain cloth seam ambiguous.',
     '384: shoe marking compatible;392: toe silhouette uncertain;399: seam detail compatible.'],
    ['400: background becomes jacket;412: chair/foreground occlusion confounds identity;423: leaf region compatible, motion not determined.',
     '424: lip silhouette uncertain;426: mouth corner compatible;428: nearby skin feature lacks exact texture identity.',
     '429: hair/temple highlight ambiguous;435 and440: glasses-arm location compatible but precise reflective point uncertain.',
     '441: jacket detail compatible;449: visible knuckle match with motion uncertainty;456: jacket feature compatible.',
     '457 and465: shoe markings compatible;472: smooth fold lacks unique material identity.']]
review=dict(updated=now,visual_review_completed=True,selection='All90preregistered fixed first/middle/last probes reviewed in six full-resolution sheets.',
    images=[f'fixed-original-review-{i+1}.jpg' for i in range(6)],scope='Original source crops, no image warping or Gaussian model change',
    block_notes=[dict(triple=i,region_notes=x) for i,x in enumerate(notes)],
    key_observations=['Occluded face/body/background points can map onto unrelated surfaces. Fixed probes192,282,298,367 show explicit losses to background.',
      'Other fixed face probes have small2Dcycles yet retain several-pixel held3Dprojection residuals. Probes105,119,213,219 are compatible-looking examples, not calibrated physical correspondences.',
      'Outside-target masks contain other people and repeated patterns. Background location alone is not a static correctness label.',
      'Many compatible clothing and shoe matches fail the predeclared0.95confidence threshold. Therefore4/473 is the gate yield, not matching accuracy.'],
    physical_camera_accuracy_verified=False,subject_motion_measured=False,root_cause_resolved=False,quality_accepted=False)
save(D/'material-track-review.json',review)
for row in fixed['rows']:
    row['visual_material_identity']='reviewed_descriptive_only_see_block_notes';row['visual_review_note']=notes[row['triple_index']][row['group']]
fixed['visual_review_completed']=True;save(D/'visual-sample-manifest.json',fixed)

# One explanatory example selected after the complete fixed90 review. Not a new acceptance sample.
with np.load(D/'triple-02.npz') as z:data={k:z[k] for k in z.files}
i=int(np.flatnonzero(data['probe_ids']==213)[0]);pts=data['points'][i];pred=data['projection'][i];frames=data['raw_frames']
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
board=Image.new('RGB',(1150,384),'white');draw=ImageDraw.Draw(board);origins=[]
for j,(xy,frame) in enumerate(zip(pts,frames)):
    origin=np.floor(xy-[140,95]).astype(int);origins.append(origin.tolist())
    with Image.open(S/'images'/f'raw_{int(frame):06d}.png') as im:crop=im.convert('RGB').crop((*origin,*(origin+[280,320])))
    x=38+j*383;y=29;board.paste(crop,(x,y));draw.text((x,3),f'{"ABC"[j]}  raw{int(frame):06d}',font=font,fill='#18384b')
    u,v=xy-origin+[x,y];draw.ellipse((u-5,v-5,u+5,v+5),outline='#009f70',width=2)
    if j==1:
        u,v=pred[j]-origin+[x,y];draw.line((u-6,v,u+6,v),fill='#ed7300',width=2);draw.line((u,v-6,u,v+6),fill='#ed7300',width=2)
    draw.text((x,355),'Held B: 3.28 original px' if j==1 else 'Source correspondence',font=small,fill='#ed7300' if j==1 else '#18384b')
board.save(D/'slide52-evidence.jpg',quality=97)
save(D/'slide-example.json',dict(probe_id=213,triple=2,selection='Explanatory example after full fixed review; not used to alter acceptance',crop_origins=origins,
    original_crop_shape=[320,280],no_crop_rescale=True,cycle_raw_px=float(data['cycle'][i]),held_raw_px=float(np.linalg.norm(pred[1]-pts[1])),confidence=float(data['confidence'][i])))
gate=read(D/'gate-decomposition.json')
decision=dict(updated=now,state='reviewed_not_adopted_insufficient_support',visual_review_completed=True,quality_accepted=False,new_camera_candidate=False,new_gaussian=False,
    official_runtime_verified=True,scope='Pinned RoMaOutdoor pretrained model on this indoor sequential13-image subset at the preregistered resolutions. Not all RoMa configurations or full175-view camera BA.',
    all_primary_pass=4,total_probes=473,face_primary_pass=0,primary_failure='Preregistered confidence0.95 plus FB<=4/cycle<=6 coverage fails; confidence alone removes many otherwise consistent points.',
    threshold_limitation='Absolute matcher confidence is not a calibrated probability of correct material identity. The conservative gate limits this candidate conclusion. No threshold tuning after outcomes.',
    geometry_observation='Among descriptive valid FB/cycle-consistent points without changing acceptance, lower face median2.879rawpx, upper2.370, torso0.835, legs0.365. Different regions/observations are not controlled method comparisons.',
    root_cause='Unresolved mixture of correspondence uncertainty, physical calibration and subject micro-motion. Both compatible-looking residuals and explicit visibility mismatches exist; no proof of a unique extrinsic error.',
    next='Do not run BA or Gaussian training from this support. Proceed to bounded HSfM data-assumption/source feasibility with the original175 identities and frozen074; no automatic alternative weights or threshold sweep.',
    evidence=['result.json','gate-decomposition.json','material-track-review.json','visual-sample-manifest.json'],
    slide_title='RoMa跨视角对应：三维投影仍有偏差',
    slide_summary='074相机固定。RoMa对应（绿），中间视角预测（橙）',
    slide_observation='镜框样本：循环差0.14 px，留出投影差3.28 px。',
    slide_conclusion='预设高置信度覆盖不足，暂不改相机。下一步核验HSfM联合几何。',
    slide_notes=[
      'Official RoMa source https://github.com/Parskatt/RoMa commit77f8d68803526dcddfd9b7a46bc76125bdc25f15. Outdoor weight andDINOv2ViT-L/14 obtained from official URLs; strict state loading passed. No subject upload.',
      'One preregistered run:13originalfullimages,18pairs,473fixedprobes,74.225seconds,peak5829224448bytes. All90preselectedvisualexamples reviewed. Source074/masks/originalimages unchanged.',
      'Primary gateconfidence>=.95,FB<=4rawpx,cycle<=6rawpx yields4/473and0facialpoints. The0.95threshold is conservative and uncalibrated for this subject; low yield is not measured matching accuracy. No post-outcome threshold modification.',
      'Explanatory probe213,raw063/075/088. Each280x320sourcecrop follows its own image correspondence without rescaling. Green points are RoMa estimates, not ground truth. Only B is held out of theA/C3Dpoint fit; orange andgreen are overlaid on the same unchangedBpixels.',
      'Small cycle0.1436px andheld3.2802px alone cannot distinguish physical camera error from matching bias or micro-motion. Explicit occlusion mismatches appear elsewhere. New camera candidate rejected for insufficient validated support; no newGaussiantrain/PLY or00deployment.',
      'P1previouslyfailedfacevalidation. Preserve source074 and114model. Next bounded candidateHSfM requires prior data-assumption andsourcecheck. Local evidence: portrait-alignment-open-source-116/roma/smoke/decision.json andmaterial-track-review.json.'])
save(D/'decision.json',decision)
print(json.dumps(dict(state=decision['state'],reviewed_tracks=90,primary_pass=4,new_training=False)))
