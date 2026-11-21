"""Native scientific overlays and actual generic input consumption; no model run."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,subprocess
import numpy as np
from PIL import Image,ImageDraw
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'grass-preflight';C=PRE/'coverage';O=PRE/'generic-input'
P=Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118');sys.path.insert(0,str(P/'src'))
from shuolee_portrait.alignment_inputs import load_alignment_inputs,read,sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def link(p,**kw):return dict(path=str(p),sha256=sha(p),**kw)
assert not O.exists();O.mkdir()
plan=read(PRE/'coverage-plan.json');result=read(C/'result.json');assert result['frames']==result['resolved_subject_frames']==94
rows=read(C/'observations.json')['rows'];z=np.load(C/'observations.npz');xy=z['keypoints'];assert xy.shape==(94,133,2)
assert z['keypoint_names'].astype(str).tolist()[40:91]==[f'face-{i}' for i in range(17,68)]
faces=Image.new('RGB',(512,1600));feet=Image.new('RGB',(1800,1280));items=[]
for i,slot in enumerate(plan['review_slots']):
    row=rows[slot];raw=Image.open(row['image']).convert('RGB');assert sha(Path(row['image']))==row['source_sha256'] and raw.size==tuple(row['wh'])
    fb=np.array(row['face_box']);mid=(fb[:2]+fb[2:])/2;left,top=np.floor(mid-[128,160]).astype(int)
    box=[int(left),int(top),int(left+256),int(top+320)];p=raw.crop(box);m=p.copy();draw=ImageDraw.Draw(m)
    for j in [30,33,36,39,42,45,48,54]:
        u,v=xy[slot,23+j]-[left,top];draw.ellipse((u-2,v-2,u+2,v+2),outline='#168aff',width=1)
    faces.paste(p,(0,i*320));faces.paste(m,(256,i*320))
    bb=np.array(row['body_box']);cx=int(np.floor((bb[0]+bb[2])/2));footbox=[cx-450,1264,cx+450,1520]
    p=raw.crop(footbox);m=p.copy();draw=ImageDraw.Draw(m)
    for j in range(15,23):
        u,v=xy[slot,j]-np.array(footbox[:2])
        if 0<=u<900 and 0<=v<256:draw.ellipse((u-3,v-3,u+3,v+3),outline='#168aff',width=1)
    feet.paste(p,(0,i*256));feet.paste(m,(900,i*256))
    items.append(dict(slot=slot,source_frame=row['source_frame_index'],source_sha256=row['source_sha256'],head_crop=box,feet_crop=footbox,
        face_native_xy={str(j):xy[slot,23+j].tolist() for j in [30,33,36,39,42,45,48,54]},face_scores=row['regions']['face']))
faces.save(O/'five_native_heads_raw_left_predictions_right.png');feet.save(O/'five_native_feet_raw_left_predictions_right.png')
# Compact evidence uses the middle3 registered crops, unchanged scale.
sheet=Image.new('RGB',(576,384))
for col,slot in enumerate(plan['review_slots'][1:4]):
    row=rows[slot];raw=Image.open(row['image']).convert('RGB');fb=np.array(row['face_box']);mid=(fb[:2]+fb[2:])/2;x,y=np.floor(mid-96).astype(int)
    crop=raw.crop((x,y,x+192,y+192));marked=crop.copy();draw=ImageDraw.Draw(marked)
    for j in [30,33,36,39,42,45,48,54]:
        u,v=xy[slot,23+j]-[x,y];draw.ellipse((u-2,v-2,u+2,v+2),outline='#168aff',width=1)
    sheet.paste(crop,(col*192,0));sheet.paste(marked,(col*192,192))
sheet.save(O/'native_three_views_raw_above_predictions_below.png')
inventory=dict(observations_sha256=sha(C/'observations.npz'),inputs_sha256=sha(C/'observations.json'),
    coordinate_domain='native_integer_pixel_centres',synthetic_control=False,annotation_status='unreviewed_missing_evidence_inventory',records=[])
for i,row in enumerate(rows):
    for j in range(51):
        ibug=j+17
        inventory['records'].append(dict(frame=row['frame'],anchor_id=f'ibug68:{ibug}',ibug68_index=ibug,uv_native=xy[i,j+40].tolist(),
            visibility='unknown',split='UNASSIGNED',physical_family_id=f'unverified_semantic_id:ibug68:{ibug}',semantic_identity_verified=False,
            localization_sigma_px=None,reliability=None,review_evidence=None,visibility_source='not_reviewed'))
save(O/'unreviewed-inventory.json',inventory)
desc=link(C/'observations.npz',convention='coco_wholebody_133',names_key='keypoint_names',coordinate_domain='native_integer_pixel_centres')
manifest=dict(schema='shuolee-alignment-input/1',purpose='real',camera_model='rectified_pinhole',frames=link(C/'observations.json',rows_key='rows'),wholebody=desc,
    face=dict(**desc,coordinates_key='keypoints',metadata=link(C/'observations.json',rows_key='rows')),
    reviews=link(O/'unreviewed-inventory.json',format='reviewed_anatomical_v1'))
save(O/'alignment-inputs.json',manifest);cfg=read(REC/'generic-2d3d-research/adapter-a0b/plan.json')['config'];save(O/'config.json',cfg)
bundle=load_alignment_inputs(O/'alignment-inputs.json');assert np.array_equal(bundle.face_uv,xy[:,40:91]) and np.array_equal(bundle.body['keypoints'],xy)
run=subprocess.run([sys.executable,str(P/'run.py'),'fit-inputs','--output',str(O)],capture_output=True,text=True,timeout=90)
(O/'CLI.stdout.txt').write_text(run.stdout,encoding='utf-8');(O/'CLI.stderr.txt').write_text(run.stderr,encoding='utf-8');assert run.returncode==0,run.stderr
report=json.loads(run.stdout);assert report['frames']==94 and report['active_fit']==report['active_held']==0 and not report['fitting_allowed'];bundle.assert_unchanged()
save(O/'CLI-result.json',report)
save(O/'review-inventory.json',dict(items=items,order='Source0,115,235,350,465. Rawleft, predictionright. Exactnativepixels without resampling or text.',all_predictions_inactive=True))
summary=dict(updated=datetime.now(timezone.utc).isoformat(),state='grass94_complete_generic_input_consumed_visual_review_pending',frames=94,body_observations=94*133,face_observations=94*51,
    detector_parameters_same_as_other_person=True,actual_CLI_read=True,coordinates_unchanged=True,all_source_frames_preserved=True,
    config_sha256=sha(O/'config.json'),source_hashes={name:sha(P/'src/shuolee_portrait'/name) for name in ['alignment_inputs.py','fit_backend.py','initialization_inputs.py','smplx_correspondence.py']},
    camera_records=report['camera_records'],missing_body_arrays=[k for k in ['boxes','flip_disagreement'] if k not in bundle.body],
    raw_responses_preserved=True,score_or_in_image_is_not_visibility=True,all_observations_inactive=True,
    physical_camera_accepted=False,physical_face_accuracy_accepted=False,real_fit=False,training=False,quality_accepted=False)
save(O/'result.json',summary);print(json.dumps(summary),flush=True)
