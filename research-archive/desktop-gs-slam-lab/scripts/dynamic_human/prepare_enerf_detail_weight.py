"""Image-window loss weights only; no asserted hand identity or 3-D labels."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-detail-weight-076';B=L/'outputs/dynamic-human-enerf-rgb-pose-correction-074';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
R.mkdir(exist_ok=True);(R/'regions').mkdir(exist_ok=True);(R/'review').mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'preparation.json').exists()
records=[r for r in read(I/'input-audit.json')['records'] if r['split']=='train'];assert len(records)==105 and all(r['frame']%2==0 for r in records)
obs={(r['camera'],r['frame']):r for r in read(B/'observations.json')['records']}
save(R/'protocol.json',dict(status='fixed_before_region_inspection_and_training',
    hypothesis='Small hand and shoe pixels may receive too little photometric weight in whole-person image loss. Test native image-window emphasis without changing body/graph capacity or inventing finger labels.',
    loss='Original074photometric+alpha+graph regularization +0.2*mean_L1_RGB_on_union_of_detail_windows. Per-image mean over valid pixels and3channels. Exact original premultiplied reference and random background; no extra keypoint/flow loss.',
    region_rule='For each of2actors:144x144px square centered at predicted body wrist9and10. Each shoe:boundingbox of corresponding3wholebody foot points17:20or20:23, expanded12px, minimum96px side. Clip to nativeimage and existingtraining crop. Union avoids doublecount. CorrectRGB074observations used only to position windows.',
    revision='Before training, source-camera01/02 review showed initial96px wrist windows sometimes cut fingertips. Enlarged hand windows to144px and minimum shoe windows to96px. Previous preparation preserved in candidates-v1; no model/validation results used for this change.',
    qualification='Image windows include background, clothing, occluders and hands/shoes. They are loss-selection windows, not semantic masks, identity labels or visibility truth. Invalid/rejected pose observations are not introduced as3Dconstraints.',
    unchanged='Original105RGB/mattes/valid masks,074frozen correctedRGB body motion,52109initialGaussians,384nodes,seed73,sampling/backgrounds,learning rates,15000updates,densityschedule. No075flow supervision,071finergraph or072maskrepair.',
    acceptance='Compare to074 at identical update budget, evaluate against067 retained reference as context. All90oddtimeknown-camera and18other-camera dev views; fixed native hand/foot/fullperson and worst cases. Development,notfinaltest.',
    ROI_weight=.2,model_updates=0,goal_complete=False))
out=[]
for r in records:
    c,t=r['camera'],r['frame'];f=I/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(f)==r['input_sha256'];z=np.load(f)
    boxes=[]
    for person,det in enumerate(obs[c,t]['detections']):
        points=np.array(det['keypoints'])
        for name,index in [('left_hand_window',9),('right_hand_window',10)]:
            ctr=points[index];boxes.append(dict(person=person,name=name,bbox=np.r_[ctr-72,ctr+72]))
        for name,start in [('left_shoe_window',17),('right_shoe_window',20)]:
            pts=points[start:start+3];ctr=(pts.min(0)+pts.max(0))/2;size=np.maximum(pts.max(0)-pts.min(0)+24,96);boxes.append(dict(person=person,name=name,bbox=np.r_[ctr-size/2,ctr+size/2]))
    mask=np.zeros((1080,1920),np.uint8);bx=z['bbox'];cx0,cy0,cx1,cy1=map(int,bx)
    for box in boxes:
        x0,y0,x1,y1=np.rint(box['bbox']).astype(int)
        x0=max(cx0,min(cx1,x0));x1=max(cx0,min(cx1,x1));y0=max(cy0,min(cy1,y0));y1=max(cy0,min(cy1,y1))
        box['bbox']=[int(x0),int(y0),int(x1),int(y1)];mask[y0:y1,x0:x1]=1
    mask&=z['valid'].astype(np.uint8);assert mask.sum()>1000
    np.savez_compressed(R/'regions'/f'camera{c}-frame{t:06d}.npz',mask=mask)
    out.append(dict(camera=c,frame=t,input_sha256=r['input_sha256'],boxes=boxes,pixels=int(mask.sum()),crop_fraction=float(mask.sum()/((cx1-cx0)*(cy1-cy0))),region_sha256=sha(R/'regions'/f'camera{c}-frame{t:06d}.npz')))
    if t in [744,750,756]:
        im=Image.fromarray(z['rgb'][cy0:cy1,cx0:cx1]);d=ImageDraw.Draw(im)
        for box in boxes:
            a,b,cc,dd=box['bbox'];d.rectangle([a-cx0,b-cy0,cc-cx0-1,dd-cy0-1],outline=(255,170,20) if 'hand' in box['name'] else (0,180,240),width=1)
        sheet=Image.new('RGB',(im.width,im.height+30),'white');sheet.paste(im,(0,30));ImageDraw.Draw(sheet).text((5,8),f'Camera{c} frame{t}: orange hand windows / blue shoe windows',fill='black');sheet.save(R/'review'/f'camera{c}-frame{t:06d}.png')
save(R/'preparation.json',dict(status='105native_image_detail_window_candidates_ready_visual_review_pending',records=out,visual_preview_count=45,
    union_crop_fraction_min=float(min(r['crop_fraction'] for r in out)),union_crop_fraction_max=float(max(r['crop_fraction'] for r in out)),
    observations_sha256=sha(B/'observations.json'),model_updates=0,goal_complete=False))
shutil.copy2(B/'graph.npz',R/'graph.npz');shutil.copy2(__file__,R/Path(__file__).name)
print(json.dumps(dict(status='105detail windows ready',preview_count=45,crop_fraction_range=[min(r['crop_fraction'] for r in out),max(r['crop_fraction'] for r in out)])))
