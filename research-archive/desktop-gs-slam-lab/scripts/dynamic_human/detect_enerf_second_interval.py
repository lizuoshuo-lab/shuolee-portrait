"""Training-only keyframe detections for new pose crops; not body GT."""
from pathlib import Path
import json,time,hashlib,shutil
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from torchvision.models.detection import keypointrcnn_resnet50_fpn,KeypointRCNN_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
R=L/'outputs/dynamic-human-enerf-second-interval-079'
O=R/'box-detection';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'observations.json').exists(),'Preserve completed detection'
protocol=read(R/'protocol.json');inputs={(r['camera'],r['frame']):r for r in read(R/'input-audit.json')['records']}
weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
save(O/'protocol.json',dict(status='fixed_before_keyframe_inference',frames=[720,726,732],cameras=protocol['training_cameras'],
    model='Official torchvision KeypointRCNN ResNet50 FPN COCO_V1, standard internal detector resize, CPU',weights_url=weights.url,
    detection_threshold=.8,association='No automatic person identity assignment; inspect original photographs with candidate rectangles first.',
    next='After review, interpolate padded boxes in time only for RTMW inference. Actual source pixels and times remain unchanged.',goal_complete=False))
torch.set_num_threads(4);torch.manual_seed(73)
model=keypointrcnn_resnet50_fpn(weights=weights,progress=False).cpu().eval()
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',16)
colors=[(255,100,20),(0,160,255),(200,0,200),(0,180,50)]
allrows=[];started=time.monotonic()
for c in protocol['training_cameras']:
    panels=[]
    for t in [720,726,732]:
        path=R/'native-input'/f'camera{c}-frame{t:06d}.npz';assert sha(path)==inputs[c,t]['input_sha256']
        with np.load(path) as z:rgb=z['rgb'];bbox=z['bbox'].tolist()
        assert inputs[c,t]['split']=='train'
        with torch.inference_mode():pred=model([to_tensor(Image.fromarray(rgb))])[0]
        indices=torch.where(pred['scores']>=.8)[0].tolist();dets=[]
        im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
        for ci,idx in enumerate(indices):
            b=pred['boxes'][idx].numpy();xy=pred['keypoints'][idx,:,:2].numpy();scores=pred['keypoints_scores'][idx].numpy()
            assert np.isfinite(b).all() and np.isfinite(xy).all()
            dets.append(dict(candidate=ci,index=idx,score=float(pred['scores'][idx]),box=b.tolist(),keypoints=xy.tolist(),keypoint_scores=scores.tolist()))
            col=colors[ci%len(colors)];draw.rectangle(b.tolist(),outline=col,width=3)
            draw.text((int(b[0])+5,int(b[1])+5),f'ID {ci}',font=font,fill=col,stroke_width=1,stroke_fill='white')
        im=im.crop(bbox);panel=Image.new('RGB',(im.width,im.height+38),'white');panel.paste(im,(0,38))
        ImageDraw.Draw(panel).text((5,5),f'Camera {c} frame {t}; crop x={bbox[0]} y={bbox[1]}',font=font,fill='black')
        panels.append(panel)
        row=dict(camera=c,frame=t,input_sha256=sha(path),candidates=dets,display_bbox=bbox)
        allrows.append(row);save(O/f'camera{c}-frame{t:06d}.json',row)
    sheet=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)),'white');x=0
    for p in panels:sheet.paste(p,(x,0));x+=p.width
    sheet.save(O/f'camera{c}-keyframe-boxes.png')
    save(O/'progress.json',dict(images=len(allrows),cameras_complete=len(allrows)//3,elapsed_seconds=time.monotonic()-started))
    print('SECOND_INTERVAL_DETECTION',c,[len(r['candidates']) for r in allrows[-3:]],flush=True)
save(O/'observations.json',dict(status='45training_keyframes_detected_identity_and_crop_review_pending',records=allrows,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name)
print('45training keyframe detections complete; no motion fit or appearance training.')
