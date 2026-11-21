"""Hand-crop model observations with exact RGB preprocessing and crop stability."""
from pathlib import Path
import json, hashlib, time, zipfile, shutil
import numpy as np, cv2, onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
from rtmlib import RTMPose
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab')
S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-motion-082'
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'observations.json').exists()
obs=read(S/'pose/observations.json')['records'];cams=sorted({r['camera'] for r in obs})
assert len(obs)==195 and not set(cams)&{'00','06','12'}
review=dict(status='15native_source_sheets_90hand_crops_actually_reviewed',
    files=[dict(path=f'source-panels/camera{c}-source.png',sha256=sha(O/'source-panels'/f'camera{c}-source.png')) for c in cams],
    selection='Light-clothed person, right hand only, all15known cameras,720..732. Hand identity/crop review only; not per-finger visibility labels.',
    reasons_zh=['右手在所有机位都有可见轮廓，后段离开奶牛玩偶、逐渐张开；01、03、04能看到掌面。',
      '其他机位以手背、侧面为主，手指互相遮挡，部分原图较软；预测不可自动视为精确关节真值。',
      '左手抓握玩偶、手腕和多根手指被遮挡；02、05、07等全身模型把左右手预测在同一只可见手上，全部左手排除。',
      '本轮只检查浅衣人物右手；不把这一局部结果声称为两个人全身清晰。'],
    hand_annotations_accepted=False)
save(O/'source-review.json',review)
model=Path('/mnt/d/GS-SLAM-Data/models/rtmhand/rtmpose-m-hand5-256x256.onnx');archive=model.with_suffix('.zip')
with zipfile.ZipFile(archive) as z:
    pipeline=json.loads(z.read(next(n for n in z.namelist() if n.endswith('/pipeline.json'))))
norm=next(t for t in pipeline['pipeline']['tasks'][0]['transforms'] if t['type']=='Normalize');assert norm['to_rgb'] is True
class Hand(RTMPose):
    def __init__(self):
        opt=ort.SessionOptions();opt.intra_op_num_threads=4;opt.inter_op_num_threads=1
        self.onnx_model=str(model);self.session=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
        self.model_input_size=(256,256);self.mean=np.array(norm['mean']);self.std=np.array(norm['std']);self.backend='onnxruntime';self.device='cpu';self.to_openpose=False
pose=Hand();assert pose.session.get_inputs()[0].shape[1:]==[3,256,256]
def box_for(d,factor=1.):
    p=np.array(d['keypoints']);pts=np.r_[p[112:133],p[10:11]]
    center=(pts.max(0)+pts.min(0))/2;side=float(np.clip(np.ptp(pts,axis=0).max()*1.5,96,144))*factor
    return np.r_[center-side/2,center+side/2].tolist()
r=next(r for r in obs if r['camera']=='04' and r['frame']==726);d=next(d for d in r['detections'] if d['person']=='person_light_clothing')
with np.load(S/'native-input/camera04-frame000726.npz') as z:rgb=z['rgb']
box=box_for(d);center,scale=bbox_xyxy2cs(np.array(box),padding=1.25)
warped,_=top_down_affine((256,256),scale,center,np.ascontiguousarray(rgb[...,::-1]))
official=(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)-np.array(norm['mean']))/np.array(norm['std'])
actual,_,_=pose.preprocess(rgb,box);assert np.array_equal(actual,official)
save(O/'preprocessing-audit.json',dict(model_sha256=sha(model),archive_sha256=sha(archive),pipeline=pipeline,
    tensor_equal=True,max_difference=float(abs(actual-official).max()),onnx_shape=pose.session.get_inputs()[0].shape,
    normalization='Native RGB exactly equals official BGR-read -> affine -> RGB-convert -> normalize.',
    crop_factors=[.85,1.,1.15],same_image_and_center=True,geometric_protocol='All105even training and90odd diagnostic observations kept separate. Crop variation is a stability check, not independent views or GT.'))
(O/'candidate-panels').mkdir(exist_ok=True)
font=ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf',12);colors=[(235,85,30),(210,40,160),(30,150,210),(65,180,55),(165,75,210)]
def overlay(raw,xy,crop):
    im=raw.copy();draw=ImageDraw.Draw(im);p=xy-np.array(crop[:2])
    for f in range(5):
        chain=[0]+list(range(1+4*f,5+4*f))
        for i,j in zip(chain[:-1],chain[1:]):draw.line([tuple(p[i]),tuple(p[j])],fill=colors[f],width=1)
    for x,y in p:draw.ellipse((x-1,y-1,x+1,y+1),fill=(255,20,20))
    return im
rows=[];start=time.monotonic()
for c in cams:
    sheet=Image.new('RGB',(1360,610),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,4),f'Camera{c} right hand | Native / Whole-body / Dedicated / crop perturbations',font=font,fill='black')
    for r in [r for r in obs if r['camera']==c]:
        t=r['frame'];d=next(d for d in r['detections'] if d['person']=='person_light_clothing');p=np.array(d['keypoints']);path=S/'native-input'/f'camera{c}-frame{t:06d}.npz'
        assert sha(path)==r['input_sha256']
        with np.load(path) as z:rgb=z['rgb']
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['rectified_sha256']
        xy,score=pose(np.ascontiguousarray(rgb),bboxes=[box_for(d,factor) for factor in [.85,1,1.15]])
        assert xy.shape==(3,21,2) and score.shape==(3,21) and np.isfinite(xy).all()
        deviation=np.linalg.norm(xy-xy[1:2],axis=-1).max(0)
        center=np.median(np.r_[p[112:133],p[10:11]],axis=0);x0=int(np.clip(round(center[0])-80,0,1760));y0=int(np.clip(round(center[1])-80,0,920));crop=[x0,y0,x0+160,y0+160]
        rows.append(dict(camera=c,frame=t,split='fit' if t%2==0 else 'diagnostic_time',person='person_light_clothing',hand='right',input_sha256=r['input_sha256'],rectified_sha256=r['rectified_sha256'],K=r['K'],w2c=r['w2c'],crop=crop,boxes=[box_for(d,f) for f in [.85,1,1.15]],keypoints=xy[1].tolist(),scores=score[1].tolist(),all_crop_keypoints=xy.tolist(),all_crop_scores=score.tolist(),crop_deviation_px=deviation.tolist(),wholebody_keypoints=p[112:133].tolist(),wholebody_scores=d['scores'][112:133]))
        if t in [720,726,732]:
            yy=40+[720,726,732].index(t)*190;raw=Image.fromarray(rgb).crop(crop)
            draw.text((8,yy-16),f'Frame{t}; max crop-jitter median {np.median(deviation):.2f}px; no visibility confidence threshold',font=font,fill='black')
            ims=[raw,overlay(raw,p[112:133],crop),overlay(raw,xy[1],crop),overlay(raw,xy[0],crop),overlay(raw,xy[2],crop)]
            for i,im in enumerate(ims):sheet.paste(im,(8+i*170,yy))
    sheet.crop((0,0,860,610)).save(O/'candidate-panels'/f'camera{c}-candidates.png')
    save(O/'progress.json',dict(images=len(rows),model_forward_crops=len(rows)*3,elapsed_seconds=time.monotonic()-start));print('HAND_OBSERVATION',c,len(rows),flush=True)
save(O/'observations.json',dict(status='195native_right_hand_candidates_585model_crops',records=rows,elapsed_seconds=time.monotonic()-start,accepted_training_labels=False,goal_complete=False))
shutil.copy2(__file__,O/Path(__file__).name);print('HAND_OBSERVATIONS_COMPLETE',len(rows),flush=True)
