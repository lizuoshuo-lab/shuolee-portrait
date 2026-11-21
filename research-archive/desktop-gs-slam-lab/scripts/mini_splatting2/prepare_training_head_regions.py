"""Draft training-only head attention boxes for a later controlled appearance ablation.

This is region annotation, not identity recognition or a motion model. Drafts
require visual review and are not consumed by the current training queue.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2];BASE=LAB/'outputs/mini-splatting2-computer00-001'
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'


def main():
    out=ROOT/'training-head-regions-001';out.mkdir(exist_ok=False);cv2.setNumThreads(4)
    cams=sorted(json.loads((BASE/'viewer/cameras.json').read_text()),key=lambda c:c['img_name'])
    cams=[c for index,c in enumerate(cams) if index%8!=0]
    assert len(cams)==51 and all(int(c['img_name'])%8!=0 for c in cams)
    detector_paths=[Path(cv2.data.haarcascades)/name for name in ['haarcascade_frontalface_alt2.xml','haarcascade_profileface.xml']]
    detectors=[cv2.CascadeClassifier(str(p)) for p in detector_paths]
    assert all(not d.empty() for d in detectors)
    rows=[];tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    for c in cams:
        path=BASE/'hover-sfm-001/dataset/images'/f'{c["img_name"]}.jpg';image=cv2.imread(str(path));h,w=image.shape[:2]
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        red=(((hsv[...,0]<12)|(hsv[...,0]>168))&(hsv[...,1]>100)&(hsv[...,2]>65)).astype('uint8')
        red=cv2.morphologyEx(red,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
        count,labels,stats,_=cv2.connectedComponentsWithStats(red)
        components=[r for r in stats[1:] if r[4]>2000 and r[3]>70]
        assert components,'Review missing red-coat locator'
        x,y,bw,bh,area=max(components,key=lambda row:row[4])
        sx=max(0,int(x-.55*bw));ex=min(w,int(x+1.55*bw));sy=max(0,int(y-1.5*bw));ey=min(h,int(y+.3*bw))
        crop=image[sy:ey,sx:ex];gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);candidates=[]
        for detector_index,detector in enumerate(detectors):
            for flip in ([False,True] if detector_index else [False]):
                arr=cv2.flip(gray,1) if flip else gray
                detected=detector.detectMultiScale(arr,scaleFactor=1.05,minNeighbors=4,minSize=(35,35))
                for dx,dy,dw,dh in detected:
                    if flip:dx=gray.shape[1]-dx-dw
                    cx=sx+dx+dw/2;cy=sy+dy+dh/2
                    if cy>y+.05*bw:continue
                    expected=np.array([x+bw*.5,y-bw*.32])
                    score=np.linalg.norm((np.array([cx,cy])-expected)/bw)-.25*dw/bw
                    candidates.append((score,[int(sx+dx),int(sy+dy),int(dw),int(dh)],detector_index,flip))
        if candidates:
            _,(fx,fy,fw,fh),detector_index,flip=min(candidates,key=lambda v:v[0])
            # Extend the detected face to include the hair and outer head.
            box=[max(0,round(fx-.35*fw)),max(0,round(fy-.55*fh)),min(w,round(fx+1.35*fw)),min(h,round(fy+1.3*fh))]
            method='frontal' if detector_index==0 else 'profile_flipped' if flip else 'profile'
        else:
            box=[max(0,int(x-.1*bw)),max(0,int(y-1.1*bw)),min(w,int(x+1.1*bw)),min(h,int(y+.05*bw))]
            method='red_coat_fallback_requires_review'
        rows.append(dict(frame=c['img_name'],box=box,method=method,red_coat_box=[int(x),int(y),int(bw),int(bh)],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),approved=False))
        rgb=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(rgb)
        draw.rectangle(box,outline='#20ff70',width=3)
        pad=35;view=rgb.crop((max(0,box[0]-pad),max(0,box[1]-pad),min(w,box[2]+pad),min(h,box[3]+pad)))
        view.thumbnail((176,185));tile=Image.new('RGB',(180,212),'white');tile.paste(view,((180-view.width)//2,24))
        ImageDraw.Draw(tile).text((5,4),c['img_name']+' '+method[:13],font=font,fill='black');tiles.append(tile)
    for sheet_index,start in enumerate(range(0,len(tiles),18)):
        sheet=Image.new('RGB',(1080,636),'#d7dfe2')
        for index,tile in enumerate(tiles[start:start+18]):sheet.paste(tile,((index%6)*180,(index//6)*212))
        sheet.save(out/f'review-{sheet_index+1}.jpg',quality=94)
    protocol=dict(status='draft_regions_pending_visual_review',training_only=True,heldout_images_read=False,
        frames=len(rows),used_for_training=False,rows=rows,
        detectors={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in detector_paths},
        purpose='Potential later region-weighted RGB ablation, only if current diagnostics justify it',
        caveats=['Head attention rectangles include some background, not a segmentation',
            'Red-coat cue is specific to this video',
            'No masks or losses in current training/evaluation have been changed'])
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(status=protocol['status'],frames=len(rows),
        methods={key:sum(r['method']==key for r in rows) for key in set(r['method'] for r in rows)})))


if __name__=='__main__':main()
