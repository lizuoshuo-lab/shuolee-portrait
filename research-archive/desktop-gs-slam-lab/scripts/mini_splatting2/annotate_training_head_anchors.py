"""Manually anchored head attention proposals from training photographs only.

Coordinates below were read on the saved review sheets, not predicted renders.
Convert them back to source pixels, interpolate nearby training frames and then
review every frame. This does not alter any running training job or evaluation.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw,ImageFont

LAB=Path(__file__).resolve().parents[2]
ROOT=LAB/'outputs/mini-splatting2-mip-depth-001'
BASE=LAB/'outputs/mini-splatting2-computer00-001'
ANCHORS={1:[50,91,153,220],7:[49,81,154,219],14:[46,99,150,228],
    21:[45,70,151,207],28:[42,63,150,207],35:[53,84,153,214],
    42:[49,79,162,224],49:[23,62,145,216],58:[26,78,147,230]}


def main():
    source=ROOT/'training-head-regions-002/manifest.json';prior=json.loads(source.read_text())
    out=ROOT/'training-head-regions-003';out.mkdir(exist_ok=False)
    anchors={};records={int(r['frame']):r for r in prior['rows']}
    for frame,tile_box in ANCHORS.items():
        row=records[frame];path=BASE/'hover-sfm-001/dataset/images'/(row['frame']+'.jpg')
        image=Image.open(path);w,h=image.size;b=row['box']
        cropbox=(max(0,b[0]-40),max(0,b[1]-40),min(w,b[2]+40),min(h,b[3]+40))
        crop=image.crop(cropbox);original_size=crop.size;crop.thumbnail((196,224))
        dx=(204-crop.width)//2;dy=26
        scale=np.array([original_size[0]/crop.width,original_size[1]/crop.height]*2)
        converted=(np.array(tile_box)-np.array([dx,dy,dx,dy]))*scale+np.array([cropbox[0],cropbox[1]]*2)
        # A fixed small source-pixel margin accommodates annotation precision.
        converted+=np.array([-6,-6,6,6])
        anchors[frame]=np.rint(converted).astype(int).tolist()
    xs=np.array(sorted(anchors));ys=np.array([anchors[x] for x in xs]);rows=[];tiles=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for frame,row in records.items():
        assert frame%8!=0
        path=BASE/'hover-sfm-001/dataset/images'/(row['frame']+'.jpg')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256']
        image=Image.open(path).convert('RGB');w,h=image.size
        box=[round(float(np.interp(frame,xs,ys[:,i]))) for i in range(4)]
        assert 0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h
        rows.append(dict(frame=row['frame'],box=box,source_sha256=row['source_sha256'],
            method='manual_anchor' if frame in anchors else 'interpolated_training_anchors'))
        preview=image.copy();ImageDraw.Draw(preview).rectangle(box,outline='#14d867',width=3)
        crop=preview.crop((max(0,box[0]-40),max(0,box[1]-40),min(w,box[2]+40),min(h,box[3]+40)))
        crop.thumbnail((196,224));tile=Image.new('RGB',(204,252),'white');tile.paste(crop,((204-crop.width)//2,26))
        ImageDraw.Draw(tile).text((6,5),row['frame']+(' anchor' if frame in anchors else ''),font=font,fill='black')
        tiles.append(tile)
    for sheet,start in enumerate(range(0,len(tiles),18),1):
        contact=Image.new('RGB',(1224,756),'#dce3e2')
        for i,tile in enumerate(tiles[start:start+18]):contact.paste(tile,((i%6)*204,(i//6)*252))
        contact.save(out/f'review-{sheet}.jpg',quality=96)
    manifest=dict(status='manual_anchor_proposal_pending_all_frame_review',created=time.time(),
        training_only=True,heldout_images_read=False,used_for_training=False,frames=len(rows),rows=rows,
        anchors_source_pixels=anchors,anchors_review_tile_pixels=ANCHORS,
        review_source_sha256={name:hashlib.sha256((source.parent/name).read_bytes()).hexdigest()
            for name in ['review-1.jpg','review-2.jpg','review-3.jpg']},
        purpose='Prepared head-specific attention ablation if geometry-only constraints fail to improve clarity',
        caveats=['Rectangles include hair, face, neck and small background margin; no segmentation or identity inference',
            'Not yet approved for training until all51source-photo rectangles are visually inspected'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(folder=str(out),frames=len(rows))))


if __name__=='__main__':main()
