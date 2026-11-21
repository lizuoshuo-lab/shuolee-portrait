"""Correct rejected training-only head-box drafts without touching held-out images.

These rectangles are a prepared annotation for a separate potential ablation;
the running variance experiment never reads them.
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


def linear_at(x,xs,values):
    if x<xs[0]:lo,hi=0,1
    elif x>xs[-1]:lo,hi=len(xs)-2,len(xs)-1
    else:
        hi=int(np.searchsorted(xs,x,side='right'));hi=min(hi,len(xs)-1);lo=max(0,hi-1)
    if xs[hi]==xs[lo]:return values[lo]
    return values[lo]+(values[hi]-values[lo])*(x-xs[lo])/(xs[hi]-xs[lo])


def main():
    source=ROOT/'training-head-regions-001/manifest.json'
    old=json.loads(source.read_text());rows=old['rows'];assert len(rows)==51
    rejected={3,43}|{int(r['frame']) for r in rows if 'fallback' in r['method']}
    anchors=[r for r in rows if int(r['frame']) not in rejected]
    xs=np.array([int(r['frame']) for r in anchors]);values=np.array([r['box'] for r in anchors],dtype=float)
    out=ROOT/'training-head-regions-002';out.mkdir(exist_ok=False)
    proposed=[];tiles=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for row in rows:
        frame=row['frame'];index=int(frame);assert index%8!=0
        path=BASE/'hover-sfm-001/dataset/images'/(frame+'.jpg')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['source_sha256']
        image=Image.open(path).convert('RGB');w,h=image.size
        box=np.array(row['box'],dtype=float) if index not in rejected else linear_at(index,xs,values)
        bw,bh=box[2]-box[0],box[3]-box[1]
        # Consistent extra top margin covers hair clipped by the face detector.
        box=box+np.array([-.05*bw,-.13*bh,.05*bw,.05*bh])
        box=np.rint(np.clip(box,[0,0,0,0],[w,h,w,h])).astype(int).tolist()
        assert box[0]<box[2] and box[1]<box[3]
        method='temporal_interpolation' if index in rejected else 'accepted_detector_anchor_with_margin'
        proposed.append(dict(frame=frame,box=box,source_sha256=row['source_sha256'],method=method,
            original_box=row['box'],approved=False))
        preview=image.copy();draw=ImageDraw.Draw(preview);draw.rectangle(box,outline='#14d867',width=3)
        pad=40;crop=preview.crop((max(0,box[0]-pad),max(0,box[1]-pad),min(w,box[2]+pad),min(h,box[3]+pad)))
        crop.thumbnail((196,224));tile=Image.new('RGB',(204,252),'white')
        tile.paste(crop,((204-crop.width)//2,26))
        ImageDraw.Draw(tile).text((6,5),frame+(' corrected' if index in rejected else ''),fill='black',font=font)
        tiles.append(tile)
    for sheet,start in enumerate(range(0,len(tiles),18),1):
        contact=Image.new('RGB',(1224,756),'#dce3e2')
        for i,tile in enumerate(tiles[start:start+18]):contact.paste(tile,((i%6)*204,(i//6)*252))
        contact.save(out/f'review-{sheet}.jpg',quality=96)
    protocol=dict(status='corrected_proposal_pending_visual_review',created=time.time(),rows=proposed,
        training_only=True,heldout_images_read=False,used_for_training=False,
        source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rejected_draft_frames=sorted(rejected),
        correction='Interpolate neighboring accepted detections; first four frames use linear extrapolation from frames5/6; add fixed padding to cover hair',
        caveats=['Regions include neck and background; they are attention rectangles, not segmentation',
            'Based only on source training photographs; no rendered candidate images were used for box placement',
            'Manual visual approval of all51 rectangles is required before any training use'])
    (out/'manifest.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(dict(folder=str(out),frames=51,rejected=sorted(rejected))))


if __name__=='__main__':main()
