"""All-frame, unannotated evidence for visibility labels before optimization."""
from pathlib import Path
import numpy as np
from PIL import Image
from preprocess_demo_fps3_123 import DATA,REPORT
from review_demo_fps3_123 import join,bounded
from run_official_gaussians123 import read,save,sha,now


def main():
    out=REPORT/'visibility-review';out.mkdir(exist_ok=False)
    rows=read(DATA/'data/inputs.json')['records']
    with np.load(DATA/'data/observations.npz',allow_pickle=False) as z:obs=dict(z)
    entries=[];panels=[]
    for i,r in enumerate(rows):
        raw=Image.open(r['image']).convert('RGB');p=obs['keypoints'][i]
        body=raw.crop(bounded(obs['boxes'][i],raw.size,10));body.thumbnail((240,300))
        # Include right elbow, wrist, hip and predicted fingers so incorrect
        # predictions cannot hide the actual pocket/arm behind a tight crop.
        hands=p[np.r_[8,10,12,np.arange(112,133)]]
        boxes=dict(right_hand=bounded(np.r_[hands.min(0),hands.max(0)],raw.size,40),
            left_hand=bounded(np.r_[p[91:112].min(0),p[91:112].max(0)],raw.size,30),
            face=bounded(np.r_[p[23:91].min(0),p[23:91].max(0)],raw.size,15))
        right=raw.crop(boxes['right_hand']);right.thumbnail((240,300))
        left=raw.crop(boxes['left_hand']);left.thumbnail((180,140))
        face=raw.crop(boxes['face']);face.thumbnail((180,150))
        side=Image.new('RGB',(180,300),'white');side.paste(face,(0,0));side.paste(left,(0,155))
        panel=join([body,right,side]);canvas=Image.new('RGB',(660,300),'white');canvas.paste(panel,(0,0));panels.append(canvas)
        raw.crop(boxes['right_hand']).save(out/f'{i:03d}-right.png')
        entries.append(dict(slot=i,frame=r['frame'],source_index=r['source_frame_index'],source_sha256=r['source_sha256'],
            page=i//10,cell=i%10,boxes=boxes))
    for first in range(0,len(rows),10):
        canvas=Image.new('RGB',(1320,1500),'white')
        for k,panel in enumerate(panels[first:first+10]):canvas.paste(panel,((k%2)*660,(k//2)*300))
        canvas.save(out/f'page-{first//10:02d}.jpg',quality=95)
    save(out/'manifest.json',dict(updated=now(),layout='Page ascending, row-major; each cell body / right-hand region / face above left-hand',
        observations_sha256=sha(DATA/'data/observations.npz'),entries=entries,all_pixels_unannotated=True))
    print(str(out))


if __name__=='__main__':main()
