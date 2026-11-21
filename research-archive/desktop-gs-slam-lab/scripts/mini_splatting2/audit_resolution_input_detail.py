"""Measure information changed by2x input reduction; this is not a3D quality bound."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from evaluate_head_detail import metric

ROOT=Path(__file__).resolve().parents[2]/'outputs/mini-splatting2-mip-depth-001'


def main():
    cv2.setNumThreads(2);out=ROOT/'resolution-input-detail-001';out.mkdir(exist_ok=False)
    data=ROOT/'source-resolution-rectified-002'
    source=Path(__file__).with_name('local_refinement.py')
    assignment=next(node for node in ast.parse(source.read_text()).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SUBJECT_BOX' for t in node.targets))
    boxes=ast.literal_eval(assignment.value);rows=[];panels=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    for frame,box in sorted(boxes.items()):
        name=f'{frame:06d}';assert frame%8==0
        fullpath=data/'dataset/images'/(name+'.png');lowpath=data/'matched-low-resolution-control/images'/(name+'.png')
        full=np.asarray(Image.open(fullpath).convert('RGB'));low=np.asarray(Image.open(lowpath).convert('RGB'))
        assert np.array_equal(cv2.resize(full,(757,1347),interpolation=cv2.INTER_AREA),low)
        old=[round(value*size) for value,size in zip(box,[757,1347,757,1347])]
        x0,y0,x1,_=[2*v for v in old];y1=2*(old[1]+round((old[3]-old[1])/3))
        target=full[y0:y1,x0:x1];frames=[('Original rectified frame',target)];scores={}
        for interpolation,label in [(cv2.INTER_LINEAR,'Bilinear'),(cv2.INTER_CUBIC,'Bicubic')]:
            restored=cv2.resize(low,(1514,2694),interpolation=interpolation)[y0:y1,x0:x1]
            scores[label]=metric(restored.astype('float32')/255,target.astype('float32')/255)
            frames.append((f'Low-res input, {label.lower()}',restored))
        row=dict(frame=name,region=[x0,y0,x1,y1],source_sha256=hashlib.sha256(fullpath.read_bytes()).hexdigest(),metrics=scores);rows.append(row)
        panel=Image.new('RGB',(1104,312),'#f1f3f4');draw=ImageDraw.Draw(panel)
        for j,(label,arr) in enumerate(frames):
            thumb=Image.fromarray(arr);thumb.thumbnail((360,278));panel.paste(thumb,(j*368,32))
            draw.text((j*368+4,7),name+' '+label,font=font,fill='#172d3a')
        panels.append(panel)
        Image.fromarray(target).save(out/(name+'-source-head.png'))
    for start in [0,4]:
        sheet=Image.new('RGB',(1104,1248),'#f1f3f4')
        for index,panel in enumerate(panels[start:start+4]):sheet.paste(panel,(0,index*312))
        sheet.save(out/f'review-{1+start//4}.png')
    summary=dict(status='complete_pending_visual_review',created=time.time(),rows=rows,
        aggregate={label:{key:statistics.mean(row['metrics'][label][key] for row in rows) for key in rows[0]['metrics'][label]} for label in ['Bilinear','Bicubic']},
        frames=len(rows),source='Same rectified video images and frozen primary head/shoulder boxes',
        hypothesis='Distinguish loss of sampled image detail from larger reconstruction errors',
        interpretation='Resampling diagnostic only. Not a3D quality bound, camera-accuracy test or proof of the cause of blur.',
        training_updated=False,gpu_used=False,independent_capture=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary['aggregate']))


if __name__=='__main__':main()
