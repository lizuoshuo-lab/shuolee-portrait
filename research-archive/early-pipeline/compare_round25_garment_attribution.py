"""Compare the same R24/R25 novel-camera rays and render a marked review board."""

import argparse
from collections import defaultdict
import json

import numpy as np
from PIL import Image, ImageDraw

from trace_round22_face_pixels import checked, need, pin, read, sha
from trace_round25_garment_pixels import OUT


R24_ATTRIBUTION = 'bdcc7e62dfe3a34ef746ebf22409da052bbae7b392e69df19e492621aa2b3bcd'


def groups(pair):
    result=defaultdict(lambda:np.zeros(2))
    for item in pair['contributors']:
        name=item['part']
        trusted=item['r25_geometry_anchor_trusted']
        if trusted is not None:
            name='garment_trusted' if trusted else 'garment_untrusted'
        result[name]+=(item['alpha_weight'],item['direct_luma'])
    return {name:dict(alpha=float(values[0]),direct_luma=float(values[1])) for name,values in result.items()}


def board(output, subject, old, new):
    margin,label_height=16,24
    arms=('R24 18k','R25 18k')
    panels=[]
    for key in old:
        group=[]
        for source in (old,new):
            item=source[key]
            panel=item['panel']
            x0,y0,x1,y1=panel['crop_xyxy']
            with Image.open(checked(panel)) as im:
                image=im.convert('RGB').crop((0,0,x1-x0,y1-y0))
            draw=ImageDraw.Draw(image)
            for i,pair in enumerate(item['pairs'],1):
                for name,color in (('artifact_xy_px','#3ee4ef'),('control_xy_px','#ffd45b')):
                    x,y=pair[name]
                    x,y=x-x0,y-y0
                    draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=1)
                    draw.text((x+5,y-7),str(i),fill=color,stroke_width=1,stroke_fill='#101010')
            group.append((image,key))
        panels.append(group)
    width=max(image.width for group in panels for image,_ in group)
    height=sum(max(image.height for image,_ in group)+label_height for group in panels)+margin*3
    sheet=Image.new('RGB',(width*2+margin*3,height),'#191919')
    draw=ImageDraw.Draw(sheet)
    top=margin
    for group in panels:
        for col,((image,key),arm) in enumerate(zip(group,arms,strict=True)):
            left=margin+col*(width+margin)
            draw.text((left,top),f'{subject} | {arm} | {key[11:14]} | cyan sample / gold control',fill='white')
            sheet.paste(image,(left,top+label_height))
        top+=label_height+max(image.height for image,_ in group)+margin
    path=output/f'{subject}-r24-r25-side-waist-hem-picks.png'
    with path.open('xb') as f:
        sheet.save(f,format='PNG')
    return dict(file=str(path),sha256=sha(path))


def run(r25_sha):
    old=pin(OUT/'smpl-delivery-round24/input-audit/side-waist-hem-attribution.json',R24_ATTRIBUTION)
    output=OUT/'smpl-delivery-round25/input-audit'
    new=pin(output/old.name,r25_sha)
    reports=read(old)['subjects'],read(new)['subjects']
    subjects={}
    for subject in reports[0]:
        first,last=[r[subject]['views'] for r in reports]
        views={}
        for key in first:
            baseline,candidate=first[key],last[key]
            # The portable w2c may differ after its coordinate export. The
            # preceding diagnostic already checked exact common world cameras.
            need(all(baseline['camera'][name]==candidate['camera'][name] for name in ('K','wh')) and
                 baseline['panel']['crop_xyxy']==candidate['panel']['crop_xyxy'], 'R25 comparison camera grid changed.')
            comparisons=[]
            for a,b in zip(baseline['pairs'],candidate['pairs'],strict=True):
                ids=('region','artifact_xy_px','control_xy_px')
                need(all(a[k]==b[k] for k in ids), 'R25 comparison picked a different ray.')
                item={k:a[k] for k in ids}
                item['parity']=[a['parity'],b['parity']]
                if all(row['parity']['passed'] for row in (a,b)):
                    values=('artifact_alpha','control_alpha','artifact_rgb','control_rgb','delta_luma','background_luma')
                    item['r24']={k:a[k] for k in values}|dict(layer_contributions=groups(a))
                    item['r25']={k:b[k] for k in values}|dict(layer_contributions=groups(b))
                    item['r25_minus_r24']=dict(artifact_alpha=b['artifact_alpha']-a['artifact_alpha'],
                        control_alpha=b['control_alpha']-a['control_alpha'],
                        abs_luma_contrast=abs(b['delta_luma'])-abs(a['delta_luma']))
                comparisons.append(item)
            views[key]=comparisons
        subjects[subject]=dict(views=views,pick_board=board(output,subject,first,last))
    result=output/'r24-r25-side-waist-hem-paired.json'
    with result.open('x',encoding='utf-8') as f:
        json.dump(dict(schema='shuolee-r24-r25-fixed-side-ray-pair/1',r24_sha256=sha(old),r25_sha256=sha(new),
            source_sha256=sha(__file__),subjects=subjects,
            interpretation='The same stored cameras and nearby controls are used for the measured contrast. A smaller contrast magnitude alone does not establish corrected geometry or quality.'),
            f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')
    print(dict(file=str(result),sha256=sha(result)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('r25_attribution_sha256')
    run(parser.parse_args().r25_attribution_sha256)
