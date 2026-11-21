"""Native-pixel comparisons. Composition only: no sharpening/resampling."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-time-control-027';dest=out/'comparison';dest.mkdir(exist_ok=True)
conditions=['same-index','offset-index'];reports=[json.loads((out/c/'first-frame/evaluation/metrics.json').read_text()) for c in conditions];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);summary=[]
for c in ['00','06','12','01','13']:
    split='heldout_view' if c in ['00','06','12'] else 'train';name=f'{split}-camera{c}';rows=[next(r for r in report['views'] if r['camera']==c) for report in reports];p=[out/g/'first-frame/evaluation' for g in conditions]
    gt=Image.open(p[0]/f'{name}-reference.png').convert('RGB');assert np.array_equal(gt,Image.open(p[1]/f'{name}-reference.png'))
    sources=[gt]+[Image.open(q/f'{name}-render.png').convert('RGB') for q in p];labels=['Reference','Same frame index','Per-camera index offset']
    for person in rows[0]['people']:
        box=person['bbox'];x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
        # Full person retained, with a separate upper-body crop for detail.
        for kind,crop in [('person',(x0,y0,x1,y1)),('upper-body',(x0,y0,x1,min(y1,y0+int(.5*h))))]:
            pw=crop[2]-crop[0];ph=crop[3]-crop[1];can=Image.new('RGB',(3*pw,ph+44),'white');d=ImageDraw.Draw(can)
            for k,(im,label) in enumerate(zip(sources,labels)):can.paste(im.crop(crop),(k*pw,44));d.text((k*pw+8,12),label,font=font,fill='black')
            can.save(dest/f'camera{c}-person{person["person_left_to_right"]}-{kind}.png')
    summary.append(dict(camera=c,split=split,frame=49,same_index_psnr_db=rows[0]['foreground_psnr_db'],offset_index_psnr_db=rows[1]['foreground_psnr_db'],difference_db=rows[1]['foreground_psnr_db']-rows[0]['foreground_psnr_db']))
result=dict(status='paired_single_instant_control_evaluated_visual_review_pending',conditions=conditions,views=summary,heldout_mean_difference_db=float(np.mean([r['difference_db'] for r in summary if r['split']=='heldout_view'])),common_settings='6000 steps, seed73, exact same initial points/colors, training cameras and density strategy; observations at selected per-camera frame changed',limits=['Single instant only','Heldout-camera time offsets are unestimated; their frame49 references remain untouched','Train07 omitted from paired-image panels because its target frame changes49 to48','Images use author matte on gray; no static scene learned','Upper-body crops do not measure isolated hand accuracy'])
(dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
