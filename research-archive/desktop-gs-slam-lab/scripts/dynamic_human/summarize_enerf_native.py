from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw
import numpy as np

lab=Path(__file__).resolve().parents[2];out=lab/'outputs/dynamic-human-enerf-native-025';last=out/'last-frame-capacity-control-v2/first-frame/evaluation';motion=out/'motion/evaluation';m=json.loads((motion/'metrics.json').read_text());s=json.loads((last/'metrics.json').read_text());dest=out/'comparison';dest.mkdir(exist_ok=True)
rows=[]
for c in ['00','06','12']:
    row=next(x for x in s['views'] if x['split']=='heldout_view' and x['camera']==c);dyn=next(x for x in m['views'] if x['camera']==c and x['frame']==59);box=row['people'][1]['bbox'];ref=Image.open(last/f'heldout_view-camera{c}-reference.png');assert np.array_equal(np.array(ref),np.array(Image.open(motion/f'camera{c}-frame059-reference.png')))
    images=[ref.crop(box),Image.open(motion/f'camera{c}-frame059-render.png').crop(box),Image.open(last/f'heldout_view-camera{c}-render.png').crop(box)];w,h=images[0].size;canvas=Image.new('RGB',(w*3,h+44),(245,245,245));draw=ImageDraw.Draw(canvas)
    for i,(im,label) in enumerate(zip(images,['Reference','Persistent motion model','Independent frame59 fit'])):canvas.paste(im,(i*w,44));draw.text((i*w+8,12),label,fill='black')
    canvas.save(dest/f'camera{c}-frame059-capacity-comparison.png');rows.append(dict(camera=c,frame=59,foreground_motion_psnr_db=dyn['foreground_psnr_db'],foreground_independent_fit_psnr_db=row['foreground_psnr_db'],display_bbox_person2=box))
result=dict(status='independent_difficult_frame_fit_does_not_resolve_moving_hand_failure',views=rows,scope='Diagnostic with separate 6000-step 3DGS initialization at frame59; changes initialization, density, optimized parameters, iteration allocation and temporal constraints. Not a single-factor ablation or continuous model replacement.',selection='Frame59 is fixed segment endpoint; heldout cams00/06/12 never used in fitting; final6000-step checkpoint fixed in advance',inference='Motion accumulation alone is not a sufficient explanation of the observed failure. Model optimization/capacity and actual moving-region multiview consistency remain to test; no proven synchronization defect.')
(dest/'capacity-summary.json').write_text(json.dumps(result,indent=2)+'\n')
review=dict(status='movement_and_detail_partial_goal_not_accepted',viewed=['first-frame all3heldout cameras bothpeople native crops','motion contacts cameras00/06/12 at15,30,45,59','motion native camera00 frame59 full reference andrender','motion camera06 frame59 andcamera12 frame45 native paired crops','training cameras01/07 frame21 native paired crops','independent frame59 fit all3heldout cameras person2 native crops'],conclusion='Torso text/stripes and legs remain recognizable. Raised hands and held rotating objects blur/break; native camera00 frame59 has a far-background floater. Independent frame59 fit also fails these moving details. No full-scene clarity acceptance.',review_scope='Sampled visual inspection, not every video frame',playback='30fps inspection, capture fps unverified')
(out/'motion-visual-review.json').write_text(json.dumps(review,indent=2)+'\n')
(out/'last-frame-capacity-control/failure-receipt.json').write_text(json.dumps(dict(status='not_used_for_training_or_evaluation',reason='Initial frame-selectable data preparation still wrote hardcoded frame0 metadata; preserved and rerun from corrected frame-aware scripts into last-frame-capacity-control-v2'),indent=2)+'\n')
print(json.dumps(result,indent=2))
