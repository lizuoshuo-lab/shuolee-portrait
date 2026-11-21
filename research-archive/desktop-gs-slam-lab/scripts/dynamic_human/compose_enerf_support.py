"""Unenhanced native comparison panels and continuous fixed-box videos."""
from pathlib import Path
import json,subprocess,argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--split',choices=['heldout_view','train'],default='heldout_view');args=ap.parse_args()
LAB=Path(__file__).resolve().parents[2];out=LAB/'outputs/dynamic-human-enerf-support-030';dest=out/('evaluation' if args.split=='heldout_view' else 'training-evaluation');report=json.loads((dest/'metrics.json').read_text());panels=out/('comparison' if args.split=='heldout_view' else 'training-comparison');panels.mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18);records=[]
groups=['reference']+report['models'];labels=['Reference','Fixed scale / opacity','Scale adapts','Opacity adapts','Both adapt'];ffmpeg=Path('C:/Users/Administrator/ffmpeg/bin/ffmpeg.exe')
for cam in report['cameras']:
    rows=[r for r in report['views'] if r['camera']==cam and r['model']=='frozen'];boxes=np.array([p['bbox'] for r in rows for p in r['people']]);union=[int(boxes[:,0].min()),int(boxes[:,1].min()),int(boxes[:,2].max()),int(boxes[:,3].max())];frames_dir=panels/f'camera{cam}-frames';frames_dir.mkdir(exist_ok=True)
    for index,t in enumerate(report['frames']):
        images=[Image.open(dest/f'camera{cam}-frame{t:03d}-{g}.png').convert('RGB') for g in groups];row=next(r for r in rows if r['common_frame']==t);w=union[2]-union[0];h=union[3]-union[1];canvas=Image.new('RGB',(w*len(groups),h+44),'white');draw=ImageDraw.Draw(canvas)
        for j,(im,label) in enumerate(zip(images,labels)):canvas.paste(im.crop(union),(j*w,44));draw.text((j*w+8,12),f'{label} | {t}',font=font,fill='black')
        canvas.save(frames_dir/f'{index:03d}.png',compress_level=1)
        if t in [34,46,50,58]:
            for p in row['people']:
                box=p['upper_body_bbox'];pw=box[2]-box[0];ph=box[3]-box[1];can=Image.new('RGB',(len(groups)*pw,ph+44),'white');d=ImageDraw.Draw(can)
                for j,(im,label) in enumerate(zip(images,labels)):can.paste(im.crop(box),(j*pw,44));d.text((j*pw+6,12),label,font=font,fill='black')
                name=f'camera{cam}-frame{t:03d}-person{p["person"]}.png';can.save(panels/name);records.append(dict(file=name,camera=cam,common_frame=t,source_frame=row['source_frame'],person=p['person'],crop=box,resized=False))
                compact=Image.new('RGB',(3*pw,ph+44),'white');d=ImageDraw.Draw(compact)
                for j,k in enumerate([0,1,4]):compact.paste(images[k].crop(box),(j*pw,44));d.text((j*pw+6,12),labels[k],font=font,fill='black')
                compact.save(panels/f'camera{cam}-frame{t:03d}-person{p["person"]}-compact.png')
    if args.split=='heldout_view':subprocess.run([str(ffmpeg),'-y','-loglevel','error','-framerate','12','-i',str(frames_dir/'%03d.png'),'-vf','scale=2500:-2','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(panels/f'camera{cam}-continuous-comparison.mp4')],check=True)
spec=dict(status='native_stills_and_fixed_union_videos_composed_review_pending',stills=records,columns=labels,compact_columns=['Reference','Fixed scale / opacity','Both adapt'],frames=report['frames'],videos='25frames per camera, fixed union of both people. Encoded2500px wide at12fps for inspection; capture FPS not independently verified. Native full images and unscaled crop PNGs retained.',enhancement=False)
spec['split']=args.split
if args.split=='train':spec['videos']='None; three separated training times are still probes, not continuous playback.'
(panels/'composition.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(stills=len(records),compact_stills=len(records),videos=3 if args.split=='heldout_view' else 0)))
