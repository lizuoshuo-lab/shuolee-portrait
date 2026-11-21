"""Build native correspondence review after model-independent uniqueness screening."""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075';I=L/'outputs/dynamic-human-enerf-articulated-control-064';O=R/'unique-patches/review';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
rows=read(R/'endpoint-records.json')['records'];masks={c:dict(np.load(R/'unique-patches'/f'camera{c}.npz')) for c in read(R/'protocol.json')['cameras']};tracks={c:dict(np.load(R/'tracks'/f'camera{c}.npz')) for c in masks}
rows=[r for r in rows if masks[r['camera']]['accepted'][list(tracks[r['camera']]['target_frame']).index(r['frame']),r['feature']]]
def stats(v):return dict(n=len(v),median=float(np.median(v)),p90=float(np.percentile(v,90)),mean=float(np.mean(v)),max=float(np.max(v))) if len(v) else dict(n=0)
summary=dict(status='unique_photo_tracks_model_comparison_ready_visual_review_pending',common_observations=len(rows),
    baseline_endpoint_px=stats([r['before_endpoint_px'] for r in rows]),correctRGB_endpoint_px=stats([r['after_endpoint_px'] for r in rows]),
    hand_box_observations=sum(r['region']=='right_hand_box' for r in rows),
    hand_box_endpoint_px=stats([r['after_endpoint_px'] for r in rows if r['region']=='right_hand_box']),
    worst=max(rows,key=lambda r:r['after_endpoint_px']),model_updates=0,training_labels_accepted=False,goal_complete=False)
save(R/'unique-patches/model-summary.json',summary)
save(R/'unique-patches/paired-endpoints.json',dict(records=rows,goal_complete=False))
cache={}
def get(c,t):
    if (c,t) not in cache:cache[c,t]=np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz')['rgb']
    return cache[c,t]
def cross(im,xy,color,size=3):
    d=ImageDraw.Draw(im);x,y=map(float,xy);d.line((x-size,y,x+size,y),fill=color,width=1);d.line((x,y-size,x,y+size),fill=color,width=1)
def make(rs,box,path,title):
    c,t=rs[0]['camera'],rs[0]['frame'];x0,y0,x1,y1=box;w=x1-x0;h=y1-y0;cw=max(w,240)
    plain=[Image.fromarray(get(c,750)[y0:y1,x0:x1]),Image.fromarray(get(c,t)[y0:y1,x0:x1])]
    marked=[x.copy() for x in plain]
    for r in rs:
        for pic,key,color in [(marked[0],'source_xy',(255,180,0)),(marked[1],'tracked_xy',(0,220,70)),(marked[1],'model_xy',(255,70,20))]:cross(pic,np.array(r[key])-[x0,y0],color)
    sheet=Image.new('RGB',(4*cw,h+48),'white');d=ImageDraw.Draw(sheet)
    labels=['Original750',f'Original{t}','Source seeds',f'Photo{t}: green track / red model']
    for j,(im,label) in enumerate(zip(plain+marked,labels)):
        sheet.paste(im,(j*cw,48));d.text((j*cw+4,6),label,fill='black')
    d.text((4,27),title,fill='black');sheet.save(path)
    return dict(file=str(path.relative_to(R)),camera=c,frame=t,features=[r['feature'] for r in rs],bbox=box)
panels=[]
for c in ['01','04','07','08']:
    for t in [744,756]:
        available=[r for r in rows if r['camera']==c and r['frame']==t];sel=[]
        for actor in [0,1]:
            picked=[]
            for r in available:
                if r['actor']!=actor:continue
                if all(np.linalg.norm(np.array(r['source_xy'])-p['source_xy'])>=35 for p in picked):picked.append(r)
                if len(picked)>=10:break
            sel+=picked
        if not sel:continue
        pts=np.array([r[k] for r in sel for k in ['source_xy','tracked_xy','model_xy']]);lo=np.floor(pts.min(0)-30).astype(int);hi=np.ceil(pts.max(0)+30).astype(int)
        box=[max(0,int(lo[0])),max(0,int(lo[1])),min(1920,int(hi[0])),min(1080,int(hi[1]))]
        panels.append(make(sel,box,O/f'camera{c}-frame{t:06d}-coverage.png',f'Camera{c} | {len(sel)} spatially spaced features | all columns original photos with optional annotation'))
for n,r in enumerate([r for r in rows if r['region']=='right_hand_box']+[summary['worst']]):
    pts=np.array([r[k] for k in ['source_xy','tracked_xy','model_xy']]);lo=np.floor(pts.min(0)-55).astype(int);hi=np.ceil(pts.max(0)+55).astype(int)
    box=[max(0,int(lo[0])),max(0,int(lo[1])),min(1920,int(hi[0])),min(1080,int(hi[1]))]
    filename=f"detail{n:02d}-camera{r['camera']}-frame{r['frame']:06d}.png"
    panels.append(make([r],box,O/filename,f"Camera{r['camera']} | feature{r['feature']} | {'hand-box candidate' if n<summary['hand_box_observations'] else 'largest remaining disagreement'}"))
save(R/'unique-patches/visual-review-plan.json',dict(status='eight_fixed_camera_time_coverage_views_all_surviving_hand_box_observations_and_worst',records=panels,actual_review_complete=False,goal_complete=False))
print(json.dumps(summary));print('Panels',len(panels))
