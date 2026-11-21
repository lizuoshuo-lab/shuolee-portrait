"""Build raw-pixel material and mask evidence; no optimization or image warping."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).parent))
from capture118_all_subjects import ROOT,REC,rt
ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True);args=ap.parse_args();root=ROOT/args.subject
rows=rt.read(root/'data/training-inputs.json')['records'];out=REC/(args.subject+'-input-review');out.mkdir(exist_ok=True)
assert not (out/'index.json').exists(),'Completed review assets are immutable.'
obs=dict(np.load(root/'data/observations.npz'));images={};masks={};features={}
def im(i):
 if i not in images:images[i]=Image.open(rows[i]['image']).convert('RGB')
 return images[i]
def xy(i,j):
 if i not in features:features[i]=np.load(root/f'camera/features-{i:03d}.npz')['xy']
 return features[i][j]
def region(i,j):
 u,v=xy(i,j);r=rows[i];b=r['face_box_xyxy']
 if i not in masks:masks[i]=np.asarray(Image.open(r['mask']).convert('L'))>127
 x,y=np.rint([u,v]).astype(int);fg=0<=x<r['wh'][0] and 0<=y<r['wh'][1] and masks[i][y,x]
 if fg and b and b[0]<=u<=b[2] and b[1]<=v<=b[3]:return 'face'
 return 'body' if fg else 'background'
held=rt.read(root/'camera/held-evaluation.json')['records'];groups={k:[] for k in ['face','body','background','mixed']}
for tr in held:
 labels=[region(i,j) for i,j in tr['nodes']];l=max(set(labels),key=labels.count)
 if labels.count(l)<.8*len(labels):l='mixed'
 tr['region_heuristic']=l;groups[l].append(tr)
summary={}
for label,items in groups.items():
 e=np.asarray([e for t in items for e in t['errors']]);summary[label]=dict(tracks=len(items),test_observations=len(e),median=float(np.median(e)) if len(e) else None,p90=float(np.quantile(e,.9)) if len(e) else None,
   over3px=int((e>3).sum()),negative_depth_tracks=sum(not t['positive_depth'] for t in items))
selected=[]
for label in ['face','body']:
 selected += sorted(groups[label],key=lambda t:(-len(t['nodes']),t['id']))[:6]
selected+=sorted(groups['background'],key=lambda t:-max(t['errors']))[:3]
selected+=sorted(groups['background'],key=lambda t:(abs(np.median(t['errors'])-summary['background']['median']),t['id']))[:3]
index=[]
for ti,tr in enumerate(selected):
 nodes=tr['nodes'];inds=np.unique(np.linspace(0,len(nodes)-1,min(len(nodes),6)).astype(int));tiles=[];selected_nodes=[]
 for oi in inds:
  i,j=nodes[oi];u,v=xy(i,j);x,y=np.rint([u,v]).astype(int);box=(x-48,y-48,x+48,y+48)
  tile=im(i).crop(box).resize((192,192),Image.Resampling.NEAREST);draw=ImageDraw.Draw(tile);draw.line((88,96,104,96),fill=(255,20,80),width=1);draw.line((96,88,96,104),fill=(255,20,80),width=1)
  tiles.append(tile);selected_nodes.append(dict(frame=rows[i]['frame'],xy=[float(u),float(v)],crop_xyxy=[int(z) for z in box]))
 sheet=Image.new('RGB',(192*6,192),(25,25,25))
 for i,tile in enumerate(tiles):sheet.paste(tile,(i*192,0))
 name=f'{ti:02d}_{tr["region_heuristic"]}_{tr["id"][:10]}_原像素特征点.jpg';sheet.save(out/name,quality=94)
 index.append(dict(file=name,track_id=tr['id'],region_heuristic=tr['region_heuristic'],nodes=selected_nodes,all_test_errors=tr['errors'],positive_depth=tr['positive_depth']))
for i in np.unique(np.linspace(0,len(rows)-1,6).astype(int)):
 r=rows[i];b=np.array(r['initial_box_xyxy']);b[:2]-=20;b[2:]+=20;b=np.rint(b).astype(int);b=np.clip(b,[0,0,0,0],[r['wh'][0],r['wh'][1],r['wh'][0],r['wh'][1]])
 rgb=im(i).crop(tuple(b));mask=Image.open(r['mask']).convert('RGB').crop(tuple(b));scale=650/rgb.height;size=(round(rgb.width*scale),650)
 panel=Image.new('RGB',(size[0]*2,650));panel.paste(rgb.resize(size,Image.Resampling.LANCZOS),(0,0));panel.paste(mask.resize(size,Image.Resampling.NEAREST),(size[0],0))
 panel.save(out/f'{r["frame"]}_原图及人物mask.png')
rt.save(out/'index.json',dict(created=rt.now(),subject=args.subject,metrics=summary,selected_tracks=index,
 camera_sha256=rt.sha(root/'camera/candidate-cameras.json'),observations_sha256=rt.sha(root/'data/observations.npz'),
 note='Original96px windows at2x with a small point marker, no text. Crop centers align for inspection only, not geometric alignment. Category uses mask and detector face ROI, not ground truth.',physical_identity_reviewed=False))
print(json.dumps(summary),flush=True)
