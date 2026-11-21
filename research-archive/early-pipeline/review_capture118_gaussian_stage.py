"""Prepare unwarped raw/mask/render crops for actual stage review; no text burned in."""
import argparse,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).parent))
from capture118_all_subjects import ROOT,REC,rt
ap=argparse.ArgumentParser();ap.add_argument('--subject',required=True);ap.add_argument('--step',type=int,required=True);a=ap.parse_args()
root=ROOT/a.subject;snapshot=root/'gaussian-run'/f'step-{a.step:06d}'
assert (snapshot/'native-ply-parity.json').exists()
out=REC/f'{a.subject}-gaussian-{a.step:06d}-review';out.mkdir(exist_ok=True)
assert not (out/'index.json').exists()
rows={r['frame']:r for r in rt.read(root/'data/training-inputs.json')['records']};index=[]
for raw in sorted(snapshot.glob('*_input.png')):
 if raw.name.endswith('_masked_input.png'):continue
 name=raw.name.removesuffix('_input.png');r=rows[name]
 ims=[Image.open(raw).convert('RGB'),Image.open(snapshot/(name+'_masked_input.png')).convert('RGB'),Image.open(snapshot/(name+'_render.png')).convert('RGB')]
 assert len(set(im.size for im in ims))==1
 w,h=ims[0].size;s=np.array([w/r['wh'][0],h/r['wh'][1]]*2)
 for region,b in [('body',r['initial_box_xyxy']),('face',r['face_box_xyxy'])]:
  if b is None:continue
  box=np.asarray(b,dtype=float)*s;pad=max(8,(box[3]-box[1])*.1);box+=np.array([-pad,-pad,pad,pad]);box=np.rint(np.clip(box,[0,0,0,0],[w,h,w,h])).astype(int)
  tiles=[im.crop(tuple(box)) for im in ims];ratio=(550 if region=='body' else 300)/tiles[0].height
  size=(round(tiles[0].width*ratio),round(tiles[0].height*ratio));tiles=[im.resize(size,Image.Resampling.LANCZOS) for im in tiles]
  panel=Image.new('RGB',(size[0]*3,size[1]),'white')
  for j,im in enumerate(tiles):panel.paste(im,(j*size[0],0))
  filename=f'{name}_{region}_raw_mask_render.png';panel.save(out/filename)
  index.append(dict(file=filename,frame=name,region=region,crop_xyxy=[int(v) for v in box],render_size=[w,h],columns=['raw','masked_input','render'],same_crop_all_columns=True,aspect_preserved=True))
rt.save(out/'index.json',dict(created=rt.now(),subject=a.subject,step=a.step,snapshot_sha256=rt.sha(snapshot/'world.ply'),images=index,actual_review_pending=True))
print(str(out),flush=True)
