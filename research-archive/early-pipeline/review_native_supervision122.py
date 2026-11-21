"""Read completed T5 proposals and prepare unannotated audit images.

No editing of masks, cameras or observations. Thumbnail review is coarse only.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,time
import numpy as np
from PIL import Image
O=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-supervision')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
r=read(O/'result.json');rows=read(O/'mask-rows.json')
assert r['state']=='draft_supervision_complete_pending_actual_review' and len(rows)==464
out=O/'review-inputs';assert not out.exists();out.mkdir()
start=time.monotonic();previous=None;metrics=[];pages=[]
for offset in range(0,len(rows),32):
 # Every mask appears once, in frame order. This is only a coarse envelope
 # review; full-resolution RGB/masks remain the required source for detail.
 canvas=Image.new('RGB',(152*8,270*4),(12,14,13))
 for j,row in enumerate(rows[offset:offset+32]):
  path=Path(row['alpha_proposal']['path']);assert sha(path)==row['alpha_proposal']['sha256']
  with Image.open(path) as im:
   mask=np.array(im)>0;thumb=im.resize((152,270),Image.Resampling.NEAREST).convert('RGB')
  canvas.paste(thumb,(j%8*152,j//8*270))
  inter=None if previous is None else int(np.count_nonzero(mask&previous))
  union=None if previous is None else int(np.count_nonzero(mask|previous))
  metrics.append(dict(frame_id=row['frame_id'],mask_pixels=int(mask.sum()),previous_mask_iou=None if previous is None else inter/union))
  previous=mask
 name=f'all_masks_{offset:03d}_{min(offset+31,463):03d}.png';canvas.save(out/name)
 pages.append(dict(file=name,frames=[x['frame_id'] for x in rows[offset:offset+32]],layout='8columns4rows, lefttoright then topdown',scale='152x270 coarsepreview; original1520x2704 remainsunchanged'))
for f in [0,116,232,348,463]:
 row=rows[f];p=Path(row['rgb']['path']);assert sha(p)==row['rgb']['sha256']
 rgb=np.array(Image.open(p).convert('RGB'));mask=np.array(Image.open(row['alpha_proposal']['path']))>0
 cut=rgb.copy();cut[~mask]=240
 # Side-by-side complete original and cutout, same fixed display scaling.
 pair=Image.new('RGB',(608,541));pair.paste(Image.fromarray(rgb).resize((304,541),Image.Resampling.LANCZOS),(0,0));pair.paste(Image.fromarray(cut).resize((304,541),Image.Resampling.LANCZOS),(304,0));pair.save(out/f'full_{f:06d}.png')
 x0,y0,x1,y1=row['mask_bbox_xyxy'];cx=(x0+x1)//2
 waist=int(y0+.52*(y1-y0))
 regions={'head':(cx-256,y0-16,cx+256,y0+496),
          'hands_clothing':(cx-384,waist-384,cx+384,waist+384),
          'feet':(cx-384,y1-464,cx+384,y1+48)}
 for name,box in regions.items():
  box=(max(0,box[0]),max(0,box[1]),min(1520,box[2]),min(2704,box[3]));a=Image.fromarray(rgb).crop(box);b=Image.fromarray(cut).crop(box)
  pairs=Image.new('RGB',(a.width*2,a.height));pairs.paste(a,(0,0));pairs.paste(b,(a.width,0));pairs.save(out/f'{name}_{f:06d}.png')
save(out/'index.json',dict(pages=pages,native_detail_frames=[0,116,232,348,463],native_detail_regions=['head','hands_clothing','feet'],
 image_operations='All images are actual masks/source crops; no text in pixels. Fullpair/overview scales only for display. head/feet crops have no resize.',
 not_complete_material_or_edge_review=True,generated_seconds=time.monotonic()-start))
save(out/'temporal-mask-metrics.json',metrics)
save(out/'summary.json',dict(masks=464,pages=len(pages),native_detail_frames=5,minimum_adjacent_mask_iou=min(m['previous_mask_iou'] for m in metrics[1:]),
 minimum_iou_frame=min(metrics[1:],key=lambda m:m['previous_mask_iou'])['frame_id'],maximum_mask_pixels=max(m['mask_pixels'] for m in metrics),minimum_mask_pixels=min(m['mask_pixels'] for m in metrics),
 geometric_alignment_claim=False,physical_supervision_accepted=False))
print(json.dumps(read(out/'summary.json')),flush=True)
