"""Readonly D4 numerical summary and original-pixel review sheets; no coordinate cleanup."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'residual-distortion-direction';O=R/'native-edge-profile';DATA=Path('D:/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'review-inputs').exists();(O/'review-inputs').mkdir()
r=read(O/'result.json');old={x['id']:x for x in read(D/'measurements.json')['records']};p=read(D/'plan.json');cache={};rows=[]
def rmsline(uv):
 uv=np.array(uv);v=uv-uv.mean(0);_,_,vh=np.linalg.svd(v,full_matrices=False)
 return float(np.sqrt(np.mean((v@vh[-1])**2)))
def crop(image,uv,side):
 x,y=np.rint(uv).astype(int);box=[int(x-side//2),int(y-side//2),int(x+side//2),int(y+side//2)]
 return image.crop(box),box
for l in r['records']:
 f=l['frame'];path=DATA/f'frame_{f:06d}.png'
 if f not in cache:
  assert sha(path)==p['input_hashes'][path.name];cache[f]=Image.open(path).convert('RGB')
 b=old[l['id']];samples=l['samples'];uv=np.array([x['uv'] for x in samples]);old_uv=np.array(b['points']);valid=np.array(b['valid'])
 shift=np.linalg.norm(uv-old_uv,axis=1);indices=np.flatnonzero(valid);extreme=indices[np.argsort(-shift[indices],kind='stable')[:2]]
 rms=np.array([x['rms_gray'] for x in samples]);bounds=np.array([x['bound_hit'] for x in samples]);std=np.array([x['formal_offset_std_px'] for x in samples])
 canvas=Image.new('RGB',(672,96));cells=[]
 for j,i in enumerate([0,20,40,60,80,100,119]):
  im,box=crop(cache[f],uv[i],96);canvas.paste(im,(j*96,0));cells.append(dict(index=i,new_uv=uv[i].tolist(),box=box,old_eligible=bool(valid[i]),model_rms_gray=float(rms[i]),bound_hit=bool(bounds[i])))
 uniform=f'{l["id"]}_uniform.png';canvas.save(O/'review-inputs'/uniform)
 differences=[]
 for rank,i in enumerate(extreme):
  image=Image.new('RGB',(384,192));tiles=[]
  for j,pt in enumerate([old_uv[i],uv[i]]):
   im,box=crop(cache[f],pt,192);image.paste(im,(192*j,0));tiles.append(dict(uv=pt.tolist(),box=box))
  name=f'{l["id"]}_difference_{rank}.png';image.save(O/'review-inputs'/name)
  differences.append(dict(index=int(i),shift_px=float(shift[i]),file=name,tiles=tiles,model=samples[i]))
 rows.append(dict(id=l['id'],frame=f,original_eligible=int(valid.sum()),all_fits=len(samples),
  numerical_success=sum(x['numerical_success'] for x in samples),bound_hits=int(bounds.sum()),
  gray_rms_median=float(np.median(rms)),gray_rms_max=float(rms.max()),formal_std_median=float(np.median(std)),
  old_eligibility_frozen_rms_before=rmsline(old_uv[valid]),old_eligibility_frozen_rms_after=rmsline(uv[valid]),
  max_location_disagreement_px=float(shift[valid].max()),median_location_disagreement_px=float(np.median(shift[valid])),
  uniform_file=uniform,cells=cells,differences=differences))
pages=[]
for start in [0,6]:
 group=rows[start:start+6];img=Image.new('RGB',(672,96*len(group)))
 for j,row in enumerate(group):img.paste(Image.open(O/'review-inputs'/row['uniform_file']),(0,96*j))
 name=f'uniform_page_{start:02d}.png';img.save(O/'review-inputs'/name);pages.append(dict(file=name,ids=[x['id'] for x in group]))
for start in [0,4,8]:
 group=rows[start:start+4];img=Image.new('RGB',(768,192*len(group)))
 for j,row in enumerate(group):
  for k,dd in enumerate(row['differences']):img.paste(Image.open(O/'review-inputs'/dd['file']),(384*k,192*j))
 name=f'difference_page_{start:02d}.png';img.save(O/'review-inputs'/name);pages.append(dict(file=name,ids=[x['id'] for x in group]))
save(O/'summary.json',dict(rows=rows,pages=pages,all_estimates_retained=True,
 comparison='Coordinates and native local intensities only. TLS diagnostic uses exactly frozen D1 eligibility, no new point deletion. Not a fit of distortion/cameras, or physical truth.',
 review_selection='All11uniform rows at original fixedindices plus2largest method disagreements per line, selected only to inspect failures, never refit or clean calibration.'))
print(json.dumps([{k:v for k,v in row.items() if k not in ['cells','differences','uniform_file']} for row in rows]))
