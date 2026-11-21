"""Create lossless source crops for D3 review, without refitting or selecting a new threshold."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
D=R/'residual-distortion-direction';O=R/'residual-edge-association'
DATA=Path('D:/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'review-inputs').exists()
(O/'review-inputs').mkdir()
result=read(O/'result.json');plan=read(D/'plan.json');old={x['id']:x for x in read(D/'measurements.json')['records']};cache={};rows=[]
for r in result['records']:
 f=r['frame'];path=DATA/f'frame_{f:06d}.png'
 if f not in cache:
  assert sha(path)==plan['input_hashes'][path.name];cache[f]=Image.open(path).convert('RGB')
 l=old[r['id']];a,b=np.array(l['a']),np.array(l['b']);normal=np.array(l['normal'])
 inds=[r['anchor']['index']]
 for key in ['-1','1']:
  st=r['stops'].get(key)
  if st:inds.append(st['index'])
 chosen=[i for i,v in enumerate(r['selected_uv']) if v is not None]
 if chosen:inds.extend([chosen[0],chosen[-1]])
 inds=list(dict.fromkeys(inds))
 canvas=Image.new('RGB',(5*160,160),(242,242,242));cells=[]
 for j,i in enumerate(inds):
  uv=a+(b-a)*i/119+normal*r['anchor']['normal_offset_px']
  x,y=np.rint(uv).astype(int);box=[int(x-80),int(y-80),int(x+80),int(y+80)]
  canvas.paste(cache[f].crop(box),(160*j,0))
  cells.append(dict(profile=i,box=box,peaks=r['all_peaks'][i],selected=r['selected_peak_indices'][i],state=r['states'][i]))
 filename=f'{r["id"]}_anchor_stops_endpoints.png';canvas.save(O/'review-inputs'/filename)
 rows.append(dict(id=r['id'],frame=f,file=filename,cells=cells,selected_count=r['selected_count'],old_eligible_count=r['old_eligible_count']))
pages=[]
for start in [0,6]:
 subset=rows[start:start+6];canvas=Image.new('RGB',(800,160*len(subset)),(242,242,242))
 for i,r in enumerate(subset):canvas.paste(Image.open(O/'review-inputs'/r['file']),(0,i*160))
 name=f'page_{start:02d}.png';canvas.save(O/'review-inputs'/name);pages.append(dict(file=name,rows=[r['id'] for r in subset]))
save(O/'review-inputs/index.json',dict(rows=rows,pages=pages,native_pixels=True,images_annotated=False,crops_only=True,selected=sum(r['selected_count'] for r in rows),old_eligible=sum(r['old_eligible_count'] for r in rows)))
print(json.dumps(dict(pages=pages,selected=sum(r['selected_count'] for r in rows),old_eligible=sum(r['old_eligible_count'] for r in rows))))
