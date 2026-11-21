"""Read-only inspection of the frozen D1 samples. Never refit coefficients or remove points."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np
from PIL import Image

T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
R=T/'shuolee/current/robust-portrait-122'; D=R/'residual-distortion-direction'; O=R/'residual-edge-identity'
DATA=Path('D:/GS-SLAM-Data/robust122-whiteT-native-supervision/rgb')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()

assert not (O/'plan.json').exists()
O.mkdir(exist_ok=False);(O/'crops').mkdir()
source=[D/'plan.json',D/'measurements.json',D/'line-review.json',D/'result.json']
hashes={p.name:sha(p) for p in source}
P=read(D/'plan.json');M=read(D/'measurements.json');L=read(D/'line-review.json')
save(O/'plan.json',dict(stage='D2_edge_identity_audit',updated=now(),source_hashes=hashes,
 hypothesis='Frozen D1 strongest-gradient samples may jump between adjacent physical borders or respond to cross-joints. This can contaminate the lens-model test.',
 single_variable='None. Read-only measurement audit; no replacement coordinates or coefficients.',
 crop_selection=dict(all_registered_lines=13,uniform_indices=[0,20,40,60,80,100,119],
   additional_diagnostic='Two largest absolute adjacent normal-offset changes with both samples originally eligible; retain both source patches. This is diagnosis only, never a new fitting/holdout selection.'),
 statistics='All original eligible observations retained. Report adjacent offset change and second difference; not a camera-error estimate or an automatic exclusion rule.',
 budget='One CPU pass over13 existing line records and3 original PNGs; no full video decode, newfeatures, camera, distortion fitting or training.',
 stop='Report actual boundary identity/occlusion ambiguity and measurement limitations. Do not tune thresholds, clean residuals and rerun D1.',
 no_new_calibration=True,no_training=True))
cache={};rows=[];pages=[]
for li,l in enumerate(M['records']):
 f=l['frame'];path=DATA/f'frame_{f:06d}.png'
 if f not in cache:
  assert sha(path)==P['input_hashes'][path.name]
  cache[f]=Image.open(path).convert('RGB')
 uv=np.array(l['points']);valid=np.array(l['valid']);offset=np.array(l['offsets'])
 ok=valid[:-1]&valid[1:];dif=np.diff(offset)
 indices=np.flatnonzero(ok);order=indices[np.argsort(-np.abs(dif[ok]),kind='stable')[:2]]
 cells=[];canvas=Image.new('RGB',(7*96,96),(240,240,240))
 for col,idx in enumerate([0,20,40,60,80,100,119]):
  x,y=np.rint(uv[idx]).astype(int);box=list(map(int,[x-48,y-48,x+48,y+48]))
  canvas.paste(cache[f].crop(box),(col*96,0));cells.append(dict(sample=idx,uv=uv[idx].tolist(),eligible=bool(valid[idx]),reason=l['reasons'][idx],box=box))
 name=f'{li:02d}_{l["id"]}_uniform.png';canvas.save(O/'crops'/name)
 details=[]
 for rank,idx in enumerate(order):
  pair=Image.new('RGB',(256,128));paircells=[]
  for col,k in enumerate([int(idx),int(idx+1)]):
   x,y=np.rint(uv[k]).astype(int);box=list(map(int,[x-64,y-64,x+64,y+64]))
   pair.paste(cache[f].crop(box),(col*128,0));paircells.append(dict(sample=k,uv=uv[k].tolist(),box=box))
  filename=f'{li:02d}_{l["id"]}_jump{rank}.png';pair.save(O/'crops'/filename)
  details.append(dict(first_sample=int(idx),second_sample=int(idx+1),normal_offset_change_px=float(dif[idx]),file=filename,cells=paircells))
 tri=valid[:-2]&valid[1:-1]&valid[2:];dd=np.diff(offset,n=2)
 rows.append(dict(id=l['id'],frame=f,group=l['physical_group'],split=l['split'],originally_used=l['id'] in L['accepted_for_measurement'],
  uniform_file=name,cells=cells,details=details,eligible=l['valid_count'],adjacent_pairs=int(ok.sum()),
  max_adjacent_offset_change_px=float(np.abs(dif[ok]).max()) if ok.any() else None,
  median_adjacent_offset_change_px=float(np.median(np.abs(dif[ok]))) if ok.any() else None,
  max_second_difference_px=float(np.abs(dd[tri]).max()) if tri.any() else None,
  nominal_arc_spacing_px=float(np.linalg.norm(np.array(l['b'])-l['a'])/119)))
for start in [0,5,10]:
 group=rows[start:start+5];page=Image.new('RGB',(672,96*len(group)),(240,240,240))
 for j,row in enumerate(group):page.paste(Image.open(O/'crops'/row['uniform_file']),(0,96*j))
 name=f'page_{start:02d}_{start+len(group)-1:02d}.png';page.save(O/'crops'/name);pages.append(dict(file=name,ids=[r['id'] for r in group]))
assert hashes=={p.name:sha(p) for p in source}
save(O/'result.json',dict(stage='D2',updated=now(),state='complete_pending_actual_material_review',rows=rows,pages=pages,
  source_hashes_unchanged=True,K_changed=False,coordinates_changed=False,new_fit=False,training=False))
q=read(R/'queue-state.json');q.update(updated=now(),phase='D2_edge_identity_pending_review',running=None,
  next='Review13 frozenedge sample strips and26 diagnostic adjacent pairs; no D1 refit. Determine measurement identity failure before any new compensation candidate.')
save(R/'queue-state.json',q)
print(json.dumps([{k:r[k] for k in ['id','originally_used','eligible','nominal_arc_spacing_px','max_adjacent_offset_change_px','max_second_difference_px']} for r in rows]))
