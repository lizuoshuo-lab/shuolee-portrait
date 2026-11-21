"""N12b registered bounded context inspection; no feature or geometry changes."""
from pathlib import Path
from datetime import datetime, timezone
from PIL import Image
import hashlib, json

I=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027')
O=I/'clip03-native-matching'; M=I/'clip03-scene-support'
selected=[7,12,18,19,20,21,22,23,26,28,57,58,59,60,61,62]
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'context-index.json').exists()
plan=read(I/'clip03-native-matching-plan.json'); idx=read(O/'visual-index.json')
assert len(selected)<=plan['budget']['context_rows_max']
frozen={name:sha(O/name) for name in ['observations.json','families.json','visual-index.json']}
imgs={f:Image.open(M/f'frame_{f:06d}.png').convert('RGB') for f in idx['columns']}
out=[]
for rid in selected:
    row=idx['rows'][rid]; assert row['category']=='background'
    strip=Image.new('RGB',(256*7,256),(80,80,80))
    originals=Image.open(O/row['file']).convert('RGB')
    for obs in row['observations']:
        col=idx['columns'].index(obs['frame']);x,y=map(round,obs['uv'])
        source=imgs[obs['frame']]
        assert source.crop((x-64,y-64,x+64,y+64)).tobytes()==originals.crop((128*col,0,128*(col+1),128)).tobytes(), 'pixel decoding mismatch'
        strip.paste(source.crop((x-128,y-128,x+128,y+128)),(256*col,0))
    name=f'context_{rid:03d}_family_{row["family"]}.png';strip.save(O/name)
    out.append(dict(row=rid,family=row['family'],split=row['split'],file=name,observations=row['observations']))
assert frozen=={name:sha(O/name) for name in frozen}
(O/'context-index.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),rows=out,columns=idx['columns'],cell_size=256,annotations_in_image=False,
  basis='Selected after128px material review, before any geometry; ambiguous rigid-object/tree-trunk context only. Same native pixels; no resampling, detections or point edits.',
  frozen_hashes=frozen,all128px_center_crops_byte_equal=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('N12B_CONTEXT_COMPLETE',len(out))
