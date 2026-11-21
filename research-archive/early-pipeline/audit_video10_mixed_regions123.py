"""Read-only causal case audit within remaining8native-context slots; no promotion."""
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
import json,hashlib,numpy as np
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge')
B=O.parent/'video9-10-native-intake/video10'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def rank(s):return hashlib.sha256(str(s).encode()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
tracks=read(O/'run/observations.json')['tracks'];families={r['family']:r for r in read(O/'run/families.json')['rows']};chosen=[]
used={r['family'] for r in read(O/'visual-review.json')['rows']}
for t in sorted(tracks,key=lambda t:(-len(t['observations']),rank(str(t['track'])+'|mixed-region349'))):
 fa=families[t['family']]
 if t['family'] in used or fa['categories']==['background'] or not any(ob['frame']==349 and ob['role']<40 for ob in t['observations']):continue
 chosen.append(t);used.add(t['family'])
 if len(chosen)==8:break
assert len(chosen)==8
out=O/'mixed-region-cases';out.mkdir()
save(out/'plan.json',dict(registered=datetime.now(timezone.utc).isoformat(),hypothesis='Frame349 has38tracks with a background observation but36belong to role-mixed families. Inspect eight deterministic longest-track cases to distinguish actual moving/reflective/occlusion content from conservative body-rectangle misclassification. This is a diagnostic of existing classification, not additional camera admission.',selection='Family-unique tracks with bg observation in349, non-pure-background family, not previously96reviewed; longest then SHA(track|mixed-region349).',tracks=[t['track'] for t in chosen],new_features=0,camera_changes=0,split_changes=0,extra_review_rows=8,total_review_rows=104,total_context_rows=16,auto_promotion=False))
rows=[]
for j,t in enumerate(chosen):
 obs=t['observations'];im=Image.new('RGB',(1152,384*((len(obs)+2)//3)),(80,80,80))
 for k,ob in enumerate(obs):
  p=O/'native'/f'native_frame_{ob["frame"]:06d}.png'
  if not p.exists():p=B/p.name
  with Image.open(p) as src:
   x,y=np.rint(ob['uv']).astype(int);im.paste(src.convert('RGB').crop((x-192,y-192,x+192,y+192)),(384*(k%3),384*(k//3)))
 name=f'case_{j:02d}_family_{t["family"]}.png';im.save(out/name)
 rows.append(dict(**t,case=j,file=name,cell_frames=[ob['frame'] for ob in obs],cell_roles=[ob['role'] for ob in obs],cell_size=384,columns=3))
save(out/'index.json',dict(rows=rows,burned_text=False,native_pixels=True,automatic_admission=False))
print(json.dumps([{k:r[k] for k in ['case','family','track','split','file','cell_frames','cell_roles']} for r in rows]))
