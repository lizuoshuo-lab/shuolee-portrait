from pathlib import Path
from PIL import Image
import json
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
D=B/'current/portrait-causal-diagnosis-117/D2-material-correspondence'
m=json.loads((D/'review/image-manifest.json').read_text(encoding='utf-8'))
pages=[]
for group,rows in [('face',[r for r in m if r['bank']=='material' and r['group'] in ['face_lower','face_upper']]),
                   ('scene_body',[r for r in m if r['bank']=='material' and r['group'] not in ['face_lower','face_upper']]),
                   ('semantic',[r for r in m if r['bank']=='semantic'])]:
 for start in range(0,len(rows),6):
  block=rows[start:start+6];im=Image.new('RGB',(768,256*len(block)),(255,255,255))
  for i,r in enumerate(block):
   with Image.open(D/'review'/r['file']) as src:im.paste(src,(0,i*256))
  name=f'{group}-sheet-{start//6+1:02d}.png';im.save(D/'review'/name)
  pages.append(dict(file=name,rows=[dict(bank=r['bank'],id=r['track'],file=r['file'],group=r['group'],semantic_name=r['semantic_name']) for r in block],text_burned=False))
(D/'review/sheets.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(pages,ensure_ascii=False))
