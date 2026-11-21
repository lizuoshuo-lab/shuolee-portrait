from pathlib import Path
import json
from PIL import Image
r=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/grass-temporal/review-013')
p=json.loads((r/'result.json').read_text())
def local(v):
 return Path(v[5].upper()+':/'+v[7:]) if v.startswith('/mnt/') else Path(v)
rows=[]
for q in p['records']:
 panels=[Image.open(local(q['images'][k]['path'])).convert('RGB') for k in ['face','feet']]
 im=Image.new('RGB',(sum(x.width for x in panels),max(x.height for x in panels)))
 x=0
 for panel in panels: im.paste(panel,(x,0));x+=panel.width
 rows.append(im)
paths=[]
for start in range(0,len(rows),4):
 ims=rows[start:start+4];canvas=Image.new('RGB',(max(x.width for x in ims),sum(x.height for x in ims)));y=0
 for im in ims:canvas.paste(im,(0,y));y+=im.height
 dst=r/f'face-feet-{start//4+1:02d}.png';assert not dst.exists();canvas.save(dst);paths.append(str(dst))
print(json.dumps(dict(body_pages=p['pages'],face_feet_pages=paths,frame_order=[q['frame'] for q in p['records']],metrics=[dict(frame=q['frame'],old=q['metrics']['independent']['body_weighted_mean_px'],new=q['metrics']['temporal']['body_weighted_mean_px']) for q in p['records']],max_angle_old=max(x['independent']['max_local_joint_deg'] for x in p['transitions']),max_angle_new=max(x['temporal']['max_local_joint_deg'] for x in p['transitions'])),indent=2))