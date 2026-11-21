"""Freeze original-resolution full-person, complete-hand and foot review windows."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';O=R/'review-plan';O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'protocol.json').exists()
assert not (R/'global_cubic/native-evaluation/metrics.json').exists() and not (R/'local_bspline/native-evaluation/metrics.json').exists()
z=np.load(R/'motion/fixed_clock-fit.npz');co=z['coefficients'];clocks={str(c):float(d) for c,d in zip(z['cameras'],z['offsets'])};clocks.update(read(R/'heldout/protocol.json')['offsets_frames'])
full=[dict(camera=c,frame=f) for c in ['01','04','07','13','17','00','06','12'] for f in [721,727,731]]
details=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for c in ['04','07','06','12']:
 for f in [721,727,731]:
  path=R/'native-input'/f'camera{c}-frame{f:06d}.npz';z=np.load(path);tau=(f+clocks[c]-726)/6;X=np.einsum('k,nkd->nd',np.array([tau**k for k in range(4)]),co);P=z['K']@z['w2c'][:3];q=X@P[:,:3].T+P[:,3];uv=q[:,:2]/q[:,2:]
  for name,indices,pad in [('both_hands_and_objects',[4,5,22,23],96),('both_person_feet',list(range(10,18))+list(range(28,36)),40)]:
   lo=np.floor(uv[indices].min(0)-pad);hi=np.ceil(uv[indices].max(0)+pad);bb=np.r_[np.maximum(lo,[0,0]),np.minimum(hi,[1920,1080])].astype(int).tolist();assert bb[2]>bb[0] and bb[3]>bb[1]
   raw=Image.fromarray(z['rgb']).crop(bb);im=Image.new('RGB',(raw.width,raw.height+30),'white');im.paste(raw,(0,30));ImageDraw.Draw(im).text((4,5),f'Original camera{c} frame{f} {name}',font=font,fill='black');namefile=f'camera{c}-frame{f:06d}-{name}-original.png';im.save(O/namefile)
   details.append(dict(camera=c,frame=f,name=name,bbox=bb,original=namefile,input_sha256=sha(path)))
save(O/'protocol.json',dict(status='fixed_before_any_candidate_render_inspection',full_person=full,details=details,
 rule='24fullpersonviews and24detailwindows at original pixels; plus worstchanges from complete120view scores; preserve regressions. No comparison images inspected before selection.',
 geometry='Newtrainingonly cubic joint tracks and frozen camera clocks, wrists plus96px for fullhands/objects, ankles/toes plus40px forfeet. Thesearecontextwindows, notsegmentationGT.',
 evaluation='90knownodd+18otherodd scored separately;12even diagnostics separate; same-scene second interval, not independent scene generalization.',goal_complete=False))
print('24full and24native detail windows fixed before candidate evaluation.')
