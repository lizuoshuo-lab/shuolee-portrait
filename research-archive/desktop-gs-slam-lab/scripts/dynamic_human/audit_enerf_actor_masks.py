from pathlib import Path
import json,hashlib,shutil,numpy as np,cv2
from PIL import Image
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-actor-supervision-070';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=read(R/'mask-preparation.json')['records'];assert len(rows)==15
for r in rows:
 c=r['camera'];f=I/'native-input'/f'camera{c}-frame000750.npz';assert sha(f)==r['input_sha256'];z=np.load(f);m=R/'masks'/f'camera{c}-frame000750.npz';assert sha(m)==r['labels_sha256'];d=np.load(m);labels=d['labels'];masks=d['raw_masks'];expected=np.full((1080,1920),-1,np.int8)
 for a in [0,1]:
  own=cv2.erode(masks[a].astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);other=cv2.dilate((masks[1-a]|masks[2]).astype(np.uint8),np.ones((7,7),np.uint8)).astype(bool);expected[own&~other&(z['matte']>.95)&z['valid']]=a
 assert np.array_equal(labels,expected) and [int((labels==a).sum()) for a in [0,1]]==r['counts']
 x0,y0,x1,y1=z['bbox'];w=x1-x0;panel=np.array(Image.open(R/'masks'/f'camera{c}-labels-review.png'));assert np.array_equal(panel[36:,:w],z['rgb'][y0:y1,x0:x1])
 overlay=z['rgb'].copy();palette=np.array([[255,135,25],[40,140,255]])
 for a in [0,1]:s=labels==a;overlay[s]=np.rint(.55*z['rgb'][s]+.45*palette[a]).astype(np.uint8)
 assert np.array_equal(panel[36:,w:],overlay[y0:y1,x0:x1])
accepted=[r['camera'] for r in rows if r['camera'] not in ['13','17']]
save(R/'mask-visual-review.json',dict(status='all15native_label_panels_actually_viewed_13accepted_as_weak_supervision',accepted_for_weak_training=True,
 viewed_cameras=[r['camera'] for r in rows],accepted_cameras=accepted,rejected_cameras={'13':'Visible blue class remnants on the dark persons rear shoe; ambiguous overlapping lower body. Whole view excluded from ownership loss.',
 '17':'Small blue class island on dark persons thigh in overlap; whole view excluded from ownership loss.'},
 findings='Other13cores visibly follow assigned people; missing forearms/holes remain ignored, not background. Toys excluded. This review is not dense pixel annotation or measured segmentation accuracy.',
 training_scope='RGB/matte training remains105images; only13frame750images add weak ownership loss. Rejected views still contribute originalRGB/matte.',goal_complete=False))
save(R/'mask-independent-audit.json',dict(status='15input_hashes_masks_rules_and_native_overlay_panels_verified',records=rows,accepted_cameras=accepted,goal_complete=False));shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(dict(accepted_cameras=accepted,known_pixels=sum(sum(r['counts']) for r in rows if r['camera'] in accepted))))
