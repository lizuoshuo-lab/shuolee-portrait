"""Record source-target failures actually seen in six training views."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-matte-diagnosis-080'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
q=read(O/'protocol.json')
for p in q['panels']:assert sha(O/p['file'])==p['sha256']
z=np.load(R/'native-input/camera04-frame000732.npz');bb=[756,367,1287,574];x0,y0,x1,y1=bb;rgb=z['rgb'][y0:y1,x0:x1];a=z['matte'][y0:y1,x0:x1];target=np.rint(rgb.astype(np.float32)*a[...,None]+127.5*(1-a[...,None])).clip(0,255).astype(np.uint8)
arrays=[rgb,np.repeat(np.rint(a[...,None]*255).astype(np.uint8),3,2),target];w=x1-x0;h=y1-y0;im=Image.new('RGB',(w*3,h+32),'white');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
for k,(arr,label) in enumerate(zip(arrays,['Original training photograph','Provided matte','Exact training target on grey'])):
 im.paste(Image.fromarray(arr),(k*w,32));d.text((k*w+4,7),label,font=font,fill='black')
ip=O/'camera04-frame000732-hands-training-target.png';im.save(ip)
for k,arr in enumerate(arrays):assert np.array_equal(np.array(Image.open(ip))[32:,k*w:(k+1)*w],arr)
out=dict(status='six_native_training_targets_visually_reviewed_background_false_positives_confirmed',actual_full_images_reviewed=q['panels'],detail_prepared=dict(file=ip.name,sha256=sha(ip),bbox=bb),
 findings=[
 'Camera04/frame726/732: disconnected wall-shadow fragments and floor pixels around feet are retained by provided bgmtv2 matte and enter the composited training target.',
 'Camera07/frame726/732: a rounded wall-shadow patch above the green toy, wall below the hands, and ground/leg-gap pixels are nonzero in the matte. The exact target contains visible wall texture at these locations.',
 'Camera13/frame732: a large wall-shadow area left of the nearer person is incorrectly foreground. Camera13/frame726 retains wall in the leg gap.',
 'Camera04/frame732: the original and composited training photograph still show the opening palm and curved fingers. Therefore these inspected training observations contain useful hand structure; not all rendered hand blur is explained by missing source pixels.'
 ],
 causal_scope='The false foreground supervision is directly observed. Its contribution to rendered floaters is consistent with the image locations but not isolated by a new training intervention; this diagnosis does not prove it is the only cause of blur.',
 metric_scope='079foreground PSNR uses the supplied matte>0.5, so false foreground pixels also affect its support. Preserve those paired metrics; do not relabel them as exact human-semantic accuracy.',
 next='Prepare training-only foreground support candidates preserving both persons and carried toys; visually inspect occlusion, open palms and feet. Keep old evaluation and add fixed source-defined detail checks rather than changing test masks to improve scores.',
 previous_related='Earlier750mask diagnostics remain applicable;080confirms the defect in the newly evaluated726-centered interval, not a first discovery of the general issue.',
 training_changed=False,goal_complete=False,updated_at=datetime.now(timezone.utc).isoformat())
save(O/'summary.json',out)
m=read(L/'manifests/dynamic-human-work.json');m.update(status='second_interval_complete_training_matte_false_positives_confirmed_clarity_unaccepted',current_runs=[],outdoor_matte_diagnosis='outputs/'+O.name+'/summary.json',next_action=out['next'],updated_at=out['updated_at'])
entry='080: Six actual training source/matte/target comparisons reviewed. Shadow/wall/ground false foreground confirmed at726/732; opening hand remains visible in actual04/732target. No retraining or clarity acceptance from this diagnosis.'
if entry not in m['completed']:m['completed'].append(entry)
save(L/'manifests/dynamic-human-work.json',m)
print(out['status'])
