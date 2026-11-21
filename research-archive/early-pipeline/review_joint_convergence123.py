"""Read-only same-observation comparison and native material review preparation."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import match_video10_temporal_bridge123 as b
from global_track_initializer123 import angle_between_rotations
G=b.R/'video10-global-initializer';P=G/'joint-track-candidate';O=P/'convergence-repair';D=O/'visual-review';D.mkdir(exist_ok=True)
read=b.read;save=b.save
assert not (O/'explanation-plan.json').exists()
files=[G/'minimal-root-diagnosis/result.json',P/'initialization.npz',P/'diagnostic-candidate.npz',O/'diagnostic-candidate.npz',O/'all-observation-comparison.json']
save(O/'explanation-plan.json',dict(registered=b.now(),operation='Read-only posthoc comparison of same161HELD observations and the already diagnosed698/752 five-point roots against initial,80-evaluation and repaired relative rotation. No root choice or optimization.',
    source_sha256={str(p):b.sha(p) for p in files},review='Largest3newHELD-error families834,77,978 plus previous worst1126. All source views for each; native384px crops, no resizing or text. This is posthoc diagnosis, not independent test selection.'))
roots=read(files[0])['roots'];rotation=[]
for name,path in [('fresh_initializer',files[1]),('previous80',files[2]),('analytic_repair',files[3])]:
    with np.load(path,allow_pickle=False) as d:R=d['R'];frames=d['frames']
    a=int(np.flatnonzero(frames==698)[0]);c=int(np.flatnonzero(frames==752)[0]);rel=R[c]@R[a].T
    rotation.append(dict(stage=name,angle_to_roots_deg=[float(np.degrees(angle_between_rotations(rel,np.asarray(r['R'])))) for r in roots]))
cmp=read(files[4]);valid=[r for r in cmp if r['new_state']=='valid' and r['old_state']=='valid'];assert len(valid)==161
summary=dict(updated=b.now(),same_observations=161,improved=sum(r['new_error_px']<r['old_error_px'] for r in valid),worsened=sum(r['new_error_px']>r['old_error_px'] for r in valid),
    previously_subpixel_now_fail=sum(r['old_error_px']<1 and r['new_error_px']>=1 for r in valid),previously_fail_now_subpixel=sum(r['old_error_px']>=1 and r['new_error_px']<1 for r in valid),
    rotation_comparison=rotation,HELD_used_to_select_root=False,causal_scope='SharedXYZ optimization changed these rotations without changingK or choosing roots fromHELD. Posthoc comparison only; no assertion that closest root equals physicaltruth.')
save(O/'explanation.json',summary)
tracks={t['family']:t for t in read(b.R/'video10-remaining-material-review/global-input-preflight/heldout-observations.json')['tracks']};rows=[]
for family in [834,77,978,1126]:
    observations=tracks[family]['observations'];sources=[];files_out=[]
    for start in range(0,len(observations),3):
        chosen=observations[start:start+3];canvas=Image.new('RGB',(1152,384),(235,235,235))
        for j,ob in enumerate(chosen):
            p=b.imgpath(ob['frame']);im=Image.open(p).convert('RGB');x,y=np.rint(ob['uv']).astype(int);box=(x-192,y-192,x+192,y+192)
            canvas.paste(im.crop(box),(j*384,0));sources.append(dict(frame=ob['frame'],uv=ob['uv'],crop=list(map(int,box)),source=str(p),sha256=b.sha(p),outside_image=bool(box[0]<0 or box[1]<0 or box[2]>im.width or box[3]>im.height)))
        fn=f'heldout_{family}_part_{start//3+1}.png';canvas.save(D/fn);files_out.append(fn)
    rows.append(dict(family=family,files=files_out,sources=sources,observations=len(observations)))
save(D/'index.json',dict(updated=b.now(),posthoc=True,native_crop_size=384,no_resize=True,no_text_added=True,rows=rows))
print(json.dumps(dict(summary=summary,images=[f for r in rows for f in r['files']]),ensure_ascii=False))
