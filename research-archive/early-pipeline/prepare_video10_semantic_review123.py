"""Register once-only review allocation, then render unchanged native observations."""
from pathlib import Path
import json,hashlib
from PIL import Image
import numpy as np
import match_video10_material123 as base
import match_video10_temporal_bridge123 as bridge
N=base.R/'video10-semantic-role';O=bridge.O
read=base.read;save=base.save;sha=base.sha;rank=base.rank
assert not (N/'material-review-plan.json').exists()
r=read(N/'classification-result.json');obs=read(N/'secondary-observations.json');fs=read(N/'secondary-families.json')['rows'];tracks=obs['tracks']
recovered=set(r['recovered_ids']);chosen=[];used=set();representatives={}
for fa in fs:
    if fa['conflict'] or fa['inherited_contradiction'] or fa['split']=='quarantine':continue
    ts=[tracks[i] for i in fa['tracks']]
    representatives[fa['family']]=sorted(ts,key=lambda t:(-len(t['observations']),rank(str(t['track'])+'|semantic-representative')))[0]
limits={'fit':64,'heldout':32};counts={'fit':0,'heldout':0}
for turn in range(33):
    for f in obs['frames']:
        for split in ['fit','heldout']:
            if counts[split]>=limits[split]:continue
            pool=[t for fam,t in representatives.items() if fam in recovered and fam not in used and t['split']==split and any(o['frame']==f for o in t['observations'])]
            if not pool:continue
            t=sorted(pool,key=lambda t:(-len(t['observations']),rank(str(t['family'])+'|semantic-review')))[0]
            chosen.append(dict(**t,review_reason='newly_recovered_family',allocation_frame=f));used.add(t['family']);counts[split]+=1
    if all(counts[s]>=limits[s] for s in limits):break
# Failed mask is not used. Read-only controls use the ORIGINAL conservative background labels.
for split,limit in [('fit',12),('heldout',5)]:
    pool=[]
    for fa in fs:
        if fa['family'] in used or fa['family'] not in representatives or not fa['candidate_background'] or fa['split']!=split:continue
        ts=[tracks[i] for i in fa['tracks'] if any(o['frame']==617 for o in tracks[i]['observations'])]
        if not ts:continue
        t=sorted(ts,key=lambda t:(-len(t['observations']),rank(str(t['track'])+'|617-control')))[0];pool.append(t)
    for t in sorted(pool,key=lambda t:(-len(t['observations']),rank(str(t['family'])+'|617-review')))[:limit]:
        chosen.append(dict(**t,review_reason='original_background_failed_mask_control',allocation_frame=617));used.add(t['family'])
assert len(chosen)<=113 and len(chosen)==len(used)
plan=dict(registered=base.now(),state='registered_before_material_images',source_classification_sha256=sha(N/'classification-result.json'),
    selection='Deterministic temporal round-robin over33frames, longest representative per frozen family. Newly recovered families max64FIT32HELD. Up to12FIT5HELD ORIGINAL background families containing617 are separate controls; the failed SAM mask remains unused.',
    hypothesis='Pixel semantic masks recover potential static material that bbox roles discarded; independent visual review determines physical support. This allocation is not an unbiased matching error rate.',
    frozen=['native keypoints and descriptors','original99pairs and all tracks','family identities, splits,quarantine and inherited contradictions','conditionalK and no poses'],
    budget=dict(max_representatives=113,max_expanded_contexts=16,new_features=0,new_masks=0,new_matching=0,camera_operations=0,training_steps=0),
    stop='Review each selected native observation and identity, without reassigning FIT/HELD. Unknown stays unknown. No extra masking/threshold sweep. Camera gates unchanged:20FIT8HELD,6FIT3HELDnonground,initial12sharedFIT,each subsequent8FIT2D3D; allfullframes and independentface<1px remain separate.',
    rows=[dict(row=i,track=t['track'],family=t['family'],split=t['split'],frames=[o['frame'] for o in t['observations']],reason=t['review_reason']) for i,t in enumerate(chosen)])
save(N/'material-review-plan.json',plan)
out=N/'material-review';out.mkdir();imgs={f:Image.open(bridge.imgpath(f)).convert('RGB') for f in obs['frames']}
rows=[];pages=[];page=Image.new('RGB',(1152,1584),(235,235,235));cursor=0;page_rows=[]
def finish():
    global page,cursor,page_rows
    name=f'page_{len(pages):02d}.png';page.crop((0,0,1152,cursor)).save(out/name);pages.append(dict(file=name,rows=page_rows))
    page=Image.new('RGB',(1152,1584),(235,235,235));cursor=0;page_rows=[]
for i,t in enumerate(chosen):
    observations=t['observations'];height=((len(observations)+8)//9)*132;strip=Image.new('RGB',(1152,height),(80,80,80))
    for j,o in enumerate(observations):
        x,y=np.rint(o['uv']).astype(int);strip.paste(imgs[o['frame']].crop((x-64,y-64,x+64,y+64)),(128*(j%9),132*(j//9)))
    file=f'row_{i:03d}_family_{t["family"]}.png';strip.save(out/file);rows.append(dict(**t,row=i,file=file,frames=[o['frame'] for o in observations],cell_center=[64,64]))
    if cursor+height+4>1584:finish()
    page.paste(strip,(0,cursor));page_rows.append(dict(row=i,y=cursor,height=height));cursor+=height+4
if cursor:finish()
save(out/'index.json',dict(rows=rows,pages=pages,all_observations=True,native_crop_px=128,burned_text=False))
save(N/'progress.json',dict(updated=base.now(),state='semantic_material_review_ready',pid=None,review_rows=len(rows),pages=len(pages),new_cameras=0,training_started=False))
print(json.dumps(dict(review_rows=len(rows),pages=len(pages),new_recovered_by_split=counts,material_reviewed=False)))
