"""Fixed, error-independent native crops for reviewing held material candidates."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-local-surface-123')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def native(p):return Path('//wsl.localhost/Ubuntu-20.04'+p) if p.startswith('/opt/') else Path(p[5].upper()+':/'+p[7:])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--subject',required=True);a=p.parse_args()
root=ROOT/a.subject;audit=REC/f'{a.subject}-regional-support';out=REC/f'{a.subject}-material-samples-review'
if (out/'index.json').exists():raise RuntimeError('Completed review exists; do not regenerate.')
out.mkdir(exist_ok=True)
obs=read(audit/'observations.json');held=read(root/'camera/held-evaluation.json')['records'];rows=read(root/'data/training-inputs.json')['records']
ids=sorted(set(x['track'] for x in obs if 'RTMW_face_ROI_only' in x['regions']))
selected=[ids[i] for i in np.unique(np.linspace(0,len(ids)-1,min(6,len(ids))).astype(int))]
plan=dict(selection='Six evenly spaced lexicographic track IDs among all held candidates having at least one face-ROI check; no residual-based filtering',
    selected=selected,eligible_tracks=len(ids),observations_sha256=sha(audit/'observations.json'),crop_native_pixels=128,
    view_selection='earliest, middle, latest by source frame within each frozen track',
    projection_or_refinement_performed=False,physical_identity_is_pending=True)
if (out/'plan.json').exists():assert read(out/'plan.json')==plan
else:(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
images=[]
for n,ident in enumerate(selected):
    tr=next(t for t in held if t['id']==ident);nodes=sorted(tr['nodes']);chosen=[nodes[i] for i in np.unique(np.linspace(0,len(nodes)-1,3).astype(int))]
    patches=[];sources=[]
    for i,j in chosen:
        xy=np.load(root/f'camera/features-{i:03d}.npz')['xy'][j];cx,cy=map(int,np.rint(xy))
        with Image.open(native(rows[i]['image'])) as im:
            crop=(cx-64,cy-64,cx+64,cy+64);patches.append(im.convert('RGB').crop(crop))
        sources.append(dict(frame=rows[i]['frame'],source_index=rows[i]['source_frame_index'],feature_xy=xy.tolist(),crop=crop))
    sheet=Image.new('RGB',(128*len(patches),128))
    for i,im in enumerate(patches):sheet.paste(im,(128*i,0))
    name=f'track_{n+1:02d}_{ident[:10]}.png'
    if (out/name).exists():
        with Image.open(out/name) as previous:assert np.array_equal(np.asarray(previous),np.asarray(sheet))
    else:sheet.save(out/name)
    images.append(dict(frame=ident,region='held_native_material_candidate',file=name,columns=[s['frame'] for s in sources],
        sources=sources,sha256=sha(out/name),native_pixels_no_warp=True,labels_in_pixels=False))
(out/'index.json').write_text(json.dumps(dict(subject=a.subject,images=images,actual_review_pending=True),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(images,ensure_ascii=False))
