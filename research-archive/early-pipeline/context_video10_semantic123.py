"""One bounded native context review of already frozen observations; no matching."""
from pathlib import Path
from PIL import Image
import numpy as np
import match_video10_temporal_bridge123 as bridge
b=bridge.base; N=b.R/'video10-semantic-role'
selected=[3,13,14,16,20,21,24,28,31,36,38,42,68,70,96,107]
out=N/'material-review'/'contexts'
assert not out.exists(), 'Already prepared: review existing contexts'
idx=b.read(N/'material-review'/'index.json'); plan=b.read(N/'material-review-plan.json')
assert len(selected)<=plan['budget']['max_expanded_contexts']
out.mkdir()
b.save(out/'allocation.json',dict(registered=b.now(),rows=selected,native_crop_px=384,
    reason='Resolve repeated material, reflections and physical background identity observed in the113 preallocated rows; no new tracks/selection by residual.',
    source_index_sha256=b.sha(N/'material-review'/'index.json'),images_reviewed=False))
imgs={}; entries=[]
for n in selected:
    row=idx['rows'][n]; obs=row['observations']; im=Image.new('RGB',(1152,384*((len(obs)+2)//3)),(80,80,80))
    for j,o in enumerate(obs):
        f=o['frame']
        if f not in imgs:imgs[f]=Image.open(bridge.imgpath(f)).convert('RGB')
        x,y=np.rint(o['uv']).astype(int)
        im.paste(imgs[f].crop((x-192,y-192,x+192,y+192)),((j%3)*384,(j//3)*384))
    name=f'row_{n:03d}_family_{row["family"]}.png'; im.save(out/name)
    entries.append(dict(row=n,family=row['family'],file=name,frames=row['frames'],all_observations=True,center=[192,192]))
b.save(out/'index.json',dict(rows=entries,burned_text=False,coordinates_changed=False))
print(dict(contexts=len(entries),new_observations=0,new_features=0,geometry=0))
