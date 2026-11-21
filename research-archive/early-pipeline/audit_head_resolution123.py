"""Read-only pixel-budget audit. No inference, camera fit, or observation changes."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
import numpy as np

R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122')
Q=R/'video10-correspondence-contrast'; N=R/'video10-semantic-role'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(Q/'plan.json'); regions=read(N/'execution-admission.json')['regions']
nodes={}
for t in read(N/'secondary-observations.json')['tracks']:
    for o in t['observations']:
        key=(o['frame'],o['feature'])
        val=(t['family'],t['split'])
        assert key not in nodes or nodes[key]==val
        nodes[key]=val
rows=[]; sources={}
for f in [644,671,698]:
    fp=next(Path(n) for n in p['input_sha256'] if Path(n).name==f'features_{f:06d}.npz')
    assert sha(fp)==p['input_sha256'][str(fp)]
    with np.load(fp) as z:uv=z['uv'].copy()
    x0,y0,x1,y1=regions[str(f)]['head']
    ids=np.flatnonzero((uv[:,0]>=x0)&(uv[:,0]<x1)&(uv[:,1]>=y0)&(uv[:,1]<y1))
    counts={s:0 for s in ['fit','heldout','quarantine','unassigned']}
    families={s:set() for s in counts}
    for i in ids:
        fam,split=nodes.get((f,int(i)),(-1,'unassigned'))
        counts[split]+=1
        if fam>=0:families[split].add(fam)
    rows.append(dict(frame=f,head_native_xyxy=[x0,y0,x1,y1],native_wh=[x1-x0,y1-y0],
        coarse_head_wh=[(x1-x0)*798/2704,(y1-y0)*448/1520],
        fine_head_wh=[(x1-x0)*1344/2704,(y1-y0)*756/1520],
        frozen_head_queries=len(ids),query_split_counts=counts,
        families_by_split={s:sorted(v) for s,v in families.items()},
        warning='Head rectangle includes hair/background. Query/family counts are not physical face coverage.'))
    sources[str(fp)]=sha(fp)
out=dict(updated=datetime.now(timezone.utc).isoformat(),frames=[644,671,698],rows=rows,
    source_sha256=sources,original_image_wh=[2704,1520],model=p['model'],
    finding='The previous whole-frame matcher used 798x448 coarse and1344x756 fine samples. Native-coordinate roundtrip is correct, but its face pixel budget is smaller than the native crop. Resolution and local context are an untested combined variable.',
    native_crop_proposal=dict(size=768,coarse_hw=[448,448],fine_hw=[768,768],queries='Reuse every frozen query in the existing head rectangle. No new detector or split changes.'),
    no_inference=True,no_camera_change=True,no_physical_subpixel_claim=True)
dest=Q/'head-resolution-audit.json';assert not dest.exists()
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'rows':rows,'result':str(dest)},ensure_ascii=False))
