"""Post-hoc trace of an actually reviewed cloth-to-eye match; no new matching."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,itertools
import numpy as np
from PIL import Image
T=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=T/'shuolee/current/local-clothed-surface-123/pattern-preflight/native-material';V=O/'visual-review'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert not (O/'false-track-trace.json').exists()
save(O/'false-track-trace-plan.json',dict(registered=datetime.now(timezone.utc).isoformat(),posthoc=True,selected_row=7,
    selection_reason='Actual original-pixel review found clothing at source0 and eye region at source120/125 in oneHELDtrack.',
    hypothesis='A long temporal descriptor edge can join unrelated material segments even with mutual matching and onefeatureperframe. Check exact saved edges, common-neighbour triangles and connectivity, without changing observations or thresholds.',
    limit='One knownfailuretrack, read saved descriptors/edges; no new detection, inference, matching, poses or optimization. Small triangle error/triangle support alone cannot prove material identity. RetainHELDmembership.'))
review=read(V/'index.json');tr=review['rows'][7];assert tr['split']=='HELD'
manifest=read(O/'feature-manifest.json')['frames'];nodes=[tuple(n) for n in tr['track_nodes']];byframe=dict(nodes)
data={slot:dict(np.load(O/'features'/manifest[slot]['feature_file'],allow_pickle=False)) for slot,j in nodes}
edges=[];adj={n:set() for n in nodes}
for item in read(O/'pair-counts.json')['records']:
    i,j=item['i'],item['j']
    if i not in byframe or j not in byframe:continue
    p=O/'pairs'/f'{i:03d}_{j:03d}.npz';d=np.load(p,allow_pickle=False);a,b=byframe[i],byframe[j]
    if not np.any((d['accepted_pairs']==[a,b]).all(1)):continue
    u,v=(i,a),(j,b);adj[u].add(v);adj[v].add(u)
    edges.append(dict(a=list(u),b=list(v),source_frames=[manifest[i]['source_frame'],manifest[j]['source_frame']],
        forward_ratio=float(d['forward_distances'][a,0]/d['forward_distances'][a,1]),
        reverse_ratio=float(d['reverse_distances'][b,0]/d['reverse_distances'][b,1]),native_displacement_px=float(np.linalg.norm(data[i]['xy'][a]-data[j]['xy'][b])),
        pair_sha256=sha(p)))
triangles=set()
for e in edges:
    a,b=tuple(e['a']),tuple(e['b']);common=adj[a]&adj[b];e['triangles']=len(common)
    for c in common:triangles.add(tuple(sorted([a,b,c])))
    reached={a};pending=[a]
    while pending:
        x=pending.pop()
        for y in adj[x]:
            if {x,y}=={a,b} or y in reached:continue
            reached.add(y);pending.append(y)
    e['bridge']=b not in reached
sheet=Image.new('RGB',(960,384),(32,32,32));views=[]
for k,(slot,j) in enumerate(nodes):
    p=Path(manifest[slot]['image']);assert sha(p)==manifest[slot]['source_sha256']
    uv=data[slot]['xy'][j];left,top=np.floor(uv).astype(int)-96;box=[int(left),int(top),int(left+192),int(top+192)]
    sheet.paste(Image.open(p).convert('RGB').crop(box),((k%5)*192,(k//5)*192))
    views.append(dict(slot=slot,source_frame=manifest[slot]['source_frame'],keypoint=j,uv=uv.tolist(),crop_xyxy=box,
        physical_label='pending_all_node_review',source_sha256=manifest[slot]['source_sha256']))
sheet.save(V/'false_track_all10_native.png')
res=dict(updated=datetime.now(timezone.utc).isoformat(),posthoc=True,family_id=tr['family_id'],track_id=tr['track_id'],split='HELD',
    FIT_membership_changed=False,observations_changed=False,edges=edges,triangles=[[list(n) for n in tri] for tri in sorted(triangles)],views=views,
    physical_false_match_confirmed_from_initial3views=True,physical_all10nodes_review_pending=True,
    no_camera_or_fit_causal_claim='This track isHELDandhasnotenteredcameraorSMPLX. It diagnoses why descriptorconsistencyisnotphysicaltruth; doesnotproveallFITtrackswrong.',
    no_new_matching_or_geometry=True,training_started=False)
save(O/'false-track-trace.json',res)
print(json.dumps(dict(nodes=len(nodes),edges=len(edges),triangles=len(triangles),cross_cloth_eye_edges=[e for e in edges if (e['a'][0]<2)!=(e['b'][0]<2)]),ensure_ascii=False))
