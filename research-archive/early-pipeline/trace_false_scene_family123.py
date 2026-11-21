"""Replay only existing descriptor decisions inside the visually contradictory held family."""
from pathlib import Path
from collections import deque
import json,time,hashlib
import numpy as np
import cv2
from PIL import Image,ImageDraw
import audit_smplify_observation_contract123 as a

O=a.REF/'solver-track-membership'
OUT=O/'false-family-bridge'
tid='82e918800e2525059cba07a2f837fbae3023c1fa09371cd75f34a9b12d1901b7'
assert (O/'result.json').exists() and not OUT.exists()
OUT.mkdir()
tracks=a.read(a.ROOT/'camera/appearance-tracks.json')
tr=next(r for r in tracks['records'] if r['id']==tid)
assert tr['held'] and len(tr['nodes'])==22
nodes=dict(tr['nodes']);rows=a.read(a.ROOT/'data/training-inputs.json')['records']
inputs=[a.ROOT/'camera/appearance-tracks.json',a.H/'local_surface123_camera.py']
inputs += [a.ROOT/f'camera/features-{s:03d}.npz' for s in nodes]
hashes={str(p):a.sha(p) for p in inputs}
a.save(OUT/'plan.json',dict(registered=a.now(),hypothesis='A raw connected component joins the communication mast at source155 to foliage at source310. Identify which original reciprocal descriptor decisions connect the two identities.',
    track_id=tid,whole_track_split='HELD remains HELD',scope='Read saved SIFT arrays only for the22 existing nodes; recompute nearest2 distances for these nodes and original scheduled pairs. No new keypoint detection, full graph rebuild, camera fit or label transfer.',
    budget=dict(cpu_seconds=120,track_families=1,nodes=22,new_features=0,new_cameras=0),
    matching='Original L2 mutual nearest and ratio0.75 unchanged; edges shown without epipolar filtering.',
    selection='This family was identified by the complete pre-registered13-family image review as metal mast versus foliage, not selected solely by its residual.',
    source_hashes=hashes))
start=time.monotonic();cv2.setNumThreads(2)
features={s:dict(np.load(a.ROOT/f'camera/features-{s:03d}.npz')) for s in nodes}
schedule={(p['i'],p['j']) for p in tracks['pairs']}
edges=[];decisions=[];matcher=cv2.BFMatcher(cv2.NORM_L2)
for i in nodes:
    for j in nodes:
        if i>=j or (i,j) not in schedule:continue
        fw=matcher.knnMatch(features[i]['descriptors'][nodes[i]:nodes[i]+1],features[j]['descriptors'],k=2)[0]
        bw=matcher.knnMatch(features[j]['descriptors'][nodes[j]:nodes[j]+1],features[i]['descriptors'],k=2)[0]
        mutual=fw[0].trainIdx==nodes[j] and bw[0].trainIdx==nodes[i]
        ratios=[float(fw[0].distance/fw[1].distance),float(bw[0].distance/bw[1].distance)]
        accepted=bool(mutual and max(ratios)<.75)
        record=dict(slots=[i,j],source_frames=[rows[s]['source_frame_index'] for s in [i,j]],
            mutual=mutual,ratios=ratios,original_rule_accepts=accepted)
        decisions.append(record)
        if accepted:edges.append(record)
adj={s:[] for s in nodes}
for e in edges:
    i,j=e['slots'];adj[i].append(j);adj[j].append(i)
def path_between(start,end,removed=None):
    q=deque([[start]]);seen={start}
    while q:
        p=q.popleft()
        if p[-1]==end:return p
        for z in sorted(adj[p[-1]]):
            if removed is not None and {p[-1],z}==set(removed):continue
            if z not in seen:seen.add(z);q.append(p+[z])
    return None
path=path_between(31,62)
assert path is not None,'Original graph could not be reconstructed; stop without inferring a bridge.'
bridges=[e['slots'] for e in edges if path_between(e['slots'][0],e['slots'][1],e['slots']) is None]
recordroot=Path('/mnt/d/00/记录/005/分析/静止场景支持')
images=[];tiles=[];slot_order=sorted(nodes)
for s in slot_order:
    uv=features[s]['xy'][nodes[s]];c=np.rint(uv).astype(int)
    box=[int(c[0]-96),int(c[1]-96),int(c[0]+96),int(c[1]+96)]
    raw=Image.open(rows[s]['image']).convert('RGB').crop(tuple(box));marked=raw.copy();d=ImageDraw.Draw(marked)
    x,y=uv-np.array(box[:2]);d.line((x-5,y,x+5,y),fill=(255,30,210),width=1);d.line((x,y-5,x,y+5),fill=(255,30,210),width=1)
    tile=Image.new('RGB',(192,384));tile.paste(raw);tile.paste(marked,(0,192));tiles.append(tile)
    filename=f'20261029_错误轨迹82e918_原帧{rows[s]["source_frame_index"]:06d}_上原图下紫色中心.png'
    p=recordroot/filename;assert not p.exists();tile.save(p)
    images.append(dict(slot=s,source_frame=rows[s]['source_frame_index'],uv=uv.tolist(),path=str(p),sha256=a.sha(p),crop=box))
for k in range(4):
    tt=tiles[6*k:6*(k+1)];sheet=Image.new('RGB',(576,384*((len(tt)+2)//3)),(240,240,240))
    for i,tile in enumerate(tt):sheet.paste(tile,((i%3)*192,(i//3)*384))
    sheet.save(a.H/f'false-family123-{k}.png')
for p,h in hashes.items():assert a.sha(Path(p))==h
assert time.monotonic()-start<120
result=dict(updated=a.now(),track_id=tid,node_count=len(nodes),original_pair_decisions=len(decisions),
    accepted_edges=edges,all_decisions=decisions,shortest_existing_path_31_to62=path,
    graph_bridges=bridges,images=images,source_hashes_unchanged=True,all22_nodes_represented=True,
    original_split_preserved=True,cameras_unchanged=True,feature_detection_run=False,
    no_geometric_filter_or_correction_applied=True,seconds=time.monotonic()-start)
a.save(OUT/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['images','all_decisions']},ensure_ascii=False))
