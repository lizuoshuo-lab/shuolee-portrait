"""Read-only native material support and leakage audit; never adjusts image geometry."""
from pathlib import Path
from datetime import datetime,timezone
import json,argparse
import cv2
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
R=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/raw-domain-121')
OUT=R/'visual-review'
ap=argparse.ArgumentParser();ap.add_argument('--supplement',action='store_true');args=ap.parse_args()
prefix='supplement-' if args.supplement else ''
if args.supplement:OUT=OUT/'roi-supplement';OUT.mkdir(exist_ok=True)
# Manually selected in the displayed originals. ROIs are diagnostic boxes, not masks or tracking labels.
ROI={'grass':dict(frame='000224',face=[1315,610,1395,715],body=[1235,575,1480,1390]),
     'sofa':dict(frame='000211',face=[1275,445,1435,625],body=[1080,400,1630,1520]),
     'portrait':dict(frame='000266',face=[495,810,580,895],body=[390,760,690,1570]),
     'indoor':dict(frame='000883',face=[1230,335,1490,650],body=[1050,270,1810,1520])}
def inside(p,b):return b[0]<=p[0]<b[2] and b[1]<=p[1]<b[3]
def crop(im,uv,size=112):
    x,y=np.rint(uv).astype(int);r=size//2;out=np.full((size,size,3),240,np.uint8)
    h,w=im.shape[:2];x0,x1=max(0,x-r),min(w,x+r);y0,y1=max(0,y-r),min(h,y+r)
    out[y0-y+r:y1-y+r,x0-x+r:x1-x+r]=im[y0:y1,x0:x1];return out
summary=[]
for subject,roi in ROI.items():
    base=ROOT/subject/('roi-supplement' if args.supplement else '')
    obs=json.loads((base/'observations.json').read_text());tracks=obs['tracks'];frames={r['frame']:r for r in obs['records']}
    parent=list(range(len(tracks)));edge_votes={}
    def find(i):
        while i!=parent[i]:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    perframe={f:[] for f in frames}
    for i,t in enumerate(tracks):
        for o in t['observations']:perframe[o['frame']].append((i,o['uv']))
    for f,items in perframe.items():
        if len(items)<2:continue
        for i,j in cKDTree(np.array([p for _,p in items])).query_pairs(1.0):
            key=tuple(sorted([items[i][0],items[j][0]]));edge_votes[key]=edge_votes.get(key,0)+1
    for (i,j),n in edge_votes.items():
        if n>=2:union(i,j)
    families={}
    for i in range(len(tracks)):families.setdefault(find(i),[]).append(i)
    cross=[ids for ids in families.values() if len({tracks[i]['split'] for i in ids})>1 or any(tracks[i]['split']=='quarantine' for i in ids)]
    quarantined={i for family in cross for i in family};selected=[];duplicates=[]
    for ids in families.values():
        if any(i in quarantined for i in ids):continue
        ids=sorted(ids,key=lambda i:(-len(tracks[i]['observations']),tracks[i]['track_id']));selected.append(ids[0]);duplicates.extend(ids[1:])
    report=dict(subject=subject,roi=roi,region_labels_are_rectangles_not_masks=True,original_tracks=len(tracks),near_duplicate_families=sum(len(x)>1 for x in families.values()),
        cross_split_families=len(cross),quarantined_tracks=len(quarantined),redundant_same_split_tracks=len(duplicates),
        retained_fit=sum(tracks[i]['split']=='fit' for i in selected),retained_heldout=sum(tracks[i]['split']=='heldout' for i in selected),
        overlap_rule='Within 1 native pixel in at least 2 common frames. Conservative quarantine of all mixed-split families; never moves heldout observations to fit.',
        selection_uses_geometry_errors=False,physical_identity_still_requires_review=True,regions={})
    (R/f'{prefix}{subject}-track-family-audit.json').write_text(json.dumps(dict(**report,retained_track_ids=[tracks[i]['track_id'] for i in selected],quarantined_track_ids=[tracks[i]['track_id'] for i in sorted(quarantined)]),indent=2)+'\n')
    source=cv2.imread(frames[roi['frame']]['image'])
    for region in ['face','body']:
        box=roi[region];x0,y0,x1,y1=box;cv2.imwrite(str(OUT/f'{subject}_{region}_native_frame{roi["frame"]}.png'),source[y0:y1,x0:x1])
        allhits=[i for i,t in enumerate(tracks) if any(o['frame']==roi['frame'] and inside(o['uv'],box) for o in t['observations'])]
        hits=[i for i in allhits if i in set(selected)]
        report['regions'][region]=dict(before=len(allhits),after=len(hits),fit=sum(tracks[i]['split']=='fit' for i in hits),heldout=sum(tracks[i]['split']=='heldout' for i in hits))
        sampled=sorted(hits,key=lambda i:tracks[i]['track_id'])[:8];rows=[];identity=[]
        for i in sampled:
            t=tracks[i];oo=sorted(t['observations'],key=lambda x:x['frame']);picks=[oo[0],oo[len(oo)//2],oo[-1]]
            panels=[crop(cv2.imread(frames[o['frame']]['image']),o['uv']) for o in picks];rows.append(np.concatenate(panels,axis=1))
            identity.append(dict(track_id=t['track_id'],split=t['split'],columns=picks))
        if rows:cv2.imwrite(str(OUT/f'{subject}_{region}_material_patches.png'),np.concatenate(rows,axis=0))
        (OUT/f'{subject}_{region}_material_patches.json').write_text(json.dumps(dict(order='rows: track_id ascending; columns: first/middle/last observation; each112px patch centered at rounded observation; no warp, text or point overlay',tracks=identity),indent=2)+'\n')
    summary.append(report)
    print(json.dumps(report),flush=True)
(R/f'{prefix}track-support-audit.json').write_text(json.dumps(dict(updated=datetime.now(timezone.utc).isoformat(),subjects=summary,
    prior_pair_metrics_note='Exploratory only: initial ID-level holdout may contain near-identical SIFT variants across splits. Do not claim independent physical-track validation until quarantine applied.'),indent=2)+'\n')
