"""Inventory unread old candidates at the frozen graph gaps; no new matching."""
from pathlib import Path
from datetime import datetime,timezone
import json,collections,hashlib
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001');R=T/'shuolee/current/robust-portrait-122';N=R/'video10-semantic-role';O=R/'video10-temporal-bridge';Q=R/'video10-correspondence-contrast'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=R/'video10-remaining-material-review';out.mkdir(exist_ok=True)
assert not (out/'inventory.json').exists()
fs=read(N/'secondary-families.json')['rows'];tracks=read(N/'secondary-observations.json')['tracks']
full_review=[O/'visual-review.json',N/'material-visual-review.json'];reviewed=set()
for p in full_review:reviewed.update(x['family'] for x in read(p)['rows'])
reviewed.update(x['family'] for x in read(N/'material-support.json')['combined_rows'])
# Inherited9frame reviews remain evidence, but not full reviews of added33frame observations.
gaps=set(read(N/'graph-audit-result.json')['frames_below8_fit']) if 'frames_below8_fit' in read(N/'graph-audit-result.json') else set(map(int,read(N/'material-support.json')['frames_below8_fit']))
eligible=[f for f in fs if f['secondary_candidate_background'] and f['split']!='quarantine' and not f['conflict'] and not f['inherited_contradiction']]
rows=[]
for f in eligible:
    if f['family'] in reviewed:continue
    pool=[tracks[t] for t in f['tracks'] if len({o['frame'] for o in tracks[t]['observations']})==len(tracks[t]['observations'])]
    pool=[t for t in pool if gaps.intersection(o['frame'] for o in t['observations'])]
    if not pool:continue
    t=sorted(pool,key=lambda t:(-len(gaps.intersection(o['frame'] for o in t['observations'])),-len(t['observations']),t['track']))[0]
    rows.append(dict(row=len(rows),family=f['family'],split=f['split'],track=t['track'],frames=[o['frame'] for o in t['observations']],gap_frames=sorted(gaps.intersection(o['frame'] for o in t['observations'])),observations=t['observations'],inherited_reviews=f['inherited_reviews']))
coverage={str(f):dict(fit=sum(r['split']=='fit' and f in r['frames'] for r in rows),heldout=sum(r['split']=='heldout' and f in r['frames'] for r in rows)) for f in sorted(gaps)}
summary=dict(updated=datetime.now(timezone.utc).isoformat(),eligible_old_families=len(eligible),previously_reviewed_representative_families=len(reviewed),
    remaining_gap_families=len(rows),split_counts=dict(collections.Counter(r['split'] for r in rows)),gap_frame_potential_coverage=coverage,
    frozen_gap_frames=sorted(gaps),rows=rows,
    scope='Allremaining originaleligible families touching any ofthe18 previously under-supportedframes. One representative per frozenfamily, maximumgapcoverage thenobservationcount thenoriginaltrackid. No newsource observations, features, matches, semanticmask, cameras or threshold changes.',
    significance='Previous selected reviews do not prove all636oldbackground candidates unusable. This is an inventory of unseen evidence, not acceptedmaterial or physicalindependence.',
    input_sha256={str(p):sha(p) for p in [N/'secondary-families.json',N/'secondary-observations.json',N/'material-support.json',N/'graph-audit-result.json',*full_review]})
save(out/'inventory.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','input_sha256']},ensure_ascii=False))
