"""Read-only canonical bindings, initialized and trained geometry; no heldout RGB."""
from pathlib import Path
import json,hashlib,numpy as np,torch,shutil
from scipy.spatial import cKDTree
L=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab');R=L/'outputs/dynamic-human-enerf-actor-supervision-070';R.mkdir(exist_ok=True)
source=L/'outputs/dynamic-human-enerf-residual-graph-067/residual_graph/resume.pt';cp=torch.load(source,map_location='cpu',weights_only=False);p={k:v.numpy() for k,v in cp['params'].items()};z=np.load(L/'outputs/dynamic-human-enerf-articulated-control-064/initial.npz');ref=np.load(L/'outputs/dynamic-human-enerf-continuous-body-063/reference.npz')['vertices'].reshape(-1,3);trees=[cKDTree(ref[i*6890:(i+1)*6890]) for i in [0,1]]
def summary(points,anchors):
 owner=anchors//6890;dist=np.stack([tr.query(points,workers=4)[0] for tr in trees],1);bound=np.linalg.norm(points-ref[anchors],axis=1);own=dist[np.arange(len(points)),owner];other=dist[np.arange(len(points)),1-owner]
 return dict(count=len(points),bound_anchor_distance_quantiles_world=np.quantile(bound,[0,.5,.9,.99,1]).tolist(),nearest_own_body_distance_quantiles_world=np.quantile(own,[0,.5,.9,.99,1]).tolist(),other_body_nearer_by_005_fraction=float((other+.05<own).mean()),other_body_nearer_by_010_fraction=float((other+.10<own).mean()),
 per_person=[dict(person=i,count=int((owner==i).sum()),other_body_nearer_by_005_fraction=float((other[owner==i]+.05<own[owner==i]).mean())) for i in [0,1]])
report=dict(status='training_initialization_and_trained_canonical_anchor_distances_checked',initial=summary(z['points'],z['anchor']),trained=summary(p['means'],p['anchor']),scene_scale=cp['scene_scale'],
 trained_opacity_quantiles=np.quantile(torch.sigmoid(torch.from_numpy(p['opacities'])).numpy(),[.5,.9,.99]).tolist(),trained_max_scale_quantiles_world=np.quantile(np.exp(p['scales']).max(-1),[.5,.9,.99,1]).tolist(),
 qualification='Nearest-body distances do not certify semantic identity; heldobjects/clothes legitimately lie off body. No reassignment or deletion performed. Referenceframe750 has residualgraphidentity.',goal_complete=False)
(R/'anchor-diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');shutil.copy2(__file__,R/Path(__file__).name);print(json.dumps(report))
