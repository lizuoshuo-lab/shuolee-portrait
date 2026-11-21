"""Evaluate the new face deformation on reserved skin observations and native overlays."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('/mnt/d/Projects/Shuolee-analysis-20261028');ROOT=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
parser=argparse.ArgumentParser();parser.add_argument('--detector',choices=['rtmw','fan'],default='rtmw');args=parser.parse_args()
OUT=BASE/('dynamic-face-control' if args.detector=='rtmw' else 'dynamic-face-fan-control')
motion=dict(np.load(OUT/'motion.npz'));fitted=dict(np.load(OUT/'fitted.npz'))
old=dict(np.load(ROOT/'fit/fitted.npz'));obs=dict(np.load(ROOT/'data/observations.npz'));weights=dict(np.load(ROOT/'fit/observation-weights.npz'))
face_targets=obs['keypoints'][:,40:91]
if args.detector=='fan':
    detector=dict(np.load(BASE/'face-local-detector/observations.npz'))
    assert np.array_equal(detector['frame_ids'],fitted['frame_ids'])
    face_targets=detector['decoded_face68'][:,17:68]
rows=json.loads((ROOT/'data/training-inputs.json').read_text())['records']
skin_source=json.loads((BASE/'head-motion-control/report.json').read_text())['independent_skin_track_before']
skin=skin_source['observations'];ids=np.array([r['frame_number']-1 for r in skin]);anchors=np.array([r['role']=='ANCHOR' for r in skin])
targets=np.array([r['target'] for r in skin]);ratio=np.array([1520,2704])/np.array(rows[0]['wh']);decoded=(targets+.5)/ratio-.5
E=np.array([r['w2c'] for r in rows])[ids];K=np.array([r['K'] for r in rows])[ids]
reference=motion['vertices_world'][0];sequence=motion['vertices_world'][1:][ids];affine=motion['vertex_affine'][1:][ids]
tree=cKDTree(reference);point=np.asarray(skin_source['point']);scale=float(motion['world_scale'])
for iteration in range(3):
    dist,nearest=tree.query(point,k=4);w=1/(dist**2+(.015*scale)**2);w/=w.sum()
    A=(affine[:,nearest]*w[None,:,None,None]).sum(1)
    b=((sequence[:,nearest]-np.einsum('nkij,kj->nki',affine[:,nearest],reference[nearest]))*w[None,:,None]).sum(1)
    def project(p):
        world=np.einsum('nij,j->ni',A,p)+b;cam=np.einsum('nij,nj->ni',E[:,:3,:3],world)+E[:,:3,3]
        q=np.einsum('nij,nj->ni',K,cam);return q[:,:2]/q[:,2:]-.5
    result=least_squares(lambda p:(project(p)[anchors]-decoded[anchors]).ravel(),point,loss='huber',f_scale=1.,max_nfev=200)
    point=result.x
uv=(project(point)+.5)*ratio-.5;errors=np.linalg.norm(uv-targets,axis=1)
stats=lambda x:dict(count=len(x),median=float(np.median(x)),p90=float(np.quantile(x,.9)),maximum=float(np.max(x)))
report=dict(protocol='Same 6 anchor and 35 reserved skin observations; head/expression fit never uses this track. Three anchor-only nearest-surface support updates.',
            anchors=stats(errors[anchors]),held=stats(errors[~anchors]),before=skin_source['held'],
            point=point.tolist(),support_vertices=nearest.tolist(),support_weights=w.tolist(),
            affine_min_determinant=float(np.linalg.det(motion['vertex_affine']).min()),
            observations=[dict(frame=int(i+1),role='ANCHOR' if a else 'HELD',target=t.tolist(),projected=p.tolist(),error_px=float(e)) for i,a,t,p,e in zip(ids,anchors,targets,uv,errors)],
            physical_accuracy_accepted=False,quality_accepted=False)
(OUT/'skin-validation.json').write_text(json.dumps(report,indent=2)+'\n')
selected=np.unique(np.linspace(0,len(rows)-1,6).astype(int));fig,axes=plt.subplots(6,3,figsize=(10,13.2),squeeze=False)
for line,i in enumerate(selected):
    row=rows[i];im=np.asarray(Image.open(row['image']).convert('RGB'));keep=weights['face'][i]>0
    y=face_targets[i,keep];p0=old['projected_face51'][i,keep];p1=fitted['projected_face51'][i,keep]
    points=np.concatenate([y,p0,p1]);lo=points.min(0)-20;hi=points.max(0)+20
    for col,(p,title) in enumerate(zip([None,p0,p1],['Original + observations','Original static SMPL-X','Dynamic SMPL-X face'])):
        ax=axes[line,col];ax.imshow(im)
        if p is None:ax.scatter(y[:,0],y[:,1],s=13,facecolors='none',edgecolors='#00c584')
        else:ax.scatter(p[:,0],p[:,1],s=20,marker='+',c='#ff7d24' if col==1 else '#008eff')
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(hi[1],lo[1]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        if line==0:ax.set_title(title,fontsize=10)
        if col==0:ax.set_ylabel(row['frame'],fontsize=9)
fig.suptitle(f'Fixed background cameras | per-frame neck, head, jaw and expression | {args.detector.upper()}',fontsize=12)
fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/'face-alignment-comparison.png',dpi=160);plt.close(fig)
print(json.dumps({k:report[k] for k in ['anchors','held','before','affine_min_determinant']}),flush=True)
