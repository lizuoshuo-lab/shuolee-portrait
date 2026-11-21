from pathlib import Path
import json,numpy as np
from audit_enerf_camera_time import project_F
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-actor56-time-049'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
manifest=read(OUT/'manifest.json');report=read(OUT/'timing-diagnostic.json');checks=[]
for row in report['pairs']:
 c,d,t=row['source'],row['target'],row['anchor'];z=np.load(OUT/f'tracks-{c}-{d}-t{t:03d}.npz')
 a=np.c_[z['source_xy'],np.ones(len(z['source_xy']))];b=np.concatenate([z['target_xy'],np.ones(z['target_xy'].shape[:-1]+(1,))],axis=-1)
 F=project_F(manifest['cameras'][c],manifest['cameras'][d]);ly=a@F.T;lx=b@F
 v=np.einsum('ni,nki->nk',ly,b)/np.sqrt(np.sum(lx[...,:2]**2,axis=-1)+np.sum(ly[:,:2]**2,axis=-1)[:,None])
 assert np.allclose(v,z['signed_sampson'],rtol=1e-9,atol=1e-9)
 for label,mask in [('moving',z['source_speed']>1),('low_motion',z['source_speed']<.3),('all',np.ones(len(a),bool))]:
  if mask.any():assert np.allclose(np.median(abs(v[mask]),axis=0),row[label]['scores_px'],rtol=1e-9,atol=1e-9)
 checks.append(dict(source=c,target=d,anchor=t,matches=len(a)))
candidates=[r for r in report['selected_candidates'] if r['status']=='train_anchor_candidate_needs_identity_review']
nonzero=[r for r in candidates if r['selected_lag']!=0];covered=[r for r in nonzero if r['check_moving_count']>=12]
better=[r for r in covered if r['check_selected_px']<r['check_zero_px']]
out=dict(status='117track_arrays_and_saved_per_lag_scores_recomputed_identity_review_pending',track_files=len(checks),matched_tracks=sum(r['matches'] for r in checks),
 candidate_pairs=len(candidates),zero_lag_candidates=sum(r['selected_lag']==0 for r in candidates),nonzero_candidates=len(nonzero),
 nonzero_with_check_coverage=len(covered),nonzero_improve_check=len(better),nonzero_worsen_check=len(covered)-len(better),
 covered_nonzero_candidates=covered,qualification='Independent vectorized score recomputation; optical flow not independently reimplemented. Positive lag pairs do not establish a globally consistent timing correction.',goal_complete=False)
save(OUT/'independent-audit.json',out)
windows=dict(manifest,root='D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');save(OUT/'manifest-windows.json',windows)
print(json.dumps({k:v for k,v in out.items() if k!='covered_nonzero_candidates'}))
