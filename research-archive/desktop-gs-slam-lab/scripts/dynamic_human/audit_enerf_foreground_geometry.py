"""Independent camera-matrix form and saved native-match arithmetic checks."""
from pathlib import Path
import json,hashlib,numpy as np
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-foreground-geometry-045'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
summary=read(OUT/'summary.json');checks=[]
for row in summary['records']:
 p=OUT/f'{row["region"]}-{row["a"]}-{row["b"]}.npz';z=np.load(p);P,Q=z['P'],z['Q']
 _,_,vh=np.linalg.svd(P);C=vh[-1];e=Q@C
 skew=np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
 F=skew@Q@np.linalg.pinv(P)
 old=z['F']/np.linalg.norm(z['F']);new=F/np.linalg.norm(F)
 assert min(np.max(abs(old-new)),np.max(abs(old+new)))<1e-9
 x=np.c_[z['xy_a'],np.ones(len(z['xy_a']))];y=np.c_[z['xy_b'],np.ones(len(z['xy_b']))]
 ly=x@F.T;lx=y@F;err=np.abs(np.einsum('ij,ij->i',y,ly))/np.sqrt(np.sum(ly[:,:2]**2+lx[:,:2]**2,axis=1))
 assert np.allclose(err,abs(z['signed_sampson']),rtol=1e-8,atol=1e-8)
 assert abs(np.median(err)-row['median_px'])<1e-8
 X=np.c_[z['world_points'],np.ones(len(x))];a=X@P.T;b=X@Q.T
 reproj=np.stack([np.linalg.norm(a[:,:2]/a[:,2:]-x[:,:2],axis=1),np.linalg.norm(b[:,:2]/b[:,2:]-y[:,:2],axis=1)],1)
 assert np.allclose(reproj,z['reprojection'],atol=1e-10)
 checks.append(dict(file=p.name,matches=len(x),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
fg=[r for r in summary['records'] if r['region']=='foreground'];bg=[r for r in summary['records'] if r['region']=='background']
report=dict(status='all_saved_matches_camera_matrix_epipolar_and_reprojection_verified',records=checks,
 foreground_pair_count=len(fg),foreground_match_count=sum(r['matches'] for r in fg),
 foreground_pair_median_range_px=[min(r['median_px'] for r in fg),max(r['median_px'] for r in fg)],
 foreground_median_of_pair_medians_px=float(np.median([r['median_px'] for r in fg])),
 foreground_pairs_median_below1px=sum(r['median_px']<1 for r in fg),
 background_median_of_pair_medians_px=float(np.median([r['median_px'] for r in bg])),
 qualification='Independent fundamental-matrix derivation from projection-matrix nullspace; descriptor matching itself not independently reimplemented. No timing or calibration causality inferred.',goal_complete=False)
save(OUT/'independent-audit.json',report)
review=dict(status='six_native_patch_sheets_72_correspondences_visually_reviewed',panels=summary['panels'],
 findings=['Most sampled lower-residual points are consistent shoe, toy, hair, face or clothing texture identities.',
 'High residual outliers include clear mismatches:08/14 match37 toy versus shoe;11/14 match100 different sleeves;01/04 match72 shoe versus sleeve;02/15 match59 opposite footwear.',
 'Some visually plausible moving-shoe matches have several-pixel residuals, e.g.01/03 match42 and02/15 matches25/63. Single-frame evidence does not establish clock offset.',
 'Observed good correspondence support is texture-biased and cannot certify untextured skin, fingers or all18cameras.'],goal_complete=False)
save(OUT/'visual-review.json',review)
save(OUT/'final-summary.json',dict(status='many_train_camera_pairs_subpixel_consistent_with_local_dynamic_residuals_and_match_outliers',
 numerical=report,conclusion='Inputs support further geometry/radius diagnosis; no blanket synchronisation failure established, no camera or timestamp adjustment adopted.',goal_complete=False))
print(json.dumps({k:v for k,v in report.items() if k!='records'}))
