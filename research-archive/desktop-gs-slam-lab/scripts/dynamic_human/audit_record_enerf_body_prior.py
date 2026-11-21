from pathlib import Path
import json,hashlib,shutil
import numpy as np
LAB=Path(__file__).resolve().parents[2];A=LAB/'outputs/dynamic-human-enerf-body-observations-057';B=LAB/'outputs/dynamic-human-enerf-body-prior-058';BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
r=read(A/'observations.json');t=read(A/'triangulation.json');z=np.load(A/'triangulated-body-joints.npz');cam={v['camera']:v for v in r['records']}
assert len(cam)==15 and not set(cam)&{'00','06','12'}
for c,row in cam.items():assert hashlib.sha256((BASE/f'train-camera-{c}.npz').read_bytes()).hexdigest()==row['input_sha256']
errs=[]
for row in t['records']:
 if not row.get('accepted'):continue
 pp=np.r_[row['point'],1.];count=0
 for obs in row['observations']:
  c=cam[obs['camera']];W=np.array(c['w2c']);K=np.array(c['K']);p=K@(W@pp)[:3];error=float(np.linalg.norm(p[:2]/p[2]-obs['xy']))
  assert abs(error-obs['error_px'])<1e-7
  if obs['inlier']:assert error<8. and obs['confidence']>=3.;count+=1;errs.append(error)
 assert count==row['inlier_count'] and count>=4
assert sum(x['accepted'] for x in t['records'])==23==int(z['valid'].sum())
save(A/'independent-audit.json',dict(status='15training_inputs_and23joint_hypotheses_all_reprojection_errors_recomputed',accepted_joints=23,inlier_observations=len(errs),median_inlier_reprojection_px=float(np.median(errs)),goal_complete=False))
save(A/'visual-review.json',dict(status='15detector_overlays_and5triangulated_projection_overlays_reviewed',triangulated_projection_cameras=['01','07','08','11','17'],
 findings='Darkperson13 face-only false detection; scores reject it. Darkoccluded shoulders/elbows/wrists in07/08/14/16/17 often inaccurate or onotherperson. Robustsupport yields23/24body joints; darkleftelbow missing. Hips and knees still uncertain under looseclothes; left/right ambiguity not certified.',
 decision='Use as soft coarse-body fitting hypotheses with missingjoint preserved. No silhouette or limbclarity acceptance.',goal_complete=False))
fit=np.load(B/'fit.npz');summary=read(B/'fit-summary.json');errors=np.linalg.norm(fit['joints'][:,fit['smpl_joint_ids']]-fit['target'],axis=-1)
assert np.max(abs(errors-np.array(summary['per_joint_error_world'])))<1e-8
assert all(np.isfinite(fit[k]).all() for k in ['vertices','joints','body_pose','betas','scale','translation','global_orient'])
save(B/'independent-audit.json',dict(status='finite_fitted_parameters_and_target_errors_recomputed_fullFK_consistency_record_verified',accepted_joint_mean_error_world=float(errors[fit['valid']].mean()),
 fullFK_max_error=summary['max_fk_vs_full_smpl_joint_error'],target_origin='057predictions, notGT',goal_complete=False))
save(B/'visual-review.json',dict(status='5of15training_projection_panels_actually_reviewed',cameras=['01','04','07','08','17'],
 findings='Torso placement and crossing-leg centers broadly follow observations. Darkhead remains too upright/displaced; missingdarkleftelbow produces wrongarmbend. Hands are open/default and notaligned togrip; feetorientation mismatches shoes. Clothingvolume, hair, toys absent. Directhardconstraint wouldencodevisibleerrors.',
 decision='Coarseprior prepared, not adopted as hard geometry. Refine head/hand/foot orientations and occludedarm observations before constrained appearance fit.',goal_complete=False))
save(A/'final-summary.json',dict(status='training_body_observations_and23joint_hypotheses_complete',cameras=15,accepted_joints=23,missing_joint='person_dark_clothing:left_elbow',
 review='visual-review.json',audit='independent-audit.json',next='Refine ambiguous body observations and fit soft articulated prior; no clarity claim.',goal_complete=False))
save(B/'final-summary.json',dict(status='coarse_SMPL_prior_fitted_but_head_arm_hand_foot_alignment_unaccepted',updates=2000,accepted_joint_mean_error_world=float(errors[fit['valid']].mean()),
 review='visual-review.json',audit='independent-audit.json',publication='Model archive, model parameters and mesh remainlocal; no template export.',goal_complete=False))
for folder in [A,B]:shutil.copy2(__file__,folder/Path(__file__).name)
print('057/058 audited and bounded visual findings recorded')
