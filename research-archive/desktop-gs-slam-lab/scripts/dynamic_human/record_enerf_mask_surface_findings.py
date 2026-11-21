from pathlib import Path
import json,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];A=LAB/'outputs/dynamic-human-enerf-mask-uncertainty-053';B=LAB/'outputs/dynamic-human-enerf-surface-tracks-054'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
r=read(A/'paired-comparison/summary.json');inputs=read(A/'input-audit.json')
save(A/'visual-review.json',dict(status='six_whole_person_and17fixed_native_detail_panels_reviewed',
 cameras=['00','06','12','01','08','17'],detail_regions=read(LAB/'outputs/dynamic-human-enerf-camera10-time-050/fixed-detail-regions.json')['regions'],
 findings='Wall fragments near hangingarm partly disappear in08, but00/12arm remains stretched and wrist detail degrades;06crossedforearms and occludedshoulder stillwrong;toy andairborneshoe remain smeared. No coherent limb recovery.',
 decision='Do not adopt uncertainty omission as a clarity fix. Disagreement is real but its loss omission alone is insufficient.',goal_complete=False))
save(A/'final-summary.json',dict(status='mask_disagreement_loss_omission_mixed_no_clear_limb_recovery',
 updates=6000,points=137951,ignored_pixels=inputs['ignored_pixels'],
 fraction_range=[min(x['ignored_fraction_of_foreground'] for x in inputs['records']),max(x['ignored_fraction_of_foreground'] for x in inputs['records'])],
 heldout_foreground_delta_db={x['camera']:x['foreground_delta_db'] for x in r['records'] if x['split']=='heldout'},
 independent_state_audit='paired-training-state-audit.json',visual_review='visual-review.json',
 note='Only loss validity changes, not RGB/matte/cameras/initialization. Mask disagreement is not newgroundtruth. Existing maskedimageSSIM changes neighboringwindowstatistics.',goal_complete=False))
r=read(B/'depth-comparison.json');assert read(B/'independent-audit.json')['status'].startswith('90tracks375')
save(B/'visual-review.json',dict(status='12native_threeview_photo_render_track_sheets_and3coverage_maps_reviewed',
 track_ids=[x['track'] for x in r['selected']],coverage_cameras=['01','08','11'],
 findings='Sampled identities mostly coherent texturedshoes, shirtfold/hem, toys. Coverage maps show dominant plantedshoe points and scattered face/toy/garmentpoints; little directsupport onbareforearms/hands/airborneshoe. Highmodeldepthdisagreement cases include bootsole, shirtfold, toyedge; some edge/occlusion correspondenceambiguity remains.',
 decision='Do not equate low texture-track depth disagreement with clear human geometry. Sparsepoints do not sufficiently supervise the failing limbs.',goal_complete=False))
save(B/'final-summary.json',dict(status='sparse_texture_depth_mostly_consistent_but_weak_failed_limb_coverage',
 tracks=90,observations=375,median_relative_depth_difference=r['median_abs_relative_depth_error'],
 p90_relative_depth_difference=r['p90_abs_relative_depth_error'],median_model_crossview_error_px=r['median_model_cross_view_reprojection_px'],
 limits='Alltraining-only. Triangulated hypotheses are notdepthGT. Barelimbs have weaktexturecoverage. No sparse-depthtraining launched from these incompletepoints.',
 next='Evaluate surface-oriented Gaussian representation and explicit surface regularization under paired inputs, then inspect limbs and heldoutviews.',goal_complete=False))
for folder in [A,B]:shutil.copy2(__file__,folder/Path(__file__).name)
mfile=LAB/'manifests/dynamic-human-work.json';m=read(mfile)
m['status']='mask_uncertainty_and_sparse_surface_diagnosis_complete_surface_method_prepared'
for entry in ['053sameinitial6000update uncertainty-loss control completed,24nativePNGs12panels6wholeperson17details verified/reviewed; no clarity acceptance.',
 '05490train-only multiviewtracks375observations and15modeldepthmaps audited;12identitysheets3coverage maps reviewed. Sparse texture-depth agreement doesnotcoverfailedbarelimbs.']:
 if entry not in m['completed']:m['completed'].append(entry)
m['outdoor_mask_uncertainty']='outputs/dynamic-human-enerf-mask-uncertainty-053/final-summary.json'
m['outdoor_surface_tracks']='outputs/dynamic-human-enerf-surface-tracks-054/final-summary.json'
m['next_actions']=['Complete isolatedgsplat1.5.3CUDA2Dsurface smoke, train055matchedsurface-only andsurface-regularized6000update branches, inspectsameheldout limbregions.',
 'Preserve failed3D/mask/time controls and distinguish surface regularizer effect from entire2Dvs3Dpipeline change.']
m['goal_turn_classification']='Progress:053actualtraining/controlreview and054surfacegeometrydiagnosis completed;055newmethod runtimeandpairedinputs prepared. No external blocker.'
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(mfile,m)
print('053054results recorded;055surface method next')
