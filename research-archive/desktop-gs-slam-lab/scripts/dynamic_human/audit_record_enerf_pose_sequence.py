from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
import numpy as np
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-enerf-pose-sequence-062';RAW=Path('D:/GS-SLAM-Data/enerf_outdoor/extracted/actor5_6');BASE=LAB/'outputs/dynamic-human-enerf-camera10-time-050/same_frame'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
records=read(OUT/'observations.json')['records'];inputs={(r['camera'],r['frame']):r for r in read(OUT/'native-input/input-audit.json')['records']};assert len(records)==195 and len(inputs)==195
for r in records:
 c,t=r['camera'],r['frame'];assert c not in ['00','06','12'] and 744<=t<=756
 raw=RAW/'images'/c/f'{t:06d}.jpg';f=OUT/'native-input'/f'camera{c}-frame{t:06d}.npz';z=np.load(f);ir=inputs[c,t]
 assert sha(raw)==r['source_sha256']==ir['source_sha256'] and sha(f)==ir['input_sha256']
 assert hashlib.sha256(z['rgb'].tobytes()).hexdigest()==r['rectified_sha256']==ir['rectified_sha256']
 assert r['K']==ir['K'] and r['w2c']==ir['w2c']
 if t==750:assert np.array_equal(z['rgb'],np.load(BASE/f'train-camera-{c}.npz')['rgb'])
 assert len(r['detections'])==2 and all(np.array(d['keypoints']).shape==(133,2) and np.isfinite(d['keypoints']).all() for d in r['detections'])
summary=read(OUT/'timing-summary.json');checks=[]
for label in ['fixed_offsets','estimated_offsets']:
 z=np.load(OUT/f'{label}-fit.npz');c=z['coefficients'];t=(z['frame']-750+z['offsets'][z['camera_index']])/6
 X=c[z['joint_index'],0]+t[:,None]*c[z['joint_index'],1]+t[:,None]**2*c[z['joint_index'],2]+t[:,None]**3*c[z['joint_index'],3]
 h=np.concatenate([X,np.ones((len(X),1))],1);p=(z['P']@h[...,None])[...,0];e=np.linalg.norm(p[:,:2]/p[:,2:]-z['xy'],axis=-1)
 assert np.max(abs(e-z['error_px']))<1e-8 and np.array_equal(z['fitmask'],z['frame']%2==0)
 v=e[~z['fitmask']];target=summary['records'][label]['validation'];assert len(v)==2262 and abs(float(np.median(v))-target['median_px'])<1e-8 and abs(float(np.sqrt(np.mean(v*v)))-target['rmse_px'])<1e-8
 checks.append(dict(branch=label,fit_sha256=sha(OUT/f'{label}-fit.npz'),validation_observations=len(v),median_px=float(np.median(v)),rmse_px=float(np.sqrt(np.mean(v*v)))))
base=np.load(OUT/'fixed_offsets-fit.npz');transfer=read(OUT/'person-transfer-summary.json');types=['person_dark_clothing','person_light_clothing'];transferchecks=[]
for person in range(2):
 sel=(base['joint_index']//18==person)&~base['fitmask'];j=base['joint_index'][sel]%18
 for label,key in [('fixed','fixed'),('own','own_offsets'),('transfer','other_person_offsets')]:
  z=np.load(OUT/f'person{person}-{label}-fit.npz');c=z['coefficients'];t=(base['frame'][sel]-750+z['offsets'][base['camera_index'][sel]])/6
  X=c[j,0]+t[:,None]*c[j,1]+t[:,None]**2*c[j,2]+t[:,None]**3*c[j,3];h=np.c_[X,np.ones(len(X))];p=(base['P'][sel]@h[...,None])[...,0];e=np.linalg.norm(p[:,:2]/p[:,2:]-base['xy'][sel],axis=-1)
  assert np.max(abs(e-z['validation_errors']))<1e-8 and abs(float(np.median(e))-transfer['records'][types[person]][key]['median_px'])<1e-8
  transferchecks.append(dict(person=person,branch=label,observations=len(e),median_px=float(np.median(e))))
save(OUT/'independent-audit.json',dict(status='195source_inputs_and_all_timing_validation_errors_recomputed',native_images=195,person_inferences=390,frame750_exact_input_matches=15,frame750_keypoints_exact_match059=read(OUT/'observations.json')['frame750_max_difference_from059_px']==0,branches=checks,transfer_checks=transferchecks,heldout_camera_photos_used=False,goal_complete=False))
save(OUT/'input-recovery.json',dict(status='rectification_convention_mismatch_resolved_before_sequence_fitting',failure='Initial directfloat64remap didnotbytewisematch050trainingRGB; assertstoppedbeforefitting. SameassertalsofailedonWindows, rulingoutanOSdecoderassumption.',cause='039officialloaderand048/050training usecv2.undistort withfloat32K/D. Independentimageevaluatorusesfloat64remap; sparse subpixel differences werealreadydocumentedin048inputaudit.',repair='All195sequenceimagesregeneratedwithoriginaltrainingfloat32undistort convention;all15frame750inputsandRTMWoutputsareexactoldmatches.',preserved='attempt-raw-decode/run.log, protocol.json, originalscript',goal_complete=False))
save(OUT/'visual-review.json',dict(status='4boundary_time_observation_panels_actually_reviewed',views=[dict(camera=c,frame=t) for c in ['04','17'] for t in [744,756]],findings='Personboxescovermainbodies acrosstheinterval. Forearms/feetmovevisibly, butdarkpersonoccludedhead/arm canstillattachtolightperson. Camera17falsecrosspersonpointsremain; frame750multiviewsupportandrobustlossreducebutdonotproveabsenceofoutliers. HighvalidationRMSE18pxmustnotbehiddenbylower2.64pxmedian.',goal_complete=False))
save(OUT/'final-summary.json',dict(status='native_pose_sequence_and_crossperson_time_offset_validation_complete_RGB_clarity_unproven',frames=13,training_camera_images=195,person_inferences=390,fit_times=list(range(744,757,2)),validation_times=list(range(745,756,2)),validation_observations=2262,validation_median_before_px=checks[0]['median_px'],validation_median_after_px=checks[1]['median_px'],validation_rmse_before_px=checks[0]['rmse_px'],validation_rmse_after_px=checks[1]['rmse_px'],offset_disagreement_between_people_rmse_frames=transfer['offset_disagreement_rmse_frames'],boundary_offsets=summary['boundary_cameras'],scope='Predictedkeypoint checkonly. SharedphysicaltimeoffsetplausiblebutnotGT. Camera01setsarbitraryreference; offsetsforheldout00/06/12unknown. NoRGBretiming/trainingperformedwiththeseoffsetsyet.',next='TestcontinuousarticulatedGaussianmodel withsameRGBtrainingcameras/evenframes andexplicitcapturetime; evaluateoriginalnativeoddtimeimages. Do notclaimoldheldoutcamerasareclockcalibrated.',goal_complete=False))
p=LAB/'manifests/dynamic-human-work.json';m=read(p);m['current_runs']=[];m['outdoor_pose_sequence_timing']='outputs/dynamic-human-enerf-pose-sequence-062/final-summary.json';m['outdoor_wholebody_website_qa']='outputs/dynamic-human-enerf-forearm-control-061/website-http-audit.json';m['status']='body_prior_and_forearm_training_complete_pose_time_hypothesis_supported_RGB_validation_pending';m['goal_turn_classification']='Progress:059060061completed,062195nativeimages390poseinferences andcubicmotion/timeoffset fits pluscrosspersontransfer independentlyverified. Clarityremainsunaccepted;noexternalblocker.'
m['next_actions']=['Buildcontinuousarticulatedmotion fromeven-time trainingkeypoints; use062cameracaptureoffsets asqualifiedhypothesis, notGT. Preserveoriginalcameracalibration/images.','RunpairedGaussianappearance reconstruction withzerooffset versusqualifiedcapturetime, sameeven-timeRGBtraining budget; compareoriginalnativeodd-timephotos. Unknownclockof00/06/12requiresseparatequalification, nohiddenheldoutresampling.','Keepcloth/hands/toysrepresentableoutsideSMPL; nohardtemplateclamping. Oncehumanclarityandmotionpass, integratebackgroundandfullscene.']
s='062195trainingimages390wholebodyinferences;7evenfit6oddvalidationtimes;median3.228to2.636px,crosspersonoffsettransferimprovesboth;18.18pxRMSEandunknownheldoutclocksremain.'
if s not in m['completed']:m['completed'].append(s)
m['last_recorded_utc']=datetime.now(timezone.utc).isoformat();save(p,m);shutil.copy2(__file__,OUT/Path(__file__).name)
print(json.dumps(dict(status='062inputsfitsandtransfer_verified',native_images=195,validation_rows=2262,clarity_complete=False)))
