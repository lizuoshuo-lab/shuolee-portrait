"""Record actual second-interval preparation and current live/file-backed progress."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-second-interval-079'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reviewed=['motion-audit/camera04-frame000726-smpl.png','motion-audit/camera07-frame000720-smpl.png','motion-audit/camera01-frame000732-smpl.png','motion-audit/camera02-frame000732-smpl.png','motion-audit/camera13-frame000726-smpl.png',
 'motion/camera02-frame000720-trajectory.png','motion/camera07-frame000732-trajectory.png','motion/camera04-frame000726-trajectory.png','motion/camera13-frame000720-trajectory.png','pose/camera04-frame000726-pose.png',
 'review-plan/camera04-frame000721-both_hands_and_objects-original.png','review-plan/camera06-frame000721-both_hands_and_objects-original.png','review-plan/camera12-frame000731-both_person_feet-original.png']
save(R/'preparation-visual-review.json',dict(status='actual_photos_and_geometry_projections_reviewed_before_appearance_evaluation',
 detection_contact_sheets=15,detection_photos=45,identity_review='box-detection/identity-review.json',
 actual_images=[dict(path=p,sha256=sha(R/p)) for p in reviewed],
 findings=['Dark-person crops missing/severelyoccluded in08/09/10/11/14/16 excluded; wallshadow and duplicate detections not used. Dark lowerleg/foot observations01/03 conservativelydisabled.',
 'Five directSMPL projectionimages sampledfront/side/overlap: principal body/head andshoe locations follow the correct people. Sparse mesh shows unoccluded andoccluded vertices alike; not a visibility-aware appearance render.',
 'SMPL does not match garment outline, toys or individual finger articulation. Approximately7pxmedian body projection error remains; no geometric-perfectness claim.',
 'Three original-only detail crops reviewed: fullhands andcarriedobjects visible in04/06;12foot window includes near shoes andfar-person occlusion. These windows areforinspection, notpartGT.'],
 appearance_predictions_reviewed=False,goal_complete=False))
branches={}
for b in ['global_cubic','local_bspline']:
 branches[b]=dict(completion=read(R/b/'completion.json') if (R/b/'completion.json').exists() else None,progress=read(R/b/'progress.json') if (R/b/'progress.json').exists() else None,evaluation_complete=(R/b/'native-evaluation/metrics.json').exists())
both=all(r['completion'] is not None for r in branches.values());status='matched_training_complete_evaluation_pending' if both else 'matched_new_interval_training_in_progress'
out=dict(status=status,source_images=213,pose_images=195,pose_predictions=312,body_updates=2500,head_updates=3000,affine_captures_verified=195,initial_points=57678,nodes=384,
 branches=branches,pose_geometry='New720..732motion fitted; old074clock priors frozen. Canonical geometry rebasedto726, no750motionextrapolation orappearancecheckpointreuse.',
 novel_camera_motion='39frozenmotion queries,old195capturetablesexactlypreserved; camera00/06/12clockscalibratedpreviously onoldinterval, notfittedfromnewimages.',
 next='Complete120nativeviewsperbranch,independentCPUrecomputationandactualfullhand/shoe/occlusionreview. No clarity acceptance yet.',
 goal_complete=False,updated_at=datetime.now(timezone.utc).isoformat())
save(R/'progress-summary.json',out)
m=read(L/'manifests/dynamic-human-work.json');m['status']=status;m['outdoor_second_interval']='outputs/dynamic-human-enerf-second-interval-079/progress-summary.json'
m['current_runs']=[] if both else [dict(kind='paired_Gaussian_training',session_id=84588,runner='scripts/dynamic_human/run_enerf_second_training.sh',note='Sequentialbranches;checkliveprocessandcompletionfilesbeforeanyrestart')]
m['goal_turn_classification']='Progress:45keyframephotos reviewed;312RGBpose predictions;newjoint/body/headmotion fits,195directSMPL/affinechecks,new57678pointinitialization,CUDAzeroextension/gradientchecksandmatchedtraining. Clarity stillunaccepted;noexternalblocker.'
m['next_action']=out['next'];m['updated_at']=datetime.now(timezone.utc).isoformat();save(L/'manifests/dynamic-human-work.json',m)
print(json.dumps(dict(status=status,steps={b:(r['completion'] or r['progress'] or {}).get('step',0) for b,r in branches.items()})))
