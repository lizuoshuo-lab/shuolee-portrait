"""Summarize completed033/034 without converting failed fits into acceptance."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
LAB=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab')
OUT=LAB/'outputs'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
b=OUT/'dynamic-human-enerf-background-fit-033'; r=b/'density-capped'
comp=read(r/'completion.json'); audit=read(r/'independent-audit.json'); metrics=read(r/'evaluation/metrics.json')
assert comp['step']==6000 and comp['points']==532803 and 'verified' in audit['status']
viewed=[f'evaluation/camera{c}-trained.png' for c in ['00','06','12']]
# Full render names are resolved from actual files rather than invented.
files=list((r/'evaluation').glob('*.png'))
for i,c in enumerate(['00','06','12']):
    matches=[p for p in files if p.stem==f'camera{c}-trained']
    if not matches: matches=[p for p in files if f'camera{c}' in p.name and 'trained' in p.name and 'alpha' not in p.name and 'comparison' not in p.name]
    assert len(matches)==1;viewed[i]=matches[0].relative_to(r).as_posix()
viewed += [f'evaluation/camera{c}-wall-comparison.png' for c in ['00','06','12','01','08','17']]
viewed += [f'evaluation/camera{c}-bottom-comparison.png' for c in ['00','06','12']]
for f in viewed:assert (r/f).is_file()
save(r/'visual-review.json',dict(status='not_accepted',images=viewed,actual_inspected_images=len(viewed),
    observations=['Wall grid/patches become recognizable versus sparse initialization, but long anisotropic streaks and blurred texture remain, including sampled training views.',
                  'Heldout edges include poorly reconstructed wall regions; bottom paving texture is smeared or missing.',
                  'High alpha coverage and a PSNR improvement do not establish reliable geometry or sufficient background detail.'],
    enhancement=False,scope='Static empty-scene background only. No person fit, dynamic joint reconstruction or camera correction applied.'))
rows=[dict(camera=x['camera'],split=x['split'],model=x['model'],**x['scores']['full_valid']) for x in metrics['rows']]
save(b/'final-summary.json',dict(status='background_fit_complete_quality_unaccepted',completed_utc=datetime.now(timezone.utc).isoformat(),
    training='Original6480point initialization, stop after2401updates at532803points exceeding500000guard; continue3599updates with fixed support to6000total. Not an unmodified default-density6000run.',
    final_checkpoint_sha256=comp['checkpoint_sha256'],rows=rows,independent_audit='density-capped/independent-audit.json',visual_review='density-capped/visual-review.json',
    decision='Do not freeze this model as an accepted background for subsequent human training.',hardware='No OOM; reported PyTorch peak1.71GB, not total system or process GPU footprint.',goal_complete=False))
d=OUT/'dynamic-human-enerf-matte-background-034'
pose=read(d/'static-pose-diagnostic.json'); floor=read(d/'pose-floor-check.json'); hom=read(d/'static-registration.json')
images=[f'camera08-frame{t:06d}-{kind}.png' for t in [740,750,760] for kind in ['wall-matte-comparison','static-registration']]
for f in images:assert(d/f).is_file()
save(d/'visual-review.json',dict(status='diagnostic_sources_reviewed_no_model_acceptance',images=images,
    additional_source_inspection=['Raw camera08 frame750 RGB and matte; empty background08;031selected actor5_6 camera08 contact sheet'],
    observations=['Real cast shadows exist on wall/floor in the original source.',
                  'Released matte assigns partial opacity to some wall texture away from the body; resulting gray-background composites retain floating wall patches.',
                  'The2Dhomography comparison aligns some static seams better, with residual illumination/microtexture differences. This warped reference is a diagnostic, not a model render.'],
    scope='One training camera and3alreadyselectedframes. Not a measured explanation of every prior actor1_4hand failure.'))
save(d/'final-summary.json',dict(status='matte_shadow_and_static_alignment_diagnosis_complete_not_applied',completed_utc=datetime.now(timezone.utc).isoformat(),camera='08',frames=[740,750,760],
    planar_check=[{k:x[k] for k in ['frame','matches','check_matches','check_identity_median_px','check_H_median_px']} for x in hom['rows']],
    pose_check=[{k:x[k] for k in ['frame','correspondences','check_points','original_check_median_px','pnp_check_median_px','rotation_change_deg','camera_center_change_released_units']} for x in pose['rows']],
    floor_check=floor['checks'],findings=['Source shadows are illumination changes on static surfaces, not necessarily body occupancy.',
        'Released foreground alpha is not semantic human ground truth. Independent static correspondence checks show a few-pixel mismatch between empty reference and current images.',
        'PnP candidates reduce held-correspondence errors; a separate10/11point bottom-band check also improves. Mostlyplanar geometry and the tiny check set limit calibration certainty.'],
    constraints=['No camera, original RGB or matte modified. No warped/cleaned image used as reconstruction result.','No metric scale verified; camera-center changes reported in released units.','Do not extrapolate one-camera diagnostic to whole-sequence calibration or claim it solves motion blur.'],
    next='Use official reconstruction with training-only inputs; separately validate complete camera alignment and time correspondence before adopting changes.',goal_complete=False))
for target,names in [(r,['train_enerf_background.py','enerf_background_data.py','evaluate_enerf_background.py','audit_enerf_background_fit.py','finalize_enerf_background_diagnosis.py']),
                     (d,['inspect_enerf_matte_background.py','inspect_enerf_static_registration.py','inspect_enerf_static_pose.py','check_enerf_pose_floor.py','finalize_enerf_background_diagnosis.py'])]:
    snapshot=target/'source-snapshot';snapshot.mkdir(exist_ok=True);rec=[]
    for n in names:
        src=LAB/'scripts/dynamic_human'/n;dst=snapshot/n;shutil.copy2(src,dst)
        rec.append(dict(path=dst.relative_to(target).as_posix(),sha256=digest(dst)))
    save(target/'source-snapshot.json',dict(files=rec))
mfile=LAB/'manifests/dynamic-human-work.json';m=read(mfile)
m.update(status='background_and_source_diagnosed_official4k4d_pilot_preparing',updated=datetime.now(timezone.utc).isoformat(),goal_turn_classification='progress',external_input_blocker=None,
    remaining_quality_issue='Dynamic hands/clothing remain unclear. Static033background also fails detail review. No new accepted human or fullscene model.',
    outdoor_background_fit='outputs/dynamic-human-enerf-background-fit-033/final-summary.json',outdoor_matte_alignment='outputs/dynamic-human-enerf-matte-background-034/final-summary.json',
    next_actions=['Complete official4K4D train-only native single-frame pilot and inspect heldout person crops before continuous walking training.','Validate moving-person correspondence and calibration with separate observations; no auto-adoption of034PnP.','Extend verified model to short motion window and shared-coordinate background/person composition.'])
m['pending']=m['next_actions']
for line in ['033:6000totalbackground updates afterdensitycaprecovery;12RGB12alpha independentlychecked,12imagesreviewed,qualityunaccepted.',
             '034:trainingcam08frames740/750/760source/matteandstaticregistrationdiagnosis;2DandPnPchecks,10/11pointbottomcheck;nooriginalcameraschanged.']:
    if line not in m['completed']:m['completed'].append(line)
save(mfile,m)
print(json.dumps(dict(background='notaccepted',diagnostics='complete',goal_complete=False)))
