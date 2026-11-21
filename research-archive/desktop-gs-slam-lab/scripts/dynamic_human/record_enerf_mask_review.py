"""Record actual visual reviews and verify diagnostic panels against source pixels."""
from pathlib import Path
import json,hashlib,datetime,shutil,numpy as np
from PIL import Image
LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-enerf-mask-quality-038';TRAIN=OUT/'train15';HYBRID=TRAIN/'hybrid-closed-core'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
for r in read(HYBRID/'summary.json')['records']:
    c=r['camera'];rgb=np.array(Image.open(TRAIN/f'camera{c}-source.png'));m=np.array(Image.open(HYBRID/f'camera{c}-mask.png'))>0
    old=np.load(TRAIN/f'camera{c}-released.npy');composite=np.rint(rgb*m[...,None]+127.5*(~m[...,None])).astype(np.uint8)
    assert np.array_equal(composite,np.array(Image.open(HYBRID/f'camera{c}-composite.png')))
    panel=np.array(Image.open(HYBRID/f'camera{c}-comparison.png'));x0,y0,x1,y1=r['bbox'];w=x1-x0
    original_composite=np.rint(rgb.astype(np.float32)*old[...,None]+127.5*(1-old[...,None])).astype(np.uint8)
    for j,array in enumerate([rgb,original_composite,composite]):assert np.array_equal(panel[32:,j*w:(j+1)*w],array[y0:y1,x0:x1])
    checks.append(dict(camera=c,mask_sha256=digest(HYBRID/f'camera{c}-mask.png'),original_photo_and_native_panels_exact=True))
assert len(checks)==15 and {r['camera'] for r in checks}.isdisjoint({'00','06','12'})
save(OUT/'independent-panel-audit.json',dict(status='15_native_segmentation_panels_verified',records=checks,scope='Input preprocessing diagnostics; not rendered reconstruction quality',goal_complete=False))
review=dict(status='bounded_input_candidate_selected_not_ground_truth',reviewed_hybrid_cameras=[r['camera'] for r in checks],
    observed=['Camera01 original photo shows intact forearm; released matte erases visible skin. Four manual interior samples reviewed; ambiguous fifth boundary sample excluded.',
    'One group box drops the dark-clothed person and toys; rejected.',
    'Automatic multipoint prototype retains both people but erases clothing/shoe pixels and a toy; rejected.',
    'Separate person/object prompts recover visible bodies but leave holes at occlusions in several cameras; raw candidates retained.',
    'Nearby-core24px combination adds large wall residues atcamera02; rejected.',
    'Closed-core candidate keeps observed forearm/body interiors and reduces distant wall residues; all15training panels inspected. Selected only for a bounded mask-pipeline training intervention.'],
    remaining=['Hard boundaries are not a softalpha matte','Near-body wall residues remain especially at cameras08/10/14/16 and some shoe edges','No manual pixel-perfect silhouetteGT or temporal mask consistency verified'],
    reconstruction_claim='None: these composites use source photographs, not model renders.',goal_complete=False)
save(OUT/'visual-review.json',review)
save(OUT/'final-summary.json',dict(status='input_false_negative_confirmed_15_candidate_masks_ready_for_bounded_control',
    diagnosis='Missing visible forearm in released camera01matte; strictall15visualhullintersection can carve away geometry supported by other views. Exact removedvolume and effect on finalquality not measured.',
    candidate='OfficialSAM2.1HieraTiny CPU; manual training-only boxes/points; union plus originalalpha>.95within49pxclosed SAM2support.',
    validation='15native panels independently checked and visuallyreviewed; originalfiles preserved, heldoutRGBnot used for candidate masks.',
    next='039same-budget1000update mask-pipeline intervention, unchangedcalibration;037projectioncandidate remains ready but unapplied.',goal_complete=False))
diag=read(OUT/'missing-forearm-diagnostic.json');diag['visual_review']='Confirmed interiorpoints775,495;771,515;767,535;767,555. Exclude ambiguous boundarypoint774,573 from claimedbody evidence.';save(OUT/'missing-forearm-diagnostic.json',diag)
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
names=['setup_sam2_cpu.py','acquire_sam2_checkpoint.py','inspect_enerf_missing_forearm.py','probe_enerf_sam2_mask.py','probe_enerf_sam2_group.py','probe_enerf_sam2_points.py','prepare_enerf_mask_review.py','segment_enerf_train15.py','refine_enerf_mask_candidates.py','record_enerf_mask_review.py']
for n in names:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
save(OUT/'source-snapshot.json',dict(files=[dict(file=n,sha256=digest(snap/n)) for n in names]))
# Completed baseline continuation review: actual6pairedpersonpanels+3fullnativeheldout images.
child=LAB/'outputs/dynamic-human-4k4d-official-035/continuation-10000';paired=read(child/'paired-comparison/summary.json')
review2=dict(status='10000_history_updates_completed_clarity_rejected',
    images=[f'paired-comparison/camera{c}-frame000750-full-person.png' for c in ['00','06','12','01','08','17']]+[f'native-evaluation/camera{c}-frame000750-trained.png' for c in ['00','06','12']],
    observations=['Black dotted silhouettes persist and some shoe outlines become heavier','Hands,toys and crossed lowerlimbs remain soft or broken','Blue shirt motif improves slightly but no photographicdetail acceptance','Target releasedmatte is imperfect and cannot be treated as semanticGT'],
    scope='Single frame750, development cameras already inspected. No temporal or large-scene acceptance.',goal_complete=False)
save(child/'visual-review.json',review2)
save(child/'final-summary.json',dict(status=review2['status'],history_updates=10000,new_updates=9000,
    heldout_foreground_psnr_delta_db={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in paired['records'] if r['split']=='heldout'},
    heldout_full_psnr_delta_db={r['camera']:r['psnr_delta_db']['full_valid'] for r in paired['records'] if r['split']=='heldout'},
    conclusion='Longer fitting does notresolve currentclarity; next input-mask control. ThreeforegroundPSNR smallincreases coexist with lowerfullimagePSNR and persistentartifacts.',goal_complete=False))
manifest=LAB/'manifests/dynamic-human-work.json';m=read(manifest)
m['status']='longer_4k4d_fit_rejected_mask_input_control_running'
m['outdoor_mask_quality']='outputs/dynamic-human-enerf-mask-quality-038/final-summary.json'
m['official4k4d_continuation']='outputs/dynamic-human-4k4d-official-035/continuation-10000/final-summary.json'
m['current_runs']=[dict(name='official4k4d_native_mask_pipeline_control',path='outputs/dynamic-human-4k4d-mask-control-039',session_id=34426,target_updates=1000,evaluation='Same shell runs6native views after completion')]
m['next_actions']=['Observe039session34426; no competingGPUtraining. Audit finalstate and6nativeview outputs, compare to035same1000budget and rawphotos.',
    'If mask control insufficient, use037projectioncandidate in separatelyidentified matchedcomparison; report extraall18camera staticcalibrationaccess.',
    'Only after sufficient singleframequality, extend continuouswalking/occlusion, staticbackgroundjointrendering and OmniReinterface.']
m['pending']=['039training and native evaluation, audit and visual review','Matchedprojectionoffset validation if needed','Continuouswalking,occlusion and joint large-scene quality acceptance']
m['goal_turn_classification']='progress:035continuation10000completedand9imagesreviewed,rejectedclarity;038missingforearmconfirmedand15training-onlyinputmaskcandidatesreviewed;039realtrainingstarted.'
m['remaining_quality_issue']='No accepted clear human/joint scene: baseline10000 retains blurred hands,toys and dotted silhouettes. Input-mask candidate is not a rendered improvement.'
m['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['last_recorded_utc']=m['updated'];save(manifest,m)
print(json.dumps(dict(mask_panels_verified=15,continuation_images_reviewed=9,status='goal_active_quality_unaccepted')))
