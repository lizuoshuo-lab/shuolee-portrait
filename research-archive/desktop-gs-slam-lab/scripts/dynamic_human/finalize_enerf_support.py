"""Record completed support controls, actual inspected samples and bounded conclusions."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil

LAB = Path(__file__).resolve().parents[2]
out = LAB / 'outputs/dynamic-human-enerf-support-030'
def read(name): return json.loads((out / name).read_text())
def save(name, data): (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')
stats = read('paired-statistics.json')
audit = read('independent-image-audit.json')
training = read('training-evaluation/metrics.json')
training_audit = read('independent-training-image-audit.json')
assert audit['images'] == 300 and audit['person_regions'] == 600
assert training_audit['images'] == 36 and training_audit['person_regions'] == 72
assert sum(len(m['checks']) for m in read('independent-state-audit.json')['models']) == 100
viewed = [f'comparison/camera{c}-frame{t:03d}-person{p}.png' for c in ['00','06','12'] for t in [34,46,50,58] for p in [1,2]]
viewed += [f'comparison/regression-camera12-frame{t:03d}-person{p}.png' for t in [43,44,45] for p in [1,2]]
viewed += [f'training-comparison/camera{c}-frame058-person{p}.png' for c in ['01','07','14'] for p in [1,2]]
viewed += [f'evaluation/camera{c}-frame058-both.png' for c in ['00','06','12']]
assert len(viewed) == 39 and all((out / f).is_file() for f in viewed)
review = dict(status='39_actual_images_inspected_clarity_not_accepted', files=viewed,
    heldout_native_comparison_panels=24, worst_regression_adjacent_panels=6, training_late_panels=6, full_final_renders=3,
    observations=[
        'Across all four conditions the torso, face and principal limb outline are recognizable, but moving fingers and held watermelon/panda surfaces remain blended, streaked or broken.',
        'The first shared state already shows hand/object breakup; constrained scale and opacity adaptation does not repair it throughout the subsequent interval.',
        'Six late training-view panels also show hand/object defects, so these failures cannot be assigned solely to unseen camera generalization.',
        'The most adverse paired mean-foreground result is camera12 frame44: both versus fresh frozen is -0.19905dB. Its two people and adjacent43/45 were inspected; differences are small and do not remove the existing failure.',
        'Camera06 full final output retains a small isolated fragment beside the right person and edge artifacts. No deletion or inpainting was applied.'
    ], scope='Native crop PNGs preserve pixels; the viewing tool may downscale wide five-column panels. Full frames were also inspected. Videos were fully decoded but not claimed to have been watched frame-by-frame.',
    enhancement=False, acceptance=False)
save('visual-review.json', review)
next_actions = [
    'Do not repeat the existing025 independent-frame diagnosis as a new result: that6000update fit also failed. Use its evidence plus030 supervised defects to investigate moving-part cross-view correspondence and source matte ghosts.',
    'Use actor5_6 walking/occlusion candidate750 from031; freeze separate calibration, appearance, heldout camera and time protocol before any fitting. No transfer of actor1_4 offsets.',
    'Train and evaluate the032 training-camera-only background seeds in actor5_6 coordinates, preserving raw RGB supervision around people; released mattes are not clean human ground truth.',
    'For the next continuous model, address moving-surface correspondence and adaptive coverage instead of extending the same scale/opacity parameter grid. Compare an appropriate official dynamic reconstruction baseline after auditing its data interface.',
    'Validate person identity through occlusion without assuming two disconnected silhouette components, then shared-coordinate background/person composition. No OmniRe completion claim.'
]
summary = dict(status='four_support_conditions_complete_human_clarity_not_accepted', completed_utc=datetime.now(timezone.utc).isoformat(),
    models=stats['models'], comparisons=[{k:v for k,v in c.items() if k != 'values'} for c in stats['comparisons']],
    training_probe_models=training['summary'], native_heldout_renders=300, native_training_probe_renders=36,
    alpha_images=336, references=84, person_regions=672, audited_states=100, temporal_transitions=288,
    videos=stats['videos'], new_motion_updates=57600, shared_first_frame_updates=6000, discarded_reproduction_updates=600,
    per_condition_history_updates=20400, gaussian_count_each=150543,
    baseline_repeat={k:v for k,v in stats['baseline_repeat'].items() if k != 'pairs'},
    visual_review='visual-review.json', image_audit='independent-image-audit.json', training_image_audit='independent-training-image-audit.json', state_audit='independent-state-audit.json',
    scope='One-seed constrained adaptation on029 development observations. Same75 appearance-heldout photos per model; all18 cameras earlier calibration-visible, no time holdout. Gray foreground with held objects, not a learned complete scene.',
    decision='Small descriptive gain (+0.14054dB both versus fresh frozen), not a sufficient hand/detail improvement or statistical significance claim. Preserve all controls and failures; no acceptance.',
    next_actions=next_actions, goal_complete=False)
save('final-summary.json', summary)
snapshot = out / 'source-snapshot'; snapshot.mkdir(exist_ok=True)
scripts = ['train_enerf_motion.py','run_enerf_support_study.py','audit_enerf_support_reproduction.py','evaluate_enerf_support.py','audit_enerf_support_states.py','audit_enerf_support_images.py','combine_enerf_support_audits.py','compose_enerf_support.py','compose_enerf_support_regression.py','summarize_enerf_support.py','finalize_enerf_support.py']
receipts=[]
for name in scripts:
    src=LAB/'scripts/dynamic_human'/name; target=snapshot/name
    shutil.copy2(src,target); receipts.append(dict(file=target.relative_to(out).as_posix(),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
assert hashlib.sha256((snapshot/'train_enerf_motion.py').read_bytes()).hexdigest() == read('protocol.json')['trainer_sha256']
save('source-snapshot.json', dict(files=receipts, notes='Training source hash matches the pre-run protocol. Supplementary evaluation and reporting source frozen after completed checks.'))
for folder, names in [
    ('dynamic-human-enerf-performer-selection-031',['select_enerf_performer_validation.py','compose_enerf_performer_selection.py','record_enerf_performer_review.py']),
    ('dynamic-human-enerf-background-032',['triangulate_enerf_background.py','audit_enerf_background.py'])]:
    target=out.parent/folder/'source-snapshot'; target.mkdir(exist_ok=True)
    for name in names: shutil.copy2(LAB/'scripts/dynamic_human'/name,target/name)
manifest_path=LAB/'manifests/dynamic-human-work.json'; manifest=json.loads(manifest_path.read_text())
manifest.update(status='support_control_complete_clarity_unaccepted_walking_background_next', updated=summary['completed_utc'], current_runs=[], goal_turn_classification='progress', external_input_blocker=None,
    remaining_quality_issue='Motion hands/toys remain blended and streaked in heldout and supervised views; scale/opacity adaptation gives only+0.14054dB. Walking/background assets prepared but not trained or accepted.',
    outdoor_support_control='outputs/dynamic-human-enerf-support-030/final-summary.json', pending=next_actions, next_actions=next_actions)
for line in [
    'Support030: four14400motion-update continuations from shared6000first state;100states,336RGB/alpha renders672regions288transitions3videos audited;39images inspected. Both support changes+0.14054dB,63/75improve9regress3initialequal; moving fingers remain unclear.',
    'Selection031: six training-camera sheets fromactor2_3/actor5_6 reviewed after preview overflow correction; actor5_6 walking/overlap750 selected, no fitting/heldout inspection.',
    'Background032:6480known-camera training-only actor5_6 background seeds triangulated and independently reprojected. Static geometry only, no background or joint human training yet.'
]:
    if line not in manifest['completed']: manifest['completed'].append(line)
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=summary['status'],reviewed_images=len(viewed),new_motion_updates=57600,goal_complete=False)))
