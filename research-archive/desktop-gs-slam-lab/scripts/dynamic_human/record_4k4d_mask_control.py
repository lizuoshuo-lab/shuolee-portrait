"""Record the completed mask-pipeline experiment and actual visual review."""
from pathlib import Path
import json, hashlib, shutil, datetime
LAB=Path(__file__).resolve().parents[2]
OUT=LAB/'outputs/dynamic-human-4k4d-mask-control-039'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v): p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
paired=read(OUT/'paired-comparison/summary.json')
assert read(OUT/'independent-state-audit.json')['optimizer_max_step']==1000
assert read(OUT/'native-evaluation/independent-image-audit.json')['full_pngs']==24
images=[f'paired-comparison/camera{c}-frame000750-full-person.png' for c in ['00','06','12','01','08','17']]
images+=['paired-comparison/camera01-frame000750-forearm.png']
images+=[f'native-evaluation/camera{c}-frame000750-trained.png' for c in ['00','06','12']]
assert all((OUT/p).is_file() for p in images)
review=dict(status='partial_silhouette_recovery_clarity_not_accepted',images=images,
    observed=['Camera01 forearm is more continuous than the released-mask render, but the hand remains thick and blurred without separated fingers.',
    'Some shoe-edge black artifacts are reduced. Heads and shoulders still have dotted black outlines.',
    'Toys, crossed legs and the blue shirt motif remain blurred or noisy; camera12 toy structure remains broken.',
    'Three complete native held-out images were inspected in addition to six full-person comparisons and one forearm comparison.'],
    scope='Frame750 only; already-inspected development views. No temporal or joint large-scene acceptance.',goal_complete=False)
save(OUT/'visual-review.json',review)
summary=dict(status=review['status'],training_updates=1000,comparison_updates=1000,
    intervention='Whole training-mask pipeline including hull, crop, source premultiplication and target supervision; not a loss-only ablation.',
    heldout_foreground_psnr_delta_db={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in paired['records'] if r['split']=='heldout'},
    heldout_full_psnr_delta_db={r['camera']:r['psnr_delta_db']['full_valid'] for r in paired['records'] if r['split']=='heldout'},
    qualification='Metrics use imperfect released target mattes. Original photographs were also inspected. Input segmentation composites are not reconstruction renders.',
    conclusion='Some forearm and boundary recovery, mixed held-out metrics, remaining hand/clothing/occlusion blur. No clarity acceptance.',
    next='Matched projection-offset control040 from this exact model and Adam state.',goal_complete=False)
save(OUT/'final-summary.json',summary)
snap=OUT/'mask-control-source-snapshot';snap.mkdir(exist_ok=True)
names=['prepare_4k4d_mask_control.py','audit_4k4d_mask_inputs.py','run_4k4d_mask_control.sh','compare_4k4d_masks.py','record_4k4d_mask_control.py']
for n in names:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
save(OUT/'mask-control-source-snapshot.json',dict(files=[dict(file=n,sha256=hashlib.sha256((snap/n).read_bytes()).hexdigest()) for n in names]))
manifest=LAB/'manifests/dynamic-human-work.json';m=read(manifest)
m.update(status='mask_control_complete_projection_pair_running',official4k4d_mask_control='outputs/dynamic-human-4k4d-mask-control-039/final-summary.json',
    current_runs=[dict(name='matched_4k4d_projection_control',path='outputs/dynamic-human-4k4d-projection-control-040',session_id=11675,new_updates_each=1000,history_updates_each=2000,branches=['original_projection','calibrated_projection'])],
    next_actions=['Audit040 matched checkpoints and native renders; inspect paired source photographs and predictions.',
                  'Only after sufficient single-frame quality, extend continuous walking, occlusion and joint scene rendering.'],
    pending=['040 matched projection training/evaluation/audit/visual review','Clear hand, clothing and occlusion reconstruction','Continuous motion and joint large-scene acceptance'],
    goal_turn_classification='Progress:039 completed1000updates; state and6native views independently audited;10images inspected. Partial forearm/boundary recovery only.040 paired control running.',
    remaining_quality_issue='Hands, clothes and occlusions remain blurred. No clear moving-human or joint scene acceptance.',external_input_blocker=None)
m['completed'].append('039:1000update mask-pipeline control completed;6native views24PNGs+12panels independently audited,7paired panels exact,10images viewed. Partial forearm/boundary recovery, mixed heldout metrics; clarity not accepted.')
m['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['last_recorded_utc']=m['updated'];save(manifest,m)
print(json.dumps(summary))
