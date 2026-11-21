from pathlib import Path
import json,hashlib,shutil,datetime
LAB=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for stage,slug in [(42,'source-support'),(43,'source-selection')]:
    out=LAB/f'outputs/dynamic-human-4k4d-{slug}-{stage:03d}';rows=read(out/'summary.json')['records'];assert len(read(out/'independent-audit.json')['records'])==6
    if stage==42:
        images=[f'camera{c}-frame000750-comparison.png' for c in ['00','06','12','01','08','17']]
        observations=['Source-mask gating reduces some dark dotted shoulder and sleeve-edge colors.',
            'Clothing motifs, toy structure, hands and overlapping limbs remain blurred. Point geometry and alpha were fixed, so this intervention cannot repair geometry.',
            'Additional unpremultiplication does not consistently improve on support gating.']
        conclusion='Select source-mask support gate for matched training044; do not claim accepted clarity from inference-only controls.'
        names=['diagnose_4k4d_source_support.py','audit_4k4d_source_support.py']
    else:
        images=[f'camera{c}-frame000750-{v}-comparison.png' for v in ['nearest2_supported','temperature4_supported','hardmax_supported'] for c in ['00','06','12']]
        observations=['Nearest2 occasionally retains stronger shirt markings but gives inconsistent views and patchier limbs/clothes.',
            'Weight power4 does not recover clear hands and introduces stronger patch structure.',
            'Hardmax visibly produces strong mottling on the white shirt and skin. No new geometry or opacity is created.']
        conclusion='Do not adopt nearest2, weightpower4 or hardmax as a clarity fix. Nearest2 improves2/6foregroundmetrics; power4 andhardmax worsenall6. Keep source-support-only candidate for retraining.'
        names=['prepare_4k4d_source_selection.py','diagnose_4k4d_source_selection.py','prepare_4k4d_source_selection_audit.py','audit_4k4d_source_selection.py']
    save(out/'visual-review.json',dict(status='partial_or_negative_result_clarity_unaccepted',images=images,observations=observations,scope='Frozen-model color controls; only listed panels visually reviewed. All6views numerically audited.',goal_complete=False))
    save(out/'final-summary.json',dict(status='frozen_model_source_color_controls_complete_clarity_not_accepted',conclusion=conclusion,
        foreground_psnr_delta_db={r['camera']:{k:v['foreground']-r['metrics']['original']['foreground'] for k,v in r['metrics'].items() if k!='original'} for r in rows},
        limits='Imperfect released target matte; development views already inspected. Foreground support does not establish visibility of the correct surface. All variants use a shared depth ordering and alpha; geometry and pretrained model unchanged.',goal_complete=False))
    snap=out/'source-snapshot';snap.mkdir(exist_ok=True)
    for n in names+['record_4k4d_source_diagnostics.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
    save(out/'source-snapshot.json',dict(files=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(snap.iterdir())]))
mfile=LAB/'manifests/dynamic-human-work.json';m=read(mfile)
m.update(status='source_support_matched_training_running',
    official4k4d_alpha_diagnostic='outputs/dynamic-human-4k4d-alpha-composition-041/final-summary.json',
    official4k4d_source_support='outputs/dynamic-human-4k4d-source-support-042/final-summary.json',
    official4k4d_source_selection='outputs/dynamic-human-4k4d-source-selection-043/final-summary.json',
    current_runs=[dict(name='source_support_matched_training',path='outputs/dynamic-human-4k4d-source-gate-training-044',session_id=71113,parent_updates=2000,new_updates_each=1000,history_updates_each=3000)],
    next_actions=['Finish044 sequential original/support training and6nativeviews each; validate exact parent, matched sampling and final tensors.',
        'Compare044 same-budget real renders against raw photographs; no acceptance fromPSNRalone.',
        'If remaining blur persists, inspect surface geometry and cross-camera moving-feature consistency before temporal/fullscene extension.'],
    pending=['044 paired training/evaluation/audit/visual review','Clear hands, clothing and occlusion','Continuous movement and joint large scene acceptance'],
    remaining_quality_issue='042source-mask gating reduces some dark edge colors and improves6foregroundmetrics, but hands/clothes/occlusion remain blurred.043stronger source selection is inconsistent or worse.',
    goal_turn_classification='Progress:040 rejected;041 exact-fragment alpha difference too small;042/043 all48nativePNGs30panels independently audited,6+9panels visually reviewed.044matched training active.',external_input_blocker=None)
m['completed']+=['041:actualCPUcompositor and14136realfragment samples checked;30PNGs6panels audited,3heldoutpanels viewed. Alpha formula effect<0.0007dB; not blur explanation.',
    '042:18nativePNGs6panels audited and6panels viewed; source-mask gate improvesforeground0.09to0.365dB with partial dark-edge recovery; no clarity acceptance.',
    '043:30nativePNGs24panels audited;9heldoutpanels viewed. Nearest2 inconsistent; weightpower4/hardmax worsenall6foregroundmetrics. Not adopted.']
m['updated']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['last_recorded_utc']=m['updated'];save(mfile,m)
print('042/043 findings recorded;044active')
