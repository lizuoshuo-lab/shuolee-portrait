from pathlib import Path
import json,hashlib,shutil
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-alpha-composition-041'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
rows=read(OUT/'real-model-summary.json')['records'];assert len(read(OUT/'independent-audit.json')['records'])==6
review=dict(status='no_visible_clarity_recovery_in_reviewed_views',images=[f'camera{c}-frame000750-comparison.png' for c in ['00','06','12']],
    observations=['Three held-out photo/legacy/standard-alpha panels inspected. Dotted contours, soft fingers, shirt motifs and occluded toy structure remain essentially unchanged.'],
    scope='Sixviews numerically audited;threeheldoutcomparisonpanels visually reviewed. Frozen-model background composition only, not retraining or geometric refinement.',goal_complete=False)
save(OUT/'visual-review.json',review)
save(OUT/'final-summary.json',dict(status='formula_difference_confirmed_but_not_current_blur_explanation',
    legacy_formula='sum_i T_i a_i o_i',standard_formula='sum_i T_i a_i = 1-product_i(1-a_i)',
    point_opacity_quantiles=rows[0]['opacity_quantiles'],max_native_alpha_change=max(r['metrics']['valid']['max_alpha_delta'] for r in rows),
    foreground_psnr_delta_db={r['camera']:r['metrics']['foreground']['after_psnr']-r['metrics']['foreground']['before_psnr'] for r in rows},
    conclusion='All learned point opacities exceed0.9978; actual alpha changes at most0.00206. Frozen-model correction changes foregroundPSNR by less than0.0007dB and does not recover clarity.',
    scope='Official PyTorch3D and differentiable OpenGL contain the same opacity-valued feature. This check does not prove equivalent optimization behavior from fresh initialization.',
    reproduction='Earlier native render is not bit-exact at a few channels. Camera00 has6channels above1u8 andmean1.35e-8RGBdifference. Both alpha variants use exactly the same current rasterization fragments;14136fragment samples independently recomputed.',goal_complete=False))
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
for n in ['probe_4k4d_alpha_composition.py','diagnose_4k4d_alpha_composition.py','audit_4k4d_alpha_composition.py','record_4k4d_alpha_result.py']:shutil.copy2(LAB/'scripts/dynamic_human'/n,snap/n)
save(OUT/'source-snapshot.json',dict(files=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(snap.iterdir())]))
print('041 mathematical and real-model finding recorded; clarity not accepted')
