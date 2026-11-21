from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
LAB=Path(__file__).resolve().parents[2];OUT=LAB/'outputs/dynamic-human-4k4d-source-gate-training-044'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
paired=read(OUT/'paired-comparison/summary.json')
assert read(OUT/'independent-state-audit.json')['matched_recorded_sampling']
reviews={
 '00':'Shoulder and sleeve dark contamination decreases; black-shirt print, face, hand and toy remain blurred or mottled. Dotted silhouettes remain.',
 '06':'Some shoulder coloring improves. Overlapped people, back arm, hanging hand and airborne shoe retain incorrect/dotted surfaces.',
 '12':'Shirt and upper-arm edge colors change mildly. Fingers, black-shirt texture and toy/waist occlusion are unresolved.',
 '01':'Shirt pattern slightly more legible in places; isolated shoulder dots and smeared hands remain.',
 '08':'Small edge color change; hidden actor arm, toy and clothing detail remain inadequate.',
 '17':'Partial shoulder color improvement; isolated light dots and blurred hidden person/toy persist.'}
items=[]
for r in paired['records']:
 p=OUT/'paired-comparison'/r['panel'];items.append(dict(file=str(p.relative_to(OUT)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finding=reviews[r['camera']]))
for c in ['00','06','12']:
 p=OUT/'source_support_blending/native-evaluation'/f'camera{c}-frame000750-trained.png'
 items.append(dict(file=str(p.relative_to(OUT)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),finding='Full native image reviewed: foreground reconstruction only, no full-scene or clear-human acceptance.'))
review=dict(status='six_paired_person_panels_and_three_full_native_renders_visually_reviewed',items=items,
 decision='Retain support gate as a bounded edge-color improvement for further diagnosis, not a clarity solution.',goal_complete=False)
save(OUT/'visual-review.json',review)
summary=dict(status='matched_training_complete_support_gate_modestly_improves_metrics_not_human_clarity',parent_updates=2000,new_updates_per_branch=1000,
 history_updates_per_branch=3000,branches=['original_blending','source_support_blending'],matched_recorded_sampling=True,
 native_render_count=12,full_pngs_verified=48,native_panels_verified=24,paired_panels_verified=6,visually_reviewed_images=9,
 foreground_psnr_delta_db={r['camera']:r['psnr_delta_db']['foreground_matte_gt0_5'] for r in paired['records']},
 finding='All six sampled views improve foreground PSNR; heldout00/06/12 improve0.324/0.248/0.234dB. Qualitative limitation remains on hands, clothing, occlusion and dotted silhouettes.',
 next='Inspect projected point footprint and cross-camera foreground geometry; do not extend to temporal/fullscene before static-frame clarity.',
 qualification='This is an extensively inspected development case, with imperfect released-matte metrics. Not a fresh final test or complete official reproduction.',goal_complete=False)
save(OUT/'final-summary.json',summary)
shutil.copy2(__file__,OUT/'source-snapshot'/Path(__file__).name)
print(json.dumps(summary,ensure_ascii=False))
